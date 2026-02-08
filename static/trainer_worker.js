/* trainer_worker.js - CPU Transformer trainer + full model download (sampled softmax)
   Runs in a Web Worker to avoid freezing the UI.
*/

importScripts("tokenizer.js");

const CONFIG = {
    downloadFormat: "f16",
    chunkBytes: 8 * 1024 * 1024,

    // training
    seqLen: 32,
    batchSize: 1, // reference impl: 1 (keeps code smaller)
    negSamples: 64, // sampled softmax negatives
    submitEveryMs: 20_000,

    // sparsification
    maxNnzPerTensor: 32_768, // keep packets bounded
    maxNnzEmbedRows: 512, // embedding grads are already sparse by row

    // numerical
    lnEps: 1e-5,
};

function encUtf8(s) {
    return new TextEncoder().encode(s);
}

function fp16_from_fp32(f) {
    const floatView = new Float32Array(1);
    const int32View = new Uint32Array(floatView.buffer);
    floatView[0] = f;
    const x = int32View[0];

    const sign = (x >> 31) & 0x1;
    let exp = ((x >> 23) & 0xff) - 127;
    const mant = x & 0x7fffff;
    const hsign = sign << 15;

    if (((x >> 23) & 0xff) === 0xff) {
        return hsign | (0x1f << 10) | (mant ? 0x200 : 0);
    }
    if (exp > 15) return hsign | (0x1f << 10);
    if (exp >= -14) {
        const hexp = (exp + 15) & 0x1f;
        const mantRounded = mant + 0x00001000;
        let hmant = mantRounded >> 13;
        if (hmant === 0x400) return hsign | ((hexp + 1) << 10);
        return hsign | (hexp << 10) | (hmant & 0x3ff);
    }
    if (exp < -24) return hsign;

    const mantFull = mant | 0x800000;
    const shift = -14 - exp;
    let sub = mantFull >> (shift + 13);
    if ((mantFull >> (shift + 12)) & 1) sub++;
    return hsign | (sub & 0x3ff);
}

function fp32_from_fp16(h) {
    const s = (h & 0x8000) << 16;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;

    let bits = 0;
    if (e === 0) {
        if (m === 0) {
            bits = s;
        } else {
            while ((m & 0x400) === 0) {
                m <<= 1;
                e -= 1;
            }
            e += 1;
            m &= 0x3ff;
            bits = s | ((e + (127 - 15)) << 23) | (m << 13);
        }
    } else if (e === 31) {
        bits = s | 0x7f800000 | (m << 13);
    } else {
        bits = s | ((e + (127 - 15)) << 23) | (m << 13);
    }

    const u32 = new Uint32Array(1);
    const f32 = new Float32Array(u32.buffer);
    u32[0] = bits >>> 0;
    return f32[0];
}

function clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
}

function nowMs() {
    return Date.now();
}

function post(type, payload = {}) {
    self.postMessage({ type, ...payload });
}

// =========================
// DGRD builder (Mode 1 + Mode 3 support; we use Mode 1)
// =========================

function dgrdPacket({ step, nodeId, trainLoss, samples, tensorUpdates, flags }) {
    const nodeBytes = encUtf8(nodeId);

    let bytes =
        4 + // magic
        2 + // version
        2 + // flags
        4 + // step
        4 + // node_id_len
        nodeBytes.length +
        4 + // train_loss
        4 + // samples
        4; // n_tensors

    for (const tu of tensorUpdates) {
        const nnz = tu.indices.length;
        bytes += 4 + 4;
        if (nnz === 0) {
            bytes += tu.values.length * 2;
        } else if (flags === 0) {
            bytes += nnz * (4 + 4);
        } else {
            throw new Error("Mode 2 not implemented in worker");
        }
    }

    const ab = new ArrayBuffer(bytes);
    const dv = new DataView(ab);
    const u8 = new Uint8Array(ab);
    let o = 0;

    u8[o++] = 0x44;
    u8[o++] = 0x47;
    u8[o++] = 0x52;
    u8[o++] = 0x44;

    dv.setUint16(o, 1, true);
    o += 2;
    dv.setUint16(o, flags, true);
    o += 2;
    dv.setUint32(o, step >>> 0, true);
    o += 4;

    dv.setUint32(o, nodeBytes.length >>> 0, true);
    o += 4;
    u8.set(nodeBytes, o);
    o += nodeBytes.length;

    dv.setFloat32(o, trainLoss, true);
    o += 4;
    dv.setUint32(o, samples >>> 0, true);
    o += 4;

    dv.setUint32(o, tensorUpdates.length >>> 0, true);
    o += 4;

    for (const tu of tensorUpdates) {
        const nnz = tu.indices.length;
        dv.setUint32(o, tu.tensorId >>> 0, true);
        o += 4;
        dv.setUint32(o, nnz >>> 0, true);
        o += 4;

        if (nnz === 0) {
            for (let i = 0; i < tu.values.length; i++) {
                dv.setUint16(o, fp16_from_fp32(tu.values[i]), true);
                o += 2;
            }
        } else {
            for (let i = 0; i < nnz; i++) {
                dv.setUint32(o, tu.indices[i] >>> 0, true);
                o += 4;
                dv.setFloat32(o, tu.values[i], true);
                o += 4;
            }
        }
    }

    return ab;
}

// =========================
// API
// =========================

let API_URL = "";
let NODE_ID = "";

function api(path) {
    return `${API_URL}${path}`;
}

async function fetchJson(path) {
    const res = await fetch(api(path), { method: "GET" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
}

async function fetchBuf(path) {
    const res = await fetch(api(path), { method: "GET" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.arrayBuffer();
}

// =========================
// Model storage (FP16) + decoding cache (FP32)
// =========================

const model = {
    info: null,
    manifest: null,
    tensors: new Map(), // id -> { meta, f16: Uint16Array }
    f32cache: new Map(), // id -> Float32Array
    bytesDone: 0,
    bytesTotal: 0,
};

function f32FromTensor(id) {
    const hit = model.f32cache.get(id);
    if (hit) return hit;

    const rec = model.tensors.get(id);
    if (!rec) throw new Error(`tensor ${id} missing`);
    const out = new Float32Array(rec.f16.length);
    for (let i = 0; i < out.length; i++) out[i] = fp32_from_fp16(rec.f16[i]);

    // simple cache cap to avoid unbounded growth
    model.f32cache.set(id, out);
    if (model.f32cache.size > 24) {
        const firstKey = model.f32cache.keys().next().value;
        model.f32cache.delete(firstKey);
    }
    return out;
}

async function downloadFullModel() {
    model.bytesDone = 0;
    model.bytesTotal = 0;

    model.info = await fetchJson("/api/v1/model/info");
    model.manifest = await fetchJson("/api/v1/model/manifest");

    for (const t of model.manifest.tensors) model.bytesTotal += t.bytes_f16;

    post("downloadProgress", {
        done: model.bytesDone,
        total: model.bytesTotal,
    });

    for (const t of model.manifest.tensors) {
        const bytesPerEl = 2;
        const chunkEls = Math.max(
            1,
            Math.floor(CONFIG.chunkBytes / bytesPerEl),
        );
        const totalEls = t.elements >>> 0;
        const out = new Uint16Array(totalEls);

        for (let off = 0; off < totalEls; off += chunkEls) {
            const count = Math.min(chunkEls, totalEls - off);
            const path =
                `/api/v1/model/tensor/${t.id}` +
                `?format=f16&offset=${off}&count=${count}`;
            const buf = await fetchBuf(path);
            out.set(new Uint16Array(buf), off);

            model.bytesDone += buf.byteLength;
            post("downloadProgress", {
                done: model.bytesDone,
                total: model.bytesTotal,
            });

            await new Promise((r) => setTimeout(r, 0));
        }

        model.tensors.set(t.id, { meta: t, f16: out });
        post("log", { msg: `Downloaded tensor [${t.id}] ${t.name}` });
    }

    post("log", { msg: "Full model download complete." });
}

// =========================
// Tensor mapping (adapt if your manifest names differ)
// =========================

function extractLayerIndex(name) {
    const s = String(name || "");

    const patterns = [
        /blocks\.(\d+)/,
        /layers\.(\d+)/,
        /h\.(\d+)/,
        /layer[_\-](\d+)/,
        /(\d+)\.attn/,
    ];

    for (const re of patterns) {
        const m = s.match(re);
        if (m) return Number(m[1]);
    }
    return null;
}

function mapTensors() {
    const cfg = model.info.config;
    const C = cfg.d_model;
    const L = cfg.n_layers;
    const V = cfg.vocab_size;
    const T = cfg.max_seq_len;
    const FF = cfg.d_ff;

    const tensors = model.manifest.tensors;

    function find2D(rows, cols, hint) {
        const cands = tensors.filter(
            (t) =>
                Array.isArray(t.shape) &&
                t.shape.length === 2 &&
                t.shape[0] === rows &&
                t.shape[1] === cols,
        );
        if (!hint) return cands[0] || null;
        const byName = cands.find((t) =>
            String(t.name || "").toLowerCase().includes(hint),
        );
        return byName || cands[0] || null;
    }

    const tokenEmbed =
        tensors.find(
            (t) =>
                Array.isArray(t.shape) &&
                t.shape.length === 2 &&
                t.shape[0] === V &&
                t.shape[1] === C &&
                String(t.name || "").toLowerCase().includes("token"),
        ) || find2D(V, C, null);

    const posEmbed =
        tensors.find(
            (t) =>
                Array.isArray(t.shape) &&
                t.shape.length === 2 &&
                t.shape[0] === T &&
                t.shape[1] === C &&
                String(t.name || "").toLowerCase().includes("pos"),
        ) || find2D(T, C, null);

    if (!tokenEmbed) throw new Error("Could not find token embedding tensor");
    if (!posEmbed) post("log", { msg: "No positional embedding found (warn)" });

    // Layer groups by index
    const layers = Array.from({ length: L }, () => ({}));
    for (const t of tensors) {
        const li = extractLayerIndex(t.name);
        if (li === null || li < 0 || li >= L) continue;
        layers[li].all = layers[li].all || [];
        layers[li].all.push(t);
    }

    // Match per-layer tensors by shapes + name hints.
    function pick1D(li, len, hint) {
        const all = layers[li].all || [];
        const cands = all.filter(
            (t) => Array.isArray(t.shape) && t.shape.length === 1 && t.shape[0] === len,
        );
        if (!hint) return cands[0] || null;
        const byName = cands.find((t) =>
            String(t.name || "").toLowerCase().includes(hint),
        );
        return byName || cands[0] || null;
    }

    function pick2D(li, a, b, hint) {
        const all = layers[li].all || [];
        const cands = all.filter(
            (t) =>
                Array.isArray(t.shape) &&
                t.shape.length === 2 &&
                ((t.shape[0] === a && t.shape[1] === b) ||
                    (t.shape[0] === b && t.shape[1] === a)),
        );
        if (!hint) return cands[0] || null;
        const byName = cands.find((t) =>
            String(t.name || "").toLowerCase().includes(hint),
        );
        return byName || cands[0] || null;
    }

    const blocks = [];
    for (let i = 0; i < L; i++) {
        const ln1w = pick1D(i, C, "ln_1") || pick1D(i, C, "ln1");
        const ln1b = pick1D(i, C, "ln_1.bias") || pick1D(i, C, "ln1.bias");

        const qkvW = pick2D(i, C, 3 * C, "c_attn") || pick2D(i, C, 3 * C, "qkv");
        const qkvB = pick1D(i, 3 * C, "c_attn") || pick1D(i, 3 * C, "qkv");

        const attnPW = pick2D(i, C, C, "c_proj") || pick2D(i, C, C, "attn");
        const attnPB = pick1D(i, C, "c_proj") || pick1D(i, C, "attn");

        const ln2w = pick1D(i, C, "ln_2") || pick1D(i, C, "ln2");
        const ln2b = pick1D(i, C, "ln_2.bias") || pick1D(i, C, "ln2.bias");

        const fcW = pick2D(i, C, FF, "c_fc") || pick2D(i, C, FF, "fc");
        const fcB = pick1D(i, FF, "c_fc") || pick1D(i, FF, "fc");

        const prW = pick2D(i, FF, C, "c_proj") || pick2D(i, FF, C, "proj");
        const prB = pick1D(i, C, "mlp") || pick1D(i, C, "proj");

        if (!qkvW || !attnPW || !fcW || !prW) {
            throw new Error(
                `Layer ${i} mapping failed. Paste /manifest tensors so we can adapt mapTensors().`,
            );
        }

        blocks.push({
            ln1w,
            ln1b,
            qkvW,
            qkvB,
            attnPW,
            attnPB,
            ln2w,
            ln2b,
            fcW,
            fcB,
            prW,
            prB,
        });
    }

    const lnFw =
        tensors.find((t) => String(t.name || "").toLowerCase().includes("ln_f")) ||
        tensors.find((t) => String(t.name || "").toLowerCase().includes("final_ln"));
    const lnFb =
        tensors.find(
            (t) =>
                String(t.name || "").toLowerCase().includes("ln_f") &&
                String(t.name || "").toLowerCase().includes("bias"),
        ) || null;

    if (!lnFw) post("log", { msg: "Final layernorm not found (warn)" });

    post("log", {
        msg: `Mapped tensors: tokenEmbed=${tokenEmbed.id}, posEmbed=${posEmbed ? posEmbed.id : "none"
            }, layers=${blocks.length}`,
    });

    return {
        cfg,
        tokenEmbedId: tokenEmbed.id,
        posEmbedId: posEmbed ? posEmbed.id : null,
        blocks,
        lnFwId: lnFw ? lnFw.id : null,
        lnFbId: lnFb ? lnFb.id : null,
    };
}

// =========================
// CPU kernels (Float32, batch=1)
// =========================

function gelu(x) {
    // tanh approximation
    const a = 0.044715;
    const s = Math.sqrt(2 / Math.PI);
    return 0.5 * x * (1 + Math.tanh(s * (x + a * x * x * x)));
}

function geluBackward(dy, x) {
    // derivative of tanh-approx gelu
    const a = 0.044715;
    const s = Math.sqrt(2 / Math.PI);
    const u = s * (x + a * x * x * x);
    const t = Math.tanh(u);
    const sech2 = 1 - t * t;
    const du = s * (1 + 3 * a * x * x);
    const dx =
        0.5 * (1 + t) * dy + 0.5 * x * sech2 * du * dy;
    return dx;
}

function layerNormForward(x, gamma, beta, T, C, eps) {
    const y = new Float32Array(T * C);
    const mean = new Float32Array(T);
    const rstd = new Float32Array(T);

    for (let t = 0; t < T; t++) {
        let m = 0;
        for (let i = 0; i < C; i++) m += x[t * C + i];
        m /= C;
        mean[t] = m;

        let v = 0;
        for (let i = 0; i < C; i++) {
            const d = x[t * C + i] - m;
            v += d * d;
        }
        v /= C;
        const rs = 1 / Math.sqrt(v + eps);
        rstd[t] = rs;

        for (let i = 0; i < C; i++) {
            const xn = (x[t * C + i] - m) * rs;
            y[t * C + i] = xn * gamma[i] + beta[i];
        }
    }

    return { y, mean, rstd };
}

function layerNormBackward(dy, x, gamma, mean, rstd, T, C) {
    const dx = new Float32Array(T * C);
    const dgamma = new Float32Array(C);
    const dbeta = new Float32Array(C);

    for (let i = 0; i < C; i++) {
        let dg = 0;
        let db = 0;
        for (let t = 0; t < T; t++) {
            const xn = (x[t * C + i] - mean[t]) * rstd[t];
            dg += dy[t * C + i] * xn;
            db += dy[t * C + i];
        }
        dgamma[i] = dg;
        dbeta[i] = db;
    }

    for (let t = 0; t < T; t++) {
        let sumDxhat = 0;
        let sumDxhatXhat = 0;

        for (let i = 0; i < C; i++) {
            const xhat = (x[t * C + i] - mean[t]) * rstd[t];
            const dxhat = dy[t * C + i] * gamma[i];
            sumDxhat += dxhat;
            sumDxhatXhat += dxhat * xhat;
        }

        for (let i = 0; i < C; i++) {
            const xhat = (x[t * C + i] - mean[t]) * rstd[t];
            const dxhat = dy[t * C + i] * gamma[i];
            const v =
                (C * dxhat - sumDxhat - xhat * sumDxhatXhat) * (rstd[t] / C);
            dx[t * C + i] = v;
        }
    }

    return { dx, dgamma, dbeta };
}

// =========================
// Linear / Matmul helpers (supports W as [in,out] or [out,in])
// =========================

function linearForward(x, W, b, T, inDim, outDim, Wrows, Wcols) {
    const y = new Float32Array(T * outDim);

    const wInOut = Wrows === inDim && Wcols === outDim;
    const wOutIn = Wrows === outDim && Wcols === inDim;
    if (!wInOut && !wOutIn) {
        throw new Error(
            `linearForward bad W shape: got [${Wrows},${Wcols}] want ` +
            `[${inDim},${outDim}] or [${outDim},${inDim}]`,
        );
    }

    for (let t = 0; t < T; t++) {
        for (let j = 0; j < outDim; j++) {
            let s = b ? b[j] : 0;
            if (wInOut) {
                // y = x @ W
                for (let i = 0; i < inDim; i++) {
                    s += x[t * inDim + i] * W[i * outDim + j];
                }
            } else {
                // W is [out,in], use transpose: y = x @ W^T
                for (let i = 0; i < inDim; i++) {
                    s += x[t * inDim + i] * W[j * inDim + i];
                }
            }
            y[t * outDim + j] = s;
        }
    }

    return y;
}

function linearBackward(dy, x, W, T, inDim, outDim, Wrows, Wcols) {
    const dx = new Float32Array(T * inDim);
    const dW = new Float32Array(W.length);
    const db = new Float32Array(outDim);

    const wInOut = Wrows === inDim && Wcols === outDim;
    const wOutIn = Wrows === outDim && Wcols === inDim;
    if (!wInOut && !wOutIn) {
        throw new Error(
            `linearBackward bad W shape: got [${Wrows},${Wcols}] want ` +
            `[${inDim},${outDim}] or [${outDim},${inDim}]`,
        );
    }

    for (let t = 0; t < T; t++) {
        for (let j = 0; j < outDim; j++) {
            const g = dy[t * outDim + j];
            db[j] += g;

            if (wInOut) {
                for (let i = 0; i < inDim; i++) {
                    dx[t * inDim + i] += g * W[i * outDim + j];
                    dW[i * outDim + j] += x[t * inDim + i] * g;
                }
            } else {
                for (let i = 0; i < inDim; i++) {
                    dx[t * inDim + i] += g * W[j * inDim + i];
                    dW[j * inDim + i] += x[t * inDim + i] * g;
                }
            }
        }
    }

    return { dx, dW, db };
}

// =========================
// Attention (causal) forward/backward (batch=1)
// =========================

function softmaxCausal(scores, T) {
    // scores: Float32Array(T*T) row-major [t,s]
    const probs = new Float32Array(T * T);

    for (let t = 0; t < T; t++) {
        let m = -Infinity;
        for (let s = 0; s <= t; s++) {
            const v = scores[t * T + s];
            if (v > m) m = v;
        }
        let sum = 0;
        for (let s = 0; s <= t; s++) {
            const e = Math.exp(scores[t * T + s] - m);
            probs[t * T + s] = e;
            sum += e;
        }
        const inv = 1 / Math.max(1e-12, sum);
        for (let s = 0; s <= t; s++) probs[t * T + s] *= inv;
    }

    return probs;
}

function mhaForward(x, params, cfg, T) {
    const C = cfg.d_model;
    const H = cfg.n_heads;
    const Hd = C / H;
    const scale = 1 / Math.sqrt(Hd);

    const Wqkv = f32FromTensor(params.qkvW.id);
    const bqkv = params.qkvB ? f32FromTensor(params.qkvB.id) : null;

    const Wp = f32FromTensor(params.attnPW.id);
    const bp = params.attnPB ? f32FromTensor(params.attnPB.id) : null;

    const qkv = linearForward(
        x,
        Wqkv,
        bqkv,
        T,
        C,
        3 * C,
        params.qkvW.shape[0],
        params.qkvW.shape[1],
    );

    const q = new Float32Array(T * C);
    const k = new Float32Array(T * C);
    const v = new Float32Array(T * C);

    for (let t = 0; t < T; t++) {
        const base = t * 3 * C;
        q.set(qkv.subarray(base, base + C), t * C);
        k.set(qkv.subarray(base + C, base + 2 * C), t * C);
        v.set(qkv.subarray(base + 2 * C, base + 3 * C), t * C);
    }

    // Per-head attention: store probs for backward
    const probsAll = new Float32Array(H * T * T);
    const ctx = new Float32Array(T * C);

    for (let h = 0; h < H; h++) {
        const scores = new Float32Array(T * T);

        for (let t = 0; t < T; t++) {
            for (let s = 0; s <= t; s++) {
                let dot = 0;
                const qt = (t * C + h * Hd) | 0;
                const ks = (s * C + h * Hd) | 0;
                for (let i = 0; i < Hd; i++) dot += q[qt + i] * k[ks + i];
                scores[t * T + s] = dot * scale;
            }
            for (let s = t + 1; s < T; s++) scores[t * T + s] = -1e9;
        }

        const probs = softmaxCausal(scores, T);
        probsAll.set(probs, h * T * T);

        // ctx[t,hd] = sum_s probs[t,s] * v[s,hd]
        for (let t = 0; t < T; t++) {
            const outBase = t * C + h * Hd;
            for (let i = 0; i < Hd; i++) {
                let ssum = 0;
                for (let s = 0; s <= t; s++) {
                    const vb = s * C + h * Hd;
                    ssum += probs[t * T + s] * v[vb + i];
                }
                ctx[outBase + i] = ssum;
            }
        }
    }

    const y = linearForward(
        ctx,
        Wp,
        bp,
        T,
        C,
        C,
        params.attnPW.shape[0],
        params.attnPW.shape[1],
    );

    return { y, cache: { x, q, k, v, ctx, probsAll } };
}

function mhaBackward(dy, params, cfg, T, cache) {
    const C = cfg.d_model;
    const H = cfg.n_heads;
    const Hd = C / H;
    const scale = 1 / Math.sqrt(Hd);

    const Wqkv = f32FromTensor(params.qkvW.id);
    const Wp = f32FromTensor(params.attnPW.id);

    // Back through proj: ctx -> dy
    const bProj = linearBackward(
        dy,
        cache.ctx,
        Wp,
        T,
        C,
        C,
        params.attnPW.shape[0],
        params.attnPW.shape[1],
    );

    const dCtx = bProj.dx;
    const dWp = bProj.dW;
    const dbp = bProj.db;

    const dq = new Float32Array(T * C);
    const dk = new Float32Array(T * C);
    const dv = new Float32Array(T * C);

    // For each head, backprop attention
    for (let h = 0; h < H; h++) {
        const probs = cache.probsAll.subarray(h * T * T, (h + 1) * T * T);

        // dp[t,s] = sum_i dCtx[t,i] * v[s,i]
        const dp = new Float32Array(T * T);
        for (let t = 0; t < T; t++) {
            const dcb = t * C + h * Hd;
            for (let s = 0; s <= t; s++) {
                const vb = s * C + h * Hd;
                let acc = 0;
                for (let i = 0; i < Hd; i++) acc += dCtx[dcb + i] * cache.v[vb + i];
                dp[t * T + s] = acc;
            }
        }

        // dv[s,i] += sum_t probs[t,s] * dCtx[t,i]
        for (let s = 0; s < T; s++) {
            const vb = s * C + h * Hd;
            for (let i = 0; i < Hd; i++) {
                let acc = 0;
                for (let t = s; t < T; t++) {
                    const dcb = t * C + h * Hd;
                    acc += probs[t * T + s] * dCtx[dcb + i];
                }
                dv[vb + i] += acc;
            }
        }

        // dscores from softmax Jacobian (causal row-wise)
        const dScores = new Float32Array(T * T);
        for (let t = 0; t < T; t++) {
            let sum = 0;
            for (let s = 0; s <= t; s++) sum += dp[t * T + s] * probs[t * T + s];
            for (let s = 0; s <= t; s++) {
                dScores[t * T + s] = probs[t * T + s] * (dp[t * T + s] - sum);
            }
        }

        // scores[t,s] = (q[t]·k[s]) * scale
        // dq[t] += sum_s dScores[t,s] * k[s] * scale
        // dk[s] += sum_t dScores[t,s] * q[t] * scale
        for (let t = 0; t < T; t++) {
            const qb = t * C + h * Hd;
            for (let s = 0; s <= t; s++) {
                const kb = s * C + h * Hd;
                const g = dScores[t * T + s] * scale;
                for (let i = 0; i < Hd; i++) {
                    dq[qb + i] += g * cache.k[kb + i];
                    dk[kb + i] += g * cache.q[qb + i];
                }
            }
        }
    }

    // Pack dq,dk,dv into dqkv then back through qkv linear
    const dqkv = new Float32Array(T * 3 * C);
    for (let t = 0; t < T; t++) {
        const base = t * 3 * C;
        dqkv.set(dq.subarray(t * C, (t + 1) * C), base);
        dqkv.set(dk.subarray(t * C, (t + 1) * C), base + C);
        dqkv.set(dv.subarray(t * C, (t + 1) * C), base + 2 * C);
    }

    const bQkv = linearBackward(
        dqkv,
        cache.x,
        Wqkv,
        T,
        C,
        3 * C,
        params.qkvW.shape[0],
        params.qkvW.shape[1],
    );

    return {
        dx: bQkv.dx,
        grads: {
            qkvW: bQkv.dW,
            qkvB: bQkv.db,
            attnPW: dWp,
            attnPB: dbp,
        },
    };
}

// =========================
// Transformer block forward/backward (GPT-2 style)
// =========================

function add2(a, b) {
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
    return out;
}

function addInPlace(dst, src) {
    for (let i = 0; i < dst.length; i++) dst[i] += src[i];
}

function mlpForward(x, params, cfg, T) {
    const C = cfg.d_model;
    const FF = cfg.d_ff;

    const Wfc = f32FromTensor(params.fcW.id);
    const bfc = params.fcB ? f32FromTensor(params.fcB.id) : null;
    const Wpr = f32FromTensor(params.prW.id);
    const bpr = params.prB ? f32FromTensor(params.prB.id) : null;

    const h = linearForward(
        x,
        Wfc,
        bfc,
        T,
        C,
        FF,
        params.fcW.shape[0],
        params.fcW.shape[1],
    );

    const g = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) g[i] = gelu(h[i]);

    const y = linearForward(
        g,
        Wpr,
        bpr,
        T,
        FF,
        C,
        params.prW.shape[0],
        params.prW.shape[1],
    );

    return { y, cache: { x, h, g } };
}

function mlpBackward(dy, params, cfg, T, cache) {
    const C = cfg.d_model;
    const FF = cfg.d_ff;

    const Wfc = f32FromTensor(params.fcW.id);
    const Wpr = f32FromTensor(params.prW.id);

    const bPr = linearBackward(
        dy,
        cache.g,
        Wpr,
        T,
        FF,
        C,
        params.prW.shape[0],
        params.prW.shape[1],
    );

    const dg = bPr.dx;
    const dWpr = bPr.dW;
    const dbpr = bPr.db;

    const dh = new Float32Array(cache.h.length);
    for (let i = 0; i < dh.length; i++) dh[i] = geluBackward(dg[i], cache.h[i]);

    const bFc = linearBackward(
        dh,
        cache.x,
        Wfc,
        T,
        C,
        FF,
        params.fcW.shape[0],
        params.fcW.shape[1],
    );

    return {
        dx: bFc.dx,
        grads: {
            fcW: bFc.dW,
            fcB: bFc.db,
            prW: dWpr,
            prB: dbpr,
        },
    };
}

function blockForward(x, block, cfg, T) {
    const C = cfg.d_model;

    const ln1w = block.ln1w ? f32FromTensor(block.ln1w.id) : new Float32Array(C).fill(1);
    const ln1b = block.ln1b ? f32FromTensor(block.ln1b.id) : new Float32Array(C);

    const ln1 = layerNormForward(x, ln1w, ln1b, T, C, CONFIG.lnEps);
    const attn = mhaForward(ln1.y, block, cfg, T);
    const x2 = add2(x, attn.y);

    const ln2w = block.ln2w ? f32FromTensor(block.ln2w.id) : new Float32Array(C).fill(1);
    const ln2b = block.ln2b ? f32FromTensor(block.ln2b.id) : new Float32Array(C);

    const ln2 = layerNormForward(x2, ln2w, ln2b, T, C, CONFIG.lnEps);
    const mlp = mlpForward(ln2.y, block, cfg, T);
    const out = add2(x2, mlp.y);

    return {
        out,
        cache: {
            x,
            x2,
            ln1,
            ln2,
            attn,
            mlp,
            ln1w,
            ln2w,
        },
    };
}

function blockBackward(dout, block, cfg, T, cache) {
    const C = cfg.d_model;

    // out = x2 + mlp
    const dX2 = dout.slice();
    const dMlp = dout.slice();

    const bMlp = mlpBackward(dMlp, block, cfg, T, cache.mlp.cache);

    // ln2 backward
    const ln2w = block.ln2w ? f32FromTensor(block.ln2w.id) : cache.ln2w;
    const bLn2 = layerNormBackward(
        bMlp.dx,
        cache.x2,
        ln2w,
        cache.ln2.mean,
        cache.ln2.rstd,
        T,
        C,
    );

    addInPlace(dX2, bLn2.dx);

    // x2 = x + attn
    const dX = dX2.slice();
    const dAttnOut = dX2.slice();

    const bAttn = mhaBackward(dAttnOut, block, cfg, T, cache.attn.cache);

    // ln1 backward
    const ln1w = block.ln1w ? f32FromTensor(block.ln1w.id) : cache.ln1w;
    const bLn1 = layerNormBackward(
        bAttn.dx,
        cache.x,
        ln1w,
        cache.ln1.mean,
        cache.ln1.rstd,
        T,
        C,
    );

    addInPlace(dX, bLn1.dx);

    const grads = {
        // LN
        ln1w: bLn1.dgamma,
        ln1b: bLn1.dbeta,
        ln2w: bLn2.dgamma,
        ln2b: bLn2.dbeta,

        // attn + mlp
        ...bAttn.grads,
        ...bMlp.grads,
    };

    return { dx: dX, grads };
}

// =========================
// Embeddings + sampled softmax head
// =========================

function hash32(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    return h >>> 0;
}

// NOTE: Replace with real GPT-2 BPE tokenizer for true LLM training.
// This fallback is deterministic but NOT semantically correct.
function tokenizeFallback(text, vocabSize) {
    const toks = text.toLowerCase().match(/[a-z0-9]+|[^\s]/g) || [];
    const ids = new Uint32Array(toks.length);
    for (let i = 0; i < toks.length; i++) {
        ids[i] = (hash32(toks[i]) % vocabSize) >>> 0;
    }
    return ids;
}

function gatherEmbedRows(embed, ids, C) {
    // embed: Float32Array(V*C)
    const T = ids.length;
    const out = new Float32Array(T * C);
    for (let t = 0; t < T; t++) {
        const tok = ids[t] >>> 0;
        const base = tok * C;
        for (let i = 0; i < C; i++) out[t * C + i] = embed[base + i];
    }
    return out;
}

function addPosEmbed(x, posEmbed, T, C) {
    if (!posEmbed) return x;
    const out = new Float32Array(x.length);
    for (let t = 0; t < T; t++) {
        const pb = t * C;
        for (let i = 0; i < C; i++) out[t * C + i] = x[t * C + i] + posEmbed[pb + i];
    }
    return out;
}

function sampledSoftmaxLossAndGrad(h, targetIds, embed, cfg) {
    // h: Float32Array(T*C) hidden states
    // targetIds: Uint32Array(T) next-token targets aligned with h positions
    // embed: Float32Array(V*C) tied output weights (token embedding)
    const C = cfg.d_model;
    const V = cfg.vocab_size;
    const T = targetIds.length;

    const dh = new Float32Array(T * C);
    const dEmbedSparse = new Map(); // token -> Float32Array(C)
    let lossSum = 0;
    let count = 0;

    function addRowGrad(tok, g) {
        const ex = dEmbedSparse.get(tok);
        if (!ex) {
            dEmbedSparse.set(tok, g);
            return;
        }
        for (let i = 0; i < C; i++) ex[i] += g[i];
    }

    for (let t = 0; t < T; t++) {
        const y = targetIds[t] >>> 0;

        // sample negatives
        const toks = [y];
        while (toks.length < 1 + CONFIG.negSamples) {
            const n = (Math.floor(Math.random() * V) >>> 0);
            if (n === y) continue;
            toks.push(n);
        }

        // logits
        const logits = new Float32Array(toks.length);
        let max = -Infinity;

        const hb = t * C;
        for (let k = 0; k < toks.length; k++) {
            const tok = toks[k];
            const wb = tok * C;
            let s = 0;
            for (let i = 0; i < C; i++) s += h[hb + i] * embed[wb + i];
            logits[k] = s;
            if (s > max) max = s;
        }

        // softmax
        let sum = 0;
        for (let k = 0; k < logits.length; k++) {
            const e = Math.exp(logits[k] - max);
            logits[k] = e;
            sum += e;
        }
        const inv = 1 / Math.max(1e-12, sum);
        for (let k = 0; k < logits.length; k++) logits[k] *= inv;

        const pTrue = logits[0];
        lossSum += -Math.log(Math.max(1e-12, pTrue));
        count++;

        // dz = p - y
        for (let k = 0; k < toks.length; k++) {
            const tok = toks[k];
            const dz = logits[k] - (k === 0 ? 1 : 0);
            const wb = tok * C;

            // dh += dz * w_tok
            for (let i = 0; i < C; i++) dh[hb + i] += dz * embed[wb + i];

            // dW_tok += dz * h
            const g = new Float32Array(C);
            for (let i = 0; i < C; i++) g[i] = dz * h[hb + i];
            addRowGrad(tok, g);
        }
    }

    return {
        loss: lossSum / Math.max(1, count),
        dh,
        dEmbedSparse,
        samples: count,
    };
}

// =========================
// Gradient collection + sparsification -> DGRD updates
// =========================

function topKIndicesAbs(g, K) {
    // min-heap of [abs, idx]
    const heapAbs = new Float32Array(K);
    const heapIdx = new Uint32Array(K);
    let size = 0;

    function swap(a, b) {
        const ta = heapAbs[a];
        heapAbs[a] = heapAbs[b];
        heapAbs[b] = ta;
        const ti = heapIdx[a];
        heapIdx[a] = heapIdx[b];
        heapIdx[b] = ti;
    }

    function up(i) {
        while (i > 0) {
            const p = ((i - 1) / 2) | 0;
            if (heapAbs[p] <= heapAbs[i]) break;
            swap(p, i);
            i = p;
        }
    }

    function down(i) {
        while (true) {
            const l = i * 2 + 1;
            const r = l + 1;
            let m = i;
            if (l < size && heapAbs[l] < heapAbs[m]) m = l;
            if (r < size && heapAbs[r] < heapAbs[m]) m = r;
            if (m === i) break;
            swap(i, m);
            i = m;
        }
    }

    for (let i = 0; i < g.length; i++) {
        const a = Math.abs(g[i]);
        if (a === 0) continue;

        if (size < K) {
            heapAbs[size] = a;
            heapIdx[size] = i >>> 0;
            up(size);
            size++;
            continue;
        }

        if (a <= heapAbs[0]) continue;
        heapAbs[0] = a;
        heapIdx[0] = i >>> 0;
        down(0);
    }

    const out = new Uint32Array(size);
    for (let i = 0; i < size; i++) out[i] = heapIdx[i];
    out.sort();
    return out;
}

function buildSparseUpdateFromDenseGrad(tensorId, g, maxNnz) {
    const idx = topKIndicesAbs(g, maxNnz);
    const values = new Float32Array(idx.length);
    for (let i = 0; i < idx.length; i++) values[i] = g[idx[i]];
    return { tensorId, indices: idx, values };
}

function buildSparseUpdateForEmbedRows(tensorId, rowGrads, C, maxRows) {
    const rows = Array.from(rowGrads.keys());
    if (rows.length > maxRows) rows.length = maxRows;

    rows.sort((a, b) => a - b);

    const nnz = rows.length * C;
    const indices = new Uint32Array(nnz);
    const values = new Float32Array(nnz);

    let p = 0;
    for (const tok of rows) {
        const g = rowGrads.get(tok);
        const base = tok * C;
        for (let i = 0; i < C; i++) {
            indices[p] = (base + i) >>> 0;
            values[p] = g[i];
            p++;
        }
    }

    return { tensorId, indices, values };
}

// =========================
// Training loop (worker)
// =========================

let running = false;

async function submitPacket(tensorUpdates, trainLoss, samples) {
    const info = await fetchJson("/api/v1/model/info");
    const step = info.step >>> 0;

    const packet = dgrdPacket({
        step,
        nodeId: NODE_ID,
        trainLoss,
        samples,
        tensorUpdates,
        flags: 0,
    });

    const res = await fetch(api("/api/v1/train/submit"), {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: packet,
    });

    const j = await res.json().catch(() => null);
    if (res.status === 409) {
        post("log", {
            msg: `409 step mismatch (server_step=${j?.server_step}). Will continue.`,
        });
        return { ok: false, server_step: j?.server_step };
    }
    if (!res.ok) {
        throw new Error(
            `submit failed: ${res.status} ${res.statusText} ${JSON.stringify(j)}`,
        );
    }
    return { ok: true, server_step: j?.server_step };
}

async function runTraining(text) {
    running = true;

    await downloadFullModel();
    const mapped = mapTensors();
    const cfg = mapped.cfg;
    const C = cfg.d_model;

    const tokenEmbed = f32FromTensor(mapped.tokenEmbedId);
    const posEmbed = mapped.posEmbedId ? f32FromTensor(mapped.posEmbedId) : null;

    if (!Tokenizer.ready) {
        post("log", { msg: "Worker: Loading tokenizer assets..." });
        await Tokenizer.load("vocab.json", "merges.txt");
    }

    const idsAll = Tokenizer.encode(text);
    if (idsAll.length < CONFIG.seqLen + 1) {
        throw new Error(
            `Need at least ${CONFIG.seqLen + 1} tokens; got ${idsAll.length}`,
        );
    }

    post("log", {
        msg: "Worker: Using real GPT-2 BPE tokenizer.",
    });

    let lastSubmit = nowMs();
    let lossAcc = 0;
    let sampleAcc = 0;

    while (running) {
        const start = Math.floor(
            Math.random() * (idsAll.length - (CONFIG.seqLen + 1)),
        );
        const input = idsAll.subarray(start, start + CONFIG.seqLen);
        const target = idsAll.subarray(start + 1, start + 1 + CONFIG.seqLen);

        const xTok = gatherEmbedRows(tokenEmbed, input, C);
        const x0 = addPosEmbed(xTok, posEmbed, CONFIG.seqLen, C);

        let x = x0;
        const blockCaches = [];
        for (let li = 0; li < mapped.blocks.length; li++) {
            const b = mapped.blocks[li];

            const params = {
                ...b,
                qkvW: { id: b.qkvW.id, shape: b.qkvW.shape },
                qkvB: b.qkvB ? { id: b.qkvB.id } : null,
                attnPW: { id: b.attnPW.id, shape: b.attnPW.shape },
                attnPB: b.attnPB ? { id: b.attnPB.id } : null,
                fcW: { id: b.fcW.id, shape: b.fcW.shape },
                fcB: b.fcB ? { id: b.fcB.id } : null,
                prW: { id: b.prW.id, shape: b.prW.shape },
                prB: b.prB ? { id: b.prB.id } : null,
            };

            params.qkvW.shape = model.tensors.get(b.qkvW.id).meta.shape;
            params.attnPW.shape = model.tensors.get(b.attnPW.id).meta.shape;
            params.fcW.shape = model.tensors.get(b.fcW.id).meta.shape;
            params.prW.shape = model.tensors.get(b.prW.id).meta.shape;

            const f = blockForward(x, params, cfg, CONFIG.seqLen);
            x = f.out;
            blockCaches.push({ params, cache: f.cache, block: b });
        }

        let h = x;
        let lnFCache = null;
        if (mapped.lnFwId) {
            const lnFw = f32FromTensor(mapped.lnFwId);
            const lnFb = mapped.lnFbId ? f32FromTensor(mapped.lnFbId) : new Float32Array(C);
            const lnF = layerNormForward(h, lnFw, lnFb, CONFIG.seqLen, C, CONFIG.lnEps);
            lnFCache = { x: h, lnF, lnFw };
            h = lnF.y;
        }

        const head = sampledSoftmaxLossAndGrad(h, target, tokenEmbed, cfg);

        let dh = head.dh;

        const gradByTensor = new Map();
        const addDenseGrad = (tid, g) => {
            if (!tid || !g) return;
            const ex = gradByTensor.get(tid);
            if (!ex) gradByTensor.set(tid, g);
            else {
                for (let i = 0; i < ex.length; i++) ex[i] += g[i];
            }
        };

        if (lnFCache) {
            const bLnF = layerNormBackward(
                dh,
                lnFCache.x,
                lnFCache.lnFw,
                lnFCache.lnF.mean,
                lnFCache.lnF.rstd,
                CONFIG.seqLen,
                C,
            );
            dh = bLnF.dx;
            addDenseGrad(mapped.lnFwId, bLnF.dgamma);
            if (mapped.lnFbId) addDenseGrad(mapped.lnFbId, bLnF.dbeta);
        }

        for (let li = blockCaches.length - 1; li >= 0; li--) {
            const bc = blockCaches[li];
            const b = bc.block;
            const params = bc.params;

            params.ln1w = b.ln1w ? { id: b.ln1w.id } : null;
            params.ln1b = b.ln1b ? { id: b.ln1b.id } : null;
            params.ln2w = b.ln2w ? { id: b.ln2w.id } : null;
            params.ln2b = b.ln2b ? { id: b.ln2b.id } : null;

            const bb = blockBackward(dh, params, cfg, CONFIG.seqLen, bc.cache);
            dh = bb.dx;

            if (b.ln1w) addDenseGrad(b.ln1w.id, bb.grads.ln1w);
            if (b.ln1b) addDenseGrad(b.ln1b.id, bb.grads.ln1b);
            if (b.ln2w) addDenseGrad(b.ln2w.id, bb.grads.ln2w);
            if (b.ln2b) addDenseGrad(b.ln2b.id, bb.grads.ln2b);

            addDenseGrad(b.qkvW.id, bb.grads.qkvW);
            if (b.qkvB) addDenseGrad(b.qkvB.id, bb.grads.qkvB);
            addDenseGrad(b.attnPW.id, bb.grads.attnPW);
            if (b.attnPB) addDenseGrad(b.attnPB.id, bb.grads.attnPB);

            addDenseGrad(b.fcW.id, bb.grads.fcW);
            if (b.fcB) addDenseGrad(b.fcB.id, bb.grads.fcB);
            addDenseGrad(b.prW.id, bb.grads.prW);
            if (b.prB) addDenseGrad(b.prB.id, bb.grads.prB);
        }

        const dTokenRows = new Map(head.dEmbedSparse);

        for (let t = 0; t < CONFIG.seqLen; t++) {
            const tok = input[t] >>> 0;
            const g = dTokenRows.get(tok) || new Float32Array(C);
            for (let i = 0; i < C; i++) g[i] += dh[t * C + i];
            dTokenRows.set(tok, g);
        }

        let dPosRows = null;
        if (mapped.posEmbedId) {
            dPosRows = new Map();
            for (let t = 0; t < CONFIG.seqLen; t++) {
                const g = new Float32Array(C);
                for (let i = 0; i < C; i++) g[i] = dh[t * C + i];
                dPosRows.set(t, g);
            }
        }

        const tensorUpdates = [];

        tensorUpdates.push(
            buildSparseUpdateForEmbedRows(
                mapped.tokenEmbedId,
                dTokenRows,
                C,
                CONFIG.maxNnzEmbedRows,
            ),
        );

        if (mapped.posEmbedId && dPosRows) {
            tensorUpdates.push(
                buildSparseUpdateForEmbedRows(
                    mapped.posEmbedId,
                    dPosRows,
                    C,
                    CONFIG.seqLen,
                ),
            );
        }

        for (const [tid, g] of gradByTensor.entries()) {
            if (!g || g.length === 0) continue;
            const K = clamp(CONFIG.maxNnzPerTensor, 1, g.length);
            tensorUpdates.push(buildSparseUpdateFromDenseGrad(tid, g, K));
        }

        lossAcc += head.loss;
        sampleAcc += head.samples;

        post("trainStep", {
            loss: head.loss,
            samples: head.samples,
        });

        if (nowMs() - lastSubmit >= CONFIG.submitEveryMs) {
            const avgLoss = lossAcc / Math.max(1, sampleAcc);
            await submitPacket(tensorUpdates, avgLoss, sampleAcc);

            post("log", {
                msg: `Submitted: tensors=${tensorUpdates.length} avgLoss=${avgLoss.toFixed(
                    4,
                )} samples=${sampleAcc}`,
            });

            lossAcc = 0;
            sampleAcc = 0;
            lastSubmit = nowMs();
        }

        await new Promise((r) => setTimeout(r, 0));
    }
}

// =========================
// Worker messages
// =========================

self.onmessage = async (e) => {
    const { type, apiUrl, nodeId, text } = e.data || {};

    try {
        if (type === "start") {
            API_URL = apiUrl;
            NODE_ID = nodeId;
            post("log", { msg: `Worker start: api=${API_URL} node=${NODE_ID}` });
            await runTraining(text);
        } else if (type === "stop") {
            running = false;
            post("log", { msg: "Worker stop requested." });
        } else if (type === "download") {
            API_URL = apiUrl;
            NODE_ID = nodeId;
            await downloadFullModel();
        }
    } catch (err) {
        post("error", { msg: err?.message || String(err) });
    }
};
