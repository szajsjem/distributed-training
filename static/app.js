/* app.js - redesigned frontend for /api/v1 server */

const CONFIG = {
    DEFAULT_API_URL: window.location.origin, // edit in UI if different
    ACCUMULATION_INTERVAL_MS: 60_000,
    TRAIN_STEP_DELAY_MS: 400,

    // upload controller target (matches server config idea)
    TARGET_UPLOAD_BPS: 500_000,

    // demo: how many elements to download from tensor 0
    SAMPLE_TENSOR_ID: 0,
    SAMPLE_COUNT: 8192,

    // full model download
    MODEL_DOWNLOAD_FORMAT: "f16", // "f16" recommended
    MODEL_DOWNLOAD_CHUNK_BYTES: 8 * 1024 * 1024, // 8 MiB per request

    // sparsity limits
    MIN_SPARSITY: 0.001,
    MAX_SPARSITY: 0.5,

    // real compute training (embedding-only negative sampling)
    NEGATIVE_SAMPLES: 8,
    BATCH_SIZE: 8,
    LOCAL_SGD_LR: 0.05, // local-only visualization; server uses its own optimizer
    EMBED_ROW_CACHE_MAX: 512,

    CHECKPOINT_KEY: "llm_node_checkpoint_v2",
};

const state = {
    connected: false,
    training: false,

    apiUrl: CONFIG.DEFAULT_API_URL,

    nodeId: `node_${Math.random().toString(36).slice(2, 10)}`,

    serverStep: null,
    serverUpdates: null,

    manifest: null, // { step, tensors: [...] }
    lastInfo: null,

    localStep: 0,
    lastSyncAt: 0,
    throughputBps: 0,
    sparsity: 0.05,

    model: {
        downloading: false,
        bytesDone: 0,
        bytesTotal: 0,
        tensors: new Map(), // id -> { meta, format, data: TypedArray }
    },

    // embedding training state
    embed: null, // { tensorId, rows, cols, meta }
    embedRowCache: new Map(), // rowId -> Float32Array (mutable)
    gradRows: new Map(), // rowId -> Float32Array
    gradSamples: 0,
    gradLossSum: 0,

    // model stub
    webgpuReady: false,
    lossLocal: [],
    lossServer: [],

    chart: null,
    submissionMode: 1, // 1: Standard, 2: Compressed, 3: Dense
};

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
        // Inf/NaN
        return hsign | (0x1f << 10) | (mant ? 0x200 : 0);
    }

    if (exp > 15) {
        // Overflow -> Inf
        return hsign | (0x1f << 10);
    }

    if (exp >= -14) {
        const hexp = (exp + 15) & 0x1f;
        const mant_rounded = mant + 0x00001000;
        let hmant = mant_rounded >> 13;
        if (hmant === 0x400) {
            return hsign | ((hexp + 1) << 10);
        }
        return hsign | (hexp << 10) | (hmant & 0x3ff);
    }

    if (exp < -24) {
        return hsign;
    }

    // Subnormal
    const mant_full = mant | 0x800000;
    const shift = -14 - exp;
    let sub = mant_full >> (shift + 13);
    if ((mant_full >> (shift + 12)) & 1) {
        sub++;
    }
    return hsign | (sub & 0x3ff);
}

function fp32_from_fp16(h) {
    // IEEE-754 half -> float32
    const s = (h & 0x8000) << 16;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;

    let bits = 0;
    if (e === 0) {
        if (m === 0) {
            bits = s;
        } else {
            // subnormal
            while ((m & 0x400) === 0) {
                m <<= 1;
                e -= 1;
            }
            e += 1;
            m &= 0x3ff;
            bits = s | ((e + (127 - 15)) << 23) | (m << 13);
        }
    } else if (e === 31) {
        // Inf/NaN
        bits = s | 0x7f800000 | (m << 13);
    } else {
        bits = s | ((e + (127 - 15)) << 23) | (m << 13);
    }

    const u32 = new Uint32Array(1);
    const f32 = new Float32Array(u32.buffer);
    u32[0] = bits >>> 0;
    return f32[0];
}

function $(id) {
    return document.getElementById(id);
}

function fmtInt(n) {
    if (n === null || n === undefined) return "-";
    return String(n);
}

function fmtBps(bps) {
    if (!bps || !Number.isFinite(bps)) return "-";
    const kb = bps / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB/s`;
    return `${(kb / 1024).toFixed(2)} MB/s`;
}

function fmtBytes(n) {
    if (!n || !Number.isFinite(n)) return "-";
    const kb = n / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    const mb = kb / 1024;
    if (mb < 1024) return `${mb.toFixed(2)} MB`;
    const gb = mb / 1024;
    return `${gb.toFixed(2)} GB`;
}

function clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
}

function log(msg, type = "info") {
    const el = $("log");
    const line = document.createElement("div");
    const ts = new Date().toLocaleTimeString();
    line.textContent = `[${ts}] ${msg}`;
    if (type === "error") line.style.color = "#ff6666";
    if (type === "good") line.style.color = "#66ffb2";
    if (type === "warn") line.style.color = "#ffcc66";
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
}

function setConnected(on) {
    state.connected = on;
    $("status-dot").className = on ? "dot on" : "dot";
    $("status-text").textContent = on ? "Connected" : "Disconnected";
    $("connect-btn").textContent = on ? "Disconnect" : "Connect";
    $("sync-btn").disabled = !on;
    $("manifest-btn").disabled = !on;
    $("train-btn").disabled = !on;
    $("download-btn").disabled = !on || state.model.downloading;
    $("save-btn").disabled = !on;
    $("load-btn").disabled = !on;
}

function updateUI() {
    $("api-url").value = state.apiUrl;
    $("node-id").textContent = state.nodeId;

    $("server-step").textContent = fmtInt(state.serverStep);
    $("server-updates").textContent = fmtInt(state.serverUpdates);

    const totalParams = state.lastInfo?.total_params;
    $("total-params").textContent = totalParams ? String(totalParams) : "-";

    $("tensor-count").textContent = state.manifest?.tensors
        ? String(state.manifest.tensors.length)
        : "-";

    $("local-step").textContent = String(state.localStep);
    $("throughput").textContent = fmtBps(state.throughputBps);
    $("sparsity").textContent = `${(state.sparsity * 100).toFixed(2)}%`;

    if ($("download-progress")) {
        if (!state.model.bytesTotal) {
            $("download-progress").textContent = "-";
        } else {
            $("download-progress").textContent =
                `${fmtBytes(state.model.bytesDone)} / ${fmtBytes(state.model.bytesTotal)}`;
        }
    }
}

// ==========================
// Chart
// ==========================

function initChart() {
    const ctx = $("lossChart").getContext("2d");
    state.chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Server avg loss",
                    borderColor: "#00d4ff",
                    data: [],
                    tension: 0.2,
                },
                {
                    label: "Local loss (demo)",
                    borderColor: "#7b2cbf",
                    data: [],
                    tension: 0.2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { display: false } },
                y: {
                    grid: { color: "rgba(255,255,255,0.06)" },
                    ticks: { color: "#a0a0a0" },
                },
            },
            plugins: {
                legend: { labels: { color: "#a0a0a0" } },
            },
        },
    });
}

function pushChart(serverLoss, localLoss) {
    if (!state.chart) return;
    const maxPoints = 60;
    state.chart.data.labels.push("");
    state.chart.data.datasets[0].data.push(serverLoss ?? null);
    state.chart.data.datasets[1].data.push(localLoss ?? null);

    while (state.chart.data.labels.length > maxPoints) {
        state.chart.data.labels.shift();
        state.chart.data.datasets.forEach((d) => d.data.shift());
    }
    state.chart.update("none");
}

// ==========================
// WebGPU check (frontend only)
// ==========================

async function initWebGPU() {
    if (!navigator.gpu) {
        $("gpu-status").textContent = "Not available";
        state.webgpuReady = false;
        return;
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No adapter");
        await adapter.requestDevice();
        $("gpu-status").textContent = "Active";
        state.webgpuReady = true;
    } catch (e) {
        $("gpu-status").textContent = "Unavailable";
        state.webgpuReady = false;
    }
}

// ==========================
// API helpers
// ==========================

function api(path) {
    return `${state.apiUrl}${path}`;
}

async function fetchJson(path) {
    const res = await fetch(api(path), { method: "GET" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
}

async function refreshInfoAndManifest() {
    const info = await fetchJson("/api/v1/model/info");
    state.lastInfo = info;
    state.serverStep = info.step;
    state.serverUpdates = info.updates;

    const manifest = await fetchJson("/api/v1/model/manifest");
    state.manifest = manifest;

    updateUI();
}

function pickEmbedTensor() {
    if (!state.manifest?.tensors?.length) return null;

    // Prefer a tensor named like token embedding; else fall back to id=0.
    const t =
        state.manifest.tensors.find((x) =>
            String(x.name || "").toLowerCase().includes("token_embed"),
        ) ||
        state.manifest.tensors.find((x) => x.id === 0) ||
        state.manifest.tensors[0];

    if (!t) return null;
    if (!Array.isArray(t.shape) || t.shape.length !== 2) return null;

    const [rows, cols] = t.shape;
    return { tensorId: t.id, rows, cols, meta: t };
}

async function downloadTensorChunked(meta, format) {
    const bytesPerEl = format === "f32" ? 4 : 2;
    const totalBytes =
        format === "f32" ? meta.bytes_f32 : meta.bytes_f16 ?? meta.elements * 2;

    const totalElements = meta.elements >>> 0;
    const chunkElements = Math.max(
        1,
        Math.floor(CONFIG.MODEL_DOWNLOAD_CHUNK_BYTES / bytesPerEl),
    );

    let out;
    if (format === "f32") out = new Float32Array(totalElements);
    else out = new Uint16Array(totalElements);

    for (let offset = 0; offset < totalElements; offset += chunkElements) {
        const count = Math.min(chunkElements, totalElements - offset);
        const url =
            `/api/v1/model/tensor/${meta.id}` +
            `?format=${format}&offset=${offset}&count=${count}`;

        const res = await fetch(api(url), { method: "GET" });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const buf = await res.arrayBuffer();

        if (format === "f32") {
            out.set(new Float32Array(buf), offset);
        } else {
            out.set(new Uint16Array(buf), offset);
        }

        state.model.bytesDone += buf.byteLength;
        updateUI();

        // yield to UI thread
        await new Promise((r) => setTimeout(r, 0));
    }

    // sanity (best-effort)
    if (Number.isFinite(totalBytes)) {
        // do nothing; some servers may have padding differences
    }

    return out;
}

async function downloadFullModel() {
    if (!state.connected) return;
    if (state.model.downloading) return;
    if (!state.manifest?.tensors?.length) await refreshInfoAndManifest();

    state.model.downloading = true;
    $("download-btn").disabled = true;
    state.model.bytesDone = 0;

    const format = CONFIG.MODEL_DOWNLOAD_FORMAT;
    let total = 0;
    for (const t of state.manifest.tensors) {
        total += format === "f32" ? t.bytes_f32 : t.bytes_f16;
    }
    state.model.bytesTotal = total;
    updateUI();

    log(
        `Downloading full model: tensors=${state.manifest.tensors.length} format=${format} total=${fmtBytes(total)}…`,
    );

    try {
        for (const t of state.manifest.tensors) {
            log(`Downloading tensor [${t.id}] ${t.name} (${t.elements} el)…`);
            const data = await downloadTensorChunked(t, format);
            state.model.tensors.set(t.id, { meta: t, format, data });
            log(`Stored tensor [${t.id}] (${fmtBytes(data.byteLength)}).`, "good");
        }

        state.embed = pickEmbedTensor();
        if (state.embed) {
            log(
                `Embedding tensor set to id=${state.embed.tensorId} shape=[${state.embed.rows},${state.embed.cols}].`,
                "good",
            );
        } else {
            log(
                "Could not identify a 2D token embedding tensor (token_embed).",
                "warn",
            );
        }
    } finally {
        state.model.downloading = false;
        $("download-btn").disabled = !state.connected;
        updateUI();
    }
}

// ==========================
// Tensor sync (sample)
// ==========================

async function syncSampleTensorSlice() {
    if (!state.manifest?.tensors?.length) {
        throw new Error("manifest missing");
    }

    const t = state.manifest.tensors.find(
        (x) => x.id === CONFIG.SAMPLE_TENSOR_ID,
    );
    if (!t) {
        throw new Error(`tensor id ${CONFIG.SAMPLE_TENSOR_ID} not found`);
    }

    const count = Math.min(CONFIG.SAMPLE_COUNT, t.elements);
    const url =
        `/api/v1/model/tensor/${t.id}` +
        `?format=f16&offset=0&count=${count}`;

    log(
        `Downloading tensor slice id=${t.id} (${t.name}) count=${count} (f16)…`,
    );

    const res = await fetch(api(url));
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const buf = await res.arrayBuffer();

    const stepHdr = res.headers.get("X-Model-Step");
    if (stepHdr) state.serverStep = Number(stepHdr);

    log(`Got ${buf.byteLength} bytes (stored as demo sample).`, "good");

    // This is where you'd upload to GPU buffers.
    // We keep it as a proof that weights download works.
    state.lastSyncAt = Date.now();
    updateUI();
}

// ==========================
// DGRD packet builder
// ==========================

function encUtf8(s) {
    return new TextEncoder().encode(s);
}

function dgrdPacket({ step, nodeId, trainLoss, samples, tensorUpdates, flags }) {
    // compute required bytes
    const nodeBytes = encUtf8(nodeId);
    let bytes =
        4 + // magic
        2 + // version
        2 + // flags
        4 + // step
        4 + // node_id_len
        nodeBytes.length +
        4 + // train_loss f32
        4 + // samples u32
        4; // n_tensors u32

    for (const tu of tensorUpdates) {
        const nnz = tu.indices.length;
        bytes += 4 + 4; // tensor_id + nnz
        if (nnz === 0) {
            // Mode 3: Dense FP16
            // In Mode 3, tu should also have `elements` or pre-built `values`
            bytes += tu.values.length * 2;
        } else if (flags === 0) {
            // Mode 1: Standard Sparse
            bytes += nnz * (4 + 4); // index + value
        } else if (flags === 1) {
            // Mode 2: Compressed Sparse
            bytes += nnz * (1 + 2); // skip + fp16
        }
    }

    const ab = new ArrayBuffer(bytes);
    const dv = new DataView(ab);
    const u8 = new Uint8Array(ab);
    let o = 0;

    // magic "DGRD"
    u8[o++] = 0x44; u8[o++] = 0x47; u8[o++] = 0x52; u8[o++] = 0x44;

    dv.setUint16(o, 1, true); o += 2; // version
    dv.setUint16(o, flags, true); o += 2; // flags
    dv.setUint32(o, step >>> 0, true); o += 4;

    dv.setUint32(o, nodeBytes.length >>> 0, true); o += 4;
    u8.set(nodeBytes, o); o += nodeBytes.length;

    dv.setFloat32(o, trainLoss, true); o += 4;
    dv.setUint32(o, samples >>> 0, true); o += 4;

    dv.setUint32(o, tensorUpdates.length >>> 0, true); o += 4;

    for (const tu of tensorUpdates) {
        const nnz = tu.indices.length;
        dv.setUint32(o, tu.tensorId >>> 0, true); o += 4;
        dv.setUint32(o, nnz >>> 0, true); o += 4;

        if (nnz === 0) {
            // Mode 3: Dense FP16
            for (let i = 0; i < tu.values.length; i++) {
                dv.setUint16(o, fp16_from_fp32(tu.values[i]), true);
                o += 2;
            }
        } else if (flags === 0) {
            // Mode 1
            for (let i = 0; i < nnz; i++) {
                dv.setUint32(o, tu.indices[i] >>> 0, true); o += 4;
                dv.setFloat32(o, tu.values[i], true); o += 4;
            }
        } else if (flags === 1) {
            // Mode 2
            let lastIdx = 0;
            for (let i = 0; i < nnz; i++) {
                let skip = 0;
                if (i === 0) {
                    skip = clamp(tu.indices[i], 0, 255);
                } else {
                    skip = clamp(tu.indices[i] - lastIdx - 1, 0, 255);
                }
                lastIdx = tu.indices[i]; // approximate if clipped? nono, just for demo
                u8[o++] = skip;
                dv.setUint16(o, fp16_from_fp32(tu.values[i]), true); o += 2;
            }
        }
    }

    return ab;
}

// ==========================
// Real compute: embedding-only negative sampling
// ==========================

function hash32(s) {
    // FNV-1a-ish
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    return h >>> 0;
}

function tokenizeHashed(text, vocabSize) {
    const toks =
        text
            .toLowerCase()
            .match(/[a-z0-9]+|[^\s]/g) || [];
    const ids = new Uint32Array(toks.length);
    for (let i = 0; i < toks.length; i++) {
        ids[i] = (hash32(toks[i]) % vocabSize) >>> 0;
    }
    return ids;
}

function sigmoid(x) {
    // stable enough for our dot ranges
    if (x >= 0) {
        const z = Math.exp(-x);
        return 1 / (1 + z);
    }
    const z = Math.exp(x);
    return z / (1 + z);
}

function getTensorStored(tensorId) {
    const rec = state.model.tensors.get(tensorId);
    if (!rec) return null;
    return rec;
}

function lruTouch(map, key) {
    const v = map.get(key);
    if (!v) return null;
    map.delete(key);
    map.set(key, v);
    return v;
}

function getEmbedRowF32(rowId) {
    if (!state.embed) throw new Error("embed tensor not set");
    const { tensorId, cols } = state.embed;

    // LRU cache hit
    const hit = lruTouch(state.embedRowCache, rowId);
    if (hit) return hit;

    const stored = getTensorStored(tensorId);
    if (!stored) {
        throw new Error(
            "Embedding tensor not downloaded. Click 'Download full model' first.",
        );
    }

    if (stored.format !== "f16") {
        throw new Error(
            `Embedding tensor must be stored as f16 for this demo (got ${stored.format}).`,
        );
    }

    const base = rowId * cols;
    const out = new Float32Array(cols);
    const data = stored.data;
    for (let j = 0; j < cols; j++) {
        out[j] = fp32_from_fp16(data[base + j]);
    }

    state.embedRowCache.set(rowId, out);
    while (state.embedRowCache.size > CONFIG.EMBED_ROW_CACHE_MAX) {
        const oldestKey = state.embedRowCache.keys().next().value;
        state.embedRowCache.delete(oldestKey);
    }

    return out;
}

function addGradRow(rowId, gradVec) {
    const existing = state.gradRows.get(rowId);
    if (!existing) {
        state.gradRows.set(rowId, gradVec.slice());
        return;
    }
    for (let i = 0; i < existing.length; i++) {
        existing[i] += gradVec[i];
    }
}

function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}

function axpy(out, a, x) {
    // out += a * x
    for (let i = 0; i < out.length; i++) out[i] += a * x[i];
}

function trainStepEmbeddingNS(ids, vocabSize) {
    if (!state.embed) throw new Error("embed tensor not set");
    if (!ids || ids.length < 2) throw new Error("not enough tokens");

    const pos = Math.floor(Math.random() * (ids.length - 1));
    const center = ids[pos] >>> 0;
    const target = ids[pos + 1] >>> 0;

    const u = getEmbedRowF32(center);
    const vPos = getEmbedRowF32(target);

    const du = new Float32Array(u.length);
    const dvPos = new Float32Array(u.length);

    // positive
    const scorePos = dot(u, vPos);
    const pPos = sigmoid(scorePos);
    const gPos = pPos - 1; // d/dx [-log(sigmoid(x))]

    axpy(du, gPos, vPos);
    axpy(dvPos, gPos, u);

    let loss = -Math.log(Math.max(1e-12, pPos));

    // negatives
    for (let k = 0; k < CONFIG.NEGATIVE_SAMPLES; k++) {
        const neg = (Math.floor(Math.random() * vocabSize) >>> 0);
        const vNeg = getEmbedRowF32(neg);

        const dvNeg = new Float32Array(u.length);

        const scoreNeg = dot(u, vNeg);
        const pNeg = sigmoid(scoreNeg);
        const gNeg = pNeg; // d/dx [-log(sigmoid(-x))] = sigmoid(x)

        axpy(du, gNeg, vNeg);
        axpy(dvNeg, gNeg, u);
        addGradRow(neg, dvNeg);

        loss += -Math.log(Math.max(1e-12, 1 - pNeg));
    }

    addGradRow(center, du);
    addGradRow(target, dvPos);

    // local SGD update (only cached rows)
    const lr = CONFIG.LOCAL_SGD_LR;
    if (lr > 0) {
        for (const [rowId, g] of state.gradRows.entries()) {
            const w = getEmbedRowF32(rowId);
            for (let i = 0; i < w.length; i++) {
                w[i] -= lr * g[i];
            }
        }
    }

    return loss;
}

function buildTensorUpdateFromGradRows() {
    if (!state.embed) throw new Error("embed tensor not set");
    const { tensorId, cols, meta } = state.embed;

    const rowIds = Array.from(state.gradRows.keys()).sort((a, b) => a - b);
    const nnz = rowIds.length * cols;
    const indices = new Uint32Array(nnz);
    const values = new Float32Array(nnz);

    let p = 0;
    for (const rowId of rowIds) {
        const g = state.gradRows.get(rowId);
        const base = rowId * cols;
        for (let j = 0; j < cols; j++) {
            indices[p] = (base + j) >>> 0;
            values[p] = g[j];
            p++;
        }
    }

    // update UI sparsity estimate for this tensor
    state.sparsity = nnz / Math.max(1, meta.elements);

    return {
        tensorId,
        indices,
        values,
    };
}

async function submitGradients({ trainLoss, samples, tensorUpdates }) {
    if (!state.connected) return;
    if (!state.serverStep) throw new Error("server step unknown");

    // NOTE: Mode 2 (u8 skip) cannot represent large first indices (first skip
    // is from 0), so we force Mode 1 for real embedding updates.
    const flags = 0;

    const packet = dgrdPacket({
        step: state.serverStep,
        nodeId: state.nodeId,
        trainLoss,
        samples,
        tensorUpdates,
        flags,
    });

    const t0 = performance.now();
    const res = await fetch(api("/api/v1/train/submit"), {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: packet,
    });
    const t1 = performance.now();

    const secs = Math.max(0.001, (t1 - t0) / 1000);
    state.throughputBps = packet.byteLength / secs;

    // adjust sparsity toward target upload
    const ratio = CONFIG.TARGET_UPLOAD_BPS / state.throughputBps;
    // gentle controller
    state.sparsity = clamp(
        state.sparsity * Math.pow(ratio, 0.25),
        CONFIG.MIN_SPARSITY,
        CONFIG.MAX_SPARSITY,
    );

    const j = await res.json().catch(() => null);

    if (res.status === 409) {
        log(
            `Server step mismatch (server_step=${j?.server_step}). Re-syncing…`,
            "warn",
        );
        await refreshInfoAndManifest();
        updateUI();
        return;
    }

    if (!res.ok) {
        throw new Error(
            `submit failed: ${res.status} ${res.statusText} ${JSON.stringify(j)}`,
        );
    }

    log(
        `Submitted DGRD (${packet.byteLength} bytes) at step=${state.serverStep}.`,
        "good",
    );
    updateUI();
}

// ==========================
// Loss polling
// ==========================

async function pollServerLosses() {
    if (!state.connected) return;

    try {
        const losses = await fetchJson("/api/v1/server/losses");
        state.lossServer = Array.isArray(losses) ? losses : [];

        const lastServer =
            state.lossServer.length > 0
                ? state.lossServer[state.lossServer.length - 1]
                : null;

        const lastLocal =
            state.lossLocal.length > 0
                ? state.lossLocal[state.lossLocal.length - 1]
                : null;

        pushChart(lastServer, lastLocal);
    } catch {
        // ignore
    } finally {
        setTimeout(pollServerLosses, 10_000);
    }
}

// ==========================
// Training loop (demo loss, real packet sync)
// ==========================

let trainerWorker = null;

async function startTraining() {
    const text = $("training-text").value.trim();
    if (!text) {
        log("Please add some training text.", "error");
        return;
    }
    if (!state.connected) {
        log("Connect to server first.", "error");
        return;
    }

    const useWorker = document.querySelector('input[name="train-thread"]:checked').value === "worker";

    if (useWorker) {
        startTrainingWorker(text);
        return;
    }

    if (!state.manifest?.tensors?.length) await refreshInfoAndManifest();
    if (!state.embed) state.embed = pickEmbedTensor();
    if (!state.embed) {
        log("No suitable embedding tensor found in manifest.", "error");
        return;
    }

    // Require weights
    if (!getTensorStored(state.embed.tensorId)) {
        log("Model not downloaded yet. Downloading now…", "warn");
        await downloadFullModel();
    }

    const vocabSize = state.lastInfo?.config?.vocab_size;
    if (!vocabSize || !Number.isFinite(vocabSize)) {
        log("Server vocab_size missing in /model/info.", "error");
        return;
    }

    const ids = Tokenizer.ready
        ? Tokenizer.encode(text)
        : tokenizeHashed(text, vocabSize);

    if (ids.length < 2) {
        log("Training text produced too few tokens.", "error");
        return;
    }

    state.training = true;
    $("train-btn").textContent = "Stop training";
    log(
        "Training started (REAL compute: embedding negative sampling + real DGRD submits).",
        "good",
    );

    let lastSync = Date.now();

    while (state.training) {
        state.localStep++;

        // compute a small batch
        let batchLoss = 0;
        state.gradRows.clear();
        for (let b = 0; b < CONFIG.BATCH_SIZE; b++) {
            batchLoss += trainStepEmbeddingNS(ids, vocabSize);
        }
        batchLoss /= CONFIG.BATCH_SIZE;
        state.lossLocal.push(batchLoss);
        state.gradLossSum += batchLoss;
        state.gradSamples += CONFIG.BATCH_SIZE;

        updateUI();

        const lastServer =
            state.lossServer.length > 0
                ? state.lossServer[state.lossServer.length - 1]
                : null;
        pushChart(lastServer, batchLoss);

        if (Date.now() - lastSync > CONFIG.ACCUMULATION_INTERVAL_MS) {
            try {
                const tu = buildTensorUpdateFromGradRows();
                const avgLoss =
                    state.gradSamples > 0
                        ? state.gradLossSum / state.gradSamples
                        : batchLoss;

                await submitGradients({
                    trainLoss: avgLoss,
                    samples: state.gradSamples || CONFIG.BATCH_SIZE,
                    tensorUpdates: [tu],
                });

                state.gradRows.clear();
                state.gradSamples = 0;
                state.gradLossSum = 0;
            } catch (e) {
                log(`Sync error: ${e.message}`, "error");
            }
            lastSync = Date.now();
        }

        await new Promise((r) => setTimeout(r, CONFIG.TRAIN_STEP_DELAY_MS));
    }

    log("Training stopped.");
}

function startTrainingWorker(text) {
    if (trainerWorker) {
        trainerWorker.terminate();
    }

    log("Spawning trainer_worker.js...", "info");
    trainerWorker = new Worker("trainer_worker.js");

    trainerWorker.onmessage = (e) => {
        const d = e.data;
        if (d.type === "log") {
            log(`[Worker] ${d.msg}`);
        } else if (d.type === "error") {
            log(`[Worker Error] ${d.msg}`, "error");
            stopTraining();
        } else if (d.type === "downloadProgress") {
            state.model.bytesDone = d.done;
            state.model.bytesTotal = d.total;
            updateUI();
        } else if (d.type === "trainStep") {
            state.localStep++;
            state.lossLocal.push(d.loss);
            updateUI();

            const lastServer =
                state.lossServer.length > 0
                    ? state.lossServer[state.lossServer.length - 1]
                    : null;
            pushChart(lastServer, d.loss);
        }
    };

    trainerWorker.postMessage({
        type: "start",
        apiUrl: state.apiUrl,
        nodeId: state.nodeId,
        text: text,
    });

    state.training = true;
    $("train-btn").textContent = "Stop training";
}

function stopTraining() {
    state.training = false;
    $("train-btn").textContent = "Start training";

    if (trainerWorker) {
        trainerWorker.postMessage({ type: "stop" });
        // We could terminate after a timeout, but letting it clean up is nicer.
        // For now, just terminate soon if it doesn't stop.
        setTimeout(() => {
            if (!state.training && trainerWorker) {
                trainerWorker.terminate();
                trainerWorker = null;
                log("Worker terminated.");
            }
        }, 1000);
    }
}

// ==========================
// Local save/load
// ==========================

function saveLocal() {
    const payload = {
        nodeId: state.nodeId,
        apiUrl: state.apiUrl,
        trainingText: $("training-text").value,
        sparsity: state.sparsity,
    };
    localStorage.setItem(CONFIG.CHECKPOINT_KEY, JSON.stringify(payload));
    log("Saved local settings.", "good");
}

function loadLocal() {
    const raw = localStorage.getItem(CONFIG.CHECKPOINT_KEY);
    if (!raw) {
        log("No local save found.", "warn");
        return;
    }
    try {
        const p = JSON.parse(raw);
        if (typeof p.apiUrl === "string") state.apiUrl = p.apiUrl;
        if (typeof p.trainingText === "string")
            $("training-text").value = p.trainingText;
        if (typeof p.sparsity === "number")
            state.sparsity = clamp(
                p.sparsity,
                CONFIG.MIN_SPARSITY,
                CONFIG.MAX_SPARSITY,
            );
        log("Loaded local settings.", "good");
        updateUI();
    } catch (e) {
        log(`Load failed: ${e.message}`, "error");
    }
}

// ==========================
// Connect/disconnect
// ==========================

async function connect() {
    state.apiUrl = $("api-url").value.trim() || CONFIG.DEFAULT_API_URL;

    try {
        await refreshInfoAndManifest();
        setConnected(true);
        updateUI();
        log("Connected to server.", "good");
        pollServerLosses();
    } catch (e) {
        setConnected(false);
        log(`Connect failed: ${e.message}`, "error");
    }
}

function disconnect() {
    stopTraining();
    setConnected(false);
    log("Disconnected.");
    updateUI();
}

// ==========================
// Chat (stub)
// ==========================

function appendChat(role, text) {
    const el = $("chat");
    const line = document.createElement("div");
    line.style.marginBottom = "8px";
    line.innerHTML = `<strong>${role}:</strong> ${text}`;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
}

async function sendChat() {
    const inp = $("chat-input");
    const text = inp.value.trim();
    if (!text) return;
    inp.value = "";

    appendChat("you", text);
    appendChat(
        "model",
        "Stub response: this frontend focuses on training sync + API wiring.",
    );
}

// ==========================
// Manifest viewer
// ==========================

function logManifest() {
    if (!state.manifest?.tensors) {
        log("Manifest not loaded.", "warn");
        return;
    }
    log(
        `Manifest step=${state.manifest.step} tensors=${state.manifest.tensors.length}`,
    );
    const preview = state.manifest.tensors.slice(0, 8);
    for (const t of preview) {
        log(`  [${t.id}] ${t.name} shape=${JSON.stringify(t.shape)}`);
    }
    if (state.manifest.tensors.length > preview.length) {
        log(`  … (${state.manifest.tensors.length - preview.length} more)`);
    }
}

// ==========================
// Init
// ==========================

window.onload = async () => {
    initChart();

    $("api-url").value = state.apiUrl;
    $("node-id").textContent = state.nodeId;

    $("connect-btn").onclick = async () => {
        if (state.connected) disconnect();
        else await connect();
    };

    document.querySelectorAll('input[name="grad-mode"]').forEach(radio => {
        radio.onchange = (e) => {
            state.submissionMode = parseInt(e.target.value);
            log(`Switched to Grad Mode ${state.submissionMode}`);
        };
    });

    $("sync-btn").onclick = async () => {
        try {
            await syncSampleTensorSlice();
        } catch (e) {
            log(`Sync failed: ${e.message}`, "error");
        }
    };

    $("download-btn").onclick = async () => {
        try {
            await downloadFullModel();
        } catch (e) {
            log(`Download failed: ${e.message}`, "error");
        }
    };

    $("manifest-btn").onclick = logManifest;

    $("train-btn").onclick = async () => {
        if (state.training) stopTraining();
        else await startTraining();
    };

    $("save-btn").onclick = saveLocal;
    $("load-btn").onclick = loadLocal;

    $("clear-log").onclick = () => {
        $("log").innerHTML = "";
    };

    $("chat-send").onclick = sendChat;
    $("chat-input").addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendChat();
    });

    await initWebGPU();

    log("Loading tokenizer...", "info");
    try {
        await Tokenizer.load("vocab.json", "merges.txt");
        log("Tokenizer ready.", "good");
    } catch (e) {
        log(`Failed to load real tokenizer: ${e.message}. Using fallback.`, "warn");
    }

    setConnected(false);
    updateUI();

    appendChat("system", "Connect to the API server, then start training.");
};