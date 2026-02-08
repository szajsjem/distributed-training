/* tokenizer.js - Lightweight GPT-2 BPE Tokenizer for Browsers
   Ported from openai/gpt-2 and huggingface/transformers.
*/

class GPT2Tokenizer {
    constructor() {
        this.vocab = {};
        this.decoder = {};
        this.bpe_ranks = {};
        this.byte_encoder = this.create_byte_encoder();
        this.byte_decoder = Object.fromEntries(Object.entries(this.byte_encoder).map(([k, v]) => [v, k]));
        this.cache = new Map();
        this.ready = false;
    }

    async load(vocabPath, mergesPath) {
        const [v_res, m_res] = await Promise.all([
            fetch(vocabPath).then(r => r.json()),
            fetch(mergesPath).then(r => r.text())
        ]);

        this.vocab = v_res;
        this.decoder = Object.fromEntries(Object.entries(this.vocab).map(([k, v]) => [v, k]));

        const bpe_merges = m_res.split('\n').slice(1).filter(l => l.trim().length > 0);
        this.bpe_ranks = Object.fromEntries(bpe_merges.map((m, i) => [m, i]));
        this.ready = true;
    }

    create_byte_encoder() {
        const bs = [...Array(95).keys()].map(i => i + 33)
            .concat([...Array(174 - 161 + 1).keys()].map(i => i + 161))
            .concat([...Array(255 - 175 + 1).keys()].map(i => i + 175));
        const cs = bs.slice();
        let n = 0;
        for (let b = 0; b < 256; b++) {
            if (!bs.includes(b)) {
                bs.push(b);
                cs.push(256 + n);
                n++;
            }
        }
        return Object.fromEntries(bs.map((b, i) => [b, String.fromCharCode(cs[i])]));
    }

    get_pairs(word) {
        const pairs = new Set();
        let prev = word[0];
        for (let i = 1; i < word.length; i++) {
            pairs.add([prev, word[i]]);
            prev = word[i];
        }
        return pairs;
    }

    bpe(token) {
        if (this.cache.has(token)) return this.cache.get(token);

        let word = token.split('');
        let pairs = this.get_pairs(word);

        if (pairs.size === 0) return token;

        while (true) {
            let bigram = null;
            let minRank = Infinity;

            for (const pair of pairs) {
                const p_str = pair.join(' ');
                const rank = this.bpe_ranks[p_str];
                if (rank !== undefined && rank < minRank) {
                    minRank = rank;
                    bigram = pair;
                }
            }

            if (bigram === null) break;

            const [first, second] = bigram;
            const new_word = [];
            let i = 0;
            while (i < word.length) {
                const j = word.indexOf(first, i);
                if (j === -1) {
                    new_word.push(...word.slice(i));
                    break;
                }
                new_word.push(...word.slice(i, j));
                i = j;

                if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
                    new_word.push(first + second);
                    i += 2;
                } else {
                    new_word.push(word[i]);
                    i += 1;
                }
            }
            word = new_word;
            if (word.length === 1) break;
            pairs = this.get_pairs(word);
        }

        const result = word.join(' ');
        this.cache.set(token, result);
        return result;
    }

    encode(text) {
        if (!this.ready) throw new Error("Tokenizer not ready. Call load() first.");

        const bpe_tokens = [];
        const matches = text.matchAll(/'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu);

        for (const match of matches) {
            let token = match[0];
            // Byte level encoding
            token = Array.from(new TextEncoder().encode(token))
                .map(b => this.byte_encoder[b])
                .join('');

            const bpe_res = this.bpe(token).split(' ');
            for (const t of bpe_res) {
                bpe_tokens.push(this.vocab[t]);
            }
        }
        return new Uint32Array(bpe_tokens);
    }

    decode(tokens) {
        let text = Array.from(tokens)
            .map(t => this.decoder[t])
            .join('');
        const bytes = Uint8Array.from(text.split('').map(c => this.byte_decoder[c]));
        return new TextDecoder().decode(bytes);
    }
}

// Global instance
self.Tokenizer = new GPT2Tokenizer();
