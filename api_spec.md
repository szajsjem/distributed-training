# Distributed LLM Training Server API (v1)

Base URL: `http://HOST:PORT`

Protocol: HTTP/1.1
- Requests with a body MUST include `Content-Length`.
- Responses always include `Content-Length`.
- Server does not support chunked transfer encoding.
- CORS is enabled (`Access-Control-Allow-Origin: *`).

All binary values are little-endian.

## Concepts

### step
The server maintains a monotonically increasing integer `step`.
- Clients fetch model tensors at a given `step`.
- Clients MUST submit gradients for the exact same `step`.
- If the server has already advanced, gradient submissions return `409`.

### tensors + tensor_id
The model parameters are exposed as a list of tensors in a stable order.
- Clients fetch `/api/v1/model/manifest` to learn all tensor IDs, shapes, and sizes.
- Then clients fetch `/api/v1/model/tensor/{id}` to download data.

### transport formats
For downloading parameters:
- `format=f16` returns FP16 (2 bytes/element)
- `format=f32` returns raw float32 bytes (4 bytes/element)

FP16 is IEEE 754 half; used for bandwidth saving.

---

## Endpoints

### GET `/healthz`
Health check.

Response `200` (JSON):
```json
{ "ok": true }
```

---

### GET `/api/v1/model/info`
Returns global model + training info.

Response `200` (JSON):
```json
{
  "step": 1,
  "updates": 0,
  "total_params": 123456789,
  "config": {
    "vocab_size": 50257,
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "d_ff": 3072,
    "max_seq_len": 512
  },
  "train": {
    "learning_rate": 0.0003,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-08,
    "weight_decay": 0.01,
    "min_nodes_for_update": 2
  }
}
```

---

### GET `/api/v1/model/manifest`
Returns tensor metadata and the current `step`.

Response `200` (JSON):
```json
{
  "step": 1,
  "tensors": [
    {
      "id": 0,
      "name": "token_embed",
      "shape": [50257, 768],
      "elements": 38597376,
      "bytes_f32": 154389504,
      "bytes_f16": 77194752
    }
  ]
}
```

Notes:
- `id` is required for downloads and gradient submissions.
- `name` is informational (not used by the server for routing).

---

### GET `/api/v1/model/tensor/{id}`
Downloads a tensor (or a slice).

Query parameters:
- `format` (optional): `f16` (default) or `f32`
- `offset` (optional, default `0`): element offset within the 1D flattened tensor
- `count` (optional, default `elements - offset`): number of elements to return

Response `200`:
- `Content-Type: application/octet-stream`
- Body: raw bytes of requested slice
  - `f16`: `count * 2` bytes
  - `f32`: `count * 4` bytes

Response headers:
- `X-Model-Step`: the server’s current step
- `X-Tensor-Id`: the requested tensor id
- `X-Tensor-Offset`: the offset used
- `X-Tensor-Count`: the count used
- `X-Tensor-Format`: `f16` or `f32`

Errors:
- `404` if tensor id does not exist
- `416` if `offset/count` out of range
- `400` for invalid parameters

---

### POST `/api/v1/train/submit`
Submit sparse gradients for the current `step`.

Request headers:
- `Content-Type: application/octet-stream`

Request body: binary packet `DGRD` (see below).

Response:
- `200` (JSON) on success:
```json
{ "ok": true, "message": "ok", "server_step": 1 }
```

- `409` (JSON) on step mismatch:
```json
{ "ok": false, "message": "step mismatch; fetch latest model", "server_step": 7 }
```

- `400` (JSON) for malformed packet.

---

### GET `/api/v1/server/losses`
Returns the server’s loss history (average loss per applied update).

Response `200` (JSON):
```json
[2.3021, 2.2897, 2.2754]
```

---

## Static Assets & Configuration

While the server is primarily a binary aggregation engine, it also serves critical static assets for the frontend and loads its own configuration from the `static/` directory.

### Configuration Files
- **`static/model_config.json`**: Model hyperparameters (d_model, n_layers, etc.).
- **`static/train_config.json`**: Training parameters (learning_rate, min_nodes_for_update).

These files are loaded by the server on startup and also served to clients to ensure synchronization.

### Tokenizer Assets
- **`static/vocab.json`**: GPT-2 standard vocabulary mapping (JSON object of token to ID).
- **`static/merges.txt`**: GPT-2 standard BPE merges (text file with merge rules).
- **`static/tokenizer.js`**: Reusable BPE tokenizer class for use in browser main-thread or WebWorkers.

---

## Binary Gradient Packet: `DGRD` v1

All fields are little-endian.

Header:
- magic: "DGRD" (4 bytes)
- version: 1 (u16)
- flags: (u16)
  - `0`: Mode 1 (Sparse Standard)
  - `1`: Mode 2 (Sparse Compressed)
- step: (u32)
- node_id_len: (u32)
- node_id: (node_id_len bytes)
- train_loss: (f32)
- samples: (u32)
- n_tensors: (u32)

### Submission Modes

#### Mode 1 (Sparse Standard) - `flags = 0`, `nnz > 0`
For each tensor:
- u32 tensor_id
- u32 nnz
- nnz times:
  - u32 index
  - f32 value

#### Mode 2 (Sparse Compressed) - `flags = 1`, `nnz > 0`
For each tensor:
- u32 tensor_id
- u32 nnz
- nnz times:
  - u8 skippednum (delta index - 1, except first entry which is skip from 0)
  - fp16 value

#### Mode 3 (Dense FP16) - `nnz = 0`
For each tensor:
- u32 tensor_id
- u32 nnz = 0
- tensor_elements times:
  - fp16 value

## Late Submissions
The server accepts submissions where `packet_step` is in `[current_server_step - 5, current_server_step]`.
Late gradients are applied to the *current* parameter step.
If a packet is more than 5 steps stale, the server returns `409 Conflict`.
