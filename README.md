# Distributed LLM Training System

A decentralized training system for small LLMs (≤330M parameters) using browser-based WebGPU compute nodes and a C++ aggregation server.

## Features
- **Cross-platform C++ Server**: Works on Windows (Winsock2) and Linux (POSIX).
- **WebGPU Acceleration**: Local training in the browser.
- **Real BPE Tokenization**: Standard GPT-2 tokenizer integration (`vocab.json` + `merges.txt`).
- **WebWorker Training**: Full Transformer training off-main-thread via `trainer_worker.js`.
- **Dynamic Sparse Gradients**: Optimizes bandwidth usage (1-50% sparsity).
- **External Configuration**: Adjust model and training parameters via `model_config.json` and `train_config.json` without recompilation.
- **Local Inference**: Chat with the current model weights entirely in the browser.
- **Real-time Visualization**: Multi-metric loss graphs using Chart.js.
- **Checkpoint Resilience**: Save/Load states via localStorage and binary export.

## Quick Start

### 1. Build the Server
You can use CMake to build the server for your platform.

#### Windows (PowerShell)
```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

#### Linux
```bash
mkdir build
cd build
cmake ..
make -j4
```

### 2. Run the Server
```bash
./train_server 8080
```

### 3. Open the Web App
Navigate to `http://localhost:8080/` in a WebGPU-enabled browser (Chrome/Edge 113+ recommended).

## Development
- `server/server.cpp`: Main aggregation and HTTP logic.
- `static/index.html`: Main UI.
- `static/app.js`: WebGPU models, training loop, and visualization logic.
- `static/style.css`: Premium dark mode UI.

## License
GNU General Public License v3.0
