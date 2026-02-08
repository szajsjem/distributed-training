# Project Roadmap & TODO

This document outlines the planned features, improvements, and research areas for the Distributed LLM Training System.

- [ ] **Refactor**: Refactor the codebase to improve modularity, performance, and maintainability.

## Data Management
- [ ] **Redesign User-Added Data Pipeline**:
    - [ ] Structured storage for `train`, `valid`, and `annotated` datasets.
    - [ ] Robust importing mechanism for custom text files and datasets.
- [ ] **Server-Shared Training Data**: Implement a mechanism for the server to distribute specific data chunks or datasets to nodes.
- [ ] **Server Validation Set**: Add a dedicated validation dataset on the server to track global validation loss independently of training nodes.

## Networking & Transport
- [ ] **Data Compression**: Implement Brotli or Zstd compression for gradient and model transmissions to reduce bandwidth.
- [ ] **Bi-directional Communication**: Explore WebSockets or gRPC-web for lower latency and real-time node coordination.
- [ ] **Security**: 
    - [ ] TLS/SSL for all API endpoints, optionally hide behind reverse proxy.
    - [ ] Authentication/Authorization for trainer nodes (e.g., JWT).

## Frontend & Features
- [ ] **Frontend Redesign**: Modernize the UI with a more intuitive dashboard, better charts, and improved responsiveness.
- [ ] **Dedicated Chat Page**:
    - [ ] Minimal interface focused on inference.
    - [ ] Ability to load local or server-side model weights.
    - [ ] Import/Export functionality for model versions.
- [ ] **Model Downloader**: Add a button to download the current server or local model directly from the UI.
- [ ] **Model Versioning**: Implement a model registry to track, compare, and rollback to previous checkpoints.

## Optimization & Research
- [ ] **Dynamic Sparsity**: Automatically adjust gradient sparsity based on individual node bandwidth and latency.
- [ ] **Optimizer Variants**: Support for Lion, RMSprop, or Sophia optimizers.
- [ ] **Advanced Monitoring**: Real-time dashboard showing samples/sec, active node map, and resource utilization.
- [ ] **Node Contribution Analytics**: Track and reward nodes based on the quality and volume of their gradient contributions.

## Reliability
- [ ] **Automated Backups**: Periodically back up checkpoints to external storage.
- [ ] **Fault Tolerance**: Improved handling of node dropouts and "Byzantine" (malicious) gradient submissions.
