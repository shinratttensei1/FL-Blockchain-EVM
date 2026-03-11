"""FL-Blockchain-EVM: A Flower / PyTorch app.

Subpackages:
    core/        – ML pipeline (constants, model, data, training)
    infra/       – External services (EVM blockchain, IPFS storage)
    strategy/    – FL aggregation strategies (MedicalFedAvg)
    dashboard/   – Live monitoring (FastAPI server, SSE, state)

Top-level modules:
    task         – Backward-compatible re-export layer
    utils        – Shared utility functions
    client_app   – Flower ClientApp entry point
    server_app   – Flower ServerApp entry point
"""
