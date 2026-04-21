# Blockchain-Based Federated Learning for ECG Superclass Classification

> **Course project** — Federated learning over the mHealth dataset with an Base mainnet smart-contract ledger for auditability and tamper-proof model tracking.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [How It Works](#how-it-works)
   - [The Model](#the-model)
   - [Data Pipeline](#data-pipeline)
   - [Federated Learning Loop](#federated-learning-loop)
   - [Blockchain Ledger](#blockchain-ledger)
   - [Voting & Filtering](#voting--filtering)
   - [IPFS Off-Chain Storage](#ipfs-off-chain-storage)
5. [Live Dashboard](#live-dashboard)
6. [Research Background](#research-background)
7. [Requirements](#requirements)
8. [Setup & Installation](#setup--installation)
9. [Configuration](#configuration)
10. [Launching the Experiment](#launching-the-experiment)
11. [Outputs](#outputs)
12. [Smart Contract](#smart-contract)
13. [Limitations & Future Work](#limitations--future-work)

---

## Overview

This project implements a **federated learning (FL) system** for 12-lead ECG classification that records every training event — local model updates, voting decisions, and aggregated global models — to an **Ethereum blockchain** (Base main testnet) via a custom Solidity smart contract.

**Clinical task:** Classify each ECG recording into one or more of **5 superclasses** defined by the PTB-XL standard:

| Code | Superclass | Example diagnoses |
|------|-----------|-------------------|
| NORM | Normal | — |
| MI | Myocardial Infarction | IMI, AMI, ASMI … |
| STTC | ST/T-Change | NDT, ISC\_, ISCAL … |
| CD | Conduction Disturbance | LBBB, RBBB, WPW … |
| HYP | Hypertrophy | LVH, RVH … |

The dataset is partitioned across **10 simulated clients**, each training a local model on its own shard. A central server aggregates the models each round, evaluates the global model, and writes an immutable summary to the blockchain.

---

## System Architecture

```mermaid
flowchart TD
    subgraph FL_NETWORK ["Flower Framework"]
        direction TB
        SERVER[["Flower Server (MedicalFedAvg)"]]
        
        subgraph CLIENTS ["Clients (RP4 Edge Devices)"]
            C0["Client 0 (Train)"]
            C1["Client 1 (Train)"]
            CX["..."]
            C9["Client 9 (Eval)"]
        end

        SERVER <==>|"Weights & Updates"| CLIENTS
    end

    subgraph OFF_CHAIN ["IPFS Layer (Off-Chain)"]
        IPFS_BRIDGE["ipfs_storage.py"]
        PINATA["Pinata / web3.storage"]
    end

    subgraph ON_CHAIN ["Blockchain Layer"]
        EVM_BRIDGE["blockchain.py (Web3.py)"]
        CONTRACT["SimpleFLBlockchain (Solidity)"]
        NETWORK_TAGP["Base main Testnet"]
    end

    SERVER ---->|"Pin Model + Metrics"| IPFS_BRIDGE
    IPFS_BRIDGE ---->|"HTTP API"| PINATA
    PINATA -.->|"CID"| IPFS_BRIDGE
    SERVER ---->|"Write Events + CIDs"| EVM_BRIDGE
    EVM_BRIDGE ---->|"Web3 Tx"| CONTRACT
    CONTRACT -.-> NETWORK_TAGP
```

---

## Project Structure

```
fl_blockchain_evm/
├── task.py                  # Dataset, model definition, train/test logic
├── client_app.py            # Flower client (train + evaluate handlers)
├── server_app.py            # Flower server (aggregation, evaluation, blockchain writes)
├── blockchain.py            # Web3 wrapper for the smart contract
├── ipfs_storage.py          # IPFS off-chain storage (Pinata/kubo/web3.storage)
├── priority_strategy.py     # MedicalFedAvg — equal-weight aggregation strategy
├── live_state.py            # Thread-safe state container for dashboard streaming
│
├── fl_dashboard_server.py   # FastAPI backend for live dashboard (REST + SSE)
├── fl_dashboard.html        # Full dashboard — topology, metrics, blockchain, IPFS
├── fl_topology.html         # Lightweight topology-only dashboard (legacy)
│
├── SimpleFLBlockchain.sol   # Solidity smart contract (deployed via Remix IDE)
├── FLBlockchain_abi.json    # ABI copied from Remix after deployment
│
├── data/
│   └── ptb-xl/              # PTB-XL dataset (downloaded separately)
│
├── outputs/                 # Auto-created at runtime
│   ├── results.json         # Per-round training & evaluation metrics (JSONL)
│   └── cm_round_N.png       # Confusion matrix plots per round
│
├── final_model.pt           # Saved global model weights after all rounds
│
├── .env                     # Secrets (not committed)
├── pyproject.toml           # Flower project config & hyperparameters
└── README.md
```

---

## How It Works

### The Model

**`Net`** (`task.py`) is a 4-stage **SE-ResNet** (Squeeze-and-Excitation Residual Network) for 1-D time-series:

```
Input: (B, 12, 1000)   — batch × 12 leads × 1000 time-steps @ 100 Hz
  └─ InputBN
  └─ Stage 1: Conv1d(12→32, k=15) → BN → ReLU → MaxPool(4) → 2× SEResBlock(32, k=7)
  └─ Stage 2: Conv1d(32→64, k=7)  → BN → ReLU → MaxPool(4) → 2× SEResBlock(64, k=5)
  └─ Stage 3: Conv1d(64→128, k=5) → BN → ReLU → MaxPool(2) → 2× SEResBlock(128, k=3)
  └─ Stage 4: Conv1d(128→256, k=3)→ BN → ReLU → MaxPool(2) → 1× SEResBlock(256, k=3)
  └─ GlobalAvgPool → Dropout(0.3) → Linear(256→5)
Output: (B, 5) raw logits  — multi-label sigmoid classification
```

~200K parameters. The SE block applies per-channel attention after each residual pair.

**Loss:** `FocalLoss` (γ=2, class-frequency alpha weights) — addresses the severe class imbalance in PTB-XL (NORM ≫ HYP/MI).

**Optimizer:** `AdamW` with cosine-annealing LR schedule and a linear warm-up for the first 10% of steps.

**Training augmentations** (applied per mini-batch):
- Gaussian noise (σ=0.05, p=0.8)
- Per-lead amplitude scaling (×0.8–1.2, p=0.5)
- Temporal shift (±30 samples, p=0.5)
- Baseline wander (sinusoidal, p=0.3)
- Mixup (α=0.3)

### Data Pipeline

PTB-XL uses the official 10-fold stratified split:

- **Folds 1–8** → training set (partitioned across clients)
- **Folds 9–10** → held-out test set (used for global evaluation only)

Each client receives a **sequential shard** of the shuffled training set (seed=42). Per-client z-score normalization is applied using that client's own training statistics.

To address the multi-label class imbalance, each client applies **ROS+RUS** (Random Over/Under Sampling) before training:

```
target = m_s + (m_l - m_s) × β      (β=1.0 → full equalization)
```

where `m_l` = largest class count, `m_s` = smallest class count. Under-represented classes are oversampled; over-represented classes are subsampled. A `WeightedRandomSampler` further reinforces balance during batching.

**Threshold optimization:** At evaluation time, per-class thresholds are tuned over `[0.05, 0.95]` in 0.02 steps to maximize per-class F1 (rather than using a fixed 0.5).

### Federated Learning Loop

Each round:

1. **Server** broadcasts the current global model weights to all selected clients.
2. **Each client** receives the weights, fine-tunes for `local-epochs` using its local shard, and returns the updated weights + scalar metrics (loss, samples, training time, active classes).
3. **Server** aggregates via `MedicalFedAvg` — a `FedAvg` variant that forces equal contribution weight (`weight=1`) for every client regardless of shard size, preventing large-shard clients from dominating.
4. **Global evaluation** is performed on the held-out test set (partition 0 used as proxy for the full test set).
5. **Blockchain** records three block types for the round (see below).

### Blockchain Ledger

`EVMBlockchain` (`blockchain.py`) wraps the deployed Solidity contract via **web3.py**. Every round writes the following blocks on-chain:

| Block type | When written | Payload stored |
|------------|-------------|----------------|
| `LOCAL` | After each round (per client) | `client_id`, `train_loss`, `num_examples`, `training_time`, `active_classes` |
| `VOTE` | After each round (per client) | `client_id`, `vote` (ACCEPTED/REJECTED), `reason`, `loss` |
| `GLOBAL` | After each round (global evaluation) | `accuracy`, `f1_macro`, `auc_macro`, `loss`, `num_clients` |

The chain can be integrity-verified at any time via `verifyChain()` on the contract — each block stores the `keccak256` hash of its predecessor.

### Voting & Filtering

Before writing VOTE blocks, the server computes a per-round loss threshold:

```python
threshold = mean(train_losses) + std(train_losses)
```

Clients whose `train_loss > threshold` are marked **REJECTED** on-chain. This is a simple statistical outlier filter — clients with anomalously high loss (potential data quality issues or poisoning) are flagged for auditability, though all clients still contribute to aggregation in this implementation.

---

### IPFS Off-Chain Storage

The system uses **IPFS** (InterPlanetary File System) as a content-addressable off-chain storage layer, optimised for **Raspberry Pi 4** edge devices:

**What is stored on IPFS per round:**

| Artifact | IPFS Key | Approx. Size | Purpose |
|----------|----------|-------------|--------|
| Training metrics | `round_N_local` | ~5 KB | Per-client loss, samples, timing |
| Vote decisions | `round_N_votes` | ~3 KB | Accept/reject verdicts |
| Global model weights | `round_N_global_model` | ~250 KB (gzip) | Full SE-ResNet state_dict |
| Evaluation metrics | `round_N_global_metrics` | ~2 KB | Accuracy, F1, AUC, per-class |

**RP4 design decisions:**
- **No IPFS daemon on devices** — uses Pinata HTTP API (saves ~200 MB RAM vs kubo)
- **Gzip compression** — 200K-param model: 800 KB → ~250 KB
- **BytesIO streaming** — no temp files (avoids SD card wear on RP4)
- **Exponential backoff with jitter** — handles flaky Wi-Fi on edge deployments
- **Graceful degradation** — if IPFS is unreachable, FL training continues normally

**Tamper-proof linkage:** Each on-chain block payload includes the IPFS CID of its detailed off-chain data. Since the blockchain stores `keccak256(payload)`, any modification to either the IPFS content or the CID reference would be detected by `verifyChain()`.

**Supported backends:**

| Backend | Best for | RAM overhead | Setup |
|---------|----------|-------------|-------|
| `pinata` | RP4 devices (recommended) | ~2 MB | Free API key from [Pinata](https://app.pinata.cloud) |
| `web3storage` | Alternative cloud | ~2 MB | Free token from [web3.storage](https://web3.storage) |
| `local` | Server-hosted IPFS | ~200 MB (kubo) | Run `ipfs daemon` on server only |

**Verifying IPFS content:**

```python
from fl_blockchain_evm.ipfs_storage import IPFSStorage

ipfs = IPFSStorage(backend="pinata")

# Fetch a global model checkpoint from any round
sd = ipfs.fetch_model("QmXyz...")  # CID from results.json or on-chain data

# Verify content integrity
assert ipfs.verify_content("QmXyz...", expected_sha256="abc123...")
```

---

## Live Dashboard

The project includes a **real-time dashboard** (`fl_dashboard.html`) served by a FastAPI backend (`fl_dashboard_server.py`). It connects to the running simulation via SSE (Server-Sent Events) and to the Base main blockchain directly.

**Features:**

| Panel | What it shows |
|-------|---------------|
| **Network Topology** | Animated canvas with 10 device nodes, FL server, blockchain node, and IPFS node — live packet animations for weight transfers, blockchain writes, and IPFS pins |
| **Stat Strip** | Round, Accuracy, F1 Macro, AUC Macro, Chain Blocks, IPFS Pins, Active Clients |
| **Client Training Table** | Per-device loss, sample count, duration, active classes, accept/reject vote |
| **Loss Threshold Votes** | Visual bar chart of each client's loss vs. threshold |
| **Metrics History Chart** | Hand-drawn line chart of accuracy, F1, AUC over rounds |
| **Per-Superclass F1** | Bar breakdown for NORM, MI, STTC, CD, HYP |
| **Confusion Matrix** | 5×5 heatmap with intensity shading |
| **Blockchain Ledger** | Live connection to Base main — shows recent blocks (LOCAL/VOTE/GLOBAL), chain validity badge, expandable full history |
| **IPFS Off-Chain Storage** | Active/disabled status, total pin count, per-round CID list with clickable Pinata gateway links |

**Interactive controls:**
- **Round slider** — scrub through historical rounds to inspect past metrics
- **LIVE button** — snap back to the latest round
- **VIEW ALL** — expand the blockchain ledger to show all blocks
- **Node tooltips** — hover over any topology node for detailed state info

---

## Research Background

This project draws conceptually from three papers:

- **SPBFL-IoV** (Ullah et al., *Computer Networks* 2025) — blockchain integration in FL for tamper-proof model update recording; filtering/clipping mechanisms for poisoning defense.
- **BFMIL** (Wu et al., *Computers & Electrical Engineering* 2026) — three-block on-chain storage structure (local model blocks, vote score blocks, global model blocks); equal-weight aggregation for non-IID data.
- **BCS+DL** (Pajila et al., *Peer-to-Peer Networking and Applications* 2026) — blockchain security scheme for distributed learning systems.

The key differences from these works: this project does **not** implement homomorphic encryption, metric/triplet loss, or deep-learning-based anomaly detection. The on-chain filtering is simpler (loss threshold vs. Euclidean distance from global model), and the target domain is 12-lead ECG classification rather than image classification or WSN security.

---

### Papers to Compare Against

#### Category 1 — FL + Blockchain on ECG / Medical IoT (closest match)
- **Rajagopal et al., 2025 — "Leveraging Blockchain and FL in Edge-Fog-Cloud Computing for ECG Data in IoT"** Journal of Network and Computer Applications, vol. 233. A highly comparable study involving FL and blockchain for ECG data across IoT edge devices. It utilizes a three-tier edge-fog-cloud architecture and smart contracts for decision-making. Key difference: that study employs a fog layer intermediary, whereas the current system is flat (clients → server → Ethereum directly). It also lacks a per-round immutable voting mechanism. https://doi.org/10.1016/j.jnca.2024.104037
- **Otoum et al., 2024 — "Enhancing Heart Disease Prediction with Federated Learning and Blockchain Integration"** Future Internet, 16(372). Blockchain-integrated FL for cardiac data. Uses blockchain to secure model updates but does not target ECG superclass classification on PTB-XL or address class imbalance with ROS+RUS. It also lacks an on-chain voting filter. https://doi.org/10.3390/fi16100372

#### Category 2 — FL on ECG / PTB-XL (same dataset, no blockchain)
- **Jimenez et al., 2024 — "Application of FL Techniques for Arrhythmia Classification Using 12-Lead ECG Signals"** arXiv:2208.10993v3. This serves as a primary baseline, specifically for the ROS+RUS balancing strategy. It achieves F1 scores in the FL Non-IID setting on 12-lead ECG but utilizes no blockchain, no IoT edge deployment, and no immutable ledger. It provides a direct comparison point for classification performance. https://doi.org/10.48550/arXiv.2208.10993
- **FL-LSTM (Explainable FL), 2025 — "Explainable Federated Learning for Multi-Class Heart Disease Diagnosis via ECG Fiducial Features"** Diagnostics, MDPI, Dec 2025. Achieves 92% accuracy, 99% AUC, and 91% F1 score across three heterogeneous ECG datasets using an FL-LSTM framework, with SHAP and LIME for interpretability. No blockchain, IoT hardware, or specific class-imbalance handling is included. The current model is architecturally simpler but adds the blockchain audit layer. https://doi.org/10.3390/diagnostics15243110
- **FL-GAF (Gramian Angular Field), 2025 — "Federated Learning with GAF for Privacy-Preserving ECG Classification on Heterogeneous IoT Devices"** IEEE ComComAp 2025. Transforms 1D ECG signals into 2D Gramian Angular Field images and deploys FL across heterogeneous IoT devices including a Raspberry Pi 4. While it demonstrates classification accuracy of 95.18%, it lacks a blockchain audit layer. 

#### Category 3 — Blockchain + FL on IoT (same infrastructure, different domain)
- **Leash-FL, 2025 — "Lightweight ECC-Based Self-Healing FL Framework for Secure IIoT Networks"** Sensors, MDPI, Nov 2025. Records all outcomes immutably on a lightweight blockchain; each record includes the hash of the submitted update, pseudonym, timestamp, and verdict. Key difference: focused on the industrial IoT domain rather than medical ECG. It utilizes a permissioned/private blockchain, whereas this system utilizes public Ethereum. https://doi.org/10.3390/s25226867
- **FBCI-SHS, 2025 — "Blockchain Framework with IoT Device using FL for Sustainable Healthcare"** Scientific Reports, Jul 2025. Healthcare domain + FL + blockchain + IoT implementation. Uses PBFT consensus (computationally heavier than the PoA used here) and does not address ECG superclass imbalance.

#### Category 4 — PTB-XL Benchmarks (performance reference only)
- **Strodthoff et al. (PTB-XL benchmark)** — The standard reference for achievable performance on PTB-XL superclass classification. Achieves macro-AUC scores of 0.93 using ResNet- and Inception-based networks. This is the ceiling for pure classification performance.
- **MRF-CNN on PTB-XL** — Achieves F1 score of 0.72 and AUC of 0.93 in superclass classification on PTB-XL. This provides a direct F1 comparison target for the current model.

---

## Requirements

### Python

```
python >= 3.10
flwr >= 1.10
torch >= 2.0
web3 >= 6.0
requests >= 2.31
wfdb
numpy
pandas
scikit-learn
matplotlib
seaborn
python-dotenv
```

### Other

- **(Optional) IPFS pinning service** for off-chain model storage
  - [Pinata](https://app.pinata.cloud) — free tier: 500 pins, 1 GB (recommended for RP4)
  - Or [web3.storage](https://web3.storage) — free tier: 5 GB
  - Or local IPFS node (kubo) running on the server machine
- **Base main testnet ETH** for gas fees
  - Faucet: Get the main Testnet ETH first, then use any bridge to swap main ETH with Base main ETH. I used [https://superbridge.app](https://superbridge.app)
  - You'll need ~0.01-0.03 ETH for a full 10-round simulation
- **Base main RPC endpoint**
  - Public RPC: `https://main.base.org` (free, no API key needed)
  - Or use Alchemy/Infura for better reliability
- **Base main Network Details**
  - Chain ID: 84532
  - Explorer: [https://main.basescan.org](https://main.basescan.org)
  - Block time: ~2 seconds
  - Gas costs: Very low (L2 benefits)

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/shinratttensei1/FL-Blockchain-EVM.git
cd FL-Blockchain-EVM
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt .
# or
pip install flwr torch web3 wfdb numpy pandas scikit-learn matplotlib seaborn python-dotenv
```

### 4. Download mHealth manually

// needs to change to mhealth

> The code uses the **100 Hz** version (`records100` / `filename_lr` column). Make sure the `data/ptb-xl/` folder is at the project root.

A numpy cache will be built automatically on the first run and saved to `data/ptb-xl/.cache/` to speed up subsequent runs.

### 5. Deploy the smart contract with Remix

No Hardhat or local Node setup is needed. The contract is deployed directly from the browser using [Remix IDE](https://remix.ethereum.org).

1. Open [https://remix.ethereum.org](https://remix.ethereum.org).
2. Create a new file and paste the contents of `SimpleFLBlockchain.sol`.
3. In the **Solidity Compiler** tab, select compiler version `0.8.20` and compile.
4. In the **Deploy & Run Transactions** tab:
   - Set environment to **Injected Provider - MetaMask** (make sure MetaMask is connected to **Base main**).
   - Click **Deploy**.
5. After deployment, copy the **contract address** from the Remix console.
6. In the **Compilation Details** panel, copy the **ABI** and save it as `FLBlockchain_abi.json` in the project root.

> **Note:** You need Base main testnet ETH for gas. Get some from the [Coinbase Faucet](https://www.coinbase.com/faucets/base-main-faucet).

**To add Base main to MetaMask:**
- Network Name: Base main
- RPC URL: `https://main.base.org`
- Chain ID: 84532
- Currency Symbol: ETH
- Block Explorer: `https://main.basescan.org`

### 6. Configure environment variables

Create a `.env` file in the project root (never commit this file):

```env
BASE_main_RPC_URL=https://main.base.org
PRIVATE_KEY=0x<YOUR_WALLET_PRIVATE_KEY>
CONTRACT_ADDRESS=0x<DEPLOYED_CONTRACT_ADDRESS>

# IPFS off-chain storage (optional — leave IPFS_BACKEND empty to disable)
IPFS_BACKEND=pinata
PINATA_JWT=<YOUR_PINATA_JWT_TOKEN>
# PINATA_GATEWAY=https://gateway.pinata.cloud/ipfs/   # optional override
```

---

## Configuration

All FL hyperparameters live in `pyproject.toml`:

```toml
[tool.flwr.app.config]
num-server-rounds = 5
fraction-train   = 1.0      # fraction of clients selected per round
local-epochs     = 3
lr               = 2e-3
```

Partition count is set in the `[tool.flwr.federations.local-simulation]` section:

```toml
[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
```

---

## Launching the Experiment

All commands assume you are in the project root with the virtual environment activated and `.env` populated.

### Run with Flower simulation (recommended for local testing)

```bash
flwr run .
```

This launches a local simulation with 10 virtual clients. The blockchain writes are real — each round will send transactions to Base main, so ensure your wallet has sufficient testnet ETH (roughly 0.01 ETH per round with default settings).

**What happens during training:**
- Each round writes exactly **3 transactions** to Base main:
  1. `LOCAL` block - aggregated client training data
  2. `VOTE` block - acceptance/rejection decisions
  3. `GLOBAL` block - global model evaluation metrics
- Results are saved to `outputs/results.json` (JSONL format)
- Confusion matrices saved as `outputs/cm_round_N.png`
- Final model saved as `final_model.pt`

### Monitor training with the live dashboard

**Step 1:** Start the dashboard backend server (in a second terminal):

```bash
source venv/bin/activate  # or .venv/bin/activate
python -m fl_blockchain_evm.fl_dashboard_server
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Step 2:** Open the dashboard in your browser:

- Open: [http://localhost:8000](http://localhost:8000)

> **Important:** The backend server must be running for the dashboard to display blockchain data and live metrics.

The dashboard shows:
- ✅ Live network topology with animated packet flows (devices → server → blockchain → IPFS)
- 📊 Real-time training metrics (accuracy, F1, AUC) with interactive round slider
- 🎯 Per-client training loss and vote decisions
- 📈 Historical metrics charts
- 🔗 **Blockchain ledger** with live Base main connection and chain validity indicator
- 📡 **IPFS panel** with per-round CID links to Pinata gateway
- 🎨 Confusion matrix heatmap and per-superclass F1 bars

> **Tip:** Start the dashboard server before or during training to see live updates. The dashboard connects to Base main to display actual blockchain blocks and reads IPFS CIDs from training results.

### Verify blockchain integrity after the run

```python
from fl_blockchain_evm.blockchain import EVMBlockchain
bc = EVMBlockchain()
print("Chain valid:", bc.verify_chain())
print("Total blocks:", bc.get_chain_length())
bc.print_chain_summary()
```

---

## Outputs

After a successful run the following files are created:

```
outputs/
├── results.json          # JSONL — one JSON object per line, types:
│                         #   "device_training"  — per-round client metrics
│                         #   "client_eval"      — per-round client evaluation
│                         #   "global"           — global eval + ipfs_cids (if enabled)
├── cm_round_1.png        # Confusion matrix heatmap, round 1
├── cm_round_2.png        # ...
└── cm_round_N.png

final_model.pt            # PyTorch state dict of the final global model
```

When IPFS is enabled, each `"global"` record in `results.json` includes an `ipfs_cids` field:

```json
{
  "type": "global",
  "round": 3,
  "accuracy": 0.812,
  "ipfs_cids": {
    "local_cid": "QmXyz...",
    "vote_cid": "QmAbc...",
    "model_cid": "QmDef...",
    "metrics_cid": "QmGhi..."
  }
}
```

### Reading results

```python
import json

with open("outputs/results.json") as f:
    records = [json.loads(line) for line in f if line.strip()]

global_rounds = [r for r in records if r["type"] == "global"]
for r in global_rounds:
    print(f"Round {r['round']:2d} | "
          f"Acc={r['accuracy']:.3f} | "
          f"F1={r['f1_macro']:.3f} | "
          f"AUC={r['auc_macro']:.3f}")
```

### Reloading the final model

```python
import torch
from fl_blockchain_evm.task import Net

model = Net()
model.load_state_dict(torch.load("final_model.pt", map_location="cpu"))
model.eval()
```

---

## Smart Contract

**`SimpleFLBlockchain.sol`** — deployed on Base main testnet.

Inherits from OpenZeppelin's `Ownable` and `Pausable`. Key design decisions:

- **Genesis block** is created at deployment with a fixed hash, anchoring the chain.
- `addBlock(flRound, blockType, data)` — hashes `data` with `keccak256`, chains to the previous block's hash, emits a `BlockAdded` event. The contract owner (the server wallet) can write any block type; authorized clients can only write `LOCAL` blocks.
- `verifyChain()` — iterates all blocks and checks `blocks[i].previousHash == blocks[i-1].contentHash`. Returns `false` if any tampering is detected.
- `pause()` / `unpause()` — emergency stop controlled by the owner.

```
Functions:
  authorizeClient(address)   onlyOwner
  revokeClient(address)      onlyOwner
  addBlock(uint, string, bytes) → uint    whenNotPaused
  getBlock(uint) → Block
  getBlockCount() → uint
  verifyChain() → bool
  getLatestBlock() → Block
  pause() / unpause()        onlyOwner
```

The Python wrapper (`blockchain.py`) serializes all metadata as JSON, encodes it to `bytes`, and calls `addBlock`. Only the `keccak256` hash of the payload is stored on-chain; the raw data is not.

---