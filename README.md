
# FL-Blockchain-EVM: Federated Learning with Blockchain Audit

## Overview

**FL-Blockchain-EVM** is a research framework for federated learning (FL) on medical IoT data, integrating Flower (FL orchestration), PyTorch (deep learning), and an Ethereum-compatible blockchain (for model and audit logging). The project demonstrates privacy-preserving, auditable, and robust FL for healthcare.


## Main Components and Code Explanation

### Data Handling and Model (`fl_blockchain_evm/task.py`)
- **Data Partitioning:** Loads the PTB-XL ECG dataset, partitions it into 10 simulated IoT clients, and applies balancing (ROS+RUS) to address class imbalance.
- **Augmentation:** Implements realistic ECG augmentations (noise, scaling, temporal shift, baseline wander) to simulate device variability.
- **Model:** Defines a 1D SE-ResNet (`Net` class) for multi-label ECG classification into 5 superclasses (NORM, MI, STTC, CD, HYP).
- **Loss Function:** Uses Focal Loss to handle class imbalance.
- **Training/Evaluation:** Provides `train` and `test` functions for model optimization and metric computation, including per-class and macro metrics.

### Federated Learning Orchestration

#### Server Logic (`fl_blockchain_evm/server_app.py`)
- **ServerApp:** Entry point for Flower's server process.
- **Custom Aggregation:** Uses `MedicalFedAvg` to ensure equal weighting for all clients, regardless of data size.
- **Training/Evaluation Rounds:** Aggregates client updates, evaluates the global model, and logs metrics.
- **Blockchain Logging:** After each round, writes local, vote, and global model blocks to the blockchain using the `EVMBlockchain` wrapper.
- **Metrics Logging:** Saves all round metrics to `outputs/results.json` for later analysis and plotting.

#### Client Logic (`fl_blockchain_evm/client_app.py`)
- **ClientApp:** Entry point for Flower's client process.
- **Training:** Each client trains the model on its partitioned data and returns model weights and training metrics.
- **Evaluation:** Each client evaluates the global model on its local validation set and returns evaluation metrics.

#### Custom Strategy (`fl_blockchain_evm/priority_strategy.py`)
- **MedicalFedAvg:** Subclasses Flower's FedAvg to force equal aggregation weights, ensuring rare-pathology clients are not underrepresented.

### Blockchain Integration

#### Smart Contract (`FLBlockchain.sol`)
- **SimpleFLBlockchain:** Solidity contract that stores a chain of blocks, each representing a local model, vote, or global model.
- **Block Structure:** Each block contains round info, type, content hash, previous hash, timestamp, and submitter.
- **Access Control:** Only authorized clients can submit local blocks; only the owner can submit global/vote blocks.
- **Chain Verification:** Provides a function to verify the integrity of the chain.

#### Python Blockchain Wrapper (`fl_blockchain_evm/blockchain.py`)
- **EVMBlockchain:** Handles connection to Ethereum, contract interaction, and transaction management.
- **Block Submission:** Provides methods to submit local, vote, and global model blocks, serializing relevant data as JSON.
- **Chain Summary:** Can print and verify the blockchain's integrity.

### Experiment Configuration
- **`pyproject.toml`:** Central configuration for dependencies, Flower app entry points, and FL hyperparameters (rounds, learning rate, etc.).
- **`.env`:** Stores sensitive blockchain connection info (RPC URL, private key, contract address).

### Results and Visualization
- **Metrics Logging:** All training and evaluation metrics are appended to `outputs/results.json`.
- **Plotting Scripts:**
  - `plot_results.py`: Generates publication-ready figures (convergence, per-class, per-client, etc.) and a Markdown summary.
  - `plot_metrics.py`, `plot_training_curves.py`, `plot_distribution.py`: Additional scripts for quick checks and data distribution visualization.

---

## Project Workflow

1. **Install dependencies:**  
	`pip install -e .`
2. **Prepare environment:**  
	- Set up `.env` with blockchain credentials.
	- Place PTB-XL data in `data/ptb-xl/`.
3. **Run simulation:**  
	`flwr run .`
4. **Visualize results:**  
	`python plot_results.py` (and other plotting scripts as needed).

---

## Launch Instructions

### 1. Install Dependencies
```bash
pip install -e .
```

### 2. Prepare Environment
- Ensure `.env` is set with your Ethereum RPC URL, private key, and contract address.
- Place the PTB-XL dataset in `data/ptb-xl/` as required by the code.

### 3. Run Local Simulation
```bash
flwr run .
```
- This will start a Flower simulation with blockchain logging.
- Results and blockchain logs will be saved in `outputs/`.

### 4. Plot Results
```bash
python plot_results.py
```
- Generates all figures and a Markdown summary in `metrics/`.

### 5. (Optional) Plot Data Distribution
```bash
python plot_distribution.py
```

### 6. (Optional) Plot Training Curves
```bash
python plot_training_curves.py
```

---

## Notes
- For more details, see the code and comments in each script, and refer to the official Flower documentation for advanced usage.
