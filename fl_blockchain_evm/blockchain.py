"""Simplified EVM Blockchain wrapper for FLBlockchain contract.

Network: Base Sepolia (Ethereum Layer 2 testnet)
  - Chain ID : 84532
  - RPC      : https://sepolia.base.org  (or Alchemy/Infura endpoint)
  - Faucet   : https://www.coinbase.com/faucets/base-sepolia-faucet
  - Explorer : https://sepolia.basescan.org
  - Block time: ~2 seconds
  - Gas cost : very low base fees + Ethereum Sepolia security

Features:
  - _send_transaction supports fire-and-wait pattern: returns tx_hash immediately
    without blocking; call wait_for_pending() at end of round to confirm all.
  - add_round_summary_block: writes ONE LOCAL block and ONE VOTE block summarising
    ALL clients for the round, instead of one pair of blocks per client.
    This reduces blockchain writes from (2K + 1) to 3 per round.
"""

import os
import json
from typing import Dict, List, Optional
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()


class EVMBlockchain:
    """Wrapper for FLBlockchain smart contract."""

    def __init__(self):
        # Load config
        self.rpc_url = os.getenv("BASE_SEPOLIA_RPC_URL")
        self.private_key = os.getenv("PRIVATE_KEY")
        self.contract_address = os.getenv("CONTRACT_ADDRESS")

        if not all([self.rpc_url, self.private_key, self.contract_address]):
            raise ValueError("Missing env variables. Check .env file.")

        # Connect to Base Sepolia
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to {self.rpc_url}")

        print(
            f"  [EVM] Connected to network (Chain ID: {self.w3.eth.chain_id})")

        # Load account
        self.account = self.w3.eth.account.from_key(self.private_key)
        print(f"  [EVM] Using account: {self.account.address}")

        # Check balance
        balance = self.w3.eth.get_balance(self.account.address)
        balance_eth = self.w3.from_wei(balance, 'ether')
        print(f"  [EVM] Balance: {balance_eth:.4f} ETH")

        # Load contract
        with open('FLBlockchain_abi.json', 'r') as f:
            contract_abi = json.load(f)

        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.contract_address),
            abi=contract_abi
        )
        print(f"  [EVM] Contract loaded: {self.contract_address}")

        # Verify contract
        chain_length = self.contract.functions.getBlockCount().call()
        print(f"  [EVM] Current chain length: {chain_length} blocks")

        # Pending tx hashes queued for confirmation at end of round
        self._pending: List[bytes] = []

        # Nonce tracked manually so fire-and-wait works without collisions
        self._nonce: Optional[int] = None

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _next_nonce(self) -> int:
        """Return next nonce, incrementing our local counter."""
        if self._nonce is None:
            self._nonce = self.w3.eth.get_transaction_count(
                self.account.address)
        n = self._nonce
        self._nonce += 1
        return n

    def _send_transaction(self, function_call, fire_and_wait: bool = False):
        """Send a transaction.

        fire_and_wait=False (default): send and block until confirmed.
        fire_and_wait=True           : send immediately, store hash in
                                       self._pending, return without blocking.
                                       Call wait_for_pending() when ready.
        """
        nonce = self._next_nonce()

        try:
            gas_estimate = function_call.estimate_gas(
                {'from': self.account.address})
            gas_limit = int(gas_estimate * 1.2)
        except Exception:
            gas_limit = 500_000

        transaction = function_call.build_transaction({
            'from': self.account.address,
            'nonce': nonce,
            'gas': gas_limit,
            'gasPrice': self.w3.eth.gas_price,
            'chainId': self.w3.eth.chain_id,
        })

        signed = self.w3.eth.account.sign_transaction(
            transaction, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"  [EVM] Transaction sent: {tx_hash.hex()}")

        if fire_and_wait:
            self._pending.append(tx_hash)
            return tx_hash

        print(f"  [EVM] Waiting for confirmation...")
        receipt = self.w3.eth.wait_for_transaction_receipt(
            tx_hash, timeout=600)
        if receipt['status'] == 1:
            print(f"  [EVM] ✓ Confirmed in block {receipt['blockNumber']}")
            return receipt
        else:
            raise RuntimeError(f"Transaction failed: {receipt}")

    def wait_for_pending(self, timeout: int = 600):
        """Block until all pending (fire-and-wait) transactions are confirmed."""
        if not self._pending:
            return
        print(
            f"  [EVM] Waiting for {len(self._pending)} pending transaction(s)...")
        for tx_hash in self._pending:
            receipt = self.w3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=timeout)
            if receipt['status'] != 1:
                raise RuntimeError(f"Transaction failed: {receipt}")
            print(f"  [EVM] ✓ Confirmed in block {receipt['blockNumber']}")
        self._pending.clear()

    # ─────────────────────────────────────────────────────────
    # Per-round batch writes  (3 tx per round instead of 2K+1)
    # ─────────────────────────────────────────────────────────

    def add_round_summary_block(
        self,
        fl_round: int,
        clients: List[Dict],
        loss_mean: float,
        loss_std: float,
        threshold: float,
    ):
        """Write ONE LOCAL block and ONE VOTE block summarising all clients.

        clients: list of dicts with keys
            client_id, train_loss, num_examples, training_time, active_classes

        Both transactions are fired immediately (fire_and_wait=True).
        Call wait_for_pending() after add_global_model_block() to confirm
        all three round transactions together.
        """
        # ── LOCAL block: aggregated view of all clients this round ──
        local_data = json.dumps({
            "round": fl_round,
            "num_clients": len(clients),
            "clients": [
                {
                    "client_id":    c["client_id"],
                    "train_loss":   c["train_loss"],
                    "num_examples": c["num_examples"],
                    "training_time_seconds": c["training_time"],
                    "active_classes": c["active_classes"],
                }
                for c in clients
            ],
            "loss_mean": loss_mean,
            "loss_std":  loss_std,
            "threshold": threshold,
        })

        print(
            f"\n  [EVM] Firing LOCAL block (Round {fl_round}, {len(clients)} clients)...")
        self._send_transaction(
            self.contract.functions.addBlock(
                fl_round,
                "LOCAL",
                Web3.to_bytes(text=local_data),
            ),
            fire_and_wait=True,
        )

        # ── VOTE block: accepted/rejected verdict for each client ──
        votes = [
            {
                "client_id": c["client_id"],
                "vote":      "ACCEPTED" if c["train_loss"] <= threshold else "REJECTED",
                "loss":      c["train_loss"],
                "reason":    (
                    "loss within threshold"
                    if c["train_loss"] <= threshold
                    else f"loss {c['train_loss']:.4f} > threshold {threshold:.4f}"
                ),
            }
            for c in clients
        ]

        accepted = sum(1 for v in votes if v["vote"] == "ACCEPTED")
        rejected = len(votes) - accepted

        vote_data = json.dumps({
            "round":    fl_round,
            "threshold": threshold,
            "accepted": accepted,
            "rejected": rejected,
            "votes":    votes,
        })

        print(f"  [EVM] Firing VOTE block (Round {fl_round}: "
              f"{accepted} accepted / {rejected} rejected)...")
        self._send_transaction(
            self.contract.functions.addBlock(
                fl_round,
                "VOTE",
                Web3.to_bytes(text=vote_data),
            ),
            fire_and_wait=True,
        )

        return votes  # returned so server_app can print the table

    def add_global_model_block(
        self,
        fl_round: int,
        model_state_dict,
        accuracy: float,
        f1_macro: float,
        auc_macro: float,
        loss: float,
        num_clients: int,
    ):
        """Write GLOBAL block, then wait for all three round txs together."""
        data = json.dumps({
            "accuracy":    accuracy,
            "f1_macro":    f1_macro,
            "auc_macro":   auc_macro,
            "loss":        loss,
            "num_clients": num_clients,
        })

        print(f"\n  [EVM] Firing GLOBAL block (Round {fl_round})...")
        self._send_transaction(
            self.contract.functions.addBlock(
                fl_round,
                "GLOBAL",
                Web3.to_bytes(text=data),
            ),
            fire_and_wait=True,
        )

        # Now wait for LOCAL + VOTE + GLOBAL together
        print(
            f"  [EVM] Waiting for all Round {fl_round} transactions to confirm...")
        self.wait_for_pending()

    # ─────────────────────────────────────────────────────────
    # Query helpers
    # ─────────────────────────────────────────────────────────

    def verify_chain(self) -> bool:
        return self.contract.functions.verifyChain().call()

    def get_chain_length(self) -> int:
        return self.contract.functions.getBlockCount().call()

    def get_round_summary(self, fl_round: int) -> Dict:
        total_blocks = self.get_chain_length()
        return {
            "fl_round":     fl_round,
            "total_blocks": total_blocks,
        }

    def print_chain_summary(self):
        length = self.get_chain_length()
        is_valid = self.verify_chain()
        print(f"\n  {'═'*60}")
        print(f"  EVM BLOCKCHAIN SUMMARY")
        print(f"  {'═'*60}")
        print(f"  Contract:        {self.contract_address}")
        print(f"  Total blocks:    {length}")
        print(f"  Chain integrity: {'✓ VALID' if is_valid else '✗ BROKEN'}")
        print(f"  {'═'*60}\n")


FLBlockchain = EVMBlockchain
