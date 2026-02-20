"""Simplified EVM Blockchain wrapper for FLBlockchain contract."""

import os
import json
from typing import Dict
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()


class EVMBlockchain:
    """Wrapper for FLBlockchain smart contract."""

    def __init__(self):
        # Load config
        self.rpc_url = os.getenv("SEPOLIA_RPC_URL")
        self.private_key = os.getenv("PRIVATE_KEY")
        self.contract_address = os.getenv("CONTRACT_ADDRESS")

        if not all([self.rpc_url, self.private_key, self.contract_address]):
            raise ValueError("Missing env variables. Check .env file.")

        # Connect to Ethereum
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

    def _send_transaction(self, function_call):
        """Send transaction and wait for confirmation."""
        nonce = self.w3.eth.get_transaction_count(self.account.address)

        try:
            gas_estimate = function_call.estimate_gas(
                {'from': self.account.address})
            gas_limit = int(gas_estimate * 1.2)
        except:
            gas_limit = 500000

        transaction = function_call.build_transaction({
            'from': self.account.address,
            'nonce': nonce,
            'gas': gas_limit,
            'gasPrice': self.w3.eth.gas_price,
            'chainId': self.w3.eth.chain_id
        })

        signed = self.w3.eth.account.sign_transaction(
            transaction, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"  [EVM] Transaction sent: {tx_hash.hex()}")

        print(f"  [EVM] Waiting for confirmation...")
        receipt = self.w3.eth.wait_for_transaction_receipt(
            tx_hash, timeout=600)

        if receipt['status'] == 1:
            print(f"  [EVM] ✓ Confirmed in block {receipt['blockNumber']}")
            return receipt
        else:
            raise RuntimeError(f"Transaction failed: {receipt}")

    def add_local_model_block(self, fl_round, client_id, model_state_dict,
                              train_loss, num_examples, training_time, active_classes):
        """Add LOCAL block."""
        data = json.dumps({
            "client_id": client_id,
            "train_loss": train_loss,
            "num_examples": num_examples,
            "training_time": training_time,
            "active_classes": active_classes
        })

        print(
            f"\n  [EVM] Writing LOCAL block (Round {fl_round}, Client {client_id})...")
        function_call = self.contract.functions.addBlock(
            fl_round,
            "LOCAL",
            Web3.to_bytes(text=data)
        )
        self._send_transaction(function_call)

    def add_vote_block(self, fl_round, client_id, vote, reason, loss):
        """Add VOTE block."""
        data = json.dumps({
            "client_id": client_id,
            "vote": vote,
            "reason": reason,
            "loss": loss
        })

        status = "ACCEPTED" if vote else "REJECTED"
        print(f"  [EVM] Writing VOTE block (Client {client_id} → {status})...")
        function_call = self.contract.functions.addBlock(
            fl_round,
            "VOTE",
            Web3.to_bytes(text=data)
        )
        self._send_transaction(function_call)

    def add_global_model_block(self, fl_round, model_state_dict, accuracy,
                               f1_macro, auc_macro, loss, num_clients):
        """Add GLOBAL block."""
        data = json.dumps({
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "auc_macro": auc_macro,
            "loss": loss,
            "num_clients": num_clients
        })

        print(f"\n  [EVM] Writing GLOBAL block (Round {fl_round})...")
        function_call = self.contract.functions.addBlock(
            fl_round,
            "GLOBAL",
            Web3.to_bytes(text=data)
        )
        self._send_transaction(function_call)

    def verify_chain(self):
        """Verify chain integrity."""
        return self.contract.functions.verifyChain().call()

    def get_chain_length(self):
        """Get total blocks."""
        return self.contract.functions.getBlockCount().call()

    def get_round_summary(self, fl_round):
        """Get summary for a specific round (simplified for EVM)."""
        total_blocks = self.get_chain_length()
        return {
            "fl_round": fl_round,
            "total_blocks": total_blocks,
            "local_models": 0,  # Not tracked
            "accepted": 0,      # Not tracked
            "rejected": 0,      # Not tracked
        }

    def print_chain_summary(self):
        """Print summary."""
        length = self.get_chain_length()
        is_valid = self.verify_chain()

        print(f"\n  {'═'*60}")
        print(f"  EVM BLOCKCHAIN SUMMARY")
        print(f"  {'═'*60}")
        print(f"  Contract:       {self.contract_address}")
        print(f"  Total blocks:   {length}")
        print(f"  Chain integrity: {'✓ VALID' if is_valid else '✗ BROKEN'}")
        print(f"  {'═'*60}\n")


FLBlockchain = EVMBlockchain
