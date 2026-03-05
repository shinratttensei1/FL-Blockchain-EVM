#!/usr/bin/env python3
"""
Benchmark suite for FL-Blockchain-IPFS ECG Stack
=================================================
Uses YOUR actual final_model.pt and real PTB-XL test data.

Run from your project root (where task.py, final_model.pt, data/ live):

  # Minimal — model + compute benchmarks only
  python benchmark_stack.py --no-blockchain --no-ipfs

  # With real PTB-XL test evaluation
  python benchmark_stack.py --no-blockchain --no-ipfs --with-testdata

  # Full stack (blockchain + IPFS configured in .env)
  python benchmark_stack.py --with-testdata

Results → benchmark_results.json
"""

import argparse
import gc
import gzip
import io
import json
import os
import sys
import time
import traceback
import hashlib
import numpy as np
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn

# ── Import YOUR model + data + test from task.py ──────────────────
try:
    from task import Net, NUM_CLASSES, SC_NAMES, load_data, test as evaluate_model
    TASK_OK = True
except ImportError:
    try:
        from fl_blockchain_evm.task import Net, NUM_CLASSES, SC_NAMES, load_data, test as evaluate_model
        TASK_OK = True
    except ImportError:
        TASK_OK = False
        print("[WARN] Cannot import task.py — will use inline model definition")

if not TASK_OK:
    NUM_CLASSES = 5
    SC_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

    class _SEResBlock(nn.Module):
        def __init__(self, ch, ks=5):
            super().__init__()
            pad = ks // 2
            self.conv1 = nn.Conv1d(ch, ch, ks, padding=pad, bias=False)
            self.bn1 = nn.BatchNorm1d(ch)
            self.conv2 = nn.Conv1d(ch, ch, ks, padding=pad, bias=False)
            self.bn2 = nn.BatchNorm1d(ch)
            mid = max(ch // 4, 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(ch, mid), nn.ReLU(inplace=True),
                nn.Linear(mid, ch), nn.Sigmoid(),
            )

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out * self.se(out).unsqueeze(-1)
            return torch.relu(out + x)

    class Net(nn.Module):
        def __init__(self, num_classes=5):
            super().__init__()
            self.input_bn = nn.BatchNorm1d(12)
            self.conv1 = nn.Conv1d(12, 32, 15, padding=7, bias=False)
            self.bn1 = nn.BatchNorm1d(32)
            self.pool1 = nn.MaxPool1d(4)
            self.res1a = _SEResBlock(32, 7)
            self.res1b = _SEResBlock(32, 7)
            self.conv2 = nn.Conv1d(32, 64, 7, padding=3, bias=False)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool2 = nn.MaxPool1d(4)
            self.res2a = _SEResBlock(64, 5)
            self.res2b = _SEResBlock(64, 5)
            self.conv3 = nn.Conv1d(64, 128, 5, padding=2, bias=False)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool3 = nn.MaxPool1d(2)
            self.res3a = _SEResBlock(128, 3)
            self.res3b = _SEResBlock(128, 3)
            self.conv4 = nn.Conv1d(128, 256, 3, padding=1, bias=False)
            self.bn4 = nn.BatchNorm1d(256)
            self.pool4 = nn.MaxPool1d(2)
            self.res4 = _SEResBlock(256, 3)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.input_bn(x)
            x = self.res1b(self.res1a(self.pool1(
                torch.relu(self.bn1(self.conv1(x))))))
            x = self.res2b(self.res2a(self.pool2(
                torch.relu(self.bn2(self.conv2(x))))))
            x = self.res3b(self.res3a(self.pool3(
                torch.relu(self.bn3(self.conv3(x))))))
            x = self.res4(self.pool4(torch.relu(self.bn4(self.conv4(x)))))
            return self.fc(self.drop(self.gap(x).squeeze(-1)))

    load_data = None
    evaluate_model = None

try:
    from web3 import Web3
    from dotenv import load_dotenv
    load_dotenv()
    WEB3_OK = True
except ImportError:
    WEB3_OK = False


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _dev(force=None):
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _serialize_sd(sd):
    buf = io.BytesIO()
    torch.save(sd, buf)
    raw = buf.getvalue()
    buf.close()
    return raw


def _gzip_bytes(data, level=6):
    return gzip.compress(data, compresslevel=level)


def _bytes_fmt(b):
    if b < 1024:
        return f"{b} B"
    if b < 1024**2:
        return f"{b/1024:.1f} KB"
    return f"{b/1024**2:.2f} MB"


def _section(title):
    print(f"\n{'═'*60}\n  {title}\n{'═'*60}")


def _row(label, value, unit=""):
    print(f"  {label:42s}  {value:>15s} {unit}")


def _load_model(path, device):
    model = Net()
    sd = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


# ══════════════════════════════════════════════════════════════════
#  BENCHMARKS
# ══════════════════════════════════════════════════════════════════

def bench_model_profile(model):
    _section("MODEL PROFILE (final_model.pt)")
    sd = model.state_dict()
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    raw = _serialize_sd(sd)
    compressed = _gzip_bytes(raw)
    param_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    buffer_bytes = sum(b.element_size() * b.numel() for b in model.buffers())

    _row("Parameters (total)", f"{n_params:,}")
    _row("Parameters (trainable)", f"{n_train:,}")
    _row("State dict (raw torch.save)", _bytes_fmt(len(raw)))
    _row("State dict (gzip level-6)", _bytes_fmt(len(compressed)))
    _row("Compression ratio", f"{len(raw)/len(compressed):.2f}x")
    _row("Param memory (fp32)", _bytes_fmt(param_bytes))
    _row("Buffer memory (BN stats)", _bytes_fmt(buffer_bytes))
    _row("Total model memory", _bytes_fmt(param_bytes + buffer_bytes))

    return {
        "parameters": n_params, "trainable_parameters": n_train,
        "state_dict_raw_bytes": len(raw), "state_dict_gzip_bytes": len(compressed),
        "compression_ratio": round(len(raw)/len(compressed), 2),
        "param_memory_bytes": param_bytes, "buffer_memory_bytes": buffer_bytes,
    }


def bench_inference(model, device, warmup=10, repeats=50, batch_sizes=[1, 8, 32, 128]):
    _section("INFERENCE LATENCY (synthetic 12-lead ECG)")
    model.eval()
    results = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 12, 1000, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        t = np.array(times)
        _row(f"Batch={bs:>3d}  total",
             f"{t.mean()*1000:.2f} ± {t.std()*1000:.2f} ms")
        _row(f"Batch={bs:>3d}  per-sample",  f"{(t.mean()/bs)*1000:.3f} ms")
        results[f"batch_{bs}"] = {
            "mean_ms": round(t.mean()*1000, 3), "std_ms": round(t.std()*1000, 3),
            "per_sample_ms": round((t.mean()/bs)*1000, 4),
        }
    return results


def bench_inference_real(model, testloader, device, warmup=3, repeats=10):
    _section("INFERENCE ON REAL PTB-XL TEST DATA")
    model.eval()
    n_batches = len(testloader)
    n_samples = sum(x.size(0) for x, _ in testloader)
    _row("Test batches", str(n_batches))
    _row("Test samples", str(n_samples))

    for i, (x, y) in enumerate(testloader):
        if i >= warmup:
            break
        with torch.no_grad():
            _ = model(x.to(device))

    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for x, y in testloader:
                _ = model(x.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    t = np.array(times)
    _row("Full test pass",
         f"{t.mean()*1000:.1f} ± {t.std()*1000:.1f} ms")
    _row("Per-sample (real data)",   f"{(t.mean()/n_samples)*1000:.3f} ms")
    _row("Throughput",               f"{n_samples/t.mean():.0f} samples/s")

    return {
        "n_samples": n_samples, "n_batches": n_batches,
        "full_pass_mean_ms": round(t.mean()*1000, 1),
        "per_sample_ms": round((t.mean()/n_samples)*1000, 4),
        "throughput_sps": round(n_samples/t.mean(), 0),
    }


def bench_full_evaluation(model, testloader, device):
    _section("FULL EVALUATION (task.test on PTB-XL)")
    if evaluate_model is None:
        print("  [SKIPPED] task.test not importable")
        return {"skipped": True}

    t0 = time.perf_counter()
    m = evaluate_model(model, testloader, device)
    elapsed = time.perf_counter() - t0

    _row("Evaluation time", f"{elapsed*1000:.1f} ms")
    _row("─── Metrics ───", "")
    _row("Loss",              f"{m['loss']:.4f}")
    _row("Accuracy",          f"{m['accuracy']:.4f}")
    _row("F1 macro",          f"{m['f1_macro']:.4f}")
    _row("F1 weighted",       f"{m['f1_weighted']:.4f}")
    _row("Precision macro",   f"{m['precision_macro']:.4f}")
    _row("Recall macro",      f"{m['recall_macro']:.4f}")
    _row("AUC macro",         f"{m['auc_macro']:.4f}")
    _row("Specificity macro", f"{m['specificity_macro']:.4f}")
    _row("─── Per-class F1 ───", "")
    for i, name in enumerate(SC_NAMES):
        _row(f"  {name}", f"{m['per_class_f1'][i]:.4f}")
    _row("─── Optimal thresholds ───", "")
    for i, name in enumerate(SC_NAMES):
        _row(f"  {name}", f"{m['optimal_thresholds'][i]:.2f}")

    m["evaluation_time_ms"] = round(elapsed * 1000, 1)
    return m


def bench_training_throughput(device, batch_size=128, num_batches=20):
    _section("TRAINING THROUGHPUT (synthetic, 1 epoch)")
    model = Net().to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    batches = [
        (torch.randn(batch_size, 12, 1000, device=device),
         torch.randint(0, 2, (batch_size, 5), device=device).float())
        for _ in range(num_batches)
    ]
    # warmup
    for i in range(min(3, num_batches)):
        x, y = batches[i]
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    total_samples = 0
    for x, y in batches:
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        total_samples += x.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sps = total_samples / elapsed
    _row("Samples", f"{total_samples:,}")
    _row("Time", f"{elapsed:.3f} s")
    _row("Throughput", f"{sps:.0f} samples/s")
    _row("Per-batch", f"{(elapsed/num_batches)*1000:.1f} ms")

    del model, opt, batches
    gc.collect()
    return {"total_samples": total_samples, "elapsed_s": round(elapsed, 4),
            "samples_per_sec": round(sps, 0), "ms_per_batch": round((elapsed/num_batches)*1000, 1),
            "batch_size": batch_size, "device": str(device)}


def bench_serialization(model, repeats=20):
    _section("SERIALIZATION & COMPRESSION")
    sd = model.state_dict()

    def _avg(fn, n=repeats):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            r = fn()
            times.append(time.perf_counter() - t0)
        return np.mean(times)*1000, r

    ser_ms, raw = _avg(lambda: _serialize_sd(sd))
    gz_ms, compressed = _avg(lambda: _gzip_bytes(raw))
    ungz_ms, _ = _avg(lambda: gzip.decompress(compressed))
    deser_ms, _ = _avg(lambda: torch.load(io.BytesIO(
        raw), map_location="cpu", weights_only=False))

    rt = ser_ms + gz_ms + ungz_ms + deser_ms
    _row("torch.save → BytesIO",    f"{ser_ms:.2f} ms")
    _row("gzip compress",           f"{gz_ms:.2f} ms")
    _row("gzip decompress",         f"{ungz_ms:.2f} ms")
    _row("torch.load from BytesIO", f"{deser_ms:.2f} ms")
    _row("Round-trip total",        f"{rt:.2f} ms")
    _row("Compressed size",         _bytes_fmt(len(compressed)))

    return {"serialize_ms": round(ser_ms, 3), "gzip_compress_ms": round(gz_ms, 3),
            "gzip_decompress_ms": round(ungz_ms, 3), "deserialize_ms": round(deser_ms, 3),
            "round_trip_ms": round(rt, 3), "raw_bytes": len(raw), "compressed_bytes": len(compressed)}


def bench_aggregation(model, num_clients=10, repeats=20):
    _section(f"FEDAVG AGGREGATION ({num_clients} clients, equal weight)")
    base = model.state_dict()
    client_sds = [
        OrderedDict(
            (k, v + torch.randn_like(v) * 0.01 if v.is_floating_point() else v.clone())
            for k, v in base.items()
        )
        for _ in range(num_clients)
    ]

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        avg_sd = OrderedDict()
        for key in base.keys():
            avg_sd[key] = torch.stack([sd[key].float()
                                      for sd in client_sds]).mean(0)
        times.append(time.perf_counter() - t0)

    t = np.array(times)
    _row(f"FedAvg ({num_clients} clients)",
         f"{t.mean()*1000:.2f} ± {t.std()*1000:.2f} ms")
    return {"num_clients": num_clients, "mean_ms": round(t.mean()*1000, 3), "std_ms": round(t.std()*1000, 3)}


def bench_vote_computation(num_clients=10, repeats=1000):
    _section(f"VOTE COMPUTATION ({num_clients} clients)")
    times = []
    for _ in range(repeats):
        losses = np.random.uniform(0.05, 0.5, size=num_clients)
        t0 = time.perf_counter()
        mu, sigma = float(losses.mean()), float(losses.std())
        threshold = mu + sigma
        votes = [{"client_id": i, "vote": "ACCEPTED" if l <= threshold else "REJECTED",
                  "loss": round(float(l), 4)} for i, l in enumerate(losses)]
        _ = json.dumps({"votes": votes, "threshold": round(threshold, 4)})
        times.append(time.perf_counter() - t0)

    _row("Vote + JSON serialize", f"{np.mean(times)*1e6:.1f} µs")
    return {"mean_us": round(np.mean(times)*1e6, 1)}


def bench_per_round_overhead(model, device, num_clients=10):
    _section(f"PER-ROUND OVERHEAD ({num_clients} clients)")
    sd = model.state_dict()

    # 1. Aggregation
    client_sds = [
        OrderedDict(
            (k, v + torch.randn_like(v) * 0.01 if v.is_floating_point() else v.clone())
            for k, v in sd.items()
        )
        for _ in range(num_clients)
    ]
    t0 = time.perf_counter()
    avg_sd = OrderedDict()
    for key in sd.keys():
        avg_sd[key] = torch.stack([c[key].float() for c in client_sds]).mean(0)
    t_agg = time.perf_counter() - t0

    # 2. Serialize + compress
    t0 = time.perf_counter()
    raw = _serialize_sd(avg_sd)
    compressed = _gzip_bytes(raw)
    t_ser = time.perf_counter() - t0

    # 3. Build 3 JSON payloads (LOCAL, VOTE, GLOBAL)
    losses = [np.random.uniform(0.05, 0.4) for _ in range(num_clients)]
    t0 = time.perf_counter()
    local_payload = json.dumps({"type": "LOCAL", "round": 5,
                                "clients": [{"id": i, "loss": round(l, 4)} for i, l in enumerate(losses)]})
    mu, sigma = np.mean(losses), np.std(losses)
    threshold = mu + sigma
    vote_payload = json.dumps({"type": "VOTE", "round": 5,
                               "votes": [{"id": i, "vote": "ACCEPTED" if l <= threshold else "REJECTED"}
                                         for i, l in enumerate(losses)], "threshold": round(float(threshold), 4)})
    model_hash = hashlib.sha256(compressed).hexdigest()
    global_payload = json.dumps({"type": "GLOBAL", "round": 5,
                                 "accuracy": 0.85, "f1_macro": 0.82, "model_hash_sha256": model_hash,
                                 "model_size_bytes": len(compressed)})
    t_payloads = time.perf_counter() - t0

    total = t_agg + t_ser + t_payloads

    _row("FedAvg aggregation",         f"{t_agg*1000:.2f} ms")
    _row("Serialize + gzip model",     f"{t_ser*1000:.2f} ms")
    _row("Build 3 JSON payloads",      f"{t_payloads*1000:.3f} ms")
    _row("─── TOTAL compute overhead", f"{total*1000:.2f} ms")
    _row("", "")
    _row("On-chain payload sizes:", "")
    _row("  LOCAL block",              _bytes_fmt(len(local_payload.encode())))
    _row("  VOTE block",               _bytes_fmt(len(vote_payload.encode())))
    _row("  GLOBAL block",             _bytes_fmt(len(global_payload.encode())))
    _row("  Model for IPFS (gzip)",    _bytes_fmt(len(compressed)))
    _row("  Model SHA-256",            model_hash[:24] + "…")

    return {
        "aggregation_ms": round(t_agg*1000, 2), "serialization_ms": round(t_ser*1000, 2),
        "payload_build_ms": round(t_payloads*1000, 3), "total_compute_ms": round(total*1000, 2),
        "local_payload_bytes": len(local_payload.encode()),
        "vote_payload_bytes": len(vote_payload.encode()),
        "global_payload_bytes": len(global_payload.encode()),
        "model_compressed_bytes": len(compressed), "model_hash_sha256": model_hash,
    }


def bench_blockchain(skip=False):
    _section("BLOCKCHAIN (Base Sepolia L2)")
    if skip or not WEB3_OK:
        print("  [SKIPPED] --no-blockchain or web3 not installed")
        return {"skipped": True}

    rpc_url = os.getenv("BASE_SEPOLIA_RPC_URL")
    private_key = os.getenv("PRIVATE_KEY")
    contract_address = os.getenv("CONTRACT_ADDRESS")
    if not all([rpc_url, private_key, contract_address]):
        print("  [SKIPPED] Missing .env vars")
        return {"skipped": True, "reason": "missing_env"}

    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            print(f"  [SKIPPED] Cannot connect to {rpc_url}")
            return {"skipped": True, "reason": "connection_failed"}

        account = w3.eth.account.from_key(private_key)
        _row("Chain ID", str(w3.eth.chain_id))
        _row("Account", account.address)
        balance = w3.eth.get_balance(account.address)
        _row("Balance", f"{w3.from_wei(balance, 'ether'):.6f} ETH")

        abi_path = None
        for f in ["FLBlockchain_abi.json", "abi.json", "contract_abi.json"]:
            if os.path.exists(f):
                abi_path = f
                break
        if not abi_path:
            import glob
            abis = glob.glob("**/FLBlockchain*.json", recursive=True)
            if abis:
                abi_path = abis[0]
        if not abi_path:
            print("  [SKIPPED] ABI file not found")
            return {"skipped": True, "reason": "no_abi"}

        with open(abi_path) as f:
            abi = json.load(f)
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(contract_address), abi=abi)

        tx_results = []
        nonce = w3.eth.get_transaction_count(account.address)
        for tx_type in ["LOCAL", "VOTE", "GLOBAL"]:
            payload = json.dumps(
                {"type": tx_type, "benchmark": True, "ts": datetime.now().isoformat()})

            t0 = time.perf_counter()
            fn = contract.functions.addBlock(
                0, tx_type, Web3.to_bytes(text=payload))
            try:
                gas_est = fn.estimate_gas({"from": account.address})
            except:
                gas_est = 300_000
            tx = fn.build_transaction({"from": account.address, "nonce": nonce,
                                       "gas": int(gas_est*1.2), "gasPrice": w3.eth.gas_price, "chainId": w3.eth.chain_id})
            t_build = time.perf_counter() - t0

            t0 = time.perf_counter()
            signed = w3.eth.account.sign_transaction(tx, private_key)
            t_sign = time.perf_counter() - t0

            t0 = time.perf_counter()
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            t_send = time.perf_counter() - t0

            t0 = time.perf_counter()
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            t_confirm = time.perf_counter() - t0

            total = t_build + t_sign + t_send + t_confirm
            cost = float(w3.from_wei(
                receipt["gasUsed"] * tx["gasPrice"], "ether"))

            _row(f"─── {tx_type} ───", "")
            _row("  Gas used", f"{receipt['gasUsed']:,}")
            _row("  Cost", f"{cost:.8f} ETH")
            _row("  Build+Sign+Send+Confirm", f"{total*1000:.0f} ms")

            tx_results.append({"type": tx_type, "gas_used": receipt["gasUsed"],
                               "cost_eth": cost, "total_ms": round(total*1000, 0)})
            nonce += 1

        total_cost = sum(r["cost_eth"] for r in tx_results)
        total_time = sum(r["total_ms"] for r in tx_results)
        _row("", "")
        _row("3-tx round total cost", f"{total_cost:.8f} ETH")
        _row("3-tx round total time", f"{total_time:.0f} ms")

        return {"skipped": False, "transactions": tx_results,
                "total_3tx_cost_eth": total_cost, "total_3tx_time_ms": total_time}
    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        return {"skipped": True, "reason": str(e)}


def bench_ipfs(model, skip=False):
    _section("IPFS STORAGE")
    if skip:
        print("  [SKIPPED] --no-ipfs")
        return {"skipped": True}
    try:
        try:
            from ipfs_storage import IPFSStorage
        except ImportError:
            from fl_blockchain_evm.ipfs_storage import IPFSStorage
        ipfs = IPFSStorage()
    except Exception as e:
        print(f"  [SKIPPED] Cannot init IPFS: {e}")
        return {"skipped": True, "reason": str(e)}

    try:
        compressed = _gzip_bytes(_serialize_sd(model.state_dict()))

        t0 = time.perf_counter()
        cid = ipfs.pin_bytes(compressed, "benchmark_model.pt.gz")
        t_pin = time.perf_counter() - t0

        t0 = time.perf_counter()
        fetched = ipfs.fetch_bytes(cid)
        t_fetch = time.perf_counter() - t0

        integrity = hashlib.sha256(compressed).hexdigest(
        ) == hashlib.sha256(fetched).hexdigest()

        _row("Pin model (gzip)", f"{t_pin*1000:.0f} ms")
        _row("Fetch model back", f"{t_fetch*1000:.0f} ms")
        _row("Payload", _bytes_fmt(len(compressed)))
        _row("SHA-256 check", "PASS" if integrity else "FAIL")

        try:
            ipfs.unpin(cid)
        except:
            pass

        return {"skipped": False, "pin_ms": round(t_pin*1000, 0),
                "fetch_ms": round(t_fetch*1000, 0), "bytes": len(compressed),
                "integrity": integrity, "cid": cid}
    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        return {"skipped": True, "reason": str(e)}


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="FL+Blockchain+IPFS Stack Benchmark")
    p.add_argument("--model", default="final_model.pt",
                   help="Path to final_model.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--no-blockchain", action="store_true")
    p.add_argument("--no-ipfs", action="store_true")
    p.add_argument("--with-testdata", action="store_true",
                   help="Evaluate on real PTB-XL test set")
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()

    device = _dev(args.device)

    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"{'║'} FL+Blockchain+IPFS Stack Benchmark {'║':>22}")
    print(f"{'╚'+'═'*58+'╝'}")
    print(f"  Device:    {device}")
    print(f"  Model:     {args.model}")
    print(f"  Clients:   {args.num_clients}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  Time:      {datetime.now().isoformat()}")
    if device.type == "cuda":
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.model):
        print(f"\n  [ERROR] Not found: {args.model}")
        print(f"  Run from your project root, or pass --model /path/to/final_model.pt")
        sys.exit(1)

    model = _load_model(args.model, device)
    print(
        f"  Loaded:    ✓ ({sum(p.numel() for p in model.parameters()):,} params)")

    testloader = None
    if args.with_testdata:
        if load_data is None:
            print("  [WARN] --with-testdata needs task.py importable")
        else:
            print(f"  Loading PTB-XL test data...")
            try:
                _, testloader = load_data(
                    partition_id=0, num_partitions=10, beta=0, batch_size=128)
                print(
                    f"  Test data: ✓ ({sum(x.size(0) for x, _ in testloader)} samples)")
            except Exception as e:
                print(f"  [WARN] Could not load test data: {e}")

    R = {"meta": {"device": str(device), "model_path": args.model,
                  "num_clients": args.num_clients, "timestamp": datetime.now().isoformat(),
                  "pytorch": torch.__version__, "real_testdata": testloader is not None}}

    R["model_profile"] = bench_model_profile(model)
    R["inference"] = bench_inference(model, device, args.warmup, args.repeats)
    if testloader:
        R["inference_real"] = bench_inference_real(model, testloader, device)
        R["full_evaluation"] = bench_full_evaluation(model, testloader, device)
    R["training"] = bench_training_throughput(device)
    R["serialization"] = bench_serialization(model, args.repeats)
    R["aggregation"] = bench_aggregation(model, args.num_clients)
    R["vote_computation"] = bench_vote_computation(args.num_clients)
    R["per_round_overhead"] = bench_per_round_overhead(
        model, device, args.num_clients)
    R["blockchain"] = bench_blockchain(skip=args.no_blockchain)
    R["ipfs"] = bench_ipfs(model, skip=args.no_ipfs)

    # ── Summary ──
    _section("SUMMARY — Key Numbers for Paper")
    _row("Inference batch=1",
         f"{R['inference'].get('batch_1',{}).get('mean_ms','?')} ms")
    _row("Inference batch=32",
         f"{R['inference'].get('batch_32',{}).get('mean_ms','?')} ms")
    if "inference_real" in R:
        _row("Real data per-sample",
             f"{R['inference_real']['per_sample_ms']} ms")
        _row("Real data throughput",
             f"{R['inference_real']['throughput_sps']:.0f} samples/s")
    if "full_evaluation" in R and not R["full_evaluation"].get("skipped"):
        _row("F1 macro",  f"{R['full_evaluation']['f1_macro']:.4f}")
        _row("AUC macro", f"{R['full_evaluation']['auc_macro']:.4f}")
    _row("Training throughput",
         f"{R['training']['samples_per_sec']:.0f} samples/s")
    _row("Model size (gzip)",         _bytes_fmt(
        R["model_profile"]["state_dict_gzip_bytes"]))
    _row("Serialize round-trip",
         f"{R['serialization']['round_trip_ms']:.1f} ms")
    _row(f"FedAvg ({args.num_clients} clients)",
         f"{R['aggregation']['mean_ms']} ms")
    _row("Per-round compute overhead",
         f"{R['per_round_overhead']['total_compute_ms']} ms")
    _row("Vote computation",          f"{R['vote_computation']['mean_us']} µs")

    bc = R["blockchain"]
    if not bc.get("skipped"):
        _row("3-tx round total",  f"{bc['total_3tx_time_ms']:.0f} ms")
        _row("3-tx round cost",   f"{bc['total_3tx_cost_eth']:.8f} ETH")
    ipfs = R["ipfs"]
    if not ipfs.get("skipped"):
        _row("IPFS pin model",    f"{ipfs['pin_ms']:.0f} ms")
        _row("IPFS fetch model",  f"{ipfs['fetch_ms']:.0f} ms")

    out = "benchmark_results.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    print(f"\n  Saved → {out}\n")


if __name__ == "__main__":
    main()
