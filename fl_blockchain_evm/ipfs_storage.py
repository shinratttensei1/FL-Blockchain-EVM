"""Lightweight IPFS storage for Federated Learning on Raspberry Pi 4.

Design priorities for RP4 (ARM Cortex-A72, 4-8 GB RAM, Wi-Fi/Ethernet):
──────────────────────────────────────────────────────────────────────────
• Zero-daemon      — HTTP pinning APIs only (Pinata / web3.storage).
                     No kubo daemon on RP4 → saves ~200 MB RAM + no DHT churn.
• Memory-efficient — Streams uploads via BytesIO; never materializes large
                     buffers on disk.  200K-param model ≈ 250 KB compressed.
• Compressed       — gzip level-6 before upload (configurable).
• Retry + backoff  — Exponential retry with jitter for unreliable Wi-Fi links
                     common on RP4 field deployments.
• Graceful degrade — If IPFS is unreachable the FL pipeline continues;
                     IPFS is an audit layer, not on the critical training path.

Supported backends
──────────────────
  pinata       Pinata Cloud free tier (500 pins, 1 GB).  Recommended for RP4.
  local        Local kubo HTTP API (http://127.0.0.1:5001).  Run on the server
               machine if you want self-hosted IPFS without loading each RP4.
  web3storage  web3.storage free tier (5 GB).

Environment variables (set in .env)
───────────────────────────────────
  IPFS_BACKEND       pinata | local | web3storage   (empty = IPFS disabled)
  PINATA_JWT         JWT from https://app.pinata.cloud/developers/api-keys
  PINATA_GATEWAY     (optional) e.g. https://gateway.pinata.cloud/ipfs/
  LOCAL_IPFS_API     (optional) default http://127.0.0.1:5001
  WEB3STORAGE_TOKEN  Token from https://web3.storage

Quick start
───────────
    from fl_blockchain_evm.ipfs_storage import IPFSStorage

    ipfs = IPFSStorage(backend="pinata")
    cid  = ipfs.pin_model(model.state_dict(), "round_1_global")
    sd   = ipfs.fetch_model(cid)
"""

import io
import gzip
import json
import time
import random
import hashlib
import logging
import os
from typing import Optional, Dict, List, Any

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("fl.ipfs")


class IPFSStorage:
    """Content-addressable off-chain storage for FL model weights and metadata.

    Designed for Raspberry Pi 4 deployments: no IPFS daemon required,
    all operations use lightweight HTTP API calls to pinning services.

    Typical per-round overhead on RP4 (Wi-Fi, ~30 Mbps):
        • pin_json  (round metadata, ~5 KB)    →  ~0.3 s
        • pin_model (200K-param SE-ResNet)      →  ~0.8 s
        • Total per round (4 pins)              →  ~2.5 s
    """

    BACKENDS = ("pinata", "local", "web3storage")

    # Public IPFS gateways for fetching (ordered by reliability)
    GATEWAYS = [
        "https://gateway.pinata.cloud/ipfs/",
        "https://ipfs.io/ipfs/",
        "https://dweb.link/ipfs/",
        "https://cloudflare-ipfs.com/ipfs/",
    ]

    def __init__(
        self,
        backend: str = "pinata",
        pinata_jwt: Optional[str] = None,
        pinata_gateway: Optional[str] = None,
        local_api_url: Optional[str] = None,
        web3storage_token: Optional[str] = None,
        compress: bool = True,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        if backend not in self.BACKENDS:
            raise ValueError(
                f"Unknown IPFS backend '{backend}'. "
                f"Choose from: {', '.join(self.BACKENDS)}"
            )

        self.backend = backend
        self.compress = compress
        self.max_retries = max_retries
        self.timeout = timeout

        # Reuse TCP connections — reduces overhead on RP4
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "fl-blockchain-evm/1.0"

        # ── Backend-specific initialization ──────────────────

        if backend == "pinata":
            self._jwt = pinata_jwt or os.getenv("PINATA_JWT", "")
            if not self._jwt:
                raise ValueError(
                    "PINATA_JWT required. "
                    "Get one at https://app.pinata.cloud/developers/api-keys"
                )
            self._gateway = (
                pinata_gateway
                or os.getenv("PINATA_GATEWAY", "")
                or self.GATEWAYS[0]
            )
            self._session.headers["Authorization"] = f"Bearer {self._jwt}"
            self._pin_url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
            self._pin_json_url = (
                "https://api.pinata.cloud/pinning/pinJSONToIPFS"
            )
            self._unpin_url = "https://api.pinata.cloud/pinning/unpin/"

        elif backend == "local":
            self._api_url = (
                local_api_url
                or os.getenv("LOCAL_IPFS_API", "http://127.0.0.1:5001")
            )
            self._gateway = f"{self._api_url}/api/v0/cat?arg="

        elif backend == "web3storage":
            self._token = web3storage_token or os.getenv(
                "WEB3STORAGE_TOKEN", ""
            )
            if not self._token:
                raise ValueError(
                    "WEB3STORAGE_TOKEN required. "
                    "Get one at https://web3.storage"
                )
            self._session.headers["Authorization"] = f"Bearer {self._token}"
            self._gateway = "https://w3s.link/ipfs/"

        # Session tracking for audit / cleanup
        self._pinned: List[Dict[str, Any]] = []
        self._total_bytes_uploaded: int = 0

        print(f"  [IPFS] Backend: {backend} | Compress: {compress}")

    # ─────────────────────────────────────────────────────────
    #  Pin operations
    # ─────────────────────────────────────────────────────────

    def pin_bytes(self, data: bytes, name: str = "fl_data") -> str:
        """Pin raw bytes to IPFS.  Returns the CID (content identifier).

        On RP4 with gzip enabled, a 200K-param model (~800 KB)
        compresses to ~250 KB before upload.
        """
        original_size = len(data)

        if self.compress:
            data = gzip.compress(data, compresslevel=6)
            name += ".gz"

        for attempt in range(1, self.max_retries + 1):
            try:
                cid = self._backend_pin(data, name)
                self._pinned.append({
                    "cid": cid,
                    "name": name,
                    "size_raw": original_size,
                    "size_compressed": len(data),
                    "timestamp": time.time(),
                })
                self._total_bytes_uploaded += len(data)
                log.info(
                    "Pinned %s → %s (%d → %d bytes)",
                    name, cid, original_size, len(data),
                )
                print(
                    f"  [IPFS] ✓ {name} → {cid[:20]}… "
                    f"({original_size:,} → {len(data):,} bytes)"
                )
                return cid

            except Exception as e:
                if attempt == self.max_retries:
                    log.error(
                        "Pin failed after %d attempts: %s",
                        self.max_retries, e,
                    )
                    raise RuntimeError(
                        f"IPFS pin failed after {self.max_retries} "
                        f"attempts: {e}"
                    ) from e
                # Exponential backoff + jitter (RP4 may have flaky Wi-Fi)
                wait = (2 ** attempt) + random.uniform(0, 1)
                log.warning(
                    "Pin attempt %d/%d failed: %s  (retry in %.1fs)",
                    attempt, self.max_retries, e, wait,
                )
                time.sleep(wait)

        # Unreachable, but keeps type-checkers happy
        raise RuntimeError("IPFS pin failed")

    def pin_json(self, data: dict, name: str = "fl_metadata") -> str:
        """Pin a JSON-serializable dict to IPFS.  Returns CID."""
        payload = json.dumps(data, default=str, separators=(",", ":"))
        return self.pin_bytes(payload.encode("utf-8"), name)

    def pin_model(self, state_dict: dict, name: str = "fl_model") -> str:
        """Serialize a PyTorch state_dict and pin to IPFS.

        Uses BytesIO to avoid temp files — important on RP4 where
        SD-card writes are slow and wear-limited.

        Returns CID.
        """
        import torch

        buf = io.BytesIO()
        torch.save(state_dict, buf)
        raw = buf.getvalue()
        buf.close()
        return self.pin_bytes(raw, name)

    # ─────────────────────────────────────────────────────────
    #  Fetch operations
    # ─────────────────────────────────────────────────────────

    def fetch_bytes(self, cid: str) -> bytes:
        """Fetch raw bytes from IPFS by CID.  Auto-decompresses gzip."""
        for attempt in range(1, self.max_retries + 1):
            for gw in self._gateway_list():
                try:
                    url = f"{gw}{cid}"
                    resp = self._session.get(url, timeout=self.timeout)
                    resp.raise_for_status()
                    data = resp.content
                    # Auto-detect gzip magic bytes
                    if data[:2] == b"\x1f\x8b":
                        data = gzip.decompress(data)
                    print(
                        f"  [IPFS] ✓ Fetched {cid[:20]}… "
                        f"({len(data):,} bytes)"
                    )
                    return data
                except Exception as e:
                    log.debug("Fetch from %s failed: %s", gw, e)
                    continue
            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Failed to fetch CID {cid} from all gateways "
            f"after {self.max_retries} attempts"
        )

    def fetch_json(self, cid: str) -> dict:
        """Fetch and parse JSON from IPFS."""
        return json.loads(self.fetch_bytes(cid).decode("utf-8"))

    def fetch_model(self, cid: str, device: str = "cpu") -> dict:
        """Fetch and deserialize a PyTorch state_dict from IPFS."""
        import torch

        data = self.fetch_bytes(cid)
        buf = io.BytesIO(data)
        state_dict = torch.load(buf, map_location=device, weights_only=False)
        buf.close()
        return state_dict

    # ─────────────────────────────────────────────────────────
    #  Cleanup & management
    # ─────────────────────────────────────────────────────────

    def unpin(self, cid: str) -> bool:
        """Unpin content from IPFS (Pinata only).  Returns success.

        Useful on RP4 to stay within free-tier pin limits (500 pins).
        Call this to remove old round artifacts after archiving.
        """
        if self.backend != "pinata":
            log.warning("Unpin only supported for Pinata backend")
            return False
        try:
            resp = self._session.delete(
                f"{self._unpin_url}{cid}", timeout=self.timeout
            )
            resp.raise_for_status()
            print(f"  [IPFS] Unpinned {cid[:20]}…")
            return True
        except Exception as e:
            log.warning("Unpin failed for %s: %s", cid, e)
            return False

    def get_session_stats(self) -> Dict[str, Any]:
        """Return upload statistics for this session."""
        return {
            "backend": self.backend,
            "total_pins": len(self._pinned),
            "total_bytes_uploaded": self._total_bytes_uploaded,
            "pins": list(self._pinned),
        }

    # ─────────────────────────────────────────────────────────
    #  Integrity verification
    # ─────────────────────────────────────────────────────────

    def verify_content(self, cid: str, expected_sha256: str) -> bool:
        """Verify that IPFS content matches an expected SHA-256 hash.

        Use this to cross-check on-chain contentHash against IPFS data.
        """
        data = self.fetch_bytes(cid)
        actual = hashlib.sha256(data).hexdigest()
        match = actual == expected_sha256
        if not match:
            log.warning(
                "Content mismatch for %s: expected %s, got %s",
                cid, expected_sha256, actual,
            )
        return match

    @staticmethod
    def compute_sha256(data: bytes) -> str:
        """Compute SHA-256 hex digest."""
        return hashlib.sha256(data).hexdigest()

    # ─────────────────────────────────────────────────────────
    #  Backend implementations (private)
    # ─────────────────────────────────────────────────────────

    def _backend_pin(self, data: bytes, name: str) -> str:
        """Dispatch to the configured backend's pin implementation."""
        if self.backend == "pinata":
            return self._pinata_pin(data, name)
        elif self.backend == "local":
            return self._local_pin(data, name)
        elif self.backend == "web3storage":
            return self._web3storage_pin(data, name)
        raise ValueError(f"Unknown backend: {self.backend}")

    def _pinata_pin(self, data: bytes, name: str) -> str:
        """Pin via Pinata Cloud API — recommended for RP4."""
        metadata = json.dumps({
            "name": name,
            "keyvalues": {
                "app": "fl-blockchain-evm",
                "type": "fl-artifact",
            },
        })
        resp = self._session.post(
            self._pin_url,
            files={"file": (name, io.BytesIO(data))},
            data={"pinataMetadata": metadata},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["IpfsHash"]

    def _local_pin(self, data: bytes, name: str) -> str:
        """Pin via local kubo HTTP API."""
        resp = self._session.post(
            f"{self._api_url}/api/v0/add",
            files={"file": (name, io.BytesIO(data))},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["Hash"]

    def _web3storage_pin(self, data: bytes, name: str) -> str:
        """Pin via web3.storage API."""
        resp = self._session.post(
            "https://api.web3.storage/upload",
            files={"file": (name, io.BytesIO(data))},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["cid"]

    def _gateway_list(self) -> list:
        """Return ordered list of gateways to try for fetching."""
        if self.backend == "pinata" and hasattr(self, "_gateway"):
            return [self._gateway] + [
                g for g in self.GATEWAYS if g != self._gateway
            ]
        elif self.backend == "local":
            return [self._gateway]
        return list(self.GATEWAYS)
