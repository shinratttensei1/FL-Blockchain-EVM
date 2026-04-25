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
  PINATA_JWT_2       (optional) fallback JWT — used automatically when the
                     primary account hits its storage/pin limit
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
            jwt1 = pinata_jwt or os.getenv("PINATA_JWT", "")
            jwt2 = os.getenv("PINATA_JWT_2", "")

            # PINATA_ACCOUNT=1 → primary account_1, fallback account_2 (default)
            # PINATA_ACCOUNT=2 → primary account_2, fallback account_1
            # This lets the pipeline route the first half of runs to account_1
            # and the second half to account_2 for clean data separation.
            _account = os.getenv("PINATA_ACCOUNT", "1").strip()
            if _account == "2":
                _ordered = [jwt2, jwt1]
                _primary_label = "account_2"
            else:
                _ordered = [jwt1, jwt2]
                _primary_label = "account_1"

            _jwts = [j for j in _ordered if j]
            if not _jwts:
                raise ValueError(
                    "PINATA_JWT required. "
                    "Get one at https://app.pinata.cloud/developers/api-keys"
                )
            self._pinata_jwts = _jwts
            self._pinata_jwt_idx = 0
            self._gateway = (
                pinata_gateway
                or os.getenv("PINATA_GATEWAY", "")
                or self.GATEWAYS[0]
            )
            self._session.headers["Authorization"] = f"Bearer {_jwts[0]}"
            self._pin_url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
            self._pin_json_url = (
                "https://api.pinata.cloud/pinning/pinJSONToIPFS"
            )
            self._unpin_url = "https://api.pinata.cloud/pinning/unpin/"
            n_accounts = len(_jwts)
            print(
                f"  [IPFS] Pinata primary: {_primary_label}"
                f" | accounts configured: {n_accounts}"
                + (" (+ 1 fallback)" if n_accounts > 1 else ""),
                flush=True,
            )

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
                acct = (
                    f" [{self._active_account_label}]"
                    if self.backend == "pinata" else ""
                )
                print(
                    f"  [IPFS] ✓ {name} → {cid[:20]}…"
                    f" ({original_size:,} → {len(data):,} bytes){acct}",
                    flush=True,
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

    @staticmethod
    def _pinata_limit_error(resp: requests.Response) -> bool:
        """True when Pinata is rejecting the upload due to account limits.

        Pinata returns 403 Forbidden when the free-tier storage cap is
        reached — confirmed from production logs. 402 = payment required,
        429 = rate limited. All three signal account exhaustion.
        """
        # 402 = payment required (storage cap), 403 = forbidden (storage cap on Pinata)
        # 429 = rate-limited: temporary, should NOT trigger a JWT switch — let
        #       the outer pin_bytes retry loop handle it with exponential backoff
        if resp.status_code in (402, 403):
            return True
        if resp.status_code == 400:
            try:
                msg = str(resp.json().get("error", "")).lower()
                return "limit" in msg or "storage" in msg or "quota" in msg
            except Exception:
                pass
        return False

    @property
    def _active_account_label(self) -> str:
        return f"account_{self._pinata_jwt_idx + 1}/{len(self._pinata_jwts)}"

    def _pinata_pin(self, data: bytes, name: str) -> str:
        """Pin via Pinata Cloud API with automatic JWT failover.

        Iterates through all configured JWTs. On a storage-limit error
        (HTTP 402/403/429) it switches to the next JWT and retries
        immediately. On any other error it raises so the outer pin_bytes
        retry loop can back off and try again.
        """
        metadata = json.dumps({
            "name": name,
            "keyvalues": {"app": "fl-blockchain-evm", "type": "fl-artifact"},
        })
        for attempt in range(len(self._pinata_jwts)):
            account_label = f"account_{self._pinata_jwt_idx + 1}"
            resp = self._session.post(
                self._pin_url,
                files={"file": (name, io.BytesIO(data))},
                data={"pinataMetadata": metadata},
                timeout=self.timeout,
            )
            if resp.ok:
                cid = resp.json()["IpfsHash"]
                log.info("Pinned %s via %s → %s", name, account_label, cid[:20])
                return cid

            if self._pinata_limit_error(resp):
                next_idx = self._pinata_jwt_idx + 1
                if next_idx < len(self._pinata_jwts):
                    self._pinata_jwt_idx = next_idx
                    self._session.headers["Authorization"] = (
                        f"Bearer {self._pinata_jwts[self._pinata_jwt_idx]}"
                    )
                    msg = (
                        f"  [IPFS] Pinata {account_label} exhausted"
                        f" (HTTP {resp.status_code})"
                        f" → switching to account_{self._pinata_jwt_idx + 1}"
                        f"/{len(self._pinata_jwts)}"
                    )
                    print(msg, flush=True)
                    log.warning(msg.strip())
                    continue
                else:
                    log.error(
                        "All %d Pinata accounts exhausted (HTTP %d) — "
                        "IPFS upload will fail after retries",
                        len(self._pinata_jwts), resp.status_code,
                    )
                    print(
                        f"  [IPFS] WARNING: all Pinata accounts exhausted"
                        f" — {name} will not be pinned",
                        flush=True,
                    )

            resp.raise_for_status()  # non-limit error → let pin_bytes retry

        # Should not reach here, but safety net
        raise RuntimeError(
            f"Pinata pin failed for {name} after trying all "
            f"{len(self._pinata_jwts)} account(s)"
        )

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
