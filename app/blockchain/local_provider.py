"""
Concrete BlockchainProvider implementation for a local EVM development
chain.

TECHNOLOGY CHOICE
- Local node: a Hardhat development node (`npx hardhat node`), reachable
  over plain JSON-RPC at BLOCKCHAIN_RPC_URL (default
  http://127.0.0.1:8545). Hardhat was chosen over Anvil/Ganache because
  it installs cleanly via `npm install` (already a project dependency
  path in this environment) with no separate binary download, and it
  ships pre-funded, publicly-documented development accounts with
  unlocked signing — no private key configuration is required for the
  default local-mode flow (see "Signing" below). Any Anvil/Ganache-style
  node exposing the same JSON-RPC surface and unlocked dev accounts
  would work identically through this same provider.
- SDK: web3.py, already a project dependency.
- Contract: a minimal Solidity registry contract, `FingerprintRegistry`
  (source: app/blockchain/contracts/FingerprintRegistry.sol). This
  provider does NOT compile Solidity at runtime -- it loads a
  precompiled artifact (ABI + bytecode) shipped alongside the source at
  app/blockchain/contracts/FingerprintRegistry.json. This avoids adding
  a runtime `solc`/`py-solc-x` dependency (and the compiler-download
  step that entails) purely to reproduce a fixed, already-reviewed
  contract; the committed .sol file is the source of truth for what
  that bytecode does, and can be recompiled with any standard Solidity
  toolchain if you want to verify the artifact yourself.

SIGNING
Two paths, selected automatically based on configuration:
    1. No BLOCKCHAIN_PRIVATE_KEY configured (the local-mode default):
       transactions are sent via `eth_sendTransaction` using the node's
       own first unlocked account. This is how Hardhat/Anvil-style dev
       nodes are designed to be used locally and requires no key
       material on the client side at all.
    2. BLOCKCHAIN_PRIVATE_KEY configured: the transaction is signed
       locally with that key and sent via `eth_sendRawTransaction`. This
       path is what a public testnet would require (see
       TestnetBlockchainProvider, not implemented in this milestone),
       but works against a local node too if you explicitly want
       client-side signing during local development.

CONTRACT DEPLOYMENT
If BLOCKCHAIN_CONTRACT_ADDRESS is not configured, this provider deploys
a fresh FingerprintRegistry contract once, the first time it needs one
(lazily, on the first register()/retrieve() call), and reuses that same
deployed instance for the rest of the process's lifetime -- it does not
redeploy on every call. This matches Option A from the architecture
("deploy automatically if no contract address exists") because it
minimizes demo-day setup steps; the tradeoff is that restarting the
Python process without setting BLOCKCHAIN_CONTRACT_ADDRESS deploys a new
contract instance (fine for a local dev chain, which typically resets
its own state on restart too). Set BLOCKCHAIN_CONTRACT_ADDRESS explicitly
to reuse one deployed contract across process restarts.

ON-CHAIN DATA MODEL
See FingerprintRegistry.sol. Stored per content_hash: algorithm,
canonicalization_version, source_reference_hash (or the zero bytes32 if
none was provided), and the registering block's timestamp. Raw content,
images, text, and face embeddings are never sent to the contract.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from eth_account import Account
from web3 import Web3
from web3.exceptions import ContractLogicError, TransactionNotFound

from app.blockchain.interface import (
    BlockchainProvider,
    BlockchainRecord,
    OnChainPayload,
)

logger = logging.getLogger(__name__)

_CONTRACT_DIR = Path(__file__).resolve().parent / "contracts"
_ARTIFACT_PATH = _CONTRACT_DIR / "FingerprintRegistry.json"

_ZERO_BYTES32 = b"\x00" * 32


class BlockchainConnectionError(Exception):
    """Raised when the configured RPC endpoint cannot be reached."""


class BlockchainRegistrationError(Exception):
    """Raised when a register() call fails (e.g. duplicate content_hash,
    or the transaction reverted for another on-chain reason)."""


def _load_contract_artifact() -> tuple[list, str]:
    if not _ARTIFACT_PATH.exists():
        raise BlockchainConnectionError(
            f"Contract artifact not found at {_ARTIFACT_PATH}. This ships "
            "with the repository; if it's missing, re-fetch the project "
            "or recompile FingerprintRegistry.sol with a standard "
            "Solidity toolchain and place the ABI+bytecode JSON there."
        )
    data = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
    return data["abi"], data["bytecode"]


def _hex_to_bytes32(hex_str: str) -> bytes:
    raw = bytes.fromhex(hex_str)
    if len(raw) != 32:
        raise ValueError(f"Expected a 32-byte (64 hex char) value, got {len(raw)} bytes")
    return raw


def _parse_private_key(private_key: str) -> Account:
    """Parse a hex private key into an eth_account Account.

    Pure function, no network access -- can be unit tested without a
    live node.

    Raises:
        ValueError: with a clear, human-readable message if the key is
            malformed (wrong length, not hex, etc).
    """
    try:
        return Account.from_key(private_key)
    except Exception as exc:
        raise ValueError(
            "BLOCKCHAIN_PRIVATE_KEY is malformed (expected a 0x-prefixed "
            f"32-byte hex private key): {exc}"
        ) from exc


class LocalBlockchainProvider(BlockchainProvider):
    """BlockchainProvider backed by a local EVM development node.

    See module docstring for the technology, signing, and deployment
    strategy.
    """

    def __init__(
        self,
        rpc_url: str,
        private_key: str | None = None,
        contract_address: str | None = None,
    ) -> None:
        self._private_key = private_key
        self._account = _parse_private_key(private_key) if private_key else None

        self._w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self._w3.is_connected():
            raise BlockchainConnectionError(
                f"Could not connect to blockchain RPC at {rpc_url}. "
                "Start a local development node first, e.g.: "
                "`npx hardhat node` (see README 'Milestone 4' setup steps)."
            )

        if self._account:
            self._from_address = self._account.address
        else:
            accounts = self._w3.eth.accounts
            if not accounts:
                raise BlockchainConnectionError(
                    "No unlocked accounts available on the configured node "
                    "and no BLOCKCHAIN_PRIVATE_KEY was set. A local dev "
                    "node (Hardhat/Anvil-style) should expose unlocked "
                    "accounts automatically."
                )
            self._from_address = accounts[0]

        self._abi, self._bytecode = _load_contract_artifact()
        self._contract_address = contract_address
        self._contract = (
            self._w3.eth.contract(address=contract_address, abi=self._abi)
            if contract_address
            else None
        )

    @property
    def contract_address(self) -> str | None:
        """The deployed/configured contract address, if known yet (None
        until the first register()/retrieve() call triggers lazy
        deployment when no address was configured)."""
        return self._contract_address

    def _ensure_contract(self):
        if self._contract is not None:
            return self._contract

        logger.info("No contract address configured; deploying FingerprintRegistry...")
        factory = self._w3.eth.contract(abi=self._abi, bytecode=self._bytecode)
        tx_hash = self._send_transaction(factory.constructor().build_transaction)
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        address = receipt.contractAddress
        if address is None:
            raise BlockchainConnectionError(
                "Contract deployment transaction did not produce a contract address."
            )

        self._contract_address = address
        self._contract = self._w3.eth.contract(address=address, abi=self._abi)
        logger.info("FingerprintRegistry deployed at %s", address)
        return self._contract

    def _send_transaction(self, build_tx_fn, **build_kwargs):
        """Build, (optionally sign), and send a transaction, returning
        the transaction hash. `build_tx_fn` is a bound
        `<contract function>.build_transaction`-style callable.
        """
        base_tx = {"from": self._from_address}
        if self._private_key:
            base_tx["nonce"] = self._w3.eth.get_transaction_count(self._from_address)
        tx = build_tx_fn({**base_tx, **build_kwargs})

        if self._private_key:
            signed = self._w3.eth.account.sign_transaction(tx, self._private_key)
            return self._w3.eth.send_raw_transaction(signed.raw_transaction)
        return self._w3.eth.send_transaction(tx)

    def register(self, payload: OnChainPayload) -> BlockchainRecord:
        contract = self._ensure_contract()

        content_hash_bytes = _hex_to_bytes32(payload.content_hash)
        source_ref_bytes = (
            _hex_to_bytes32(payload.source_reference_hash)
            if payload.source_reference_hash
            else _ZERO_BYTES32
        )

        try:
            tx_hash = self._send_transaction(
                contract.functions.register(
                    content_hash_bytes,
                    payload.algorithm,
                    payload.version,
                    source_ref_bytes,
                ).build_transaction
            )
            receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        except ContractLogicError as exc:
            raise BlockchainRegistrationError(
                f"On-chain registration reverted for content_hash="
                f"{payload.content_hash}: {exc}"
            ) from exc

        if receipt.status != 1:
            raise BlockchainRegistrationError(
                f"Registration transaction {receipt.transactionHash.hex()} "
                "failed on-chain (status=0). This usually means "
                "content_hash was already registered."
            )

        block = self._w3.eth.get_block(receipt.blockNumber)
        return BlockchainRecord(
            tx_hash=f"0x{receipt.transactionHash.hex()}",
            block_number=receipt.blockNumber,
            timestamp=datetime.fromtimestamp(block.timestamp, tz=timezone.utc),
            on_chain_payload=payload,
        )

    def retrieve(self, tx_hash: str) -> BlockchainRecord | None:
        normalized_tx_hash = tx_hash if tx_hash.startswith("0x") else f"0x{tx_hash}"
        try:
            receipt = self._w3.eth.get_transaction_receipt(tx_hash)
        except TransactionNotFound:
            return None

        contract = self._ensure_contract()
        events = contract.events.FingerprintRegistered().process_receipt(receipt)
        if not events:
            return None

        content_hash_bytes = events[0]["args"]["contentHash"]
        record = contract.functions.getRecord(content_hash_bytes).call()
        content_hash, algorithm, version, source_ref_bytes, ts, exists = record
        if not exists:
            return None

        source_reference_hash = (
            source_ref_bytes.hex() if source_ref_bytes != _ZERO_BYTES32 else None
        )
        payload = OnChainPayload(
            content_hash=content_hash.hex(),
            algorithm=algorithm,
            version=version,
            source_reference_hash=source_reference_hash,
        )
        return BlockchainRecord(
            tx_hash=normalized_tx_hash,
            block_number=receipt.blockNumber,
            timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
            on_chain_payload=payload,
        )
