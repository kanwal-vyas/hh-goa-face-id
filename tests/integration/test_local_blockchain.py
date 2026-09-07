"""
Integration test for LocalBlockchainProvider against a REAL local EVM
development node.

ENVIRONMENT ASSUMPTION (documented explicitly): a local EVM-compatible
development node must be running and reachable at the RPC URL below
before these tests will execute. The one used to build and validate
this milestone is a Hardhat development node:

    cd <a scratch directory outside this repo>
    npm install --no-save hardhat solc@0.8.24
    npx hardhat init   # or create a minimal hardhat.config.js
    npx hardhat node

This test suite does NOT start a node itself (per the milestone's
constraint against silently starting a blockchain inside pytest) --
it only detects whether one is already reachable and skips cleanly
with a clear reason if not.

No mocking: this file talks to a real web3.py Web3 instance and a real
deployed FingerprintRegistry contract. Every assertion below is checking
a genuine on-chain round trip, not a stub.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from web3 import Web3

from app.blockchain.interface import OnChainPayload
from app.blockchain.local_provider import (
    BlockchainRegistrationError,
    LocalBlockchainProvider,
)
from app.config.settings import load_settings
from app.content.fingerprint import hash_source_reference
from main import _build_authorized_pipeline

RPC_URL = "http://127.0.0.1:8545"


def _node_available() -> bool:
    try:
        w3 = Web3(Web3.HTTPProvider(RPC_URL, request_kwargs={"timeout": 2}))
        return w3.is_connected()
    except Exception:  # noqa: BLE001 -- reachability probe: any failure means "unavailable"
        return False


pytestmark = pytest.mark.skipif(
    not _node_available(),
    reason=(
        f"No local EVM development node reachable at {RPC_URL}. Start one "
        "with `npx hardhat node` (see this file's module docstring for "
        "full setup) and re-run to execute these integration tests."
    ),
)


@pytest.fixture(scope="module")
def provider() -> LocalBlockchainProvider:
    return LocalBlockchainProvider(rpc_url=RPC_URL)


def _deterministic_payload(test_name: str) -> OnChainPayload:
    """Derive a unique-but-deterministic ContentFingerprint-like payload
    from the calling test's name, so each test registers a distinct
    content_hash (avoiding duplicate-registration collisions between
    independent tests) while remaining fully reproducible.
    """
    content_hash = hashlib.sha256(
        f"integration-test-content::{test_name}".encode()
    ).hexdigest()
    source_reference_hash = hash_source_reference(
        f"https://example.org/integration-test/{test_name}"
    )
    return OnChainPayload(
        content_hash=content_hash,
        algorithm="SHA-256",
        version="v1",
        source_reference_hash=source_reference_hash,
    )


def test_connects_to_real_local_chain(provider: LocalBlockchainProvider):
    assert provider._w3.is_connected()
    # A real chain has a real chain_id and a non-negative block number.
    assert provider._w3.eth.chain_id is not None
    assert provider._w3.eth.block_number >= 0


def test_register_returns_real_transaction_hash_and_block(
    provider: LocalBlockchainProvider,
):
    payload = _deterministic_payload("register_returns_real_tx")
    record = provider.register(payload)

    assert record.tx_hash.startswith("0x")
    assert len(record.tx_hash) == 66
    # Confirm it's real hex, not a placeholder string.
    int(record.tx_hash, 16)

    assert record.block_number >= 0
    assert record.timestamp is not None


def test_retrieve_after_register_matches_registered_payload(
    provider: LocalBlockchainProvider,
):
    payload = _deterministic_payload("retrieve_round_trip")
    record = provider.register(payload)

    retrieved = provider.retrieve(record.tx_hash)

    assert retrieved is not None
    assert retrieved.on_chain_payload.content_hash == payload.content_hash
    assert retrieved.on_chain_payload.algorithm == "SHA-256"
    assert retrieved.on_chain_payload.version == payload.version
    assert retrieved.on_chain_payload.source_reference_hash == payload.source_reference_hash
    assert retrieved.tx_hash == record.tx_hash
    assert retrieved.block_number == record.block_number
    assert retrieved.timestamp is not None


def test_duplicate_registration_is_deterministically_rejected(
    provider: LocalBlockchainProvider,
):
    payload = _deterministic_payload("duplicate_registration")
    provider.register(payload)

    with pytest.raises(BlockchainRegistrationError):
        provider.register(payload)


def test_retrieve_unknown_transaction_returns_none(
    provider: LocalBlockchainProvider,
):
    # A syntactically valid but never-broadcast transaction hash.
    unknown_tx_hash = "0x" + "ee" * 32
    assert provider.retrieve(unknown_tx_hash) is None


def test_full_authorized_pipeline_registers_and_verifies_on_real_chain():
    """The complete happy path uses the real local provider, not a mock."""
    reference = (
        Path(__file__).parents[2]
        / "examples"
        / "authorized_demo_images"
        / "garden-avatar.png"
    )

    report = _build_authorized_pipeline(load_settings(env_file=None)).run(reference)

    assert report.succeeded is True
    assert len(report.stages) == 8
    assert report.selected_match is not None
    assert report.fingerprint is not None
    assert report.blockchain_record is not None
    assert report.verification_result is not None
    assert report.verification_result.match is True
    assert report.verification_result.local_hash == report.fingerprint.hash
    assert report.verification_result.on_chain_hash == report.fingerprint.hash
