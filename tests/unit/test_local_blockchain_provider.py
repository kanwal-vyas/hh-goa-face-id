"""
Unit tests for app.blockchain.local_provider that do NOT require a live
local node. Live round-trip tests are in
tests/integration/test_local_blockchain.py.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from app.blockchain.interface import (
    BlockchainProvider,
    BlockchainRecord,
    OnChainPayload,
)
from app.blockchain.local_provider import (
    BlockchainConnectionError,
    LocalBlockchainProvider,
    _hex_to_bytes32,
    _parse_private_key,
)


def test_local_blockchain_provider_is_concrete_and_implements_interface():
    assert not inspect.isabstract(LocalBlockchainProvider)
    assert issubclass(LocalBlockchainProvider, BlockchainProvider)


def test_connection_error_on_unreachable_rpc():
    """Port 1 is a reserved/unroutable port -- nothing should ever be
    listening there, so this exercises the "RPC unreachable" path
    without needing a live node."""
    with pytest.raises(BlockchainConnectionError):
        LocalBlockchainProvider(rpc_url="http://127.0.0.1:1")


def test_empty_rpc_url_raises_connection_error():
    with pytest.raises(BlockchainConnectionError):
        LocalBlockchainProvider(rpc_url="")


# --- _parse_private_key: pure, no network ---


def test_parse_private_key_accepts_valid_key():
    # A well-known, publicly-documented Hardhat dev test key (account #0).
    # This is not a secret -- it is printed by every `npx hardhat node`
    # invocation and used purely to test key-parsing logic offline.
    dev_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    account = _parse_private_key(dev_key)
    assert account.address == "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"


def test_parse_private_key_rejects_malformed_key():
    with pytest.raises(ValueError, match="malformed"):
        _parse_private_key("not-a-private-key")


def test_parse_private_key_rejects_wrong_length():
    with pytest.raises(ValueError, match="malformed"):
        _parse_private_key("0x1234")


# --- _hex_to_bytes32: pure, no network ---


def test_hex_to_bytes32_accepts_valid_64_char_hex():
    result = _hex_to_bytes32("a" * 64)
    assert len(result) == 32


def test_hex_to_bytes32_rejects_wrong_length():
    with pytest.raises(ValueError):
        _hex_to_bytes32("a" * 10)


def test_hex_to_bytes32_rejects_non_hex():
    with pytest.raises(ValueError):
        _hex_to_bytes32("not-hex-at-all-not-hex-at-all-not-hex-at-all-x!")


# --- BlockchainRecord validation ---


def _payload() -> OnChainPayload:
    return OnChainPayload(content_hash="a" * 64, algorithm="SHA-256", version="v1")


def test_blockchain_record_accepts_valid_tx_hash():
    record = BlockchainRecord(
        tx_hash="0x" + "a" * 64,
        block_number=1,
        timestamp=datetime.now(timezone.utc),
        on_chain_payload=_payload(),
    )
    assert record.block_number == 1


def test_blockchain_record_rejects_missing_0x_prefix():
    with pytest.raises(ValueError):
        BlockchainRecord(
            tx_hash="a" * 64,
            block_number=1,
            timestamp=datetime.now(timezone.utc),
            on_chain_payload=_payload(),
        )


def test_blockchain_record_rejects_wrong_length_tx_hash():
    with pytest.raises(ValueError):
        BlockchainRecord(
            tx_hash="0x1234",
            block_number=1,
            timestamp=datetime.now(timezone.utc),
            on_chain_payload=_payload(),
        )


def test_blockchain_record_rejects_negative_block_number():
    with pytest.raises(ValueError):
        BlockchainRecord(
            tx_hash="0x" + "a" * 64,
            block_number=-1,
            timestamp=datetime.now(timezone.utc),
            on_chain_payload=_payload(),
        )
