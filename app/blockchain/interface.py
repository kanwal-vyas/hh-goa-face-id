"""
BlockchainProvider abstraction.

Defines the interface and data model used to register a content
fingerprint on-chain and later retrieve it for verification. No concrete
implementation (local dev chain or public testnet) is provided in this
milestone — see the architecture doc for LocalBlockchainProvider /
TestnetBlockchainProvider, which arrive in a later milestone.

Only a fingerprint/commitment and minimal metadata are ever intended to
go on-chain. Raw biometric data, raw images, and raw source content are
explicitly out of scope for on-chain storage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class OnChainPayload:
    """The minimal data committed on-chain for a given piece of content.

    Attributes:
        content_hash: Hex SHA-256 digest of the canonicalized content.
        algorithm: Hashing algorithm identifier, e.g. "SHA-256".
        version: Canonicalization scheme version used to produce the hash
            (see app/content/canonicalize.py). Needed so future
            verification runs know which normalization rules to
            reproduce.
        source_reference_hash: Optional hash (not the raw value) of the
            content's source reference, if the source reference itself
            should not be stored in the clear.
    """

    content_hash: str
    algorithm: str
    version: str
    source_reference_hash: str | None = None


@dataclass(frozen=True)
class BlockchainRecord:
    """Result of a successful registration, or a retrieved on-chain record.

    Attributes:
        tx_hash: Transaction hash the record was written/read under.
        block_number: Block the transaction was included in.
        timestamp: Block timestamp (UTC).
        on_chain_payload: The committed data itself.
    """

    tx_hash: str
    block_number: int
    timestamp: datetime
    on_chain_payload: OnChainPayload

    def __post_init__(self) -> None:
        if not self.tx_hash.startswith("0x"):
            raise ValueError(f"tx_hash must be 0x-prefixed, got {self.tx_hash!r}")
        if len(self.tx_hash) != 66:
            raise ValueError(
                f"tx_hash must be 66 characters (0x + 64 hex), got "
                f"{len(self.tx_hash)}: {self.tx_hash!r}"
            )
        if self.block_number < 0:
            raise ValueError(f"block_number must be non-negative, got {self.block_number}")


class BlockchainProvider(ABC):
    """Abstract interface for registering and retrieving fingerprint
    commitments on a blockchain.

    Concrete subclasses (local dev chain, public testnet) are implemented
    in a later milestone. This class must remain abstract at this stage.
    """

    @abstractmethod
    def register(self, payload: OnChainPayload) -> BlockchainRecord:
        """Write a fingerprint commitment on-chain and return the
        resulting record, including the transaction hash."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, tx_hash: str) -> BlockchainRecord | None:
        """Fetch a previously-registered record by transaction hash.

        Returns None if no record is found for the given tx_hash.
        """
        raise NotImplementedError
