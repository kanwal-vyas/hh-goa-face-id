"""
Verification scaffold.

Defines the data model and interface for the final pipeline stage:
recomputing a content fingerprint and comparing it against the on-chain
record to determine VERIFIED vs TAMPER_DETECTED.

No concrete implementation is provided at this milestone — it depends on
CanonicalContent hashing (app.content.canonicalize) and a
BlockchainProvider.retrieve() call (app.blockchain.interface), neither of
which are concretely implemented yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from app.blockchain.interface import BlockchainProvider
from app.content.canonicalize import CanonicalContent


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    TAMPER_DETECTED = "TAMPER_DETECTED"


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of comparing a locally recomputed hash against the
    on-chain record.

    Attributes:
        local_hash: Hex SHA-256 digest recomputed from current content.
        on_chain_hash: Hex SHA-256 digest read from the blockchain record.
        status: VERIFIED if the hashes match, TAMPER_DETECTED otherwise.
        tx_reference: The transaction hash the on-chain record was read
            from, so an evaluator can independently inspect it.
        checked_at: When this verification was performed.
    """

    local_hash: str
    on_chain_hash: str
    status: VerificationStatus
    tx_reference: str
    checked_at: datetime

    @property
    def match(self) -> bool:
        return self.status == VerificationStatus.VERIFIED

    def __post_init__(self) -> None:
        expected_status = (
            VerificationStatus.VERIFIED
            if self.local_hash == self.on_chain_hash
            else VerificationStatus.TAMPER_DETECTED
        )
        if self.status != expected_status:
            raise ValueError(
                "VerificationResult.status is inconsistent with "
                "local_hash vs on_chain_hash comparison — status must be "
                "derived from the actual hash comparison, never set "
                "independently."
            )


class Verifier(ABC):
    """Abstract interface for the verification stage.

    Concrete implementation (recompute hash from CanonicalContent, fetch
    BlockchainRecord via a BlockchainProvider, compare) is added in a
    later milestone once both dependencies are concretely implemented.
    """

    @abstractmethod
    def verify(
        self,
        canonical_content: CanonicalContent,
        tx_hash: str,
        blockchain_provider: BlockchainProvider,
    ) -> VerificationResult:
        raise NotImplementedError
