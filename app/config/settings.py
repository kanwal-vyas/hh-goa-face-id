"""
Typed configuration for the pipeline.

All configuration is sourced from environment variables (optionally loaded
from a local .env file via python-dotenv). Nothing sensitive is ever
hardcoded here — see .env.example for the documented variable list.

This module performs eager validation at load time so that misconfiguration
is caught immediately (e.g. at CLI startup) rather than deep inside the
pipeline during a demo run.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    from dotenv import load_dotenv

    _DOTENV_AVAILABLE = True
except ImportError:  # pragma: no cover - dotenv is a declared dependency
    _DOTENV_AVAILABLE = False


class BlockchainMode(str, Enum):
    """Which BlockchainProvider implementation should be used.

    Only the mode is defined at this milestone; concrete providers
    (LocalBlockchainProvider / TestnetBlockchainProvider) are implemented
    in a later milestone per the approved architecture.
    """

    LOCAL = "local"
    TESTNET = "testnet"


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class Settings:
    """Immutable, validated application settings.

    Fields are intentionally limited to what this milestone's modules
    actually consume. New settings should be added alongside the
    milestone that first needs them, not speculatively.
    """

    blockchain_mode: BlockchainMode
    blockchain_rpc_url: str
    blockchain_private_key: str | None
    blockchain_contract_address: str | None
    match_threshold: float
    log_level: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.match_threshold <= 1.0):
            raise ConfigError(
                f"MATCH_THRESHOLD must be between 0.0 and 1.0, got "
                f"{self.match_threshold!r}"
            )

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ConfigError(
                f"LOG_LEVEL must be one of {sorted(valid_levels)}, "
                f"got {self.log_level!r}"
            )

        if self.blockchain_mode == BlockchainMode.TESTNET and not self.blockchain_private_key:
            raise ConfigError(
                "BLOCKCHAIN_PRIVATE_KEY is required when "
                "BLOCKCHAIN_MODE=testnet (not required for local mode, "
                "which uses a pre-funded dev account)."
            )

        if not self.blockchain_rpc_url:
            raise ConfigError("BLOCKCHAIN_RPC_URL must not be empty.")

        if self.blockchain_contract_address is not None and not _ADDRESS_RE.match(
            self.blockchain_contract_address
        ):
            raise ConfigError(
                "BLOCKCHAIN_CONTRACT_ADDRESS must be a 0x-prefixed 40-hex-"
                f"character Ethereum address, got {self.blockchain_contract_address!r}"
            )


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value.strip() if isinstance(value, str) else value


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    """Load settings from environment variables.

    Args:
        env_file: Optional path to a .env file to load before reading
            os.environ. If the file does not exist, this is silently
            skipped (matches typical dotenv behavior) — the process
            environment always takes precedence over the file.

    Raises:
        ConfigError: if required variables are missing or invalid.
    """
    if env_file and _DOTENV_AVAILABLE:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)

    raw_mode = (_get_env("BLOCKCHAIN_MODE", "local") or "local").lower()
    try:
        blockchain_mode = BlockchainMode(raw_mode)
    except ValueError as exc:
        valid = [m.value for m in BlockchainMode]
        raise ConfigError(
            f"BLOCKCHAIN_MODE must be one of {valid}, got {raw_mode!r}"
        ) from exc

    blockchain_rpc_url = _get_env("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545") or ""
    blockchain_private_key = _get_env("BLOCKCHAIN_PRIVATE_KEY") or None
    blockchain_contract_address = _get_env("BLOCKCHAIN_CONTRACT_ADDRESS") or None

    raw_threshold = _get_env("MATCH_THRESHOLD", "0.6") or "0.6"
    try:
        match_threshold = float(raw_threshold)
    except ValueError as exc:
        raise ConfigError(
            f"MATCH_THRESHOLD must be a float, got {raw_threshold!r}"
        ) from exc

    log_level = _get_env("LOG_LEVEL", "INFO") or "INFO"

    return Settings(
        blockchain_mode=blockchain_mode,
        blockchain_rpc_url=blockchain_rpc_url,
        blockchain_private_key=blockchain_private_key,
        blockchain_contract_address=blockchain_contract_address,
        match_threshold=match_threshold,
        log_level=log_level,
    )
