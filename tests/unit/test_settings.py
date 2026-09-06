import pytest

from app.config.settings import BlockchainMode, ConfigError, load_settings


def _clear_env(monkeypatch):
    for var in (
        "BLOCKCHAIN_MODE",
        "BLOCKCHAIN_RPC_URL",
        "BLOCKCHAIN_PRIVATE_KEY",
        "BLOCKCHAIN_CONTRACT_ADDRESS",
        "MATCH_THRESHOLD",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_defaults_load_successfully(monkeypatch):
    _clear_env(monkeypatch)
    settings = load_settings(env_file=None)
    assert settings.blockchain_mode == BlockchainMode.LOCAL
    assert settings.blockchain_rpc_url == "http://127.0.0.1:8545"
    assert settings.blockchain_private_key is None
    assert settings.blockchain_contract_address is None
    assert settings.match_threshold == 0.6
    assert settings.log_level == "INFO"


def test_valid_contract_address_accepted(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv(
        "BLOCKCHAIN_CONTRACT_ADDRESS", "0x5FbDB2315678afecb367f032d93F642f64180aa3"
    )
    settings = load_settings(env_file=None)
    assert settings.blockchain_contract_address == "0x5FbDB2315678afecb367f032d93F642f64180aa3"


def test_malformed_contract_address_rejected(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("BLOCKCHAIN_CONTRACT_ADDRESS", "not-an-address")
    with pytest.raises(ConfigError):
        load_settings(env_file=None)


def test_invalid_blockchain_mode_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("BLOCKCHAIN_MODE", "mainnet-yolo")
    with pytest.raises(ConfigError):
        load_settings(env_file=None)


def test_invalid_match_threshold_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MATCH_THRESHOLD", "not-a-float")
    with pytest.raises(ConfigError):
        load_settings(env_file=None)


def test_match_threshold_out_of_range_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MATCH_THRESHOLD", "1.5")
    with pytest.raises(ConfigError):
        load_settings(env_file=None)


def test_invalid_log_level_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LOG_LEVEL", "SUPER_VERBOSE")
    with pytest.raises(ConfigError):
        load_settings(env_file=None)


def test_testnet_requires_private_key(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("BLOCKCHAIN_MODE", "testnet")
    with pytest.raises(ConfigError):
        load_settings(env_file=None)


def test_testnet_with_private_key_succeeds(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("BLOCKCHAIN_MODE", "testnet")
    monkeypatch.setenv("BLOCKCHAIN_PRIVATE_KEY", "0xdeadbeef")
    settings = load_settings(env_file=None)
    assert settings.blockchain_mode == BlockchainMode.TESTNET
    assert settings.blockchain_private_key == "0xdeadbeef"


def test_no_hardcoded_private_key():
    """Settings module source must not contain a literal private key
    (a long hex string assigned as a value), while still allowing
    legitimate short patterns like the address-format regex."""
    import inspect
    import re

    from app.config import settings as settings_module

    source = inspect.getsource(settings_module)
    # A private key is 64 hex characters (32 bytes); the address regex
    # only matches up to 40, so this pattern would only trip on an
    # actual hardcoded secret, not on the regex literal.
    assert re.search(r"0x[0-9a-fA-F]{64}", source) is None
