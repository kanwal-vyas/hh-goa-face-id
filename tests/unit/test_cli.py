import subprocess
import sys
from pathlib import Path

import pytest
from web3 import Web3

from main import _verification_status

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RPC_URL = "http://127.0.0.1:8545"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_help_runs_successfully():
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "Face Identification" in result.stdout


def test_no_args_prints_help_and_exits_zero():
    result = _run_cli()
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_missing_reference_image_produces_clean_error():
    result = _run_cli("--reference", "examples/does_not_exist.jpg")
    assert result.returncode == 1
    assert "not found" in result.stderr.lower()
    assert "Traceback" not in result.stderr


def test_reference_flag_with_real_but_faceless_file_reports_no_face_error(tmp_path):
    """A real, decodable image with no face should surface the
    processor's clear 'no face detected' error, not the missing-file
    error and not a crash."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    image_path = tmp_path / "no_face.jpg"
    ok, buf = cv2.imencode(".jpg", noise)
    assert ok
    image_path.write_bytes(buf.tobytes())

    result = _run_cli("--reference", str(image_path))
    assert result.returncode == 1
    assert "no face detected" in result.stderr.lower()


def test_reference_flag_with_corrupted_file_reports_clean_error(tmp_path):
    bad_image = tmp_path / "corrupt.jpg"
    bad_image.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg-body")

    result = _run_cli("--reference", str(bad_image))
    assert result.returncode == 1
    assert "unsupported or corrupted image" in result.stderr.lower()
    assert "Traceback" not in result.stderr


FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "authorized_faces"
SINGLE_FACE_IMAGE = FIXTURE_DIR / "single_face.jpg"


@pytest.mark.skipif(
    not SINGLE_FACE_IMAGE.exists(),
    reason="No authorized single-face test image supplied - see tests/fixtures/authorized_faces/README.md",
)
def test_reference_and_compare_against_itself_reports_strong_match():
    """End-to-end CLI demonstration using an authorized fixture image
    compared against itself - should report STRONG_MATCH."""
    result = _run_cli(
        "--reference", str(SINGLE_FACE_IMAGE), "--compare", str(SINGLE_FACE_IMAGE)
    )
    assert result.returncode == 0
    assert "Face processing" in result.stdout
    assert "Face comparison" in result.stdout
    assert "STRONG_MATCH" in result.stdout


# --- --blockchain-demo (Milestone 4) ---


def test_help_lists_blockchain_demo_flag():
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "--blockchain-demo" in result.stdout


def test_blockchain_demo_missing_input_file_fails_cleanly():
    result = _run_cli("--blockchain-demo", "/tmp/does-not-exist-blockchain-demo.txt")
    assert result.returncode == 1
    assert "not found" in result.stderr.lower()
    assert "Traceback" not in result.stderr


def _node_available() -> bool:
    try:
        w3 = Web3(Web3.HTTPProvider(RPC_URL, request_kwargs={"timeout": 2}))
        return w3.is_connected()
    except Exception:  # noqa: BLE001 -- reachability probe: any failure means "unavailable"
        return False


@pytest.mark.skipif(
    _node_available(),
    reason="A local EVM node IS reachable; this test specifically covers the unavailable case.",
)
def test_blockchain_demo_fails_cleanly_when_node_unavailable(tmp_path):
    content_file = tmp_path / "content.txt"
    content_file.write_text("some demo content", encoding="utf-8")

    result = _run_cli("--blockchain-demo", str(content_file))
    assert result.returncode == 1
    assert "could not connect" in result.stderr.lower()
    assert "npx hardhat node" in result.stderr
    assert "Traceback" not in result.stderr
    # Local stages before the blockchain connection attempt still ran
    # for real and were reported, rather than the whole command
    # aborting silently.
    assert "Content hash:" in result.stdout


@pytest.mark.skipif(
    not _node_available(),
    reason=(
        f"No local EVM node reachable at {RPC_URL}. Start one with "
        "`npx hardhat node` and re-run to execute this test."
    ),
)
def test_blockchain_demo_succeeds_against_live_local_node(tmp_path):
    content_file = tmp_path / "content.txt"
    content_file.write_text(
        "CLI blockchain demo test content — unique per test run "
        f"{content_file}",
        encoding="utf-8",
    )

    result = _run_cli("--blockchain-demo", str(content_file))
    assert result.returncode == 0, result.stderr
    assert "Content hash:" in result.stdout
    assert "Transaction: 0x" in result.stdout
    assert "On-chain hash:" in result.stdout
    assert "VERIFICATION: PASS" in result.stdout


# --- _verification_status: pure function, no blockchain needed ---


def test_verification_status_pass_on_matching_hashes():
    same_hash = "a" * 64
    assert _verification_status(same_hash, same_hash) == "PASS"


def test_verification_status_fail_on_mismatched_hashes():
    assert _verification_status("a" * 64, "b" * 64) == "FAIL"
