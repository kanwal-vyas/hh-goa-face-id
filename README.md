# HH Goa 2026 — Face Identification & Blockchain Verification

Hackathon shortlisting submission (Task 3). This repository implements a
modular pipeline that: processes a face image, discovers matching
content within an authorized, bounded search target, fingerprints that
content, registers the fingerprint on a blockchain, and later verifies
whether the content has remained unchanged.

## Current implementation status

**Milestone 7 is complete: bounded candidate matching.** The repository
includes a real, local `SearchProvider` implementation that retrieves
only a small preconfigured consented demo corpus, plus deterministic
candidate-image matching. It is not a social-media or open-web search
system.

Implemented and real (not stubbed, not mocked):
- **Face detection / encoding** (`app/face/opencv_processor.py`,
  `app/face/compare.py`) — OpenCV Haar cascade detection + a classical
  LBP embedding, cosine similarity, configurable NO_MATCH /
  POSSIBLE_MATCH / STRONG_MATCH bands.
- **Deterministic content canonicalization** (`app/content/canonicalize.py`)
  — `DeterministicCanonicalizer` turns `DiscoveredContent` into
  byte-for-byte reproducible `CanonicalContent`, independent of source,
  candidate, retrieval time, or metadata key order.
- **SHA-256 content fingerprinting** (`app/content/fingerprint.py`) —
  `hash_canonical_content()`.
- **Separate source-reference hashing** (`app/content/fingerprint.py`)
  — `hash_source_reference()`, intentionally independent of the content
  hash.
- **Local Ethereum-compatible blockchain registration + retrieval**
  (`app/blockchain/local_provider.py`) — a real Solidity registry
  contract (`FingerprintRegistry.sol`), deployed and called via
  `web3.py` against a real local EVM development node.
- **Local-vs-on-chain fingerprint verification** — implemented directly
  in the CLI's `--blockchain-demo` flow (see "Milestone 4 — Local
  Blockchain" below); a dedicated `Verifier` abstraction for this same
  comparison, wired into the full pipeline, is still scaffolded only
  (see Known Limitations).
- **Authorized local corpus search** (`app/search/authorized_corpus.py`)
  — deterministic filtering and ranking of committed synthetic demo
  records, with no network access and no biometric lookup.
- **Candidate matching** (`app/matching/candidate_matcher.py`) — local
  candidate-image processing, face-similarity comparison, explicit
  rejection handling, deterministic ranking, and cautious selection.
- **CLI demonstrations** for all of the above (`--reference`/
  `--compare`, `--fingerprint`, `--blockchain-demo`, and
  `scripts/run_authorized_match_demo.py`).

**Not implemented**: full end-to-end pipeline orchestration
(`PipelineRunner` remains a
standalone scaffold — see Known Limitations). This project performs no
web or social-media search of any kind.

## Face processing

### Technology chosen: OpenCV Haar cascade + classical LBP embedding

- **Detection**: `cv2.CascadeClassifier` with the Haar cascade
  `haarcascade_frontalface_default.xml`, which ships **inside** the
  `opencv-python` wheel — no separate download, no network dependency,
  identical behavior on Linux/macOS/Windows the moment `pip install
  opencv-python` succeeds.
- **Embedding**: a hand-rolled, grid-based Local Binary Pattern (LBP)
  histogram descriptor (Ahonen, Hadid & Pietikäinen, 2006), computed
  with OpenCV + NumPy only. Fixed dimension (default 8×8 grid × 8 bins =
  512), fully deterministic for a given input.
- **Model version tag**: `opencv-haar+lbp-v1` (recorded on every
  `FaceEmbedding` so future comparisons/upgrades stay consistent).
- **Similarity metric**: cosine similarity between two embedding
  vectors, clamped to `[0.0, 1.0]`.

### Why this over a DNN embedding model?

OpenCV also supports `cv2.FaceDetectorYN` (YuNet) and
`cv2.FaceRecognizerSF` (SFace) directly — a meaningfully more accurate
deep-learned detector + embedding pair, and the first choice evaluated
for this milestone. However:

- **Confirmed constraint**: the official pretrained weights for those
  models are distributed via Git LFS on the `opencv/opencv_zoo` GitHub
  repository. LFS objects are served from `media.githubusercontent.com`,
  which was **not reachable** from this project's sandboxed development
  environment (only `github.com` / `raw.githubusercontent.com` were
  reachable, and those return small Git LFS *pointer* files — not the
  real binary — for this particular repository). This was verified
  directly (see `scripts/download_face_models.py`'s docstring for the
  exact URLs and SHA-256 hashes recorded during that investigation).
- Given the hard hackathon deadline and the priority on a pipeline that
  is **reliably runnable and testable right now**, this milestone ships
  the zero-download Haar+LBP path as the default.

**`scripts/download_face_models.py`** documents the DNN upgrade path
precisely (URLs + verified SHA-256 hashes) for use on a machine with
normal internet access. It was **not** run successfully end-to-end in
this project's environment — that specific limitation is stated
explicitly here rather than silently worked around.

**Swapping in the DNN path later requires no changes outside the face
module**: implement a `YuNetSFaceFaceProcessor(FaceProcessor)` following
the same interface as `OpenCVFaceProcessor`, and a matching
`FaceComparator` if SFace's own match/cosine convention differs. The CLI
and everything downstream only depend on the abstract `FaceProcessor` /
`FaceComparator` interfaces.

### Honest tradeoff

Haar+LBP is meaningfully less accurate than a modern deep face-embedding
model, especially across pose/lighting variation. It is used here
because it is genuinely reproducible today with zero setup friction —
not because it is the most accurate option available. **Tune
`MATCH_THRESHOLD` against your own authorized test images before relying
on this for a demo** — there is no published, validated threshold for
this classical descriptor the way there is for a model like SFace.

### Thresholds / match bands

Given one `MATCH_THRESHOLD` (default `0.6`), bands are derived as:

```
similarity_score >= MATCH_THRESHOLD             -> STRONG_MATCH
MATCH_THRESHOLD*0.5 <= similarity < THRESHOLD    -> POSSIBLE_MATCH
similarity_score <  MATCH_THRESHOLD*0.5          -> NO_MATCH
```

These boundaries are a documented heuristic, not a scientifically
validated cutoff — see the tradeoff note above.

### CPU / platform requirements

CPU-only, no GPU required. `opencv-python` ships prebuilt wheels for
Windows/macOS/Linux, so no local compilation (unlike `dlib`-based
libraries such as `face_recognition`, which was evaluated and rejected
specifically for its unreliable Windows/CMake build story under a hard
deadline).

### Privacy behavior

- Embeddings are held in memory only for the duration of a CLI
  invocation; nothing is written to disk or persisted between runs.
- No face image or embedding is logged at content level — only counts,
  dimensions, and scores.
- Results are always reported as a similarity score plus a qualitative
  band, never as confirmed identity (see CLI output below).

### Exact commands

```bash
python main.py --reference examples/test_face.jpg
python main.py --reference examples/test_face.jpg --compare examples/authorized_demo_images/harbor-avatar.png
python scripts/run_authorized_match_demo.py --reference examples/test_face.jpg
```

Example output (illustrative — actual similarity/band depend on your
images):

```
[1/8] Face processing
    Reference face detected
    Model: opencv-haar+lbp-v1
    Embedding dimension: 512
    Quality heuristic (blur-based, not a model confidence): 0.842

[2/8] Face comparison
    Similarity: 0.9187
    Result: STRONG_MATCH
    (match_threshold=0.6)

Note: this is a similarity score, not proof of real-world identity.

Remaining pipeline stages are not wired into this CLI flow:
  · [3/8] Candidate matching: not wired into this CLI flow
  · [4/8] Content extraction: not wired into this CLI flow
  · [5/8] Canonicalization: not wired into this CLI flow
  · [6/8] SHA-256 fingerprint: not wired into this CLI flow
  · [7/8] Blockchain registration: not wired into this CLI flow
  · [8/8] Verification: not wired into this CLI flow
```

**Similarity ≠ confirmed identity.** This score is a face-similarity
signal only; it is never presented, here or anywhere else in this
project, as proof that two images depict the same real-world person.

**Note on the stage numbering above**: this face-processing flow does
not go through canonicalization/hashing — it only ever produces a
`FaceEmbedding`/similarity result, not `DiscoveredContent`.
Canonicalization + SHA-256 fingerprinting and blockchain registration
are real and tested, but are demonstrated independently via
`--fingerprint`/`--blockchain-demo` (see next sections) until the
pipeline-integration milestone wires every stage through one shared
run — see "Milestones" below.

## Canonicalization and Fingerprinting

### Why canonicalization exists

The same logical content can arrive as different bytes depending on
*when* or *how* it was fetched (retrieval timestamp), what order a
metadata dict happened to be built in, or which line-ending convention
a text file uses. If we hashed raw `DiscoveredContent` directly, two
fetches of the *identical* underlying post could produce two different
hashes for reasons that have nothing to do with the content actually
changing — which would make blockchain verification useless (everything
would look "tampered" all the time). Canonicalization defines one
deterministic byte representation for a given piece of content, so the
SHA-256 hash reflects genuine content changes only.

### Why raw content isn't simply concatenated

Hashing something like `text + metadata_string` is ambiguous: two
different `(text, metadata)` pairs can concatenate to the same string
(e.g. `text="ab", metadata="c"` vs `text="a", metadata="bc"`). Instead,
every field is serialized into its own explicitly-named JSON key, so
there's no way for content to "leak" across a field boundary.

### Included / excluded fields (canonicalization version `v1`)

**Included** (part of the content hash):
- `text` — Unicode NFC-normalized, newlines normalized to `\n`.
  Otherwise **unchanged**: no stripping, no lowercasing, no punctuation
  removal. The goal is content integrity, not semantic similarity, so a
  single meaningful character change must still change the hash.
- `metadata` — included as given; key order never affects the hash (see
  "Deterministic serialization" below).
- `raw_bytes` — included via base64 encoding in its own JSON field
  (`raw_content_base64`), so binary content can never collide with text
  or metadata bytes. `None`/absent raw_bytes serializes as JSON `null`;
  empty bytes (`b""`) serialize as `""` — the two cases are always
  distinguishable.

**Excluded** (never affect the content hash):
- `candidate_id` — identifies a retrieval candidate, not content.
- `source_reference` — identifies WHERE content was found, not what it
  is. **[Corrected]** An earlier version of this canonicalizer included
  `source_reference` (and `candidate_id`), which made the fingerprint
  source-dependent — the exact same content at two different URLs
  hashed differently. That was a genuine semantic bug for a
  content-integrity fingerprint: it should represent the content
  itself, not its location. This has been corrected — the same content
  now hashes identically regardless of where it was found or which
  retrieval candidate produced it. If provenance needs to be committed
  alongside a fingerprint, it is hashed *separately* — see
  "content_hash vs. source_reference_hash" below.
- `retrieved_at` — when the content was fetched, not what was fetched.

Verified directly by tests (`tests/unit/test_canonicalization.py`): the
same content with a different `source_reference`, a different
`candidate_id`, a different `retrieved_at`, or metadata built in a
different key order all produce byte-identical `canonical_bytes` and
identical hashes.

**Known conservative limitation**: `DiscoveredContent.metadata` has no
schema distinguishing content-intrinsic fields (e.g. a caption) from
volatile retrieval fields (e.g. a request ID). This milestone does not
attempt to guess which is which — the entire dict is included in the
hash as-is. Callers are responsible for only putting genuinely
content-intrinsic fields into `metadata`.

### `content_hash` vs. `source_reference_hash`

These are two different values serving two different purposes, and the
codebase keeps them structurally separate so they can never be
conflated:

| | What it identifies | Computed by |
|---|---|---|
| `content_hash` | The canonicalized **content itself** | `hash_canonical_content()` over `CanonicalContent.canonical_bytes` |
| `source_reference_hash` | **Where** the content was found (a URL, file path, etc.) | `hash_source_reference()` — a plain SHA-256 over the UTF-8 source-reference string, no canonicalization envelope needed since it's already a single unambiguous string |

`source_reference` itself remains available on `DiscoveredContent` for
logging/off-chain use, but only its *hash* — never the raw value — is
what optionally accompanies a `content_hash` in a blockchain record
(see "Milestone 4 — Local Blockchain" below).

### Deterministic serialization

Fields are assembled into a dict and serialized with:
```python
json.dumps(envelope, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
```
- `sort_keys=True` makes metadata key insertion order irrelevant (Python's
  `json` module sorts recursively at every nesting level).
- `separators=(",", ":")` strips whitespace, so no pretty-printing
  setting can ever change the byte output.
- Explicit UTF-8 encoding, not `repr()`, not implicit `str.encode()`
  defaults.

### SHA-256

`app/content/fingerprint.py`'s `hash_canonical_content()` computes
`hashlib.sha256(canonical_content.canonical_bytes).hexdigest()` — the
standard library only, no external hashing dependency, no MD5/SHA-1.

### Why `generated_at` is not hashed

`ContentFingerprint.generated_at` records *when the fingerprint was
computed* — useful metadata for a blockchain record later, but not part
of the content's identity. Only `canonical_bytes` is fed into
`hashlib.sha256()`. Verified directly by a test: hashing the same
`CanonicalContent` with two different `generated_at` values yields the
same `hash` both times.

### How tampering changes the fingerprint

Any single meaningful byte/character change in `text` or `raw_bytes`
propagates into `canonical_bytes` (since the field is included verbatim,
NFC-normalization aside) and therefore changes the SHA-256 hash. This is
the exact mechanism the Milestone 4 blockchain demo's
local-vs-on-chain comparison relies on: recompute the hash from current
content, compare against the hash registered on-chain, and a mismatch
means the content changed since registration. See "Milestone 4 — Local
Blockchain" below for the real, working implementation of this
comparison.

### Empty vs. missing content

- Empty text (`text=""`) and empty bytes (`raw_bytes=b""`) are both
  valid, hash successfully, and are distinguishable from each other and
  from "no content at all."
- `DiscoveredContent` itself refuses construction with neither
  `raw_bytes` nor `text` set (`ValueError` at the model layer) — `None`
  is never silently hashed.

### Exact command

```bash
python main.py --fingerprint path/to/local_text_file.txt
```

Example output (illustrative — the actual hash depends on your file's content):
```
Canonicalization version: v1
Algorithm: SHA-256
Fingerprint: 779e496f3cf7bca47cb2e51b235c940990b12d46e0b4cc80e119a4368c00bf04
```

Note: since `source_reference` and `candidate_id` are excluded from the
content hash (see above), running this command on two different files
with byte-identical text content will produce the **same** fingerprint —
this is the corrected, intended behavior.

## Milestone 4 — Local Blockchain

### Why local-first

A local Ethereum-compatible development chain (not a public testnet)
is the primary and only currently-implemented target. This avoids
depending on faucets, external RPC providers, browser wallets, block
explorers, or internet connectivity during a demo — the biggest
reliability risk under a hard deadline. `BlockchainProvider` is an
abstraction specifically so a `TestnetBlockchainProvider` could be
added later without touching anything else, but it does not exist yet.

### Tooling

- **Node**: a [Hardhat](https://hardhat.org) development node
  (`npx hardhat node`), exposing standard JSON-RPC at
  `http://127.0.0.1:8545` with pre-funded, unlocked development
  accounts. Any Anvil/Ganache-style node exposing the same JSON-RPC
  surface and unlocked accounts would work identically through this
  same provider — Hardhat was used here because it installs cleanly via
  plain `npm install` with no separate binary download.
- **SDK**: `web3.py`, plus `eth-account` (imported directly for
  private-key parsing when `BLOCKCHAIN_PRIVATE_KEY` is set).
- **Contract**: a minimal Solidity registry,
  `app/blockchain/contracts/FingerprintRegistry.sol`. The provider does
  **not** compile Solidity at runtime — it loads a precompiled artifact
  (`FingerprintRegistry.json`: ABI + bytecode) checked into the repo
  alongside the source. This avoids a runtime `solc`/`py-solc-x`
  dependency purely to reproduce a fixed, already-reviewed contract; the
  committed `.sol` file is the source of truth for what that bytecode
  does and can be recompiled with any standard Solidity 0.8.24 toolchain
  to verify the artifact yourself.

### What the contract does

`FingerprintRegistry` stores one record per unique `content_hash`:

```solidity
function register(
    bytes32 contentHash,
    string calldata algorithm,
    string calldata canonicalizationVersion,
    bytes32 sourceReferenceHash
) external returns (bool);

function getRecord(bytes32 contentHash) external view returns (Record memory);
```

Registering the same `content_hash` twice **reverts** (deterministic
duplicate-registration behavior — not silently overwritten, not
silently ignored). A `FingerprintRegistered` event is emitted on
successful registration.

### Registration flow

```
ContentFingerprint (content_hash, algorithm, canonicalization_version)
    + source_reference_hash (computed separately, see below)
        ↓
OnChainPayload
        ↓
LocalBlockchainProvider.register(payload)
        ↓
  1. Connect to BLOCKCHAIN_RPC_URL, verify connectivity
  2. Deploy FingerprintRegistry if no BLOCKCHAIN_CONTRACT_ADDRESS is
     configured (lazily, once per process — not redeployed per call)
  3. Build + send the register() transaction
       - no BLOCKCHAIN_PRIVATE_KEY: sent via the node's own unlocked
         dev account (eth_sendTransaction) — no client-side signing
       - BLOCKCHAIN_PRIVATE_KEY set: signed locally, sent via
         eth_sendRawTransaction
  4. Wait for the transaction receipt
        ↓
BlockchainRecord (tx_hash, block_number, timestamp, on_chain_payload)
```

### Retrieval flow

```
tx_hash
        ↓
LocalBlockchainProvider.retrieve(tx_hash)
        ↓
  1. Fetch the transaction receipt (None if not found)
  2. Decode the FingerprintRegistered event from the receipt to get
     content_hash
  3. Call the contract's getRecord(content_hash) — the canonical
     on-chain state, not just the event log
        ↓
BlockchainRecord | None
```

### On-chain data model

**Stored on-chain** (per `FingerprintRegistry.sol`):
- `content_hash` (SHA-256 of canonicalized content, as `bytes32`)
- `algorithm` (`"SHA-256"`)
- `canonicalization_version` (e.g. `"v1"`)
- `source_reference_hash` (SHA-256 of the source reference, as
  `bytes32`; the zero-bytes32 value if none was provided)
- the registering block's `timestamp`

**Never stored on-chain**:
- raw images, raw text, or any other raw content
- face embeddings or any other biometric data
- private keys
- the raw source reference / URL itself (only its hash)

### Local blockchain setup

These are the exact commands used to build and validate this milestone.

**1. Python dependencies:**
```bash
pip install -r requirements.txt
```

**2. Local EVM node** (in a separate scratch directory outside this repo):
```bash
mkdir hardhat-node && cd hardhat-node
npm init -y
npm pkg set type="module"
npm install --no-save hardhat
```
Create `hardhat.config.js`:
```javascript
export default {
  solidity: "0.8.24",
};
```

**3. Start the local node** (leave running in its own terminal):
```bash
npx hardhat node
```
This prints a JSON-RPC endpoint (`http://127.0.0.1:8545` by default)
and a list of pre-funded development accounts with their private keys —
these are the same well-known, publicly-documented Hardhat test
accounts every `npx hardhat node` invocation prints; they are safe to
reference in documentation but are not meant to hold anything of value,
and no production key should ever be handled this way.

**4. Environment**: with the node running and `BLOCKCHAIN_MODE=local`
(the default), no further configuration is required —
`BLOCKCHAIN_PRIVATE_KEY` can stay blank; the provider uses the node's
own unlocked first account automatically.

**5. Contract deployment**: automatic. The first `--blockchain-demo`
run with no `BLOCKCHAIN_CONTRACT_ADDRESS` configured deploys a fresh
`FingerprintRegistry` and logs its address
(`FingerprintRegistry deployed at 0x...`). Set
`BLOCKCHAIN_CONTRACT_ADDRESS` to that value to reuse the same deployed
contract on subsequent runs instead of deploying a new one each time.

**6. Run the CLI demo:**
```bash
python main.py --blockchain-demo path/to/local_text_file.txt
```

*(Recompiling the contract yourself, if you want to verify the
committed artifact, only needs `solc` — e.g. `npm install --no-save
solc@0.8.24` and running it against
`app/blockchain/contracts/FingerprintRegistry.sol`; this is optional
and not required to run the CLI.)*

### Environment variables (blockchain)

| Variable | Required? | Notes |
|---|---|---|
| `BLOCKCHAIN_MODE` | No (default `local`) | `local` or `testnet`. Only `local` has a concrete provider implemented. |
| `BLOCKCHAIN_RPC_URL` | No (default `http://127.0.0.1:8545`) | Must point at a reachable JSON-RPC endpoint for `--blockchain-demo` to succeed. |
| `BLOCKCHAIN_PRIVATE_KEY` | No for local mode; required if `BLOCKCHAIN_MODE=testnet` | Leave blank for local mode — the node's own unlocked dev account is used instead. **Never commit a real key here.** |
| `BLOCKCHAIN_CONTRACT_ADDRESS` | No | Leave blank to auto-deploy a fresh contract on first use; set it to reuse one deployed contract across runs. |

`.env.example` documents all of these with the same guidance and ships
with every value blank/default — no real credentials are present
anywhere in this repository.

### CLI example

```bash
python main.py --blockchain-demo examples/demo_content.txt
```

Representative output (a real run from this milestone's own
validation — see "Testing" below for how to reproduce it yourself; your
own hashes/tx/block will differ since they depend on your file content
and chain state):

```
[1/5] Canonicalizing content... ✓
[2/5] Computing SHA-256 fingerprint... ✓
    Content hash: 0236b5391f7384f78a74d4c061f7fc5592fc1275a1fd2ba8f6a13d1d68ddaf52

[3/5] Computing source-reference hash... ✓
    Source hash: 3f1bb6cb03ec85177647c9fb93f85c66d5a3bd06d3389cfd754fc1fea7265c3e

[4/5] Registering on local blockchain... ✓
    Transaction: 0x2d00e47c0ab15373225797dd38af22065f7c9aa926ba12f4c801b3f4d0ecd5c6
    Block: 2

[5/5] Retrieving and verifying blockchain record... ✓

    Local hash:    0236b5391f7384f78a74d4c061f7fc5592fc1275a1fd2ba8f6a13d1d68ddaf52
    On-chain hash: 0236b5391f7384f78a74d4c061f7fc5592fc1275a1fd2ba8f6a13d1d68ddaf52

    VERIFICATION: PASS
```

The `Content hash` and `Source hash` above are exactly reproducible if
you run this command against the committed `examples/demo_content.txt`
unchanged. `Transaction` and `Block` will differ on your machine — they
depend on your local chain's state (e.g. block number increments with
however many prior transactions your node instance has seen).

If the local node isn't running, stages [1]-[3] still run and print
real local values; stage [4] fails clearly (exit code 1) rather than
faking a transaction:
```
Error: Could not connect to blockchain RPC at http://127.0.0.1:8545.
Start a local development node first, e.g.: `npx hardhat node` ...
```

### Verification / tamper semantics

```
Local canonical content
        ↓ SHA-256
    content_hash  ──────────────┐
                                 │
Blockchain record                │
        ↓ retrieve                │
    on_chain content_hash        │
                                 ▼
                    local_hash == on_chain_hash ?
                          │              │
                        equal        not equal
                          ↓              ↓
                        PASS           FAIL
```

`VERIFICATION: PASS` means exactly one thing: **the content you just
canonicalized and hashed matches the fingerprint commitment previously
written on-chain, bit for bit.** It is a real equality check on real
SHA-256 digests (`main._verification_status()`), not a heuristic.

**What this does NOT prove**: it does not prove the identity of any
person, and it does not prove that a social-media account or post
"belongs to" anyone. There is no search or identity-matching component
wired into this comparison at all — it is a pure content-integrity
check between a local file and an on-chain commitment. Conflating
"fingerprint verified" with "identity confirmed" would misrepresent
what this milestone actually does.

## Architecture

```
Reference Image
      │
      ▼
Face Processing            (detect → embed)
      │
      ▼
SearchProvider              (authorized local corpus implementation — see Scope)
      │
      ▼
Candidate Matching          (face similarity ⊥ content relevance)
      │
      ▼
Content Extraction
      │
      ▼
Canonicalization            (deterministic, versioned)
      │
      ▼
SHA-256 Fingerprint
      │
      ▼
Blockchain Registration     (BlockchainProvider abstraction)
      │
      ▼
Blockchain Verification     (recompute → compare → VERIFIED / TAMPER_DETECTED)
```

Every stage is defined behind an abstract interface so concrete
implementations can be swapped without touching the orchestration layer.
Face similarity and content relevance are kept as separate, distinct
signals throughout — they are never collapsed into one vague confidence
score.

## Repository structure

```
project/
├── app/
│   ├── face/            # FaceProcessor/FaceComparator interfaces + models,
│   │                     # OpenCVFaceProcessor + CosineFaceComparator (real impl)
│   ├── search/           # SearchProvider interface + bounded local corpus provider
│   ├── matching/          # CandidateMatcher + authorized-corpus implementation
│   ├── content/           # ContentExtractor interface; DeterministicCanonicalizer
│   │                       # + ContentFingerprint/hash_canonical_content/
│   │                       # hash_source_reference (real impl)
│   ├── blockchain/        # BlockchainProvider interface + models;
│   │   ├── local_provider.py  # LocalBlockchainProvider (real impl, web3.py)
│   │   └── contracts/         # FingerprintRegistry.sol + compiled artifact
│   ├── verification/      # Verifier interface + VerificationResult (scaffold —
│   │                       # see Known Limitations; real comparison lives in
│   │                       # main.py's --blockchain-demo for now)
│   ├── pipeline/          # PipelineRunner orchestration (scaffold-stage)
│   └── config/            # Settings loading/validation
├── tests/
│   ├── unit/               # Config, models, interfaces, face processing,
│   │                       # canonicalization, blockchain (unit), CLI
│   ├── integration/        # test_local_blockchain.py — real local-chain
│   │                       # round trip, skip-gated on node availability
│   └── fixtures/
│       └── authorized_faces/  # Optional local-only photos (not committed)
├── scripts/
│   ├── run_demo.py         # Thin CLI wrapper, will grow into the demo entry point
│   └── download_face_models.py  # Optional DNN upgrade-path fetch (see Face processing)
├── examples/                # safe blockchain text + authorized synthetic corpus/assets
├── .env.example
├── requirements.txt
└── main.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment variables

See `.env.example` for the full documented list. Summary:

| Variable | Required | Notes |
|---|---|---|
| `BLOCKCHAIN_MODE` | No (default `local`) | `local` or `testnet`. Only `local` has a concrete provider implemented. |
| `BLOCKCHAIN_RPC_URL` | No (default `http://127.0.0.1:8545`) | RPC endpoint; must be reachable for `--blockchain-demo` to succeed |
| `BLOCKCHAIN_PRIVATE_KEY` | Only if `BLOCKCHAIN_MODE=testnet` | Leave blank for local mode — the node's own unlocked dev account is used instead. Never commit a real key |
| `BLOCKCHAIN_CONTRACT_ADDRESS` | No | Leave blank to auto-deploy a fresh contract on first use; set to reuse one across runs |
| `MATCH_THRESHOLD` | No (default `0.6`) | Float in `[0.0, 1.0]` |
| `LOG_LEVEL` | No (default `INFO`) | Standard Python log levels |

Configuration is validated eagerly at startup — invalid values fail fast
with a clear `ConfigError`, before any pipeline stage runs.

## Running the CLI

```bash
python main.py --help
python main.py --reference examples/reference.jpg
```

With no `--reference` supplied, the CLI prints help and exits `0`. With a
missing file, it prints a clear, actionable error and exits `1`. With a
valid file, it prints the eight pipeline stages and honestly marks all
of them as not yet implemented (face processing being the exception —
see "Face processing" above for the real `--compare` flow).

Two further independent commands, documented in their own sections
above: `--fingerprint <file>` (canonicalize + SHA-256 hash a local text
file) and `--blockchain-demo <file>` (the same, plus register + retrieve
+ verify against a local blockchain).

## Testing

```bash
pytest
```

Current coverage includes: configuration loading/validation, data model
validation (embeddings, candidates, canonical content, verification
results), authorized-corpus initialization, filtering, ranking,
malformed-corpus rejection, deterministic retrieval, and a guard that
the provider does not use a reference embedding handle to select a
candidate. It also covers CLI behavior (help, missing file, corrupted
file, no-face file), and import-speed guards against model loading or
network calls at import time.

**Face processing tests** (`tests/unit/test_face_processor.py`,
`tests/unit/test_face_comparison_logic.py`) split into two groups:

- **Fully exercised now, no external image needed**: LBP embedding
  determinism/shape, cosine similarity math, match-band derivation,
  comparator behavior, empty/corrupted-image handling, zero-face
  detection on random noise, no-network-call guard, no-model-load-at-
  import guard.
- **Fixture-gated, currently skipped**: detecting exactly one face on a
  real photo, and detecting multiple faces on a real photo. These
  require an actual photograph, which was not available in this
  project's environment and was not fabricated or downloaded per this
  project's scope constraints (see
  `tests/fixtures/authorized_faces/README.md`). Drop in your own
  authorized/consented test photos at the documented paths to activate
  them locally. **This is the only category of test in this repository
  that is skipped due to a missing local resource rather than a missing
  external service** (contrast with the blockchain integration tests
  below, which are skipped only when no local node is running).

**Canonicalization/fingerprint tests**
(`tests/unit/test_canonicalization.py`) are all fully exercised now, no
external resources needed: determinism across repeated runs,
source-reference/candidate-id/retrieval-timestamp independence (see
"Milestone 4" corrected semantics above), metadata-key-order
independence, one-character/one-byte tamper sensitivity, Unicode NFC
normalization (including precomposed-vs-decomposed accent equivalence),
newline-style normalization, empty-text/empty-bytes/missing-content
handling, canonicalization-version and algorithm labeling, `generated_at`
non-participation in the hash, 64-character hex hash format validation,
an explicit offline-network guard, `hash_source_reference()` behavior,
and an original-vs-modified tamper scenario.

**Blockchain tests** split across three files/categories:

- **Unit, no live node needed** (`tests/unit/test_local_blockchain_provider.py`):
  `LocalBlockchainProvider` is a concrete, non-abstract `BlockchainProvider`;
  connection errors on an unreachable/empty RPC URL; private-key parsing
  (valid key, malformed key, wrong length) as a pure offline check;
  `_hex_to_bytes32` validation; `BlockchainRecord`'s own format
  validation (tx_hash prefix/length, non-negative block number).
- **Integration, requires a live local EVM node**
  (`tests/integration/test_local_blockchain.py`): detects RPC
  reachability first and **skips cleanly with an explicit reason** if no
  node is running — it never starts one itself. When a node **is**
  running, it exercises a genuinely real round trip: connect, register
  a real fingerprint, receive a real transaction hash, retrieve the real
  record, confirm the on-chain `content_hash`/`algorithm`/
  `canonicalization_version`/`source_reference_hash` all match what was
  registered, confirm duplicate registration deterministically reverts,
  and confirm an unknown transaction hash returns `None`. Nothing here
  is mocked.
- **CLI tests** (`tests/unit/test_cli.py`): `--blockchain-demo` argument
  is accepted and listed in `--help`; a missing input file fails
  cleanly; an unreachable local node fails cleanly with an actionable
  message (self-skips if a node happens to already be running when this
  specific test runs); a full `--blockchain-demo` run against a live
  node succeeds end-to-end and prints content hash / transaction hash /
  on-chain hash / `VERIFICATION: PASS` (self-skips if no node is
  running); and `_verification_status()` — the exact PASS/FAIL equality
  check the CLI uses — is unit-tested directly for both outcomes,
  including the FAIL case, without needing a live mismatch scenario on
  an actual chain.

**Expected behavior summary**:
| Local EVM node running? | Blockchain integration tests | Blockchain CLI tests |
|---|---|---|
| No | Skip (reason: node unreachable) | "Unavailable" test passes; "success" test skips |
| Yes | Run for real, pass | "Success" test passes; "unavailable" test skips |

This project does not measure or claim a specific test-coverage
percentage — the paragraphs above describe what is and isn't exercised,
rather than a coverage number.

## On-chain vs. off-chain summary

See "Milestone 4 — Local Blockchain" above for the full data model.
Briefly: `content_hash`, `algorithm`, `canonicalization_version`,
`source_reference_hash`, and a block timestamp are on-chain. Raw
content, raw images, face embeddings, private keys, and the raw source
reference/URL itself are never sent to the contract.

## SearchProvider scope

**This is the most safety-sensitive part of the architecture, so it's
stated explicitly here.**

`app/search/interface.py` defines a provider-agnostic `SearchProvider`
abstraction. It does **not** define, imply, or include a general-purpose
open-web person-identification, reverse-face-search, or social-media
scraping capability, and no such capability will be implemented in this
project.

`AuthorizedCorpusSearchProvider` in `app/search/authorized_corpus.py`
implements that boundary now. It loads only an operator-selected local
JSON file (`examples/authorized_demo_corpus.json` for the demo); it
never calls the network, scrapes a platform, accepts raw face data, or
reads `SearchQuery.face_embedding_ref`. Search uses only the authorized
corpus's `dataset_id`, `source_reference`, `tag`, and `keywords` hints.
Keyword coverage produces a provider relevance score; results are then
ordered by score and `candidate_id` for deterministic output. That
relevance is content retrieval only, not face similarity and never an
identity claim.

Each v1 corpus record requires `candidate_id`, URL-like
`source_reference`, non-empty `text`, string-only `metadata`, an
`image_path` relative to the corpus file, and optional string `tags`.
The committed corpus contains procedural synthetic 512×512 PNG portraits
and fictional community-post text only—no real people, photographs, or
private information. The PNGs are generated locally from deterministic
shapes, gradients, and noise by
`scripts/generate_synthetic_demo_faces.py`; the generator takes no image
input. Copy the JSON file and its local assets to replace it with a
separately authorized dataset; construct
`AuthorizedCorpusSearchProvider` with the replacement path. A future
provider for an authorized real source can implement the same
`SearchProvider.search(SearchQuery) -> list[SearchCandidate]` contract
while enforcing that source's consent, authentication, retention, and
API rules. It must still keep candidate retrieval separate from
downstream face similarity (`NO_MATCH`, `POSSIBLE_MATCH`, or
`STRONG_MATCH`).

This deliberately bounded demo corpus is **not an open-web search
engine**, reverse-image service, or arbitrary-person identification
tool.

## Candidate matching

`AuthorizedCorpusCandidateMatcher` connects candidates returned by the
bounded provider to the existing local `FaceProcessor` and
`FaceComparator`. The caller supplies an already-produced reference
`FaceEmbedding`, the processor/comparator implementations, and the
provider's `candidate_image_paths`. For every returned candidate, the
matcher reads the local authorized image, requires exactly one detected
face, and compares that face's embedding with the supplied reference.
The result is an existing `MatchResult`, which retains the original
`SearchCandidate`, the independent provider relevance, face similarity
score/band, usability, rejection reason, selection flag, and selection
explanation.

No usable score is manufactured when content cannot be evaluated:
missing/unreadable paths and decoder/processor failures are rejected;
zero faces are rejected; and multiple faces are rejected without picking
one arbitrarily. Rejected candidates follow usable candidates in the
ranked results and preserve a clear reason.

Usable candidates are ranked, deterministically and without a combined
opaque confidence formula, by:

1. Match band: `STRONG_MATCH`, then `POSSIBLE_MATCH`, then `NO_MATCH`.
2. Face similarity score, highest first, within a band.
3. Stable `candidate_id`, ascending, as a tie-breaker.

Selection is deliberately stricter than ranking. A result is selected
only if it is the sole highest-scoring `STRONG_MATCH`. If no candidates
are returned, the ranked list is empty. If all are rejected, none is
selected. `NO_MATCH` and `POSSIBLE_MATCH` candidates remain visible but
unselected. When the top `STRONG_MATCH` score is tied, all tied results
are reported as ambiguous and none is selected. Multiple strong matches
with distinct scores select the unique highest score according to this
rule. A selected candidate is still only a similarity-based candidate
selection; it is never confirmation of a person's real-world identity.

This matching layer does not introduce a network source or reverse-image
lookup. `scripts/run_authorized_match_demo.py` exercises the real local
path: it processes the supplied reference, discovers every record from
the authorized corpus, requires exactly one detected face per candidate,
and prints the actual similarity score and match band. It does not alter
the still-scaffolded `PipelineRunner`; a later integration milestone can
inject the same provider and matcher into that runner.

## Privacy/security considerations

- Face embeddings are designed to be **ephemeral**: kept in memory for
  the duration of a run only. If a future milestone needs to persist an
  embedding (e.g. for caching during development), that will be
  explicitly documented here, including retention duration and how to
  clear it.
- Raw face images and raw embeddings are never sent to the blockchain
  layer — only a content fingerprint (SHA-256 hash), a separately-hashed
  source reference, and small versioning metadata are committed
  on-chain.
- Secrets (`BLOCKCHAIN_PRIVATE_KEY`) are sourced exclusively from
  environment variables / `.env` (gitignored), never hardcoded.
- Similarity results are always reported as continuous scores plus a
  qualitative band (`NO_MATCH` / `POSSIBLE_MATCH` / `STRONG_MATCH`) —
  never presented as proof of real-world identity.
- The search stage is scoped to an authorized target, not open-web
  discovery — see **SearchProvider scope** above.

## Known limitations

- The bundled corpus validates the real decode/detect/compare path with
  locally generated synthetic PNGs. This is a deterministic engineering
  demo, not an accuracy or identity benchmark; Haar+LBP still needs
  evaluation on a separately authorized dataset before any broader use.
- Haar+LBP is a classical, zero-download approach chosen for
  reproducibility under a hard deadline — it is meaningfully less
  accurate than a modern deep face-embedding model. See "Face
  processing" above for the documented DNN upgrade path and why it
  wasn't used as the default.
- Match-band thresholds are a documented heuristic, not a scientifically
  validated cutoff for this embedding — tune `MATCH_THRESHOLD` against
  your own images before a demo.
- **Search is intentionally local and bounded.** The bundled provider
  searches only the committed synthetic corpus. Candidate matching is
  available as a local component, but neither is wired into the
  scaffolded `PipelineRunner` or a concrete `ContentExtractor`. It cannot search the web,
  social-media platforms, or arbitrary people.
- **The local blockchain is development/demo infrastructure, not
  production infrastructure.** `LocalBlockchainProvider` targets a
  local Hardhat-style dev node with unlocked test accounts; it has not
  been exercised against a public testnet or mainnet
  (`TestnetBlockchainProvider` does not exist). Restarting the Python
  process without a configured `BLOCKCHAIN_CONTRACT_ADDRESS` deploys a
  fresh contract each time (see "Milestone 4" above).
- **The blockchain never stores raw content** — only hashes and small
  metadata (see "On-chain vs. off-chain summary" above). This is a
  deliberate privacy/scope decision, not an oversight, but it also means
  the blockchain record alone cannot be used to recover or display the
  original content — only to check whether locally-held content still
  matches what was committed.
- **`--blockchain-demo`'s PASS/FAIL is a content-integrity check only —
  not identity proof.** See "Verification / tamper semantics" above.
  There is currently no code path anywhere in this repository that
  claims a face match, a content match, and a blockchain verification
  together prove a real-world identity; they are three separate,
  independently-reported signals by design (per the architecture's
  "Matching Strategy" principle).
- A dedicated `Verifier` implementation
  (`app/verification/verifier.py`), wired into a shared pipeline, does
  **not** exist yet — the real local-hash-vs-on-chain-hash comparison
  currently lives directly in `main.py`'s `--blockchain-demo` handler
  (`_verification_status()`). Consolidating this into `Verifier` and
  `PipelineRunner` is future work (see Milestones).
- `app/pipeline/runner.py`'s `PipelineRunner` is currently a standalone
  scaffold not wired into any of `main.py`'s real flows
  (face-processing, `--fingerprint`, or `--blockchain-demo`); it will be
  updated to orchestrate all real stages together once `SearchProvider`
  exists (see Milestones).
- Blockchain integration tests (`tests/integration/test_local_blockchain.py`)
  require a reachable local EVM node to actually execute; they skip
  cleanly (not fail) when one isn't running. See "Testing" above for the
  exact skip/run matrix.
- No UI or website is provided, per the task's own requirements — the
  CLI is the only interface.

## Milestones

1. **Scaffold** — repo structure, config, interfaces, CLI, tests. ✅ done
2. **Face processing** — real detection + embedding + comparison on
   supplied images. ✅ done (Haar+LBP; DNN upgrade path documented but
   not run in this environment — see Known limitations)
3. **Canonicalization + hashing** — deterministic, source/candidate-
   independent SHA-256 fingerprinting, plus separate source-reference
   hashing. ✅ done (semantics corrected mid-milestone — see
   "Canonicalization and Fingerprinting" above)
4. **Local blockchain** — `LocalBlockchainProvider`, `FingerprintRegistry.sol`,
   register/retrieve round-trip, CLI `--blockchain-demo` with real
   local-vs-on-chain verification. ✅ done
5. `SearchProvider` (authorized demo corpus) — real, non-hardcoded retrieval/ranking. ✅ done
6. Candidate matching — combine face similarity + content relevance into `MatchResult`. ✅ done
7. Dedicated `Verifier` + `PipelineRunner` integration — consolidate the
   comparison currently inline in `--blockchain-demo` into the
   `Verifier` abstraction, and wire face processing, canonicalization,
   and blockchain stages through one shared pipeline run. Not started.
8. Tamper demo path as part of the integrated pipeline — scripted
   content modification + re-verification through `PipelineRunner`
   (a version of this already exists standalone in the canonicalization
   tests and is implicitly demonstrable via two `--blockchain-demo` runs
   against modified content, but not yet as one integrated flow). Not started.
9. Tests + cleanup — full unit/integration pass, dead-code removal.
10. README finalization + demo rehearsal.

## Demo-day checklist

Covers only what is currently implemented — the blockchain
fingerprinting demo, not the full Task 3 pipeline (no search stage
exists yet):

1. `pip install -r requirements.txt`
2. Start a local EVM node in its own terminal: `npx hardhat node` (see
   "Local blockchain setup" above for one-time setup)
3. Leave `.env` blockchain settings at their defaults (or `cp
   .env.example .env` if you haven't already) — no key needed for local mode
4. Have a demo content file ready (`examples/demo_content.txt` is
   provided; swap in your own text file if you prefer)
5. Run: `python main.py --blockchain-demo examples/demo_content.txt`
6. Point at the printed `Transaction:` line and `Content hash:` line
7. Point at the `[5/5]` retrieval step and the printed on-chain record
8. Point at `VERIFICATION: PASS` — and note aloud what it does and does
   not prove (see "Verification / tamper semantics" above)
