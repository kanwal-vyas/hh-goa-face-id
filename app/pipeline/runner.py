"""
Pipeline orchestration.

Thin coordinator over the eight pipeline stages defined in the
architecture:

    [1/8] Face processing
    [2/8] Search
    [3/8] Candidate matching
    [4/8] Content extraction
    [5/8] Canonicalization
    [6/8] SHA-256 fingerprint
    [7/8] Blockchain registration
    [8/8] Verification

At this milestone, none of the concrete stage implementations exist yet
(only their interfaces do — see app/face, app/search, app/matching,
app/content, app/blockchain, app/verification). This runner therefore
does NOT execute a working pipeline. Its job in this milestone is to:

    1. Accept a reference image path.
    2. Perform the one real check that IS meaningful yet (does the file
       exist / is it readable).
    3. Clearly report, stage by stage, that implementation is pending —
       never print a fabricated success status.

This keeps `main.py` honest and gives later milestones a stable place to
wire in real stage implementations one at a time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

STAGE_NAMES: tuple[str, ...] = (
    "Face processing",
    "Search",
    "Candidate matching",
    "Content extraction",
    "Canonicalization",
    "SHA-256 fingerprint",
    "Blockchain registration",
    "Verification",
)


class ReferenceImageNotFoundError(Exception):
    """Raised when the supplied reference image path does not exist."""


@dataclass(frozen=True)
class StageStatus:
    """Status of a single pipeline stage for this run."""

    name: str
    index: int
    total: int
    implemented: bool
    detail: str = ""

    @property
    def label(self) -> str:
        return f"[{self.index}/{self.total}] {self.name}"


@dataclass(frozen=True)
class PipelineRunReport:
    """Summary of a pipeline run under the current milestone.

    This is NOT a VerificationResult or a claim of pipeline success —
    it is a scaffold-stage report, explicitly named to avoid confusion
    with the eventual real result types.
    """

    reference_image: Path
    stages: tuple[StageStatus, ...] = field(default_factory=tuple)

    @property
    def any_implemented(self) -> bool:
        return any(stage.implemented for stage in self.stages)


class PipelineRunner:
    """Orchestrates the pipeline stages.

    Concrete stage implementations are injected in later milestones
    (e.g. via constructor parameters for FaceProcessor, SearchProvider,
    etc). At this milestone, no stage implementations are wired in, so
    the runner only validates its input and reports scaffold status.
    """

    def __init__(self) -> None:
        # Later milestones will accept and store concrete stage
        # implementations here (FaceProcessor, SearchProvider,
        # CandidateMatcher, ContentExtractor, Canonicalizer,
        # BlockchainProvider, Verifier). None are available yet.
        pass

    def run(self, reference_image: str | Path) -> PipelineRunReport:
        ref_path = Path(reference_image)

        logger.info("Starting pipeline scaffold run for %s", ref_path)

        if not ref_path.exists():
            raise ReferenceImageNotFoundError(
                f"Reference image not found: {ref_path}"
            )
        if not ref_path.is_file():
            raise ReferenceImageNotFoundError(
                f"Reference image path is not a file: {ref_path}"
            )

        stages: list[StageStatus] = []
        total = len(STAGE_NAMES)
        for i, name in enumerate(STAGE_NAMES, start=1):
            status = StageStatus(
                name=name,
                index=i,
                total=total,
                implemented=False,
                detail="not implemented in this milestone",
            )
            logger.info("%s — %s", status.label, status.detail)
            stages.append(status)

        report = PipelineRunReport(reference_image=ref_path, stages=tuple(stages))
        logger.info(
            "Pipeline scaffold run complete for %s. No stages are "
            "implemented yet in this milestone; this is expected.",
            ref_path,
        )
        return report
