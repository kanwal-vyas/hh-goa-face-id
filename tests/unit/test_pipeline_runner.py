import pytest

from app.pipeline.runner import (
    STAGE_NAMES,
    PipelineRunner,
    ReferenceImageNotFoundError,
)


def test_run_raises_on_missing_reference(tmp_path):
    runner = PipelineRunner()
    missing = tmp_path / "nope.jpg"
    with pytest.raises(ReferenceImageNotFoundError):
        runner.run(missing)


def test_run_raises_on_directory_reference(tmp_path):
    runner = PipelineRunner()
    with pytest.raises(ReferenceImageNotFoundError):
        runner.run(tmp_path)  # a directory, not a file


def test_run_reports_all_stages_as_pending(tmp_path):
    fake_image = tmp_path / "ref.jpg"
    fake_image.write_bytes(b"not-a-real-image-but-that-s-fine-here")

    runner = PipelineRunner()
    report = runner.run(fake_image)

    assert len(report.stages) == len(STAGE_NAMES)
    assert report.any_implemented is False
    for stage in report.stages:
        assert stage.implemented is False
        assert stage.detail  # non-empty explanation
