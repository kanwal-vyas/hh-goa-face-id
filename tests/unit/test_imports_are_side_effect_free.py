"""
Guards against a common scaffold regression: importing a module
triggering a model download, network call, or other slow/side-effecting
work at import time.

We can't fully sandbox the network in a unit test without extra
infrastructure, so this test instead asserts on a cheap, meaningful
proxy: importing every app module completes near-instantly. A model
download or network call would make this test obviously slow/flaky,
surfacing the regression.
"""

import importlib
import time

APP_MODULES = [
    "app.config.settings",
    "app.search.interface",
    "app.blockchain.interface",
    "app.face.processor",
    "app.face.compare",
    "app.face.opencv_processor",
    "app.content.extractor",
    "app.content.canonicalize",
    "app.content.fingerprint",
    "app.matching.candidate_matcher",
    "app.verification.verifier",
    "app.pipeline.runner",
]


def test_all_app_modules_import_quickly():
    start = time.monotonic()
    for module_name in APP_MODULES:
        importlib.import_module(module_name)
    elapsed = time.monotonic() - start
    # Generous ceiling for a scaffold with no heavy dependencies loaded;
    # a real model download would take far longer than this.
    assert elapsed < 5.0, (
        f"Importing app modules took {elapsed:.2f}s — investigate for "
        "unexpected network calls or model downloads at import time."
    )


def test_opencv_processor_module_does_not_load_cascade_at_import_time(monkeypatch):
    """Model initialization (loading the Haar cascade) must happen only
    when OpenCVFaceProcessor() is explicitly instantiated, never as a
    side effect of importing the module."""
    import sys

    module_name = "app.face.opencv_processor"
    sys.modules.pop(module_name, None)

    calls = []
    import cv2

    original_init = cv2.CascadeClassifier

    def _tracking_init(*args, **kwargs):
        calls.append((args, kwargs))
        return original_init(*args, **kwargs)

    monkeypatch.setattr(cv2, "CascadeClassifier", _tracking_init)
    importlib.import_module(module_name)

    assert calls == [], (
        "cv2.CascadeClassifier was constructed at import time — model "
        "loading must be deferred to OpenCVFaceProcessor.__init__."
    )
