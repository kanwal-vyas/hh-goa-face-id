# Authorized face fixtures (not committed)

This directory is where you can place your own explicitly authorized/
consented test photos to activate the detector-dependent tests in
`tests/unit/test_face_processor.py`:

- `single_face.jpg` — a photo containing exactly one face.
- `two_faces.jpg` — a photo containing two or more faces.

Neither file is committed to this repository (see `.gitignore`), and
none was available in this project's development environment, so the
tests that depend on them are skipped by default with a clear reason.

Do not add photos of unsuspecting private individuals here — use your
own image, a consenting collaborator's image (with permission), or a
synthetic/test identity.
