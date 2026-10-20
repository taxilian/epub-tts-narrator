# Legacy series utilities

These offline helpers preserve validation, reporting, packaging, and the v1-to-v2
chapter-bundle migration used by the original seven-book Soulship run. They are
kept as source code because the regression tests exercise their exact-text,
archive-integrity, and rollback guarantees. No book text, recordings, model
weights, credentials, or run results are included here.

These scripts expect the historical `book-01` through `book-07` layout and its
shared registry and audit files. They retain their original Soulship report
identifiers and filenames for compatibility. They are not required for the
current `analyze --continue-from` directory workflow.

Every executable helper requires `--root PATH` pointing to an existing legacy
run directory. For example, from the repository root:

```sh
python scripts/legacy_series/audit_continuity.py --root /path/to/legacy-run
```

`bundle_validation.py` supplies shared checks and has no CLI. The migration
preserves the original bundles and refuses to run twice; packaging writes a new
archive and refuses to replace an existing one. The original local run scripts
remain in their ignored run directory.
