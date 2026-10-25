# OVERVIEW
Offline, deterministic regression suite for the analyzer and renderer.
Tests protect failure, resume, recovery, validation, and boundary behavior.

Keep this file current: when suite conventions, helpers, or the run environment change, update it in the same commit.

# LAYOUT
Flat directory: 264 `test_*.py` files plus two shared helper modules.
`audition_fixtures.py` holds bundle/script helpers; `shared_fixtures.py` holds cross-cutting helpers (Terminal, snapshot, pcm_hash, make_epub, timed_transcript, FakeEntryPoints, PARENTHESIS_TRIM_CASES, renderer_config). Neither starts with `test_`, so discovery ignores them; add new shared helpers there, not inline copies.
`fixtures/` contains five JSON contracts: attribution semantics and speech rewrite policies v1 through v4.
`test_<module>.py` usually maps to `epub_tts_narrator/<module>.py`.
Renderer coverage uses `test_renderer_<module>.py`, with behavior files for cross module flows.
Bare helper imports work because unittest discover puts `tests/` on `sys.path`.
`test_series_bundle_tools.py` additionally inserts `scripts/legacy_series` into `sys.path`.
The real EPUB fixture is `examples/last-lantern/the-last-lantern.epub`.
Performance tests fake only planning, synthesis, and ASR boundaries; the comparison integration uses real snapshots, frozen voice stores, verification, FFmpeg assembly, timing exports, and resume. Synthetic audiobook fixtures without an EPUB explicitly set `render.require_cover=False`; production cover policy stays enabled.

# CONVENTIONS
Use stdlib `unittest` and `unittest.TestCase`; no pytest fixtures, markers, or conftest files.
Name classes `<Subject>Tests(unittest.TestCase)`.
Name methods as sentence length behavior descriptions, including the relevant condition.
Use `subTest` for malformed input matrices and related case tables.
Use `tempfile.TemporaryDirectory` with cleanup for filesystem tests.
For started patches, register cleanup: `self.addCleanup(patch.stopall)` or per-patch `self.addCleanup(patcher.stop)` — both are house style.
Keep tests deterministic, offline, and independent of execution order.
Cover failure and resume paths, including files named `test_*_resume.py`, `test_*_retry.py`, and `test_*_recovery.py`.

# FAKING PATTERNS
| Boundary | Pattern | Canonical example |
| --- | --- | --- |
| LLM | Per file FakeClient; `complete(system, payload)` dispatches on `payload["mode"]`, scripts request IDs, deep copies `self.calls`. | `test_pipeline.py` |
| HTTP | Patch `<module>.build_opener`; return a Mock opener and `FakeResponse(io.BytesIO)` with `status`; inspect real Request fields. | `test_provider.py` |
| TTS and GPU | Replace only third party boundaries: Mock model, soundfile writer, `SimpleNamespace` segments, and `ModuleType("mlx.core")` stubs. | `test_renderer_mlx_audio_engine.py` |
| Credentials | `patch.dict(os.environ, {"...": "wrong"})`; prove ambient variables are ignored in favor of TOML configuration. | Provider tests |
FakeClient variants use constructor flags such as `invalid_text=` and `mutation=`, or local subclasses such as `FailingClient`.
Assert that truncated, NaN, inf, and int16 audio never reaches disk.
Keep live model behavior out of tests; use `scripts/validate_identity.py` for manual regression.

# GATING
| Dependency or platform | Gate |
| --- | --- |
| ffmpeg and ffprobe | `unittest.skipUnless(..., "ffmpeg/ffprobe required")` |
| Optional server dependencies | `skipUnless(importlib.util.find_spec("msgpack") ...)` or `HAS_SERVER_DEPS` |
| POSIX terminal behavior | `skipUnless(os.name == "posix", "POSIX required")`; cover pty and termios here |
Do not use pytest markers. State the missing dependency or platform in every skip reason.
Diagnostics redaction is covered by the `DO NOT ECHO` fixtures in `test_cast_evidence_diagnostics.py` and `test_attribution_review_diagnostics.py`.

# HOW TO ADD A TEST
1. Choose the mapped module or the narrow behavior file; name it `test_<module>.py`.
2. Identify the user visible outcome, including an error, malformed input, retry, or resume case.
3. Keep owned logic real; fake only network, GPU, TTS, credentials, and other external boundaries.
4. Add deterministic setup, cleanup, and a behavior named `TestCase` method.
5. Run the focused test, then the relevant subset or full suite before handoff.

# COMMANDS
The suite needs the optional extras installed: verification tests use `metaphone`, plus `packaging` and `soundfile`; server tests use the chapter-server stack. On a bare interpreter ~1000 tests error instead of skipping. Setup:
```sh
python3.11 -m venv .venv
.venv/bin/pip install -e ".[chapter-server,verification,remote]" packaging soundfile httpx
```
Full suite: `.venv/bin/python -m unittest discover -s tests -q`
Single file: `.venv/bin/python -m unittest discover -s tests -p "test_epub.py" -q`
Renderer subset: `.venv/bin/python -m unittest discover -s tests -p "test_renderer_*.py" -q`

Always run through discover with `-s tests`: bare helper imports (`from test_contracts import ...`) only resolve that way; `python -m unittest tests.test_epub` fails on files with cross-imports.
