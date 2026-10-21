# Remote chapter rendering

The chapter service owns synthesis, speech verification, retries, chapter
assembly, loudness normalization, and encoding. The client submits several
chapters, watches progress, and downloads completed chapters. Closing the client
does not cancel submitted jobs.

**Only compressed audio crosses the chapter-service API.** Voice references are
registered as FLAC once, finished chapters are M4A/AAC, and lossless chapter
masters are FLAC. JSON sidecars preserve source links and timing. No WAV download
or upload route exists.

This implementation has local integration tests with real FFmpeg and test-only
speech/recognition fixtures. It has **not** been benchmarked or auditioned on
Runpod/NVIDIA hardware. Deployment and GPU billing are separate from installing
this code.

## Client setup

Install the optional transport dependency into the existing environment:

```sh
uv pip install --python .venv-renderer/bin/python -e '.[remote]'
```

Add this to the existing global `epub-tts-narrator.toml` on the client:

```toml
[remote_chapters]
url = "https://YOUR-CHAPTER-SERVICE"
api_key = "YOUR-SERVICE-KEY"
timeout_seconds = 60
poll_seconds = 2
websocket = true
```

The service key belongs to this application, not to Runpod. Keep real credentials
in the ignored local TOML file. Configuration comes from files, not environment
variables. There is no book-specific or series-specific server configuration.

Use the existing render commands:

```sh
.venv-renderer/bin/audiobook-render render dist-runs/soulship-fresh/book-01 --chapters 6 --chapters-only
.venv-renderer/bin/audiobook-render render dist-runs/soulship-fresh/book-01 --chapters 6-8 --chapters-only
.venv-renderer/bin/audiobook-render assemble dist-runs/soulship-fresh/book-01
```

`assemble` requires all selected chapters; use `--chapters 6-8` to package just
that subset. Omitting `--chapters-only` renders the selected chapters and then
packages an M4B. The client needs FFmpeg/ffprobe but does not load a local TTS or
ASR model for remote rendering. Create and audition missing voices with `cast`
before submitting remote jobs.

The CLI prints the exact path for each completed M4A. Downloaded files live in
`BOOK/render-work/remote-chapters/<selection-hash>/`. The normal audiobook output
directory still holds the final M4B, chapter files, cover, and book-wide sync
JSON. Original voice samples and inherited book/series selections are untouched.

`--chapters-only` downloads M4A and JSON. FLAC masters remain on the server until
final book assembly needs them. M4B packaging decodes those lossless masters
locally into temporary PCM, then encodes the book once; it does not transcode the
lossy M4A chapters. All network transfers remain compressed. Temporary decoded
files are removed when packaging finishes or fails.

## Server setup

For Runpod, use the [project-owned worker deployment](deploy/runpod/README.md).
It prepares pinned models and supervises the chapter service plus one private,
unmodified Fish CUDA process. Ceph can hold the compressed completed exports.

Install this project and FFmpeg/ffprobe on the worker. Python 3.11 or newer is
required. The HTTP service and Linux verification dependencies are optional:

```sh
python -m pip install -e '.[chapter-server,cuda-verification]'
audiobook-render serve-chapters --config epub-tts-narrator.toml
```

See [chapter-server.example.toml](chapter-server.example.toml) for server settings.
The service can use a local MLX engine on Apple Silicon or configured Fish HTTP
workers. On a Linux GPU host, run a CUDA-capable Drama 3 backend separately and
point a `render_pool` worker at its compatible `/v1/tts` endpoint. The chapter
service itself does not install, modify, or launch another inference library;
the optional Runpod launcher handles that process lifecycle. The
existing Fish adapter requires the inline-reference contract documented in
[RENDERER.md](RENDERER.md); compatibility with a particular CUDA backend still
needs a real deployment test. Keep that backend on the same host when avoiding
uncompressed audio between machines: its internal synthesis endpoint uses WAV,
while this chapter service exposes compressed artifacts to the client.

For the pinned official native Fish server, set the worker's
`protocol = "fish_native_drama3"` and `concurrency = 1`. This enables reference
memory caching, preserves our paragraph boundaries, and puts the seed in the
native request body. The default `fish_inline` profile remains available for
existing compatible workers. The native profile requires `engine.top_k = 30`.

For Linux recognition, `verification.type = "faster_whisper"` uses a local
CTranslate2 Whisper model directory containing `config.json`, `model.bin`, and
`tokenizer.json`. It never downloads a model implicitly. CPU recognition is the
default, leaving the GPU for synthesis; CUDA recognition is configurable and
must be included in the memory/throughput budget. Existing JEV decisions and
secondary recognition remain part of the verification pipeline.

Keep `chapter_server.data_dir` on persistent storage. It contains:

- An immutable FLAC voice registry, keyed by audio hash and exact transcript.
- A durable SQLite job queue and replayable progress events.
- Accepted paragraph checkpoints and compressed completed chapter exports.

Run one service process per data directory. A process lease prevents competing
services from recovering the same queue. Inference stays warm across jobs. A
shared engine pool admits up to two chapters so the next chapter can generate
while the previous chapter finishes CPU recognition or remote JEV verification.
The pool still enforces its configured GPU concurrency and loads each local
model once. Local MLX recognition shares the serialized local inference runtime.
Plugins without a concurrency contract process one chapter at a time. CPU
assembly is separate and bounded, so synthesis also continues while an earlier
chapter is being assembled, compressed, or uploaded.

Registered reference bytes and transcripts are reused. Model-specific reference
encoding reuse depends on the configured inference backend; this registry does
not implement or modify that backend's KV cache or batching internals.

## Resume and integrity

Submission pins the script, voice hashes, render settings, and renderer
fingerprint. Retrying a submission returns the same durable job. Reconnecting
replays numbered events, with HTTP polling available if WebSockets fail. Retrying
a failed job reuses its accepted paragraphs. An interrupted service recovers
pending jobs when restarted against the same data directory and compatible
renderer configuration.

Accepted speech has a separate cache from chapter assembly. Changing only
spacing or bitrate reuses accepted speech on the server. Already downloaded,
unchanged chapters can be reused without contacting the server. Existing fully
completed local chapters are also reused after their existing integrity checks.
Moving an unfinished local paragraph cache to the server is not currently
supported; the client blocks that migration instead of silently regenerating
those accepted takes. Finish that chapter locally first or explicitly choose a
new work directory for a separate remote run.

Downloads are staged, verified against size and SHA256, and published atomically.
The lossless master is additionally checked against the server's decoded PCM
hash before final book assembly. Saved job records preserve expiring download
links in private files; credentials are not part of job specifications or hashes.

Assembled chapter WAVs are temporary. Accepted paragraph WAV checkpoints remain
private on the server for resumability; the service does not delete or compress
those while another chapter may be using them. They never cross the public
chapter API. Voice assets are never automatically deleted.

## Output and synchronization

Each completed nonempty chapter provides:

| Artifact | Contents |
| --- | --- |
| `chapter.m4a` | Assembled, normalized AAC chapter audio |
| `chapter.flac` | Lossless master for final book packaging |
| `chapter.json` | Source/voice identity, audio hashes, sample frames, composition metadata |
| `sync.json` | Source ranges and chapter timing after trimming and inserted pauses |
| `verification.json` | Successful acceptance reports and their original provenance |
| `manifest.json` | Versioned artifact inventory and integrity hashes |

Empty chapters return metadata explicitly identifying the empty result.
Synchronization currently has **paragraph/chunk precision**, not word precision.
Reliable word alignment remains a separate feature. Codec delay/padding-aware
playback is needed for AAC timing; JSON frame ranges use the canonical PCM
timeline and exclusive range ends.

## Idle shutdown and independent downloads

Local development defaults to `idle_action = "none"`. To enable the ten-minute
Runpod stop action, configure all of the following on the server:

```toml
[chapter_server]
data_dir = "/workspace/chapter-server-data"
persistent_data = true
idle_action = "runpod_stop"
idle_timeout_seconds = 600
runpod_pod_id = "YOUR-POD-ID"
runpod_api_key = "YOUR-RUNPOD-KEY"

[chapter_storage]
bucket = "YOUR-BUCKET"
endpoint_url = "https://YOUR-S3-COMPATIBLE-ENDPOINT"
region = "YOUR-REGION"
access_key_id = "YOUR-STORAGE-KEY"
secret_access_key = "YOUR-STORAGE-SECRET"
prefix = "epub-tts-narrator"
download_expiry_seconds = 3600
```

Merge these values into the existing server tables; TOML does not allow
duplicate tables. `persistent_data = true` asserts that the operator actually
mounted storage that survives a Pod stop. It does not provision that storage.
Use a network volume if results/checkpoints must also survive Pod termination.

Completed artifacts are uploaded to independent object storage before the job
becomes complete. Signed download URLs keep working after the GPU stops, until
they expire. To renew expired URLs, restart the service against the persistent
store and rerun the command. S3 credentials and endpoint configuration are
explicit file settings; shared AWS credential profiles are not used.

The idle timer excludes queued jobs, active synthesis/verification, retries,
CPU assembly, publication, and incoming uploads. An open WebSocket and status
polling do not keep the GPU alive. Admission closes atomically before calling
Runpod's actual Pod stop API; a failed stop request reopens admission and reports
the failure. Simply exiting the Python process is not used to stop billing.

Persistent storage continues to cost money after GPU compute stops. This is an
application-level idle monitor: a crashed process cannot execute its shutdown
callback, so an independent provider watchdog is still advisable for unattended
deployments. No paid resources are provisioned by these commands.
