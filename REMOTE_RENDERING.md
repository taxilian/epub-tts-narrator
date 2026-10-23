# Remote chapter rendering

The chapter service owns synthesis, speech verification, retries, chapter
assembly, loudness normalization, and encoding. The client submits several
chapters, watches progress, and downloads completed chapters. Closing the client
does not cancel submitted jobs.

Local and remote rendering share the clip-generation, verification, and recovery
pipeline. When a paragraph uses one saved voice and delivery style, recovery can
render exact sentence slices, verify each slice, then join and verify the whole
paragraph. It retains the original chapter plan and text links, and accepted
audio (including completed recovery slices) remains reusable after interruption.
Mixed-voice or mixed-delivery paragraphs cannot use this sentence fallback;
source-boundary replanning currently requires the local book-rendering wrapper.

**Only compressed audio crosses the chapter-service API.** Voice references are
registered as FLAC once, finished chapters are M4A/AAC, and lossless chapter
masters are FLAC. JSON sidecars preserve source links and timing. No WAV download
or upload route exists.

Local integration tests use real FFmpeg and test-only speech/recognition
fixtures. A short deployment test exercises the live synthesis, verification,
and storage configuration; it does not establish full-book throughput or a
comparison between GPU models. Installing this package alone does not allocate
compute; the managed Runpod bootstrap command creates paid resources.

## Managed Runpod client

Use the [Runpod deployment setup](deploy/runpod/README.md) with your normal
global configuration and a private `runpod.local.toml` override. The managed
commands supply the service URL and application bearer key automatically;
you do not need the manual `[remote_chapters]` configuration below.

```sh
# Initial deployment: install committed code, dependencies and models, test, stop.
.venv-renderer/bin/audiobook-render runpod bootstrap --source local

# Deploy subsequent committed changes, update dependencies/configuration, test, stop.
.venv-renderer/bin/audiobook-render runpod update --source local

# Render a selected manifest chapter into a separate remote cache, then stop.
.venv-renderer/bin/audiobook-render runpod encode books/book-01 --chapters 3 \
  --chapters-only --work-dir books/book-01/render-work-remote

# Render missing chapters and package the complete book with its EPUB cover.
.venv-renderer/bin/audiobook-render runpod encode books/book-01 \
  --work-dir books/book-01/render-work-remote -o audio/book-01 \
  --epub books/book-01.epub
```

Bootstrap and Update default to `--source github`, using the configured
repository and ref. Use `--source local` for committed changes that have not
been published there: it transfers committed HEAD, excluding uncommitted files
and private configuration. Encode uses the installed revision; run Update after
committing a code change. Chapter numbers refer to the analysis manifest and
can include a cover, title page, or other front matter before the first story
chapter. Keep the same work directory when resuming.

Bootstrap and Update start the chapter service in maintenance mode. Health,
voice registry, and job status APIs remain available, but existing queued or
interrupted jobs remain untouched and new chapter submissions/retries are
rejected. Their smoke test exercises the engine directly. Encode starts normal
queue processing, including recovery of previously interrupted chapters.
Maintenance mode does not change renderer or accepted-clip cache identities.

Each managed operation stops compute on completion, error, or interruption.
`--stop-delay 300` adds a fixed five-minute grace period with the client still
attached; Ctrl-C ends it immediately. The deployment lock remains held during
that delay, so it is not an idle window for starting a second CLI operation.
The server's separate `[runpod].idle_timeout_seconds` setting defaults to 600
seconds and is an additional fallback when no work is pending.

Stopping releases the GPU. Restarting depends on capacity on the Pod's machine,
and every new inference process must load the model into GPU memory again.
If Runpod confirms that a managed Pod's host has no free GPU, the client can
create one replacement with the same GPU, datacenter, and existing network
volume, within the configured price cap. Explicit `runpod.pod_id` deployments
remain fixed. An uncertain creation is reconciled before another operation;
it is never blindly repeated. See the deployment guide for recovery details.
The persistent network volume retains model files, environments, registered
voices, and accepted checkpoints, so unchanged models are not downloaded again.
Storage remains billed while compute is stopped.

## Manual client setup

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
.venv-renderer/bin/audiobook-render render books/book-01 --chapters 6 --chapters-only
.venv-renderer/bin/audiobook-render render books/book-01 --chapters 6-8 --chapters-only
.venv-renderer/bin/audiobook-render assemble books/book-01
```

`assemble` requires all selected chapters; use `--chapters 6-8` to package just
that subset. Omitting `--chapters-only` renders the selected chapters and then
packages an M4B. The client needs FFmpeg/ffprobe but does not load a local TTS or
ASR model for remote rendering. Create and audition missing voices with `cast`
before submitting remote jobs.

The CLI prints the exact path for each completed M4A. Downloaded files live in
`WORK/remote-chapters/<selection-hash>/chapter.m4a`, where `WORK` is `--work-dir`
or its default `BOOK/render-work`. With `--chapters-only`, `-o` does not receive
chapter copies. When packaging an M4B, `-o` (default `BOOK/audiobook`) receives
`book.m4b`, `chapters/chapter-NNNN.m4a`, and book-wide sync JSON. Supply `--epub`
with the analyzed EPUB to include its cover when no EPUB path is configured.
Original voice samples and inherited book/series selections are untouched.

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
The native adapter folds generated seeds into the server's supported 31-bit
range so retries do not collapse to the same clamped seed. Take metadata retains
the requested and submitted seeds; previously accepted audio remains reusable.

The stock SGLang-Omni profile uses `protocol = "fish_sglang_drama3"`, a
`/v1/audio/speech` URL, `multi_speaker = false`, and a bounded worker concurrency
(start with four). Several requests share one loaded model. A request may retain
an entire paragraph using one saved voice, including changes of delivery, but
cannot mix different voices. References remain unchanged FLAC bytes; only
loopback synthesis responses use WAV. Stock SGLang does not support per-request
seeds, which the adapter reports explicitly in each take's metadata.

This profile requires `engine.top_k = 30` for residual codebooks and uses the
upstream stock uncapped main-token setting. Upstream warns that capping the main
token at 30 can make output fall silent. Worker identity and take metadata record
the requested and effective sampling profiles. Accepted speech remains reusable
across native and SGLang workers for the same configured model contract, text,
and voice; this retains already verified audio, not a promise of identical
sampling or deterministic reproduction. Changed model contracts, text, or
voices continue to invalidate the relevant cached takes.

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

`render_pool.max_inflight` bounds complete chunk operations, including waiting
for verification. The default `0` keeps the existing per-chapter window of total
worker concurrency plus one. An explicit value from 1 to 256 must be at least
the total worker concurrency and is shared across both chapters. For example,
eight synthesis slots with `max_inflight = 32` can keep generating while earlier
takes await recognition or text review. Pending audio is saved on disk; the
window bounds admitted chunk operations. Additional coordinator threads can
wait for admission without loading their audio. Increasing the window absorbs
verification delays but cannot make a persistently
slower verifier keep up. Scheduling changes preserve accepted audio and cache
identities. Cancellation stops waiting admissions while active takes finish.

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
deployments. The manual `serve-chapters` command does not provision resources;
managed `runpod bootstrap` does, and the other lifecycle operations can start
existing paid compute.
