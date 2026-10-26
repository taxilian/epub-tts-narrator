# Remote chapter rendering

The chapter service owns synthesis, speech verification, retries, chapter
assembly, loudness normalization, and encoding. The client submits several
chapters, watches progress, and downloads completed chapters. Only an
explicitly connected client's jobs execute, and scheduling state belongs only
to the current render invocation: a disconnect drains the in-flight take at a
durable boundary and then removes that run's queue rows, a restart clears
past-run rows, and a completed run is removed once its client acknowledges
verified delivery (see [Connected client sessions](#connected-client-sessions)).

Local and remote rendering share the clip-generation, verification, and recovery
pipeline. When a paragraph uses one saved voice and delivery style, recovery can
render exact sentence slices, verify each slice, then join and verify the whole
paragraph. It retains the original chapter plan and text links, and accepted
audio (including completed recovery slices) remains reusable after interruption.
Mixed-voice or mixed-delivery paragraphs cannot use this sentence fallback.
When such a paragraph still fails verification on a remote job, the service
reports the failed paragraph's hash identity, and the client records the split
locally, rebuilds its plan, and resubmits that chapter with the paragraph
expressed as its separately verified source turns.

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

The [Runpod deployment guide](deploy/runpod/README.md) is authoritative for the
managed worker: bootstrap, update, encode, stop semantics, replacement, and
recovery. The managed commands supply the service URL and application bearer
key automatically; you do not need the manual `[remote_chapters]` configuration
below. Bootstrap and update default to `--source github`; use `--source local`
to deploy committed-but-unpublished changes — uncommitted files and private
configuration are never transferred. Every managed operation stops compute on
completion, error, or interruption.

Operational notes specific to chapter rendering:

- Final M4B publication adds square EPUB cover art and book tags on the client. New analyses remember the EPUB path; use `--epub PATH` for older bundles or moved originals. The required cover is checked before scheduling full-book work. `audiobook-render refresh-metadata BOOK --epub PATH` repairs a finished local output without starting the Pod or regenerating audio.
- Chapter numbers refer to the analysis manifest and can include a cover, title
  page, or other front matter before the first story chapter. Keep the same
  work directory when resuming.
- Bootstrap and update leave the service in maintenance mode: health, voice
  registry, and job status APIs answer, but existing rows are left untouched and
  new chapter submissions are rejected. Maintenance mode does not change
  renderer or accepted-clip cache identities. Encode starts normal queue
  processing; a normal start first removes every persisted job and event row
  that has no live connected owner, so no interrupted or queued book from a
  past run can ever execute on its own after a restart.
- Two separate idle timeouts exist: the deployment's
  `[runpod].idle_timeout_seconds` (default 300) controls managed idle shutdown,
  while the chapter service's `[chapter_server].idle_timeout_seconds` governs
  idle shutdown while serving — see "Idle shutdown and independent downloads"
  below.
- The persistent network volume retains model files, environments, registered
  voices, and accepted checkpoints, so unchanged models are not downloaded
  again. Storage remains billed while compute is stopped.

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

`--chapters-only` skips final book packaging, but downloads M4A, JSON and FLAC
masters before acknowledging delivery. M4B packaging decodes those lossless masters
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
must be included in the memory/throughput budget. Configuring
`verification.reference_model` prepares a second copy of the same pinned
Whisper repository for reference corroboration; managed deployments bind it to
the `[runpod].reference_whisper_dir` directory. Existing JEV decisions and
secondary recognition remain part of the verification pipeline.

Keep `chapter_server.data_dir` on persistent storage. It contains:

- An immutable FLAC voice registry, keyed by audio hash and exact transcript.
- A run-scoped SQLite queue whose rows exist only while their client run is
  connected or draining; restarts and detaches remove them.
- Accepted paragraph checkpoints and compressed completed chapter exports.

Run one service process per data directory. A process lease prevents competing
services from recovering the same queue. Inference stays warm across jobs. The
shared engine pool admits up to the effective
`render_pool.chapter_concurrency` coordinators (2 by default in legacy mode,
4 under the experimental pipeline scheduler, capped at 16) so the next chapter
can generate
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

### Scheduling diagnostics (experimental pipeline opt-in)

The chapter server projects the local `[render_pool]` scheduler controls to the
worker unchanged (`project_pipeline_controls`); automatic values are resolved
on the worker from its own configured slots. Under the experimental
`render_pool.scheduler = "pipeline"`, the automatic `max_inflight` becomes a
global transaction ceiling: `16` on a single-slot local worker and
`min(256, max(64, 16 × slots))` remotely — at eight declared slots that is 128.
The legacy window keeps its slots + 1 per-chapter meaning, so a legacy pool
reporting `9` on eight slots and a pipeline pool reporting `128` are different
quantities with different bases, never a plain 2-vs-16 comparison. Managed
coordinators are additionally capped at 16 (`MAX_PARALLEL_CHAPTERS`), the
assembly pool runs `chapter_server.assembly_workers` (default 2), and the
dispatch semaphore admits `assembly_workers + coordinators` units at once.

`GET /v1/jobs` carries an additive read-only `scheduling` object (advertised as
`scheduling_status: 1` in capabilities; older servers simply omit it). It holds
the queue counts per state, the active scheduler mode and capacity basis
(`pipeline_global_transactions` or `legacy_per_chapter_window`), the logical
scheduler's pending/inflight/owner gauges, the attempt backlog including its
monotonic `peak_pending_count` and `peak_materialized_bytes` high-water marks,
and the per-stage queued/active counts from the closed render-metrics stage
set. Every value is a number, boolean, or one of those fixed labels — never
book text, transcripts, URLs, keys, or job identifiers. Observing grants
nothing: the endpoints never create or renew a session, open admission, start
compute, or grow any registry, and peak gauges survive a drained queue so a
finished burst stays visible. `runpod progress` renders this block as one
`scheduling:` line. Interpretation: a job in `generating` whose synthesis
stage shows `0` active, with `paused: no connected rendering client` absent,
is real idle inference waiting on downstream work; a job reported `paused`
with `owner_connected: false` is a stopped/disconnected render — the service
holds no execution grant and will not resume until a client reconnects.

Registered reference bytes and transcripts are reused. Model-specific reference
encoding reuse depends on the configured inference backend; this registry does
not implement or modify that backend's KV cache or batching internals.

## Connected client sessions

The service executes work only for explicitly connected clients, and every
run owns only its own scheduling state; it never automatically resumes a
queued job on restart, executes for an observer, or lets a past run influence
a new one.

- Capabilities advertise `client_sessions: 1`, `session_ttl_seconds`
  (default 30), and `ephemeral_runs: 1`. A client that does not support
  sessions or run-scoped ephemeral cleanup cannot execute against this
  server, and this client refuses servers without those capabilities.
- `POST /v1/sessions` with `{"id": <sha256>}` explicitly opens one ephemeral
  session. Sessions live in memory only: restarting the service clears every
  session and all ownership.
- `POST /v1/sessions/{id}/heartbeat` renews only an already-live session and
  never recreates an expired or revoked one. `DELETE /v1/sessions/{id}`
  revokes immediately: queued, failed, and completed rows owned by that
  session are removed at once, and in-flight takes drain at their next
  durable boundary before their rows are removed. Rows whose session silently
  expired are pruned the same way by the dispatch loop. Every session
  endpoint uses the same bearer authentication as the rest of the API.
- `POST /v1/jobs` and `POST /v1/jobs/{id}/retry` take the session id in their
  `session` payload field, separate from the existing `submission` retry
  budget. Every execution grant requires a live session. Clients derive the
  idempotency key from the chapter specification and their own session, so a
  fresh invocation never collides with a past server job — even on a
  recreated database with the same deterministic identity — while retries
  within the run stay idempotent. A job owned by another live client returns
  409 instead of being stolen; an explicit reattach of a still-draining job
  keeps the new live owner's grant instead of deleting it.
- `POST /v1/jobs/{id}/ack` retires a completed run after its connected owner
  confirmed that every artifact was downloaded, size- and SHA-256-checked,
  and durably cached. Only the live owner can acknowledge. The queue and
  event rows are removed; artifact files and published storage objects
  remain for the download links the client already holds. A lost
  acknowledgement is safe: the client's validated cache is already
  authoritative, and the rows are pruned when the session detaches.
- Reads never grant or renew anything: `GET /v1/jobs/{id}`, the event
  endpoints, the WebSocket, and artifact downloads neither require a session
  nor extend one, and they never create sessions or change run ownership.
- `GET /v1/jobs` is a read-only queue list (`limit` 1–256, default 100) with
  `id`, `state`, `paused`, `owner_connected`, `chapter_number`,
  `total_chapters`, `source_fingerprint`, `updated`, and `last_event` per job.
  It exposes no specification text, transcripts, errors, or session ids.

While a job generates, the server rechecks its owner's session at every safe
task boundary: before dispatching each chunk, at engine admission, when
reacquiring a pool slot, and before every new take. If the client disconnects,
in-flight GPU work, verification, and atomic cache writes finish and save; no
further GPU starts. The abandoned run's row is then removed — never silently
requeued for later execution — while its accepted speech cache, registered
voices, and artifact files stay intact. A real failure while the client is
still connected keeps the existing failed behavior. Client heartbeats run on
a dedicated thread for the whole invocation (submissions, waits, and
downloads), stop before the client deletes its session, and a heartbeat error
surfaces in the foreground promptly. Session identifiers are hashes and never
appear in logs or reports.

## Resume and integrity

Submission pins the script, voice hashes, render settings, and renderer
fingerprint. Within one invocation, retrying a submission returns the same
job, and retrying a failed job reuses its accepted paragraphs. There is no
cross-run job history: a rerun opens a new session, submits run-scoped jobs,
and follows progress from a fresh event cursor, so a past run's cursor or
receipt can never suppress or replay progress. An interrupted or restarted
service removes its past-run rows at startup; nothing queued ever executes on
its own.

Resumability comes from validated outputs, not run history. Each completed
chapter is downloaded, size- and SHA-256-checked, and recorded in a durable
cache descriptor (`WORK/remote-chapters/<selection-hash>/cache.json`) that
binds the exact source/voice/render selection and every artifact hash. The
descriptor carries no job id, event cursor, queue state, or session — only
integrity identity and private artifact locations. New runs download lossless
masters too, so their later assembly does not depend on a live job or an
unexpired signed link. Reruns reuse verified downloads without contacting
the service; legacy `job.json` receipts from older clients are consumed only
to validate and migrate their already-downloaded artifacts into this
descriptor, never to resume or steer a new run, and unvalidated leftovers are
cleared before a fresh submission. Existing fully completed local chapters
are also reused after their existing integrity checks. Moving an unfinished
local paragraph cache to the server is not currently supported; the client
blocks that migration instead of silently regenerating those accepted takes.
Finish that chapter locally first or explicitly choose a new work directory
for a separate remote run.

Speech retries share one bounded allowance across all chapter jobs of a client
submission and their retries, matching one local render invocation; a new
submission starts a fresh allowance. Paragraphs no longer regenerate whole
takes after their allowance is spent — like local rendering, a persistently
failing paragraph is split into its source turns and resubmitted.

Accepted speech has a separate cache from chapter assembly. Changing only
spacing or bitrate reuses accepted speech on the server.

Downloads are staged, verified against size and SHA256, and published atomically.
The lossless master is additionally checked against the server's decoded PCM
hash before final book assembly. Cache descriptors preserve expiring download
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

Local development defaults to `idle_action = "none"`. To enable the five-minute
Runpod stop action, configure all of the following on the server:

```toml
[chapter_server]
data_dir = "/workspace/chapter-server-data"
persistent_data = true
idle_action = "runpod_stop"
idle_timeout_seconds = 300
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
they expire. New runs copy the lossless masters locally before ACK, including
chapter-only renders. A legacy cache missing a master after its URL expires
requires restoring that file from storage or a fresh render in a new work
directory; existing verified files remain untouched. S3 credentials and endpoint configuration are
explicit file settings; shared AWS credential profiles are not used.

The idle timer excludes live-owned queued jobs, active synthesis/verification,
retries, CPU assembly, publication, incoming uploads and unacknowledged delivery.
Maintenance setup heartbeats keep smoke checks active. A new book reuses the
loaded service; the idle countdown starts only after its work is done.
An open WebSocket and status
polling do not keep the GPU alive. Admission closes atomically before calling
Runpod's actual Pod stop API; a failed stop request reopens admission and reports
the failure. Simply exiting the Python process is not used to stop billing.

Persistent storage continues to cost money after GPU compute stops. This is an
application-level idle monitor: a crashed process cannot execute its shutdown
callback, so an independent provider watchdog is still advisable for unattended
deployments. The manual `serve-chapters` command does not provision resources;
managed `runpod bootstrap` does, and the other lifecycle operations can start
existing paid compute.
