# Run our chapter worker on Runpod

The lifecycle CLI installs **this project's code** for voice registration, chapter
jobs, verification, assembly, and Ceph publication. CUDA inference runs on
loopback as our own `audiobook-render serve` process with
`engine.type = "fish_speech_cuda"`, inside the pinned, unmodified Fish Speech
environment. It calls Fish's library directly because chapter continuity needs
prior turns with codec tokens and returned codes, which Fish's stock
`api_server.py` does not offer. Each Drama 3 copy serves one request at a time;
`runpod.inference_replicas` can load several copies on one large GPU (below), and
several chapter coordinators overlap verification and CPU assembly with synthesis. Built-in JEV and
pronunciation reviews use a separate bounded queue so network waits do not
occupy the recognition/model thread.

## Bootstrap, update, and encode

Keep your normal global configuration. Put deployment differences in
`runpod.local.toml` beside it; the CLI merges them when preparing the private
server configuration. Your Mac's renderer settings remain unchanged. The remote
file includes rendering, verification, JEV and Ceph settings, plus analysis-provider
settings when needed for pronunciation recovery. It excludes voice-design keys
and local book/voice paths. The application bearer key is created once and kept
in private deployment state. It differs from the Runpod account API key.

Store your Runpod API key under `[chapter_server].runpod_api_key` and your Ceph
RGW endpoint, bucket, region and credentials under `[chapter_storage]` in the
normal local file. Configure the cloud preference, GPU, region, hourly price
limit and Pod disk size in the override's `[runpod]` table. Use a private HTTPS Ceph bucket reachable from
both the Pod and your Mac.

Copy [runpod.example.toml](runpod.example.toml) to `runpod.local.toml` beside your
normal configuration. Its image is pinned; GPU availability and prices are
checked when creating compute. `runpod.data_center_id` is an **optional location
hint**, not a requirement: empty, stale, or non-US values are ignored, never
errors. Every managed allocation searches all US datacenters that the chosen
cloud's live catalog reports for an eligible GPU.

### Storage: a Pod disk, no network volumes

Each worker keeps `/workspace` on its own host-local **Pod disk** (Runpod's
"volume disk", `volume_size_gb`, 150 GB in the example). It survives stop and
restart on the same host, so a warm worker keeps its code, environments,
models, voices and job checkpoints between books. It is discarded with the Pod,
and it cannot follow a Pod to another host. Network volumes are not used: they
work only on Secure Cloud, and they tie a worker to one datacenter.

A new Pod — the first one, a replacement for a Pod deleted in the console, or
a capacity replacement — therefore starts with an **empty disk** and is fully
reprovisioned: this checkout is deployed, dependencies are installed, models
are downloaded, and the smoke check runs before any book is submitted. Long-term
results live in the Ceph bucket and in the client's local work directory, not on
the Pod disk. The disk is billed while the Pod exists, including while stopped;
the hourly ceiling covers compute only.

`runpod.volume_id` is no longer accepted. Network volumes created by older
versions are never mounted by new workers, moved, or deleted: `runpod status`
lists them under `network_volumes` because they keep billing until you delete
them in the Runpod console. A worker created before this change keeps its
network volume for as long as it remains healthy.

### Cloud selection

`runpod.cloud_types` lists Runpod clouds in preference order. The default
`["secure"]` uses only Secure Cloud (Runpod-owned datacenters). Add
`"community"` (community-hosted hardware, usually cheaper and often more
available) to allow it:

| Setting | Meaning |
| --- | --- |
| `["secure"]` | Secure Cloud only (default) |
| `["community", "secure"]` | Prefer Community; Secure is the fallback |
| `["secure", "community"]` | Prefer Secure; Community is the fallback |
| `["community"]` | Community Cloud only |

Each cloud is quoted from its own live catalog (`cloud=COMMUNITY` or
`SECURE`, priced from that cloud's `price.community` or `price.secure`), and the
first cloud with an eligible GPU under the policy below wins; GPU class and
price are compared only within a cloud. The chosen cloud is saved with the GPU
and region before the only creation request. If Runpod definitely refuses the
request (HTTP 400, its no-capacity answer, or 403, no access to that pool), the
client makes **one** request on the next cloud in the list in the same run.
Ambiguous outcomes are reconciled by name exactly as before and never fall
back; other errors, such as 402 insufficient balance, stop.

Community hosts are not guaranteed a public IP, and Runpod's proxy SSH carries
only an interactive shell. If a Community worker never publishes a direct
22/tcp port before `startup_timeout_seconds`, the client confirms it stopped
and creates one replacement, on the next cloud in the list when there is one.
That counts as the operation's single replacement. A Community worker's public
IP and SSH port may change after a restart; the client reads them on each start.

A healthy existing worker is never replaced merely because the preference
changed; the new order applies to the next Pod that is created.

Provider 5xx errors include bounded, credential-redacted JSON error details,
so a placement rejection is visible instead of a blind retry message.

### GPU selection

By default (`prefer_h100 = true`) a new Pod prefers any available NVIDIA H100
variant (80GB HBM3/SXM, PCIe, or NVL) whose catalog rate is **strictly under
$2.60/hour** and within `max_hourly_price`, choosing the cheapest eligible one
across all eligible US datacenters of the cloud being quoted. If no such H100 exists — too expensive,
out of stock, or absent from the US catalog — the Pod uses the configured
`gpu_type_id` fallback (the RTX PRO 6000 in the example), which must be
available at or under `max_hourly_price` in some eligible US datacenter, or
that cloud is skipped (and nothing is created when no cloud qualifies). Set
`prefer_h100 = false` to pin the configured GPU exactly. GPU class, price, and
stock decide first; your configured `data_center_id` hint only breaks ties.
After creation the provider's actual rate is checked against the same rules
and the worker is stopped if it exceeds them. Catalog and hourly prices cover
compute only; the Pod disk is billed separately. The chosen cloud, GPU and
region are saved before the Pod is created, so a retried or restarted
deployment keeps the same hardware, and a healthy existing Pod is never
replaced just to upgrade its GPU, move its region, or change its cloud.

`[runpod].service_port` selects the chapter service's listen port and Runpod
proxy port; it defaults to **8082**. The pinned base image reserves 8081 for
nginx. Choose an unused port if using another image; ports 22 and 8080 are
reserved for SSH and native Fish inference. This deployment setting overrides
`[chapter_server].port` in the generated worker configuration. The CLI applies
changed port exposure while the Pod is stopped, preserving its Pod disk.
If the worker exits during startup, the CLI reports its logs and stops compute
without waiting for the full readiness timeout.

```sh
# Create the deployment, install code/dependencies/models, test, then stop.
.venv-renderer/bin/audiobook-render runpod bootstrap

# Start it, update code/dependencies/configuration, test, then stop.
.venv-renderer/bin/audiobook-render runpod update

# Start the service, render and download one chapter, then stop.
.venv-renderer/bin/audiobook-render runpod encode BOOK-DIRECTORY --chapters 6 --chapters-only

# Inspect the provider's current state, or explicitly stop compute.
.venv-renderer/bin/audiobook-render runpod status
.venv-renderer/bin/audiobook-render runpod stop

# Delete every Pod this deployment created, and their Pod disks (asks first).
.venv-renderer/bin/audiobook-render runpod cleanup    # or: just runpod-cleanup
```

### Deleting the workers: `runpod cleanup`

`runpod cleanup` (or `just runpod-cleanup` from the checkout) **deletes** every
Pod this deployment created: the current worker, retired replacements, and Pods
whose creation reply was lost. It finds them by this deployment's unique
generated names (`NAME-DEPLOYMENT_ID`, and `NAME-DEPLOYMENT_ID-rHEX` for
replacements) and by the Pod IDs saved in its state; other Pods on the account
are never touched, and a Pod pinned by `runpod.pod_id` is never deleted. It lists
what it found and deletes nothing unless you type `delete`; `--yes` skips the
prompt and is required without a terminal. Deleting a Pod also deletes its Pod
disk — its models, environments, voices, and unfinished job checkpoints — and
cannot be undone. Network volumes, Ceph objects, local caches, and lease and lock
files are not touched.

Each deletion is confirmed by reading the Pod back (a 404 means it is gone) and
then removed from the saved state, so an interrupted or partly failed cleanup
resumes with the remaining Pods when rerun. Cleanup holds the deployment lock,
so it cannot run while an `encode` from the same state directory is attached;
it never starts or creates compute. The next `bootstrap` or `encode` creates
and fully sets up a new worker.

To inspect chapter progress while Encode is attached, use a second terminal:

```sh
.venv-renderer/bin/audiobook-render runpod progress --watch
# Or inspect one saved chapter without resuming it:
.venv-renderer/bin/audiobook-render runpod progress --job JOB_ID --json
```

Progress is read-only and works while Encode holds the deployment lock. It
shows chapter counts, connected/paused state, and the latest completed-block
counts when available. Workers running this revision also report a
`scheduling:` line (and a `scheduling` object in `--json` mode): the queue
counts per state, the active scheduler mode and capacity basis, the logical
scheduler's pending/active/owner gauges, the attempt backlog with its
monotonic peak pending-count and peak materialized-bytes high-water marks,
and per-stage queued/active counts from the closed render-metrics stage set.
Interpretation: a `generating` chapter whose synthesis stage shows zero active
is intentional idle inference waiting on downstream recognition/review work;
`paused` plus `owner_connected: false` means the rendering client is gone and
nothing will execute until one reconnects. It never starts a Pod, creates a
session, renews an
execution lease, or resumes a job. Stop watching with Ctrl-C. After updating
the worker, only jobs explicitly owned by a live rendering client can run;
closing that client pauses unfinished work after in-flight cache writes drain.

Encode first compares the deployed revision with this checkout's committed
HEAD. If they differ, it deploys the new code, runs the full install and the
smoke check, and continues into rendering **without stopping the Pod in
between** — no separate `update` run is needed. `--no-update` skips the check
and encodes with the deployed code as-is; a new worker, or one that never
passed its smoke check, is fully bootstrapped first (see below).

### Deleted worker recovery

If the Pod was deleted in the Runpod console, the next managed `encode`
recovers by itself. It first validates the local book and configuration — a
preflight failure creates and starts nothing — and then checks the saved Pod
before any paid mutation. A confirmed-deleted managed Pod is recreated under
the cloud and placement policy above, with an empty Pod disk. Local caches,
Ceph objects, lease and lock files are never touched.

Every new Pod invalidates the saved deployed revision and smoke flag through a
durable `bootstrap_required` marker: encode redeploys this checkout, runs the
full install, downloads models, and passes the smoke check before submitting
the book. `--no-update` never skips this mandatory bootstrap; it is ignored with
a notice, and the marker is re-checked after the worker starts because a
replacement can switch workers during startup. The marker clears only after a
successful smoke check, so an interrupted setup always resumes with the full
local deploy, install, prepare, and smoke. A healthy deployment keeps the fast
path: runtime install only, no smoke check, no redeploy.

An explicit `runpod.pod_id` is a pin: a pin the provider confirms missing is a
configuration error, never a silent recreation, and a pinned Pod is never
replaced. Managed saved IDs are recreated instead. Unconfirmed API answers
(401/403/5xx, lost responses) never prove absence and never trigger
replacement; a pod creation whose response was lost is reconciled by its unique
saved name on the next run instead of duplicated, and must match its saved
cloud, GPU, region, image and Pod disk before it is adopted. An older unsent
request that would have mounted a network volume is discarded and selected
again with a Pod disk. After the worker is replaced, chapter jobs saved by an
older server do not carry scheduling permissions or event cursors forward.
Each invocation submits fresh run-scoped jobs, reusing accepted audio and
validated local chapter files rather than queue or event history.

`--config FILE` selects the normal configuration; `--override-config FILE`
selects a different deployment override. Bootstrap and Update default to
`--source github`, which checks out the configured public repository/ref.
`--source local` transfers the current **committed HEAD** in a Git bundle;
uncommitted files, private configuration and book data are not included. This
allows testing a committed change before publishing it on GitHub.

Bootstrap and Update use maintenance startup: the chapter API is available for
checks, but old run records are discarded and chapter submissions/retries are
rejected. Voices, accepted clips and finished artifact files remain intact.
After the smoke check, the API activates normal operation without reloading
the inference model. Setup heartbeats prevent idle shutdown during that check.
For manual operation, the worker's
`serve-maintenance` command (or `serve-chapters --maintenance` on the renderer)
provides the same protection. Maintenance can be activated through the
authenticated API; it does not change audio-cache identities.

Successful operations leave the service and inference model warm. The server
stops paid compute after **300 idle seconds**, so another book can use the same
loaded model within that window. A matching revision and unchanged config
reuse the ready service without stopping it, installing dependencies or
launching another model process. Code or configuration changes still require
an update. `--stop-delay` is deprecated and ignored; it no longer holds the
client or stops a ready worker. Use `runpod stop` for an explicit immediate stop.

Disconnecting or interrupting a render releases its execution grants; in-flight
accepted work drains before the idle countdown starts. Connected delivery of
completed chapters also counts as busy until the files are verified and
acknowledged. Startup failures before successful setup still stop compute
immediately, because an idle supervisor is not yet reliable. Persistent storage
continues to be billed while compute is stopped.

The first live deployment should use one short chapter. The CLI's smoke check
runs a separate short passage through CUDA synthesis on every Drama 3 copy, then
speaks a second passage on each copy in the voice of its first sample, sent as
a lossless FLAC reference exactly as the chapter service sends registered
voices. It checks both with the configured Whisper device after the same
normalization rendering uses, encodes FLAC and M4A, and verifies Ceph
upload/download checksums and deletion of its temporary objects. It does not change any book or saved character voice. A
successful smoke check does not establish throughput for a full book.

## What persists when compute stops

Runpod clears the container disk when a Pod stops; the deployment therefore
keeps source, managed Python, virtual environments, dependency cache, model
weights, voices and job checkpoints on the `/workspace` Pod disk:

| Path | Contents |
| --- | --- |
| `/workspace/epub-tts-narrator` | Application checkout and Python environment |
| `/workspace/fish-speech` | Pinned, unmodified Fish checkout/environment; also holds our package for the inference service |
| `/workspace/.epub-tts-runtime` | Installer, managed Python and dependency cache |
| `/workspace/models` | Drama 3 and Whisper weights |
| `/workspace/chapter-server-data` | Registered voices, durable jobs and resumable audio |
| `/workspace/chapter-server-data/runtime/inference.toml` | Inference service config written by the launcher |
| `/workspace/epub-tts-narrator.remote.toml` | Private merged server configuration |

The launcher places the selected backend's virtual-environment tools first in
its subprocess search path. Restarting loads model weights and captures
process-local CUDA graphs.

The Pod disk retains these files across stop and start on the same host, but
not across a Pod replacement. Ceph retains finished exports independently.
The Pod disk continues to cost money while compute is stopped. Stopping
releases the GPU, so restarting depends on capacity becoming available on the
Pod's machine; immediate restart is not guaranteed. Native model loading also
takes time on every process start.

For managed deployments, a confirmed host-capacity error (or a Community worker
without direct SSH, above) triggers at most one replacement Pod per operation.
The replacement keeps the same image and must meet the configured hourly cap;
its cloud, GPU and region are selected by the same policy as a new Pod, so they
may differ from the source. The old Pod's cloud, GPU, datacenter, image, and
storage are recorded as an immutable source snapshot before anything is
created, and the stopped Pod is validated against that snapshot independently
of the replacement target — a changed `data_center_id` hint or cloud order is
not a mismatch. The replacement starts with an empty Pod disk and requires the
full bootstrap described above. The old Pod remains stopped, is never deleted automatically,
and its ID is retained in `runpod status` under `previous_pod_ids`; **its Pod
disk keeps billing until it is deleted** with `runpod cleanup` (below) or in the
Runpod console.

Replacement intent is saved before creation. If the API response is lost, the
client looks for that uniquely named Pod and verifies its placement before
adopting it. Rerunning resumes this check instead of creating another Pod.
Pending replacements are resumed before ordinary provisioning, even if the
old Pod was subsequently deleted. Only a confirmed 404 permits treating the
source as gone; permission and service errors leave the saved intent intact.
An unresolved creation is reported by `runpod status` and shutdown diagnostics;
inspect Runpod when the CLI cannot confirm which resource was created. Keep
the private deployment state so the client can reconcile the outcome safely.

On restart the installer restores missing container system packages, then
reuses the persistent Python environments. Encode uses its `runtime` mode,
which does not resolve or download Python packages. Bootstrap and Update use
`full` mode to install requirements. Prepared model snapshots are checked and
reused without new download requests when unchanged.

The client registers missing FLAC voices by hash, submits chapter jobs, follows
progress, and downloads M4A/JSON. Only compressed audio crosses the network.
FLAC masters support final M4B assembly without generating speech again.
Timing metadata currently links text at paragraph/chunk precision, not words.
See [remote rendering](../../REMOTE_RENDERING.md) for artifacts and recovery.

## Inference backend

Inference is always our service on Fish's library. Stock SGLang-Omni batches concurrent requests through one
model, but its Drama 3 speech API accepts codes only as system-turn references,
has no assistant-turn history and never returns generated codes (checked at the
pinned commit `35e973de` and at `main` on 2026-11-05), so it cannot carry chapter
history and is rejected. A future batched backend would need custom SGLang
preprocessing and vocoder stages that accept history and return codes.

Fish's pinned Drama 3 decoder is itself single-request: it allocates caches for
one row, captures its CUDA decode graphs for one row, and its attention takes one
shared position for every row, so requests with different prompt lengths cannot
share a batch without reimplementing parts of the model.

### Several model copies on one GPU

`runpod.inference_replicas` (default 1, at most 4) loads that many independent
Drama 3 copies on the Pod's one GPU. Each copy is its own inference process on a
private loopback port (8080, then 8090, 8091, 8092) and serves one request at a
time; the chapter service sees one single-request worker per copy, so up to that
many passages from different chapters generate at once. Passages inside one
chapter stay sequential because each one's history is the previous accepted take.

Each copy needs 41.5 GB for weights and codec and grows to about 48 GB
(48,110 MiB measured) with its caches and graphs while generating. Before
loading anything the worker checks that GPU 0 has at least 47 GiB per copy and
stops with the shortfall otherwise; CUDA Whisper needs another 2.5–4.5 GB when
`verification.device = "cuda"`. An 80 GB H100 holds one copy,
so more than one replica requires `prefer_h100 = false` and a large-memory
`gpu_type_id` such as the 96 GB RTX PRO 6000. All copies load together at
startup, and the smoke check generates and verifies a sample from every copy.

Without MPS the copies time-slice the GPU. `runpod.inference_mps = true` starts a
private NVIDIA Multi-Process Service daemon (`nvidia-cuda-mps-control`, which
must be present in the container) for the inference copies only, so their
kernels can run concurrently; Whisper and the chapter service do not join it.
The daemon is stopped whenever the worker stops. Changing either setting takes
effect at the next `runpod update`, or at an `encode` that finds the
configuration changed. Measure before relying on any speedup.

#### Measured on an RTX PRO 6000 (2026-11-07)

Fixed 24-passage corpus (`scripts/benchmark_rendering.py pipeline --worker all`),
CUDA Whisper verification, one Secure Cloud RTX PRO 6000 (96 GB), same quality
thresholds as rendering:

| Configuration | Generated audio / s | Verified audio / s | Accepted | GPU memory peak |
| --- | --- | --- | --- | --- |
| 1 copy (default) | 2.10× | 2.10× | 24 / 24 | 52.7 GB |
| 2 copies | 2.09× | 1.47× | 22 / 24 | 97.1 of 97.9 GB |
| 2 copies + MPS | 2.62× | 1.84× | 22 / 24 | 97.2 of 97.9 GB |

Without MPS a second copy adds nothing: the copies time-slice a GPU that one
single-request decoder already keeps busy (96% utilization). MPS lets their
kernels overlap for about 25% more generated audio, but two copies with CUDA
Whisper fill the card to within 1 GB, and two passages were rejected in both
two-copy runs (none with one copy); that cause has not been isolated. Keep the
default single copy. For comparison, the batched SGLang backend measured 8.04×
verified on the same card and corpus on 2026-10-22, before it was removed for
lacking chapter history; that gap is the headroom a batched decoder with
history could recover.

### Optional CUDA verification

CPU recognition remains the default. To move Whisper onto the same GPU, add
this to `runpod.local.toml` and run a full Update:

```toml
[verification]
device = "cuda"
compute_type = "int8_float16"
```

Omitting `compute_type` selects `int8_float16` for CUDA or `int8` for CPU.
Explicit remote choices override these defaults; local Mac hardware settings
are not inherited. Before inference starts, the launcher loads the runtime
libraries and checks GPU 0's supported compute types without loading Whisper
weights. An unsupported device/type fails clearly without falling back to CPU.

The application environment pins CTranslate2 `4.8.3` and CUDA 12 cuBLAS
`12.8.4.1`. This CTranslate2 release uses native CUDA convolution and needs no
cuDNN; its [Linux wheel build](https://github.com/OpenNMT/CTranslate2/blob/v4.8.3/python/tools/prepare_build_environment_linux.sh)
disables cuDNN. The launcher supplies the application library path before Python
starts and removes it from the separate inference subprocess. These are
internal runtime settings; users continue configuring everything through TOML.

With CUDA verification, Whisper shares the GPU with the inference service; benchmark
the combined load and actual peak memory on the chosen GPU.

Benchmark before choosing a concurrency level. See
[benchmark_rendering.py](../../scripts/benchmark_rendering.py) for a fixed-book
corpus and separate generation/verification timings. `worker.py serve-tts`
starts only private inference, so a benchmark does not resume queued chapter
jobs. Run it under an operation that stops the Pod in a `finally` block; the
chapter service's idle shutdown is not running in this mode. Do not interpret
raw synthesis throughput as verified audiobook throughput or compare runs on
different texts as a hardware speedup.

For CUDA verification benchmarks, use the application wrapper so the same
runtime libraries are selected before the interpreter starts:

```sh
/workspace/epub-tts-narrator/.venv/bin/python /workspace/epub-tts-narrator/deploy/runpod/worker.py \
  --config /workspace/epub-tts-narrator.remote.toml run-application -- \
  /workspace/epub-tts-narrator/scripts/benchmark_rendering.py verify /workspace/benchmark/manifest.json \
  --config /workspace/epub-tts-narrator.remote.toml
```

The wrapper also accepts Python's `-m MODULE` form. It replaces itself with
the application interpreter, preserving signals and exit status. It does not
start inference, load models by itself, or stop paid compute.

For the whole-chapter pipeline benchmark, use
[benchmark_chapter_pipeline.py](../../scripts/benchmark_chapter_pipeline.py)
with its `prepare` and `run` subcommands. `prepare` freezes a deterministic
corpus from selected real chapters plus portable FLAC references before any
measurement; `run` executes the production ChapterJobs/ChapterExecutor service
with real HTTP/session/ACK behavior in a fresh namespaced cache and output
root, using `--scheduler legacy|pipeline` for A/B. Keep both arms on the same
transport and host: run the worker side through the same `worker.py
run-application --` wrapper so recognition and GPU samples come from the
production environment, and drive A/B over loopback on the host before
qualifying the ordinary remote client. A run that used `--fake-engine`, a
resumed/cached corpus, or any injected disconnect is labeled and excluded
from speed comparisons automatically; benchmarks never modify the original
book. The pipeline scheduler itself stays opt-in: default scheduling remains
`legacy` until a recorded live A/B gate (at least 20% median accepted-source-
audio throughput with unchanged acceptance rules) passes, and no speedup is
claimed before that measurement exists.

## Manual worker commands and diagnostics

For a Linux x86_64 Pod with root SSH access, `/workspace` mounted persistently,
and a CUDA 12.9-compatible NVIDIA driver:

```sh
sh /workspace/epub-tts-narrator/deploy/runpod/install.sh /workspace/epub-tts-narrator full
/workspace/epub-tts-narrator/.venv/bin/python /workspace/epub-tts-narrator/deploy/runpod/worker.py prepare
/workspace/epub-tts-narrator/.venv/bin/python /workspace/epub-tts-narrator/deploy/runpod/worker.py check
/workspace/epub-tts-narrator/.venv/bin/python -u /workspace/epub-tts-narrator/deploy/runpod/worker.py serve
```

After the chapter service starts, run `worker.py smoke` from another SSH session.
The default config is `/workspace/epub-tts-narrator.remote.toml`; use
`--config PATH` before the subcommand to override it. `prepare` resumes partial
model downloads; `check` validates offline receipts and configured CUDA support; `serve` never downloads
models. `smoke` has a ten-minute limit. Each smoke object uses a unique prefix
and is deleted even if download verification fails.

The launcher waits for native model readiness before starting the authenticated
chapter service. Its default public URL is `https://POD_ID-8082.proxy.runpod.net`;
expose HTTP port **8082**, plus TCP 22 for SSH. If changing `service_port`, use
that port in the URL and Pod exposure settings. Fish binds exclusively to
`http://127.0.0.1:8080/v1/tts`. A filesystem lease prevents duplicate launchers
from using the same data directory. Child process groups are terminated on
failure or interruption. The CLI stops paid compute after its operation, even
if the worker failed during installation or startup.

## Optional reusable image

The included Dockerfile can preinstall source and dependencies in an immutable
image, with models and job state on `/workspace`. Build on a Linux x86_64
builder or Docker Buildx with a running daemon:

```sh
docker buildx build --platform linux/amd64 -f deploy/runpod/Dockerfile -t YOUR-REGISTRY/epub-tts-narrator:YOUR-VERSION --push .
```

Its build context allowlist excludes credentials, books, audio and model files.
No SSH server is bundled. The image uses `/opt` for preinstalled immutable
runtime files; those are restored from the image rather than a manual container
installation. The lifecycle CLI uses the persistent source/SSH deployment
instead. Do not mount `/workspace` over `/opt`.

## Pinned inference contract

- Fish source: `541566932abbaebcb0a40e562a1f846738383f38`.
- Drama 3 weights: `b2b5ee998034b2975720906c815e7fb0e9f007bd`.
- CTranslate2 Whisper large-v3: `edaa852ec7e145841d8ffdb056a99866b5f0a478`.
- Fish uses managed Python 3.12 and its frozen CUDA 12.9 lock. Chapter processing
  and CPU/CUDA recognition use a separate environment.
- `install.sh` installs this stdlib-only package into Fish's environment with
  `--no-deps` (plus `msgpack`) after every frozen `uv sync`; Fish's source and
  lock stay unmodified.
- The inference service loads Drama 3 like the stock server (bfloat16, no
  compilation, codec on the tower device) and generates one chunk per request
  with `build_drama3_prompt(history=...)` and `generate_drama3`, Fish's default
  sampling (`temperature`/`top_p` 0.7, `top_k` 30) and `set_seed`. Each copy
  serves **one** request at a time; `runpod.inference_replicas` (default 1)
  loads independent copies, never a shared or batched model.
- Requests and responses use protocol `audiobook-history-wav-v2`: speaker-tagged
  references, up to two history turns with codes, and a MessagePack response
  with the WAV and the generated codes.

Primary references: [Fish installation](https://speech.fish.audio/install/),
[Fish inference](https://speech.fish.audio/inference/),
[Runpod Pod lifecycle](https://docs.runpod.io/pods/manage-pods),
[Runpod ports](https://docs.runpod.io/pods/configuration/expose-ports), and
[Ceph S3 API](https://docs.ceph.com/en/latest/radosgw/s3/).
