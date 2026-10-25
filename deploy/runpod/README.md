# Run our chapter worker on Runpod

The lifecycle CLI installs **this project's code** for voice registration, chapter
jobs, verification, assembly, and Ceph publication. One pinned, unmodified Fish
Speech or SGLang-Omni process supplies CUDA inference on loopback. Native Fish
serializes requests; SGLang batches concurrent requests through one loaded TTS
model. Verification and CPU assembly can overlap synthesis. Built-in JEV and
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
normal local file. Configure the GPU, region, hourly price limit and storage in
the override's `[runpod]` table. Use a private HTTPS Ceph bucket reachable from
both the Pod and your Mac.

Copy [runpod.example.toml](runpod.example.toml) to `runpod.local.toml` beside your
normal configuration. Its image is pinned; GPU availability and prices are
checked when creating compute. The selected region must also support network
volumes. The example's 150 GB volume costs storage fees even while stopped;
the hourly ceiling covers compute and does not cap storage charges.

The selected datacenter must support **standard network volumes**, not just
the requested GPU. The live catalog is checked before allocation; a GPU-only
location such as US-NC-2 is rejected locally with supported alternatives.
Provider 5xx errors include bounded, credential-redacted JSON error details
so a storage-placement rejection is visible instead of a blind retry message.

### GPU selection

By default (`prefer_h100 = true`) a new Pod prefers any available NVIDIA H100
variant (80GB HBM3/SXM, PCIe, or NVL) whose catalog rate is **strictly under
$2.60/hour** and within `max_hourly_price`, choosing the cheapest eligible one
in your configured `data_center_id`. If no such H100 exists — too expensive,
out of stock, or absent from your datacenter — the Pod uses the configured
`gpu_type_id` fallback (the RTX PRO 6000 in the example), which must be
available at or under `max_hourly_price` or nothing is created. Set
`prefer_h100 = false` to pin the configured GPU exactly. After creation the
provider's actual rate is checked against the same rules and the worker is
stopped if it exceeds them. Catalog and hourly prices cover compute only;
persistent storage is billed separately. The chosen GPU is saved before the
Pod is created, so a retried or restarted deployment keeps the same hardware,
and a healthy existing Pod is never replaced just to upgrade its GPU.

`[runpod].service_port` selects the chapter service's listen port and Runpod
proxy port; it defaults to **8082**. The pinned base image reserves 8081 for
nginx. Choose an unused port if using another image; ports 22 and 8080 are
reserved for SSH and native Fish inference. This deployment setting overrides
`[chapter_server].port` in the generated worker configuration. The CLI applies
changed port exposure while the Pod is stopped, preserving its network volume.
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
```

To inspect chapter progress while Encode is attached, use a second terminal:

```sh
.venv-renderer/bin/audiobook-render runpod progress --watch
# Or inspect one saved chapter without resuming it:
.venv-renderer/bin/audiobook-render runpod progress --job JOB_ID --json
```

Progress is read-only and works while Encode holds the deployment lock. It
shows chapter counts, connected/paused state, and the latest completed-block
counts when available. It never starts a Pod, creates a session, renews an
execution lease, or resumes a job. Stop watching with Ctrl-C. After updating
the worker, only jobs explicitly owned by a live rendering client can run;
closing that client pauses unfinished work after in-flight cache writes drain.

Encode first compares the deployed revision with this checkout's committed
HEAD. If they differ, it deploys the new code, runs the full install and the
smoke check, and continues into rendering **without stopping the Pod in
between** — no separate `update` run is needed. `--no-update` skips the check
and encodes with the deployed code as-is; a worker that was deleted or never
passed its smoke check is fully bootstrapped first (see below).

### Deleted worker or volume recovery

If the Pod or network volume was deleted in the Runpod console, the next
managed `encode` recovers by itself. It first validates the local book and
configuration — a preflight failure creates and starts nothing — and then
verifies both saved resources before any paid mutation. A deleted managed Pod
is recreated beside the retained volume; a deleted volume is replaced with a
new empty one, and a surviving Pod that mounted it is stopped and confirmed
stopped before any replacement is created, so nothing ever boots on dead
storage. Local caches, Ceph objects, lease and lock files are never touched.

Storage replacement invalidates the saved deployed revision and smoke flag:
encode redeploys this checkout, runs the full install, downloads models, and
passes the smoke check before submitting the book. `--no-update` never skips
this mandatory bootstrap; it is ignored with a notice when the deployment was
deleted or never bootstrapped. A healthy deployment keeps the fast path:
runtime install only, no smoke check, no redeploy.

An explicit `runpod.pod_id` or `runpod.volume_id` is a pin: a pin the provider
confirms missing is a configuration error, never a silent recreation. Managed
saved IDs are recreated instead. Unconfirmed API answers (401/403/5xx, lost
responses) never prove absence and never trigger replacement; a pod creation
whose response was lost is reconciled by its unique saved name on the next run
instead of duplicated. After the worker is replaced, chapter jobs saved by an
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
runs a separate short passage through CUDA synthesis and the configured Whisper device, encodes
FLAC and M4A, and verifies Ceph upload/download checksums and deletion of its
temporary objects. It does not change any book or saved character voice. A
successful smoke check does not establish throughput for a full book.

## What persists when compute stops

Runpod clears the container disk when a Pod stops; the deployment therefore
keeps source, managed Python, virtual environments, dependency cache, model
weights, voices and job checkpoints on `/workspace`:

| Path | Contents |
| --- | --- |
| `/workspace/epub-tts-narrator` | Application checkout and Python environment |
| `/workspace/fish-speech` | Pinned, unmodified native inference checkout/environment |
| `/workspace/sglang-omni` | Optional pinned, unmodified batching checkout/environment |
| `/workspace/.epub-tts-runtime` | Installer, managed Python and dependency cache |
| `/workspace/models` | Drama 3 and Whisper weights |
| `/workspace/chapter-server-data` | Registered voices, durable jobs and resumable audio |
| `/workspace/chapter-server-data/runtime-cache/sglang` | SGLang, Triton, Inductor, FlashInfer and CUDA compilation caches |
| `/workspace/epub-tts-narrator.remote.toml` | Private merged server configuration |

The launcher places the selected backend's virtual-environment tools first in
its subprocess search path, including `ninja` used for kernel compilation.
SGLang compilation caches are derived from `chapter_server.data_dir` and retained
across stops; no environment-variable configuration is required. Restarting
still loads model weights and captures process-local CUDA graphs.

A network volume retains these files even if the Pod is replaced. Ceph retains
finished exports independently. Persistent storage continues to cost money
while compute is stopped. Stopping releases the GPU, so restarting depends on
capacity becoming available on the Pod's machine; immediate restart is not
guaranteed. Native model loading also takes time on every process start.

For managed deployments, a confirmed host-capacity error triggers at most one
replacement Pod per operation. The replacement keeps the same image,
datacenter, and existing network volume, and must meet the configured hourly
cap. Its GPU is selected by the same policy as a new Pod: a preferred H100
under $2.60/hour when `prefer_h100` is set, otherwise the configured
`gpu_type_id`. The old Pod's GPU is recorded as the replacement's immutable
source, and the stopped Pod is validated against it even when the replacement
targets a different GPU. The previous Pod remains stopped. The client does not
allocate another volume or choose a different region. A deployment with an
explicit `runpod.pod_id` is fixed and is never replaced automatically.

Replacement intent is saved before creation. If the API response is lost, the
client looks for that uniquely named Pod and verifies its placement before
adopting it. Rerunning resumes this check instead of creating another Pod.
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

## Optional batched inference

Select the backend in the deployment override, then run a full Update:

```toml
[runpod]
tts_backend = "sglang"
tts_concurrency = 4
```

`tts_backend` defaults to `native`. For `sglang`, `tts_concurrency` accepts 1–8
simultaneous requests; these share one model, not multiple replicas. The worker
uses stock SGLang-Omni commit `35e973de917988da23eeba5706c1d6045f260104`
in its own environment and reuses the existing Drama 3 checkpoint and codec.
The pinned dependency stack requires Linux glibc 2.34+ and a CUDA-13-compatible
driver. Installation leaves the native environment and upstream source intact.

The stock speech API supports one speaker per request. Single-voice paragraph
groups keep their original text and delivery controls; mixed-speaker paragraphs
require the native backend. This stock API has no per-request seed control:
metadata explicitly records `seed_applied = false`, and repeated requests draw
new stochastic takes. Model/voice/sampling compatibility still controls reuse
of previously accepted pool clips. Switching scheduling or concurrency alone
does not discard accepted audio.

The generated SGLang configuration bounds unfinished paragraphs across all
chapter coordinators to four times `tts_concurrency`. This includes recognition
and review, so a full GPU batch can run while earlier audio is being checked.
It still uses one TTS model and one shared Whisper model.

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
starts and removes it from the separate SGLang CUDA 13 subprocess. These are
internal runtime settings; users continue configuring everything through TOML.

With CUDA verification, SGLang's engine memory share is 0.60 instead of 0.70,
using the stock `--tts_engine.gpu_memory_fraction` option. The vocoder's 0.10
share and preprocessing reserve stay unchanged. This leaves more space for
Whisper; benchmark the combined load and actual peak memory on the chosen GPU.
The native backend's memory behavior is unchanged.

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
- Native requests use `single_chunk=true`, `use_memory_cache="on"`, a seed in
  the MessagePack body, and speaker-tagged references. Native `top_k` is fixed
  at 30; inference concurrency is **1**. No second model copy is loaded.
- Native compilation is disabled. Multiple chapter coordinators overlap CPU
  stages without adding unsupported GPU batching.

Primary references: [Fish installation](https://speech.fish.audio/install/),
[Fish inference](https://speech.fish.audio/inference/),
[Runpod Pod lifecycle](https://docs.runpod.io/pods/manage-pods),
[Runpod ports](https://docs.runpod.io/pods/configuration/expose-ports), and
[Ceph S3 API](https://docs.ceph.com/en/latest/radosgw/s3/).
