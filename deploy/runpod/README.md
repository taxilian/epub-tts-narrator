# Run our chapter worker on Runpod

The lifecycle CLI installs **this project's code** for voice registration, chapter
jobs, verification, assembly, and Ceph publication. One pinned, unmodified Fish
Speech process supplies CUDA inference on loopback. GPU requests are serialized;
CPU verification and assembly can overlap subsequent synthesis.

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

`--config FILE` selects the normal configuration; `--override-config FILE`
selects a different deployment override. Bootstrap and Update default to
`--source github`, which checks out the configured public repository/ref.
`--source local` transfers the current **committed HEAD** in a Git bundle;
uncommitted files, private configuration and book data are not included. This
allows testing a committed change before publishing it on GitHub.

Each operation stops the Pod on success, failure or interruption. Optionally
pass `--stop-delay 300` to keep the client attached for a five-minute grace
period before stopping; Ctrl-C ends the grace period and stops immediately.
Server idle shutdown is an additional fallback after its configured period
without queued work, synthesis, assembly or uploads. Leaving SSH or exiting a
server process alone does **not** stop GPU billing.

The first live deployment should use one short chapter. The CLI's smoke check
runs a separate short passage through CUDA synthesis and CPU Whisper, encodes
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
| `/workspace/.epub-tts-runtime` | Installer, managed Python and dependency cache |
| `/workspace/models` | Drama 3 and CPU Whisper weights |
| `/workspace/chapter-server-data` | Registered voices, durable jobs and resumable audio |
| `/workspace/epub-tts-narrator.remote.toml` | Private merged server configuration |

A network volume retains these files even if the Pod is replaced. Ceph retains
finished exports independently. Persistent storage continues to cost money
while compute is stopped. Stopping releases the GPU, so restarting depends on
capacity becoming available on the Pod's machine; immediate restart is not
guaranteed. Native model loading also takes time on every process start.

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
model downloads; `check` validates offline receipts; `serve` never downloads
models. `smoke` has a ten-minute limit. Each smoke object uses a unique prefix
and is deleted even if download verification fails.

The launcher waits for native model readiness before starting the authenticated
chapter service. Its public URL is `https://POD_ID-8081.proxy.runpod.net`; expose
HTTP port **8081 only**, plus TCP 22 for SSH. Fish binds exclusively to
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
  and CPU recognition use a separate environment.
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
