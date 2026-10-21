# Run our chapter worker on Runpod

This deployment runs **this project's code** for job scheduling, persistent voice
registration, verification, chapter assembly, and Ceph publication. One unmodified,
pinned Fish Speech process supplies CUDA inference on loopback. No SGLang-Omni
service is required. References use Fish's memory cache, GPU requests are
serialized, and CPU verification/assembly can overlap subsequent synthesis.

The launcher and transport have automated tests. **The image, model loading,
real GPU synthesis, and Runpod/Ceph integration still need a live deployment
test.** Do that with one short chapter before queuing a book.

## First deployment

1. Add credit in [Runpod billing](https://console.runpod.io/user/billing).
   Use an on-demand Pod for the first test. An H100 with 80 GB VRAM provides
   headroom for the native Drama 3 model; throughput still needs benchmarking.
2. Create a network volume in the selected GPU region and mount it at
   `/workspace`. Start with approximately 150 GB for the models and a limited
   chapter test, then measure accepted paragraph checkpoint usage. Ceph holds
   finished exports; it does not replace the resumable working directory.
3. Start a Linux x86_64 development Pod with SSH, root access, and a CUDA 12.9
   compatible NVIDIA driver. A Runpod PyTorch development template is useful
   for the first installation; its model environment is not used. Expose
   **HTTP port 8081 only** for our application. Check `nvidia-smi` on the Pod:
   its driver-supported CUDA version must be 12.9 or newer.
4. Upload the source archive described below using `scp` and the SSH details
   from Runpod's Connect panel. On the Pod:

   ```sh
   mkdir -p /opt
   tar -xzf /tmp/epub-tts-narrator-worker.tar.gz -C /opt
   sh /opt/epub-tts-narrator/deploy/runpod/install.sh
   cp /opt/epub-tts-narrator/deploy/runpod/epub-tts-narrator.example.toml /workspace/epub-tts-narrator.toml
   chmod 600 /workspace/epub-tts-narrator.toml
   ```

   Edit that **one global server file**. Set a random application API key, Ceph
   connection details, and the existing JEV settings. Use the same application
   key in the Mac client's `[remote_chapters]` table. It is separate from the
   Runpod and Ceph keys. Do not upload the Mac's entire private config; its
   analyzer and voice-design credentials are unnecessary here.
5. Explicitly prepare the pinned models, then check the prepared files:

   ```sh
   /opt/epub-tts-narrator/.venv/bin/python /opt/epub-tts-narrator/deploy/runpod/worker.py prepare
   /opt/epub-tts-narrator/.venv/bin/python /opt/epub-tts-narrator/deploy/runpod/worker.py check
   ```

   These default to `/workspace/epub-tts-narrator.toml`. Override with
   `--config PATH` **before** the subcommand. Preparation downloads public model
   snapshots and can resume partial downloads. `check` validates config and
   prepared files; it does not test the GPU. `serve` never downloads models.
6. Start the first supervised test in the SSH terminal:

   ```sh
   /opt/epub-tts-narrator/.venv/bin/python -u /opt/epub-tts-narrator/deploy/runpod/worker.py serve
   ```

   Keep the terminal attached. The launcher waits for the native model to load,
   then starts the chapter service. The public URL is
   `https://POD_ID-8081.proxy.runpod.net`. The private inference endpoint is
   `http://127.0.0.1:8080/v1/tts`; do not expose port 8080.

Initial setup/model preparation is not covered by the application's idle timer.
Stop the Pod yourself when leaving setup unfinished. Exiting SSH or Python does
**not** stop GPU billing. Manual `/opt` installs live on the container disk and
may need reinstalling after stop/replacement; use the image below for repeated,
unattended deployments. Models and job checkpoints stay on `/workspace`.

## Ceph and automatic shutdown

Use a private bucket on an HTTPS RADOS Gateway endpoint reachable from Runpod
and the Mac. Configure its endpoint, bucket, region/zonegroup API name, and a
bucket-scoped access/secret key under `[chapter_storage]`. The publisher uses
path-style requests and Signature V4, uploads compressed outputs, checks their
size and SHA256 metadata, and issues presigned download URLs. Validate a small
upload and download before enabling unattended jobs.

After confirming the persistent mount, Ceph publication, and repeatable worker
startup, merge these into the existing server table:

```toml
[chapter_server]
persistent_data = true
idle_action = "runpod_stop"
idle_timeout_seconds = 600
runpod_pod_id = "YOUR-POD-ID"
runpod_api_key = "YOUR-RUNPOD-KEY"
```

The service calls Runpod's stop API after ten minutes without work, once outputs
are published. CPU assembly, uploads, and queued jobs prevent shutdown; status
polling does not. Ceph links work after shutdown until expiry. Restart the
service to renew them. Runpod's network-volume S3 API currently does not support
presigned URLs; use Ceph for this publisher.

Persistent storage still costs money after stopping compute. A crashed process
cannot run its idle callback; check billing/alerts during the initial test.
See [remote rendering](../../REMOTE_RENDERING.md) for recovery and artifacts.
Timing metadata currently links text at paragraph/chunk precision, not words.

## Run a chapter from the Mac

Install the client transport once:

```sh
uv pip install --python .venv-renderer/bin/python -e '.[remote]'
```

Merge into the Mac's existing global `epub-tts-narrator.toml`:

```toml
[remote_chapters]
url = "https://YOUR-POD-ID-8081.proxy.runpod.net"
api_key = "THE-SAME-APPLICATION-KEY-AS-THE-SERVER"
```

Select a chapter with no unfinished local paragraph cache:

```sh
.venv-renderer/bin/audiobook-render render BOOK-DIRECTORY --chapters 6 --chapters-only
```

The client registers missing FLAC voices by hash, submits the chapter, follows
progress, and downloads M4A/JSON. Only compressed audio crosses the network.
FLAC masters are downloaded when final M4B assembly needs them. Listen to the
result, check multi-speaker passages and resume behavior, and measure actual
throughput before submitting more chapters.

## Source archive and reusable image

From a checkout with committed deployment files, make an allowlisted archive.
This excludes private configs, books, samples, weights, Git history, and the
untracked working tree:

```sh
mkdir -p dist
git archive --format=tar.gz --prefix=epub-tts-narrator/ --output=dist/epub-tts-narrator-worker.tar.gz HEAD pyproject.toml README.md REMOTE_RENDERING.md RENDERER.md epub_tts_narrator deploy/runpod
```

For repeat deployments, build our CUDA image on a Linux x86_64 builder (or
Docker Buildx targeting Linux amd64). This requires a running Docker daemon
and downloads sizeable dependencies, but no models or credentials:

```sh
docker buildx build --platform linux/amd64 -f deploy/runpod/Dockerfile -t YOUR-REGISTRY/epub-tts-narrator:YOUR-VERSION --push .
```

The Dockerfile-specific allowlist excludes local secrets/audio. The image
defaults to `sleep infinity` for deliberate setup. Once the persistent volume
contains the prepared models and config, set the Pod container start command:

```text
/opt/epub-tts-narrator/.venv/bin/python -u /opt/epub-tts-narrator/deploy/runpod/worker.py serve
```

No SSH service is bundled in our image. Prepare the volume using the development
Pod, then attach it when deploying the image. The launcher tears down both child
process groups on failure or interruption. Run one worker per data directory;
a startup lease prevents duplicate loads.

## Pinned inference contract

- Fish Speech source: `541566932abbaebcb0a40e562a1f846738383f38`.
- Drama 3 weights: `b2b5ee998034b2975720906c815e7fb0e9f007bd`.
- CTranslate2 Whisper large-v3: `edaa852ec7e145841d8ffdb056a99866b5f0a478`.
- Fish uses its own Python 3.12 environment and frozen CUDA 12.9 lock. The
  chapter application and CPU recognition use a separate environment.
- `fish_native_drama3` sends `single_chunk=true`, `use_memory_cache="on"`, and
  the seed in the MessagePack body, preserving speaker-tagged references.
  Native `top_k` is fixed at 30; use inference concurrency **1**.
- Multiple chapter coordinators overlap non-GPU stages. This does not add
  native GPU batching or load a second model copy.

Primary references: [Fish installation](https://speech.fish.audio/install/),
[Fish inference](https://speech.fish.audio/inference/),
[Runpod ports](https://docs.runpod.io/pods/configuration/expose-ports),
[Runpod storage](https://docs.runpod.io/storage/s3-api), and
[Ceph S3 API](https://docs.ceph.com/en/latest/radosgw/s3/).
