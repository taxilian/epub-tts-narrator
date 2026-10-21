#!/bin/sh
# Install uploaded project code on a fresh Linux x86_64 worker, or in our image.
# This does not download weights, start services, or provision a paid resource.
set -eu

worker_root=/opt/epub-tts-narrator
fish_root=/opt/fish-speech
fish_revision=541566932abbaebcb0a40e562a1f846738383f38

if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ] || [ "$(id -u)" != 0 ]; then
    echo 'Run this installer as root on the Linux x86_64 Pod, not on your Mac.' >&2
    exit 1
fi
if [ ! -f "$worker_root/pyproject.toml" ]; then
    echo "Extract the project into $worker_root first." >&2
    exit 1
fi

apt-get update
apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake ffmpeg git \
    libasound2-dev libportaudio2 libportaudiocpp0 libsndfile1 libsox-dev \
    portaudio19-dev python3 python3-dev python3-venv
rm -rf /var/lib/apt/lists/*

python3 -m venv /opt/epub-tts-worker-tools
/opt/epub-tts-worker-tools/bin/python -m pip install --no-cache-dir 'uv==0.8.15'
worker_uv=/opt/epub-tts-worker-tools/bin/uv
if [ ! -x "$worker_root/.venv/bin/python" ]; then
    "$worker_uv" venv --python 3.12 "$worker_root/.venv"
fi
"$worker_root/.venv/bin/python" -c 'import sys; assert sys.version_info >= (3, 11), "Worker requires Python 3.11+"'

if [ ! -d "$fish_root/.git" ]; then
    # Refuse to reuse an unrelated nonempty directory.
    mkdir -p "$fish_root"
    if [ -n "$(ls -A "$fish_root")" ]; then
        echo "$fish_root is not an empty directory or a Git checkout." >&2
        exit 1
    fi
    git -C "$fish_root" init
    git -C "$fish_root" remote add origin https://github.com/fishaudio/fish-speech.git
    git -C "$fish_root" fetch --depth 1 origin "$fish_revision"
    git -C "$fish_root" checkout --detach FETCH_HEAD
fi
if [ "$(git -C "$fish_root" rev-parse HEAD)" != "$fish_revision" ] || \
    [ -n "$(git -C "$fish_root" status --porcelain --untracked-files=no)" ]; then
    echo 'Fish source differs from the pinned, unmodified checkout; refusing to replace it.' >&2
    exit 1
fi
(
    cd "$fish_root"
    "$worker_uv" sync --python 3.12 --extra cu129 --frozen
)
"$worker_uv" pip install --python "$worker_root/.venv/bin/python" \
    -e "${worker_root}[chapter-server,cuda-verification]" \
    -r "$worker_root/deploy/runpod/requirements.txt"
echo 'Worker installed. Supply the TOML file, then run worker.py prepare, check, and serve.'
