#!/bin/sh
# Install uploaded project code on a fresh Linux x86_64 worker, or in our image.
# This does not download weights, start services, or provision a paid resource.
set -eu

worker_root=${1:-/workspace/epub-tts-narrator}
install_mode=${2:-full}
runtime_root=${3:-/workspace/.epub-tts-runtime}
fish_root=${4:-/workspace/fish-speech}
fish_revision=541566932abbaebcb0a40e562a1f846738383f38

case "$install_mode" in full|runtime) ;; *) echo 'Mode must be full or runtime.' >&2; exit 1 ;; esac
for worker_path in "$worker_root" "$runtime_root" "$fish_root"; do
    case "$worker_path" in /*) ;; *) echo 'Installation paths must be absolute.' >&2; exit 1 ;; esac
done

if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ] || [ "$(id -u)" != 0 ]; then
    echo 'Run this installer as root on the Linux x86_64 Pod, not on your Mac.' >&2
    exit 1
fi
if [ ! -f "$worker_root/pyproject.toml" ]; then
    echo "Extract the project into $worker_root first." >&2
    exit 1
fi

# Container packages disappear when Runpod restarts a stopped Pod. Check first,
# then restore only missing system dependencies. Python, both environments,
# downloaded packages and all models live on the persistent /workspace mount.
worker_packages='build-essential ca-certificates cmake ffmpeg git libasound2-dev libportaudio2 libportaudiocpp0 libsndfile1 libsox-dev portaudio19-dev python3 python3-dev python3-venv'
missing_packages=''
for worker_package in $worker_packages; do
    if [ "$(dpkg-query -W -f='${db:Status-Status}' "$worker_package" 2>/dev/null || true)" != installed ]; then
        missing_packages="$missing_packages $worker_package"
    fi
done
if [ -n "$missing_packages" ]; then
    apt-get update
    # These words are fixed package names above, not user-controlled shell text.
    apt-get install -y --no-install-recommends $missing_packages
    rm -rf /var/lib/apt/lists/*
fi

if [ "$install_mode" = runtime ]; then
    "$worker_root/.venv/bin/python" -c 'import fastapi, uvicorn, boto3, faster_whisper, soundfile, msgpack, epub_tts_narrator'
    "$fish_root/.venv/bin/python" -c 'import torch, fish_speech'
    echo 'Persistent worker environments are ready.'
    exit 0
fi

mkdir -p "$runtime_root"
worker_uv="$runtime_root/tools/bin/uv"
if [ ! -x "$worker_uv" ] || [ "$("$worker_uv" --version)" != 'uv 0.8.15' ]; then
    python3 -m venv "$runtime_root/tools"
    "$runtime_root/tools/bin/python" -m pip install --no-cache-dir 'uv==0.8.15'
fi
if [ ! -f "$runtime_root/python-path" ]; then
    "$worker_uv" python install --install-dir "$runtime_root/python" --no-bin --cache-dir "$runtime_root/cache" 3.12
    python3 - "$runtime_root" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
interpreters = sorted((root / 'python').glob('cpython-3.12.*/bin/python3.12'))
if len(interpreters) != 1:
    raise SystemExit('Expected one persistent managed Python interpreter')
(root / 'python-path').write_text(str(interpreters[0]) + '\n')
PY
fi
worker_python=$(cat "$runtime_root/python-path")
case "$worker_python" in "$runtime_root"/python/*/bin/python3.12) ;; *) echo 'Invalid persistent Python receipt.' >&2; exit 1 ;; esac
if [ ! -x "$worker_python" ]; then
    echo 'Persistent Python is missing; repair the runtime before continuing.' >&2
    exit 1
fi
if [ ! -x "$worker_root/.venv/bin/python" ]; then
    "$worker_uv" venv --python "$worker_python" --no-python-downloads --cache-dir "$runtime_root/cache" "$worker_root/.venv"
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
    "$worker_uv" sync --python "$worker_python" --no-python-downloads --cache-dir "$runtime_root/cache" --extra cu129 --frozen
)
"$worker_uv" pip install --python "$worker_root/.venv/bin/python" \
    --cache-dir "$runtime_root/cache" \
    -e "${worker_root}[chapter-server,cuda-verification]" \
    -r "$worker_root/deploy/runpod/requirements.txt"
echo 'Worker installed on persistent storage. Run worker.py prepare, check, serve, then smoke.'
