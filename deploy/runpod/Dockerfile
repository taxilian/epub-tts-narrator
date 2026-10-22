# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

WORKDIR /opt/epub-tts-narrator
COPY pyproject.toml README.md ./
COPY epub_tts_narrator/ ./epub_tts_narrator/
COPY deploy/runpod/ ./deploy/runpod/
RUN sh deploy/runpod/install.sh /opt/epub-tts-narrator full /opt/epub-tts-runtime /opt/fish-speech

EXPOSE 8082
# First boot permits configuration upload and explicit model preparation.
# After preparation, set the Pod start command to the worker's `serve` command.
# Only the chapter service port is public; Fish binds to loopback on 8080.
CMD ["sleep", "infinity"]
