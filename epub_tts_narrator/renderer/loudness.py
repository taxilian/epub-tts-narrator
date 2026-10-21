"""Loudness-matched audition copies; immutable TTS references stay untouched."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from .assembly import _wave_info
from .common import digest, file_hash, read_json, write_json


class SampleNormalizer:
    """Cache two-pass EBU R128 previews by source bytes and playback settings."""

    def __init__(self, cache_dir, *, target_lufs=-20.0, true_peak_db=-1.5, ffmpeg="ffmpeg"):
        for value, low, high, name in (
            (target_lufs, -40, -10, "sample loudness"),
            (true_peak_db, -9, 0, "sample true peak"),
        ):
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not low <= value <= high):
                raise ValueError(f"Invalid {name} target")
        self.cache_dir = Path(cache_dir)
        self.ffmpeg = ffmpeg
        self.settings = {"version": 1, "target_lufs": float(target_lufs),
                         "true_peak_db": float(true_peak_db), "loudness_range": 50,
                         "sample_rate": 44100, "ffmpeg": str(ffmpeg)}

    def _filter(self):
        return (f"loudnorm=I={self.settings['target_lufs']}:"
                f"TP={self.settings['true_peak_db']}:LRA=50:print_format=json")

    def _run(self, source, audio_filter, output):
        command = [self.ffmpeg, "-hide_banner", "-nostats", "-nostdin", "-n",
                   "-protocol_whitelist", "file,pipe", "-i", str(source),
                   "-map", "0:a:0", "-vn", "-sn", "-dn", "-af", audio_filter]
        if output is None:
            command += ["-f", "null", "-"]
        else:
            command += ["-map_metadata", "-1", "-ac", "1", "-ar", "44100",
                        "-c:a", "pcm_s16le", "-f", "wav", str(output)]
        try:
            result = subprocess.run(command, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f"Unable to normalize voice sample with {Path(self.ffmpeg).name}: {exc}") from exc
        report = result.stderr.decode("utf-8", errors="replace")
        if result.returncode:
            raise ValueError(f"Voice sample normalization failed: {report[-2000:].strip()}")
        matches = re.findall(r'\{\s*"input_i"\s*:.*?\}', report, flags=re.DOTALL)
        try:
            stats = json.loads(matches[-1])
            for key in ("input_i", "input_tp", "input_lra", "input_thresh",
                        "output_i", "output_tp", "target_offset"):
                if not math.isfinite(float(stats[key])):
                    raise ValueError("non-finite loudness")
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Voice sample has no measurable loudness (silent or too short), "
                             "or FFmpeg returned invalid loudness measurements") from exc
        return stats

    def _cached(self, folder, identity, info):
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Voice preview cache must be a regular directory")
        audio, metadata = folder / "sample.wav", folder / "normalization.json"
        if any(path.is_symlink() or not path.is_file() for path in (audio, metadata)):
            raise ValueError("Voice preview cache is incomplete; remove this preview directory and retry: " + str(folder))
        record = read_json(metadata)
        if (not isinstance(record, dict) or record.get("identity") != identity
                or record.get("audio_sha256") != file_hash(audio)
                or _wave_info(audio, 44100) != info):
            raise ValueError("Voice preview cache failed integrity checks; remove this preview directory and retry: " + str(folder))
        return audio.resolve()

    def prepare(self, path):
        """Return a ready preview, without loading models or altering a voice."""
        source = Path(path).resolve()
        info = _wave_info(source, 44100)
        identity = {"source_sha256": file_hash(source), "settings": self.settings}
        # Book path validation protects the work root. Reject symlinks in the
        # preview directory and its work parent, while allowing OS aliases such
        # as macOS /var -> /private/var higher up the absolute path.
        for directory in (self.cache_dir, self.cache_dir.parent):
            if directory.is_symlink():
                raise ValueError("Voice preview cache directories must not be symlinks")
        folder = self.cache_dir / digest(identity)
        if folder.exists() or folder.is_symlink():
            return self._cached(folder, identity, info)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".preview-", dir=self.cache_dir) as temporary:
            # A single atomic directory publication prevents interrupted processing
            # from exposing a partial WAV or a mismatched measurement report.
            staged = Path(temporary) / "ready"
            staged.mkdir()
            audio = staged / "sample.wav"
            measured = self._run(source, self._filter(), None)
            audio_filter = (self._filter() + ":linear=true"
                            f":measured_I={float(measured['input_i'])}"
                            f":measured_TP={float(measured['input_tp'])}"
                            f":measured_LRA={float(measured['input_lra'])}"
                            f":measured_thresh={float(measured['input_thresh'])}"
                            f":offset={float(measured['target_offset'])}")
            result = self._run(source, audio_filter, audio)
            if _wave_info(audio, 44100) != info:
                raise ValueError("Voice sample normalization changed the audio length")
            if file_hash(source) != identity["source_sha256"]:
                raise ValueError("Voice sample changed while preparing its preview")
            if (abs(float(result["output_i"]) - self.settings["target_lufs"]) > 1.0
                    or float(result["output_tp"]) > self.settings["true_peak_db"] + .2):
                raise ValueError("Voice sample normalization could not meet the loudness/peak targets")
            write_json(staged / "normalization.json", {
                "identity": identity, "audio_sha256": file_hash(audio), **info,
                "input_lufs": float(measured["input_i"]),
                "output_lufs": float(result["output_i"]),
                "input_true_peak_db": float(measured["input_tp"]),
                "output_true_peak_db": float(result["output_tp"]),
                "normalization_type": result.get("normalization_type"),
            })
            try:
                os.rename(staged, folder)
            except OSError:
                # Another process may have prepared this exact content meanwhile.
                if not folder.exists():
                    raise
                return self._cached(folder, identity, info)
        return self._cached(folder, identity, info)
