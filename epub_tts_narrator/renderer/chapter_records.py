"""Format constants, signing, and record builders for remote chapter artifacts.

The chapter executor writes these records; the client validates downloads by
deriving expectations from the same builders. Nothing here trusts the records
themselves: callers supply every value the job spec determines.
"""

from __future__ import annotations

from copy import deepcopy

from .common import digest


CHAPTER_FORMAT = "epub-tts-narrator-chapter"
SYNC_FORMAT = "audiobook-renderer-chapter-sync"
VERIFICATION_FORMAT = "audiobook-renderer-chapter-verification"
MANIFEST_FORMAT = "epub-tts-narrator-chapter-artifacts"


def signed_record(record):
    """The record_sha256 envelope: digest of the record without its signature."""
    return digest({key: value for key, value in record.items() if key != "record_sha256"})


def audio_pointers(frames):
    """Status and filenames implied by whether the chapter contains audio."""
    return {"status": "complete" if frames else "empty",
            "audio_file": "chapter.m4a" if frames else None,
            "lossless_file": "chapter.flac" if frames else None}


def chapter_record(composed, *, spec, pcm_data_sha256):
    record = {key: deepcopy(composed[key]) for key in
              ("id", "number", "title", "frames", "channels", "sample_rate", "segments",
               "pcm_sha256", "encoded_sha256", "normalization", "track_number", "total_chapters")}
    pointers = audio_pointers(composed["frames"])
    record.update(format=CHAPTER_FORMAT, version=1, spec_hash=digest(spec),
                  source_fingerprint=spec["source_fingerprint"],
                  voice_bindings=deepcopy(spec["voice_bindings"]),
                  pcm_data_sha256=pcm_data_sha256,
                  audio_file=pointers["audio_file"], lossless_file=pointers["lossless_file"],
                  status=pointers["status"])
    return record


def sync_record(*, book_id, source_fingerprint, spec_hash, chapter_id, chapter_number,
                sample_rate, frames, precision, audio_file, audio_sha256, segments):
    return {"format": SYNC_FORMAT, "version": 1, "book_id": book_id,
            "source_fingerprint": source_fingerprint, "spec_hash": spec_hash,
            "chapter_id": chapter_id, "chapter_number": chapter_number,
            "sample_rate": sample_rate, "timebase": f"1/{sample_rate}",
            "range_end": "exclusive", "frames": frames,
            "timing_basis": "PCM sample frames after trimming and inserted pauses",
            "precision": precision, "audio_file": audio_file, "audio_sha256": audio_sha256,
            "playback_note": "Use codec delay/padding-aware playback; word alignment is not provided.",
            "segments": deepcopy(segments)}


def verification_record(*, spec_hash, context, acceptance):
    return {"format": VERIFICATION_FORMAT, "version": 1, "spec_hash": spec_hash, "passed": True,
            "engine": context["engine"], "verifier": context["verifier"],
            "quality_version": context["quality_version"], "quality": context["quality"],
            "acceptance": acceptance}


def manifest_record(*, identity, spec, status, artifacts):
    return {"format": MANIFEST_FORMAT, "version": 1, "identity": identity,
            "spec_hash": digest(spec), "source_fingerprint": spec["source_fingerprint"],
            "status": status, "artifacts": artifacts}


def expected_artifact_names(frames):
    return {"chapter.json", "sync.json", "verification.json"} | ({"chapter.m4a", "chapter.flac"} if frames else set())
