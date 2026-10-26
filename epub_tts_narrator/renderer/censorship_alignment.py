"""Known-text timing for accepted speech; never a speech acceptance decision.

Call on the verifier's inference thread after chapter verification has drained,
before submitting CPU assembly. MLX Whisper's shared ModelHolder is reused and
must not be accessed concurrently by another verifier in the same process.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from threading import RLock
import wave

from .common import digest, file_hash, read_json, regular_file, write_json
from .run_locks import acquire_directory_lock


_FORMAT = "epub-tts-censorship-alignment"
_VERSION = 2
_METHOD = "mlx_whisper_attention_dtw"
_MLX_LOCK = RLock()
_LEXICAL = re.compile(r"[^\W_]", re.UNICODE)
_WORD = re.compile(r"[^\W_]+(?:['’\-‐‑][^\W_]+)*", re.UNICODE)


class CensorshipAlignmentUnavailable(ValueError):
    """Accepted audio remains usable, but automatic masking needs real timings."""


class CensorshipAlignmentFailed(ValueError):
    """Alignment was attempted but cannot safely identify word boundaries."""


def _compact(text):
    return "".join(text.split())


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _covers(expected, transcript):
    """Written lexical coverage, not a fuzzy/phonetic speech acceptance rule."""
    if not isinstance(transcript, str):
        return False
    from .quality import normalize_words
    left, right = normalize_words(expected), normalize_words(transcript)
    return bool(left) and "".join(word.replace("'", "") for word in left) == "".join(word.replace("'", "") for word in right)


def _duration(path):
    try:
        with wave.open(str(path), "rb") as source:
            frames, rate = source.getnframes(), source.getframerate()
            if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getcomptype() != "NONE":
                raise CensorshipAlignmentUnavailable("Censorship alignment requires saved mono PCM16 speech")
    except (wave.Error, EOFError):
        # Saved accepted clips can be losslessly compressed. This import does
        # not allocate a model or decode the audio into a large buffer.
        try:
            import soundfile
            info = soundfile.info(str(path))
            if info.format != "FLAC" or info.subtype != "PCM_16" or info.channels != 1:
                raise ValueError
            frames, rate = info.frames, info.samplerate
        except (ImportError, RuntimeError, ValueError) as exc:
            raise CensorshipAlignmentUnavailable("Censorship alignment requires readable WAV or lossless FLAC speech") from exc
    if frames <= 0 or rate <= 0:
        raise CensorshipAlignmentUnavailable("Censorship alignment cannot locate words in empty audio")
    return frames / rate


def _primary_verifier(verifier):
    seen = set()
    while verifier is not None and id(verifier) not in seen:
        seen.add(id(verifier))
        state = vars(verifier) if hasattr(verifier, "__dict__") else {}
        if (state.get("identity", {}).get("backend") == "mlx-whisper"
                and state.get("model_path") is not None):
            return verifier
        verifier = state.get("primary", state.get("verifier", state.get("_verifier")))
    raise CensorshipAlignmentUnavailable(
        "Censorship needs reliable word timings; automatic forced alignment currently requires the configured local MLX Whisper verifier")


def _serialized_verifier(verifier):
    # Only our known facade may dispatch arbitrary alignment closures. Do not
    # assume a plugin with similarly named attributes owns this runtime API.
    from .pool import _SerializedVerifier
    seen = set()
    while verifier is not None and id(verifier) not in seen:
        seen.add(id(verifier))
        if type(verifier) is _SerializedVerifier:
            return verifier
        state = vars(verifier) if hasattr(verifier, "__dict__") else {}
        verifier = state.get("primary", state.get("verifier"))
    return None


def _model_identity(verifier):
    identity = verifier.identity
    result = {key: identity[key] for key in ("backend", "package_version", "packages", "model_sha256", "language")
              if key in identity}
    if (result.get("backend") != "mlx-whisper" or not isinstance(result.get("model_sha256"), dict)
            or not result["model_sha256"] or not isinstance(result.get("language"), str)):
        raise CensorshipAlignmentUnavailable("Censorship alignment requires a fingerprinted local Whisper model")
    for filename, state in getattr(verifier, "_model_file_state", {}).items():
        current = Path(filename).stat()
        if (current.st_size, current.st_mtime_ns) != state:
            raise CensorshipAlignmentUnavailable("ASR model changed during this run; recreate the verifier before alignment")
    return result


def _windows(text, duration, report):
    if duration <= 30:
        return [{"start": 0.0, "end": duration, "text": text}]
    segments = report.get("segments") if isinstance(report, dict) else None
    if not isinstance(segments, list) or not segments:
        raise CensorshipAlignmentUnavailable(
            "Censorship alignment needs exact transcript-bearing ASR segment boundaries for saved clips longer than 30 seconds; existing speech is retained")
    result, previous = [], 0.0
    for segment in segments:
        if not isinstance(segment, dict):
            raise CensorshipAlignmentUnavailable("Censorship alignment has invalid ASR segment boundaries")
        start, end, part = segment.get("start"), segment.get("end"), segment.get("text")
        if (not _number(start) or not _number(end) or start < previous or end > duration
                or end <= start or end - start > 30 or not isinstance(part, str) or not part.strip()):
            raise CensorshipAlignmentUnavailable("Censorship alignment needs ordered ASR segments of at most 30 seconds with exact text")
        previous = end
        result.append({"start": float(start), "end": float(end), "text": part})
    # Punctuation/case variants do not change word coverage. Rebind each real
    # ASR segment to its exact known-text slice, never to an estimated duration.
    expected = list(_WORD.finditer(text))
    def spelling(word):
        return word.casefold().replace("’", "'").replace("‐", "-").replace("‑", "-")
    spoken = [spelling(match.group()) for item in result for match in _WORD.finditer(item["text"])]
    if spoken != [spelling(match.group()) for match in expected]:
        raise CensorshipAlignmentUnavailable("ASR segment words do not exactly cover the accepted text; censorship timing cannot be guessed")
    count, offset = 0, 0
    for item in result:
        count += len(list(_WORD.finditer(item["text"])))
        end = expected[count].start() if count < len(expected) else len(text)
        item["text"], offset = text[offset:end], end
        if not item["text"].strip():
            raise CensorshipAlignmentUnavailable("ASR segment has no known words to align")
    return result


def _checked_words(words, text, duration):
    if not isinstance(words, list) or not words:
        raise CensorshipAlignmentFailed("Forced alignment returned no word boundaries")
    result, previous = [], 0.0
    for item in words:
        if not isinstance(item, dict):
            raise CensorshipAlignmentFailed("Forced alignment returned an invalid word record")
        word, start, end = item.get("word"), item.get("start"), item.get("end")
        if (not isinstance(word, str) or not word.strip() or not _number(start) or not _number(end)
                or start < previous or end < start or end > duration):
            raise CensorshipAlignmentFailed("Forced word boundaries overlap, are unordered, or leave the audio")
        lexical = _LEXICAL.search(word) is not None
        if lexical and (end - start < 0.019 or end - start > 2.0):
            raise CensorshipAlignmentFailed("Forced alignment produced an uncertain word duration; neighboring dialogue will not be muted")
        record = {"word": word, "start": float(start), "end": float(end)}
        probability = item.get("probability")
        if probability is not None:
            if not _number(probability) or not 0 <= probability <= 1:
                raise CensorshipAlignmentFailed("Forced alignment returned invalid token probabilities")
            # This is transcript likelihood, not a calibrated timestamp score.
            record["probability"] = float(probability)
        result.append(record)
        previous = end
    if _compact("".join(word["word"] for word in result)) != _compact(text):
        raise CensorshipAlignmentFailed("Forced alignment did not preserve the complete known text")
    return result


def _mlx_align(path, text, verifier, windows):
    """One teacher-forced attention pass per real ASR window; no ASR decoding."""
    try:
        import mlx.core as mx
        from mlx_whisper.audio import HOP_LENGTH, N_FRAMES, N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.timing import find_alignment, merge_punctuations
        from mlx_whisper.tokenizer import get_tokenizer
        from mlx_whisper.transcribe import ModelHolder
    except ImportError as exc:
        raise CensorshipAlignmentUnavailable("Install the existing local mlx-whisper verification requirements for censorship alignment") from exc
    with _MLX_LOCK:
        model = ModelHolder.get_model(str(verifier.model_path), mx.float16)
        tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                                  language=verifier.language, task="transcribe")
        audio = load_audio(str(path))
        words = []
        for window in windows:
            tokens = tokenizer.encode(window["text"])
            if len(tokens) + len(tokenizer.sot_sequence) + 2 > model.dims.n_text_ctx:
                raise CensorshipAlignmentUnavailable("Known text exceeds Whisper's alignment context; reliable smaller ASR segments are required")
            start = round(window["start"] * SAMPLE_RATE)
            end = round(window["end"] * SAMPLE_RATE)
            samples = audio[start:end]
            if len(samples) > N_SAMPLES or len(samples) < HOP_LENGTH * 2:
                raise CensorshipAlignmentUnavailable("ASR alignment window is outside Whisper's audio context")
            mel = log_mel_spectrogram(samples, n_mels=model.dims.n_mels, padding=N_SAMPLES)
            real_frames = len(samples) // HOP_LENGTH
            mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
            alignment = find_alignment(model, tokenizer, tokens, mel, real_frames)
            merge_punctuations(alignment, "\"'“¿([{-", "\"'.。,，!！?？:：”)]}、")
            part = [{"word": entry.word, "start": float(entry.start), "end": float(entry.end),
                     "probability": float(entry.probability)} for entry in alignment if entry.word]
            part = _checked_words(part, window["text"], len(samples) / SAMPLE_RATE)
            words.extend({**word, "start": word["start"] + window["start"],
                          "end": word["end"] + window["start"]} for word in part)
        return words


def align_saved_audio(audio_path, expected_text, verifier, cache_dir, *, report=None, runner=None):
    """Return real word timings bound to immutable accepted audio and known text.

    ``verifier=None`` permits cached-only use during CPU assembly. A supplied
    verifier additionally binds reuse to that model. The optional ``runner``
    injection substitutes only inference: (path, text, primary, windows) -> words.
    ``report['segments']`` can supply exact ASR windows for audio over 30 seconds.
    Missing windows trigger audio-only ASR for segmentation, then require exact
    ordered word coverage before known-text alignment. This never re-verifies
    or regenerates the accepted take.
    Independent recognized lexical coverage is mandatory: supplied known tokens
    must not force a missing word onto neighboring speech. A fresh audio-only
    recognition, if necessary, is saved once even when it disproves coverage.
    This report does not change original verification, transcript, or acceptance.
    """
    if not isinstance(expected_text, str) or not expected_text.strip():
        raise CensorshipAlignmentUnavailable("Censorship alignment needs the accepted nonempty spoken text")
    path = regular_file(audio_path)
    duration = _duration(path)
    binding = {"audio_sha256": file_hash(path), "text_sha256": hashlib.sha256(expected_text.encode("utf-8")).hexdigest(), "version": _VERSION}
    directory = Path(cache_dir)
    if directory.is_symlink():
        raise CensorshipAlignmentUnavailable("Censorship alignment cache must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (digest(binding) + ".json")
    primary = _primary_verifier(verifier) if verifier is not None else None
    model = _model_identity(primary) if primary is not None else None
    lock = acquire_directory_lock(target.with_suffix(".lock"))
    try:
        saved = None
        if target.exists() or target.is_symlink():
            saved = read_json(regular_file(target))
            if (not isinstance(saved, dict) or saved.get("format") != _FORMAT
                    or saved.get("binding") != binding):
                raise CensorshipAlignmentUnavailable("Saved censorship alignment belongs to different audio or text")
            if saved.get("status") == "complete" and (model is None or model == saved.get("model")):
                result = saved.get("result")
                if (not isinstance(saved.get("model"), dict) or saved["model"].get("backend") != "mlx-whisper"
                        or not saved["model"].get("model_sha256") or not isinstance(result, dict)
                        or saved.get("result_sha256") != digest(result) or result.get("text") != expected_text
                        or result.get("alignment_method") != _METHOD or result.get("word_timing_status") != "available"
                        or not _covers(expected_text, result.get("alignment_recognized_text"))):
                    raise CensorshipAlignmentFailed("Saved censorship alignment is invalid; original speech has not changed")
                _checked_words(result.get("words"), expected_text, duration)
                return result
        if primary is None:
            raise CensorshipAlignmentUnavailable("No completed alignment is cached; prepare censorship with the configured ASR verifier, without regenerating speech")
        state = {"format": _FORMAT, "binding": binding, "model": model, "status": "pending"}
        if (isinstance(saved, dict) and saved.get("model") == model
                and isinstance(saved.get("windows"), list) and saved.get("windows_sha256") == digest(saved["windows"])):
            state.update(windows=saved["windows"], windows_sha256=saved["windows_sha256"],
                         segment_source=saved.get("segment_source", "saved_alignment"))
        if isinstance(saved, dict) and saved.get("model") == model:
            for name in ("recognition", "coverage"):
                value = saved.get(name)
                if isinstance(value, dict) and saved.get(name + "_sha256") == digest(value):
                    state.update({name: value, name + "_sha256": saved[name + "_sha256"]})
        write_json(target, state)
        try:
            facade = _serialized_verifier(verifier)
            def align():
                if facade is not None and runner is None:
                    # A resolver wrapper can hide the underlying recognizer
                    # from requires_mlx; initialize on its own lane explicitly.
                    facade._runtime.initialize_mlx()
                def recognize_once():
                    if "recognition" not in state:
                        transcribe = getattr(primary, "transcribe", None)
                        if not callable(transcribe):
                            raise CensorshipAlignmentUnavailable("Censorship needs independent audio-only recognition before known-text alignment")
                        raw = transcribe(path)
                        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
                            raise CensorshipAlignmentFailed("Censorship alignment recognition returned no transcript")
                        # Retain no provider/configuration fields, only the
                        # audio-derived transcript and measured segments.
                        recognition = {"text": raw["text"], "segments": raw.get("segments")}
                        state.update(recognition=recognition, recognition_sha256=digest(recognition))
                        write_json(target, state)
                    return state["recognition"]
                coverage = state.get("coverage")
                if coverage is None:
                    transcript = report.get("transcript") if isinstance(report, dict) else None
                    if _covers(expected_text, transcript):
                        coverage = {"text": transcript, "source": "accepted_asr_report"}
                    else:
                        raw = recognize_once()
                        if not _covers(expected_text, raw["text"]):
                            raise CensorshipAlignmentFailed(
                                "Recognized words do not completely cover the known text; censorship will not force a missing word onto neighboring dialogue")
                        coverage = {"text": raw["text"], "source": "fresh_audio_only_asr"}
                    state.update(coverage=coverage, coverage_sha256=digest(coverage))
                    write_json(target, state)
                if not _covers(expected_text, coverage.get("text")):
                    raise CensorshipAlignmentFailed("Saved censorship coverage no longer proves the known words")
                guidance = {"segments": state["windows"]} if "windows" in state else report
                segment_source = state.get("segment_source", "whole_clip" if duration <= 30 else "accepted_report")
                try:
                    windows = _windows(expected_text, duration, guidance)
                except CensorshipAlignmentUnavailable:
                    if duration <= 30:
                        raise
                    # Decode audio alone solely to locate real segment windows.
                    # No known-text prompt and no acceptance-policy changes.
                    raw = recognize_once()
                    windows = _windows(expected_text, duration, raw)
                    segment_source = "fresh_audio_only_asr"
                state.update(windows=windows, windows_sha256=digest(windows), segment_source=segment_source)
                write_json(target, state)
                return (runner or _mlx_align)(path, expected_text, primary, windows)
            words = facade._call("align_saved_audio", align, (), {}) if facade is not None else align()
            words = _checked_words(words, expected_text, duration)
            if file_hash(path) != binding["audio_sha256"]:
                raise CensorshipAlignmentFailed("Saved audio changed during censorship alignment")
            if _model_identity(primary) != model:
                raise CensorshipAlignmentUnavailable("ASR model changed during censorship alignment")
            result = {"text": expected_text, "transcript": expected_text, "words": words,
                      "word_timing_status": "available", "alignment_method": _METHOD,
                      "alignment_segment_source": state["segment_source"],
                      "alignment_recognized_text": state["coverage"]["text"],
                      "alignment_coverage_source": state["coverage"]["source"],
                      "alignment_audio_sha256": binding["audio_sha256"], "alignment_text_sha256": binding["text_sha256"]}
        except Exception:
            # No book text, provider details, or credentials in failure state.
            write_json(target, {**state, "status": "failed", "error_code": "alignment_unavailable"})
            raise
        write_json(target, {**state, "status": "complete", "result": result, "result_sha256": digest(result)})
        return result
    finally:
        lock.rmdir()
