"""Chapter-owned accepted history; inference receives an immutable context snapshot."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path

from .common import digest, file_hash, regular_file


_CURRENT = ContextVar("accepted_chapter_continuity", default=None)


def enabled(config):
    mode = config.get("render", {}).get("continuity", "off")
    if mode not in ("off", "chapter"):
        raise ValueError("render.continuity must be off or chapter")
    return mode == "chapter"


def require_support(config, engine):
    if enabled(config):
        if getattr(engine, "capabilities", {}).get("chapter_continuity") is not True:
            raise ValueError("render.continuity=chapter requires an explicitly supported local runtime")
        check = getattr(engine, "check_continuity", None)
        if callable(check):
            check()


def current():
    return _CURRENT.get()


class ChapterContinuity:
    """Only the sequential chapter coordinator advances first/latest accepted takes."""

    def __init__(self, engine_identity):
        self.runtime = digest({key: value for key, value in engine_identity.items()
                               if key != "cache_model_alias"})
        self.history = {}

    def reset(self):
        self.history.clear()

    @contextmanager
    def transaction(self, request):
        assets = ([request["voice_asset_hash"]] if request["kind"] == "speech"
                  else sorted(set(request["voice_assets"].values())))
        asset = assets[0] if len(assets) == 1 else None
        barrier = ("mixed-paragraph" if asset is None else
                   "simultaneous" if "performer" in request else None)
        if barrier:
            self.reset()
        takes = deepcopy(self.history.get(asset, []))
        identity = {"version": 1, "mode": "chapter", "runtime": self.runtime,
                    "policy": "first-latest-fail-on-overflow", "voice_asset": asset,
                    "barrier": barrier,
                    "takes": [take["identity"] for take in takes]}
        bound = dict(request, continuity=identity)
        token = _CURRENT.set({"identity": identity, "takes": takes,
                              "request_binding": digest(bound)})
        try:
            yield bound
        finally:
            _CURRENT.reset(token)
            if barrier:
                self.reset()

    def accept(self, path, accepted):
        from .continuity_codes import load_codes
        scope = current()
        asset = scope["identity"]["voice_asset"]
        metadata = accepted.get("engine_metadata", {})
        marker = metadata.get("continuity_codes")
        # Joins, repairs and best-effort choices remain accepted audio, not
        # evidence for the exact logical text represented by this transaction.
        barriers = {"recovery", "clip_repair", "speech_resolution", "fragment_recovery",
                    "delivery_recovery", "direction_repair", "exhaustion_selection"}
        if asset is None or barriers.intersection(metadata):
            self.history.pop(asset, None)
            return
        if marker is None:
            raise ValueError("Accepted continuity take has no exact generated codec evidence")
        if (accepted.get("report", {}).get("passed") is not True
                or file_hash(regular_file(path)) != accepted.get("audio_sha256")):
            raise ValueError("Invalid continuity accepted audio")
        if (marker.get("context") != scope["identity"]
                or marker.get("request_binding") != scope["request_binding"]
                or marker.get("text") != scope["compiled_text"]
                or marker.get("accepted_audio_sha256") != accepted["audio_sha256"]):
            raise ValueError("Continuity generation does not match the accepted request")
        actual_engine = metadata.get("worker_engine", accepted["request"]["engine"])
        runtime = digest({key: value for key, value in actual_engine.items() if key != "cache_model_alias"})
        if marker.get("runtime") != runtime:
            raise ValueError("Continuity accepted runtime provenance changed")
        codes_path = Path(path).parent / marker["file"]
        load_codes(codes_path, marker)
        entry = {"path": str(codes_path), "marker": deepcopy(marker), "identity": {
            "codes_sha256": marker["sha256"], "text_sha256": digest(marker["text"]),
            "runtime": marker["runtime"], "request_hash": accepted["request_hash"],
            "audio_sha256": accepted["audio_sha256"], "acceptance_sha256": digest(accepted)}}
        previous = self.history.get(asset, [])
        self.history[asset] = [previous[0], entry] if previous else [entry]
