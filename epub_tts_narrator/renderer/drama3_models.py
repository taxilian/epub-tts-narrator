"""Supported Drama 3 releases, without importing either inference runtime."""

from dataclasses import dataclass


DEFAULT_DRAMA3_MODEL = "fishaudio/drama-3"
MLX_AUDIO_DRAMA3_VERSION = "0.6.1"
MLX_AUDIO_DRAMA3_REQUIREMENT = f">={MLX_AUDIO_DRAMA3_VERSION},<0.7"


def supports_mlx_audio_drama3(version):
    """Accept stable patch releases in the supported MLX-Audio minor series."""
    if not isinstance(version, str) or not version:
        return False
    # Keep importing the model catalog safe in the base, non-renderer environment.
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version

    try:
        release = Version(version)
    except InvalidVersion:
        return False
    return (not release.is_prerelease and not release.is_devrelease
            and release in SpecifierSet(MLX_AUDIO_DRAMA3_REQUIREMENT))


@dataclass(frozen=True)
class Drama3Model:
    repo: str
    revision: str
    runtime: str
    size: str
    weight_files: tuple[str, ...]


DRAMA3_MODELS = {
    DEFAULT_DRAMA3_MODEL: Drama3Model(
        DEFAULT_DRAMA3_MODEL, "b2b5ee998034b2975720906c815e7fb0e9f007bd",
        "sogni", "about 41.5 GB",
        tuple(f"model-{number:05d}-of-00008.safetensors" for number in range(1, 9)),
    ),
    "mlx-community/fish-audio-drama-3-4bit": Drama3Model(
        "mlx-community/fish-audio-drama-3-4bit", "1ac1b2108b63c2ef5bade463a121569363e293ce",
        "mlx_audio", "about 23.8 GB",
        tuple(f"model-{number:05d}.safetensors" for number in range(1, 7)),
    ),
    "mlx-community/fish-audio-drama-3-8bit": Drama3Model(
        "mlx-community/fish-audio-drama-3-8bit", "fedc7cc92bab5a180709cbaae6208d10b390bb97",
        "mlx_audio", "about 44 GB",
        tuple(f"model-{number:05d}.safetensors" for number in range(1, 11)),
    ),
    "fcmeyer/drama-3-MLX-3bit-HQ": Drama3Model(
        "fcmeyer/drama-3-MLX-3bit-HQ", "fb68eb7bb1786ed7781e5b87d0d2a381f3e7d089",
        "mlx_audio", "about 21.3 GB",
        tuple(f"model-{number:05d}.safetensors" for number in range(1, 6)),
    ),
}


def get_drama3_model(model_id):
    """Reject unsupported names before model setup, imports or downloads."""
    if not isinstance(model_id, str) or model_id not in DRAMA3_MODELS:
        raise ValueError("engine.model must name a supported Drama 3 model: "
                         + ", ".join(DRAMA3_MODELS))
    return DRAMA3_MODELS[model_id]
