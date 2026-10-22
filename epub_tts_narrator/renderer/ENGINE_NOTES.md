# Local voice design and Drama 3 rendering

`drama3_mlx` uses the unofficial [Sogni MLX port](https://github.com/Sogni-AI/drama3-mlx)
of Fish Audio Drama 3. Fish's own runtime currently requires NVIDIA GPUs; the
adapter runs the same model locally through MLX on Apple Silicon.

The vendored runtime is version `0.1.1+epub.1`, with unchanged inference code from commit
`3b76bf84714d3766d7872f3409639a438e296cc8`. It supports Python 3.10 or newer
and uses MLX 0.32.4, MLX-LM 0.31.3, and stable MLX-Audio `>=0.6.1,<0.7`.
Only upstream dependency metadata and the package version were changed; see
[provenance](../../vendor/drama3-mlx/PROVENANCE.md). The renderer itself
requires Python 3.11 or newer; Python 3.12 is suitable for a dedicated environment.
The recommended [official model](https://huggingface.co/fishaudio/drama-3)
revision is `b2b5ee998034b2975720906c815e7fb0e9f007bd`. Its weights and codec
occupy about 41.5 GB. The port reports 48–55 GB peak memory for q8 on an M5 Max.
Those are upstream measurements, not a benchmark of this renderer.

The adapter owns one model instance and caches reference conditioning by the
audio content hash and exact transcript. It exposes `close()` to release those
objects. Use one synthesis worker initially.

## Voice behavior

Drama 3 supports character casting from text descriptions through its scene
format. The description belongs in the system turn's character list, and the
user turn contains a named script line. The ordinary MLX TTS API hardcodes a
generic system instruction and adds a speaker token, so calling it without a
reference does not exercise that scene format. `drama3_design.py` now constructs
the documented scene prompt directly and uses the installed frame generator
and codec. [Drama 3 technical report, Appendix B, Figure 13](https://arxiv.org/html/2610.18010v1#A2)

Configure `[voice_design] type = "engine"` to use local Drama 3 for both casting
and later reference cloning. Casting supports one character and one script line
(which may contain several sentences). It does not implement multi-character
scene output: the pinned sampler cannot emit the required audio separators.
Whitespace in descriptions and scripts is collapsed to keep a single scene
line; words and punctuation are preserved. Each take records the full applied
prompt and a template/token hash. A separate `voice_design_identity` includes
the scene template hash without invalidating unchanged reference-clone caches.

The earlier unconditioned startup samples did not send character descriptions
to the model at all. Their failures were not evidence of a model limitation.
Existing selected recordings remain in use until explicitly regenerated.

An optional alternative designer is `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`
through the shared MLX-Audio runtime. The model and speech tokenizer
occupy about 4.52 GB; the pinned model revision is
`7d3824abff87e49756bb0f83fb5411de75d160c4`. Load the downloaded local directory
with `mlx_audio.tts.utils.load_model`, then call
`model.generate_voice_design(text=..., instruct=..., language="English")`.
The results contain mono audio at 24 kHz. [MLX API documentation](https://github.com/blaizzy/mlx-audio/blob/main/docs/models/tts/qwen3-tts.md#voicedesign-create-any-voice),
[model files](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16/tree/7d3824abff87e49756bb0f83fb5411de75d160c4)

Configure the designer independently of `[engine]`:

```toml
[voice_design]
type = "qwen3_mlx"
checkpoint = "./models/qwen3-voice-design"
language = "English"
temperature = 0.7
max_tokens = 2048
```

For either designer, the design prompt and spoken script are separate inputs. The resulting
recording passes the same acoustic and independent transcript checks as other
references before it becomes a saved voice. Later Drama 3 synthesis receives
that recording and transcript, rather than designing a new voice on each line.
Qwen documents this design-then-clone pattern for persistent characters; using
the resulting waveform as Drama 3's reference connects the two models.
[Qwen's workflow](https://github.com/QwenLM/Qwen3-TTS#voice-design-then-clone)

The voice record must retain the designer model and applied description
separately from the Drama 3 rendering identity. Applying a description means
the designer received it; it does not mean a classifier or a human verified
every audible trait. ASR verifies words, not character identity or performance.
Prompts request one clean, dry voice without music or sound effects, but the
current quality checks do not independently classify background music.

The reference clip and transcript are portable assets. Codec tokens are a
runtime cache and do not define the durable voice contract. Saved references
remain fixed across books until an explicit regeneration replaces the active
selection; previous samples remain available for recovery.

Narration, dialogue, and other roles attached to one explicit character ID use
that character's canonical reference. Additional sharing uses
`[voices.aliases]`, mapping an alias voice ID to its canonical voice ID.
Character names are not used to infer voice sharing, and voice aliases do not
merge distinct analyzer character records.

## Failure behavior

The upstream `accept` callback returns the final rejected take if all retries
fail. This adapter does not use that callback. It calls `generate` exactly once
with `single_chunk=True`, rejects missing or multiple segments, requires
`finish_reason == "eos"`, and rejects short, malformed, or non-finite audio.
Successful inference is **not** proof of correct speech: the renderer must
perform silence/content validation and decide whether to retry before accepting
or publishing the take.

Untrusted book text cannot insert square-bracket directions or angle tags.
Explicit delivery directions use a small documented allowlist: whisper,
nervous, sigh, and laugh (with simple spelling variants). Other plain-text
directions are reported as unapplied. Source text is never silently deleted
to make it acceptable to the model.

The probe script can also test separate main-token and codebook top-k caps
through `drama3_sampling.py`. This is an experimental proxy, not part of the
production engine. Removing the main-token cap fixed some bad takes but caused
other failures in the controlled test; it is not a demonstrated silence cure
for this MLX setup. Normal casting and cloning keep the configured sampling
settings and always require acoustic and transcript verification.

Take identity includes runtime version/revision, actual checkpoint file hashes,
sampling settings, and output encoding. Model file hashes are cached only while
size, nanosecond modification/change times, and inode remain unchanged. The
Hugging Face revision is recorded when local download metadata establishes it;
arbitrary local checkpoints are not labelled as the recommended release.
