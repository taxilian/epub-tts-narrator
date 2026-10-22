# Drama 3 quantized-model sanity check

`benchmark_drama_quantized.py` compares the existing Sogni q8 renderer with a
preconverted Drama 3 checkpoint loaded through MLX-Audio. It runs outside the
book pipeline, reads two existing reference voices, and writes its own WAVs and
JSON evidence. It never selects voices or changes rendering configuration.

This is a small compatibility and performance check: one narration passage and
one two-speaker exchange, each with a warmup and three measured seeds. It does
not establish long-chapter reliability or voice similarity.

## Observed result — October 21, 2026

The MLX-Audio 4-bit checkpoint passed this sanity check on an Apple M5 Max with
128 GiB unified memory. All six measured recordings and both warmups passed
strict normalized word-error-rate zero and the renderer's acoustic checks.

| Runtime/checkpoint | Measured passes | Time for six requests | Accepted audio / wall second | Peak generation memory | Load time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing Sogni q8 | 6/6 | 21.99 s | 1.50× | 49.53 GB | 8.67 s |
| MLX-Audio, `mlx-community/fish-audio-drama-3-4bit` | 6/6 | 14.62 s | 2.42× | 30.16 GB | 9.30 s |
| Existing Sogni q4 option | 4/6 | 24.98 s | 0.93× | 34.15 GB | 8.15 s |

The candidate used **33.5% less generation time for the same six requests** and
**39.1% less peak MLX memory** than q8. Its audio was slightly longer, so the
61.6% increase in audio-duration throughput includes some pacing difference.
These are local measurements of different runtime/checkpoint combinations, not
an isolated quantization experiment or an API-dollar-cost comparison.

The final comparison ran sequentially after the download completed. Earlier
exploratory q8/q4 runs overlapped the download and are retained separately, but
are excluded from this table. Sogni used the existing environment with MLX
0.32.3 / MLX-Audio 0.5.7; the candidate used an isolated MLX-Audio 0.6.0 / MLX
0.32.4 environment. The candidate's downloaded weights, including codec, occupy
23.76 GB. No production configuration or voice selection was changed.

Both Sogni q4 runs produced internal silence stalls for dialogue seeds 1729 and
1730. All other q4 measured takes passed. The candidate had no long silent tails
(at most 0.09 s under a -60 dBFS window check); dialogue seed 1731 had an internal
pause of about 1.79 s, below the existing 2.5 s rejection threshold. Listening
still needs to judge whether its pacing and voice identity are suitable.

The linked 8-bit and 3-bit HQ MLX checkpoints were researched but **not run**.
The result supports trying this 4-bit checkpoint; it does not establish full-book
reliability. The renderer now supports it through `engine.model`; see
[model selection and installation](../RENDERER.md#choose-the-speech-model).

Local evidence (generated files are intentionally ignored by Git):

- [Final q8 control manifest](../output/quantized-sanity/sogni-q8-control/manifest.json)
- [MLX-Audio 4-bit manifest](../output/quantized-sanity/mlx-4bit/manifest.json)
- [Final Sogni q4 control manifest](../output/quantized-sanity/sogni-q4-control/manifest.json)
- Narration, seed 1729: [q8](../output/quantized-sanity/sogni-q8-control/checked/narration-seed-1729.wav), [MLX 4-bit](../output/quantized-sanity/mlx-4bit/checked/narration-seed-1729.wav)
- Dialogue, seed 1729: [q8](../output/quantized-sanity/sogni-q8-control/checked/dialogue-seed-1729.wav), [MLX 4-bit](../output/quantized-sanity/mlx-4bit/checked/dialogue-seed-1729.wav)
- Candidate pause example: [dialogue, seed 1731](../output/quantized-sanity/mlx-4bit/checked/dialogue-seed-1731.wav)

Every take, including warmups and rejected q4 recordings, is retained. Reference
audio hashes match before and after every run. Seven focused offline harness
tests pass, including failure-cost accounting and real FLOAT-to-PCM conversion.

## Setup

All models now share MLX-Audio `>=0.6.1,<0.7` and the dependencies in
`renderer.requirements.txt`. These commands assume Apple Silicon, Python 3.12,
`uv`, and FFmpeg. An optional benchmark environment uses that same dependency
set; the historical measurements above retain their recorded runtime versions.

```sh
uv venv --python 3.12 .venv-quantized
uv pip install --python .venv-quantized/bin/python -r quantized.requirements.txt

.venv-quantized/bin/hf download mlx-community/fish-audio-drama-3-4bit \
  --revision 1ac1b2108b63c2ef5bade463a121569363e293ce \
  --local-dir models/fish-audio-drama-3-4bit
```

Model choices:

| Checkpoint | Runtime and recipe |
| --- | --- |
| [mlx-community/fish-audio-drama-3-4bit](https://huggingface.co/mlx-community/fish-audio-drama-3-4bit) | MLX-Audio; 4-bit routed experts, 8-bit dense projections and audio tower |
| [mlx-community/fish-audio-drama-3-8bit](https://huggingface.co/mlx-community/fish-audio-drama-3-8bit) | MLX-Audio; 8-bit routed experts, remaining weights bf16 |
| [fcmeyer/drama-3-MLX-3bit-HQ](https://huggingface.co/fcmeyer/drama-3-MLX-3bit-HQ) | MLX-Audio; mixed 3/4-bit routed experts, selected other weights at higher precision |

The runner accepts any of these local checkpoints. Substitute its directory,
model ID and exact downloaded commit when testing another model. Downloading or
reading a model card is not evidence that it passed this test. The existing
Sogni baseline already quantizes experts to q8; it is not an unquantized control.
An additional `--backend sogni --experts q4` run can check the existing runtime's
lower-precision option without downloading a new checkpoint. This is a different
recipe from MLX-Audio's published 4-bit model and must be labeled separately.

## Generate and verify

Each voice directory must contain `voice.json` and `reference.wav`. The runner
checks the saved audio SHA-256 before use and records whether the reference audio files
are unchanged afterward. Use two known synthetic demonstration voices or
appropriate consented recordings with accurate transcripts.

Set paths for your local installation:

```sh
renderer_python='/absolute/path/to/existing/.venv-renderer/bin/python'
baseline_checkpoint='/absolute/path/to/existing/models/drama-3'
mara_voice='/absolute/path/to/first/voice-directory'
lena_voice='/absolute/path/to/second/voice-directory'
whisper_checkpoint='/absolute/path/to/local/whisper-snapshot'

"$renderer_python" scripts/benchmark_drama_quantized.py generate \
  --backend sogni --checkpoint "$baseline_checkpoint" \
  --model-id fishaudio/drama-3 \
  --revision b2b5ee998034b2975720906c815e7fb0e9f007bd \
  --voice-dirs "$mara_voice" "$lena_voice" \
  --output output/quantized-sanity/sogni-q8

.venv-quantized/bin/python scripts/benchmark_drama_quantized.py generate \
  --backend mlx-audio --checkpoint models/fish-audio-drama-3-4bit \
  --model-id mlx-community/fish-audio-drama-3-4bit \
  --revision 1ac1b2108b63c2ef5bade463a121569363e293ce \
  --voice-dirs "$mara_voice" "$lena_voice" \
  --output output/quantized-sanity/mlx-4bit

# Run after both TTS processes have exited, so Whisper does not compete for RAM.
"$renderer_python" scripts/benchmark_drama_quantized.py verify \
  output/quantized-sanity/sogni-q8/manifest.json \
  output/quantized-sanity/mlx-4bit/manifest.json \
  --whisper "$whisper_checkpoint"
```

Use a fresh output directory for every generation run; existing runs are refused.
The `--revision` value records provenance supplied by the caller. Confirm it
against the download metadata; it is not a hash verification of arbitrary local
weights. Manifests record installed package versions and VCS source metadata,
hardware, references, sampling, timings, errors, audio hashes, and transcripts.

## Read the results

- **Load seconds:** imports, weight loading and GPU synchronization, before the
  first synthesis. Downloads and environment installation are separate.
- **Generation seconds:** the full public synthesis call, GPU synchronization,
  and WAV writing. The first call for each reference combination is retained as
  a warmup. It includes encoding/prefill and is excluded from steady-state speed.
- **Audio seconds per wall second:** total measured audio duration divided by
  total measured synthesis time. Higher is faster. The reciprocal is reported as
  wall seconds per audio second; this avoids ambiguity in the term “RTF.”
- **Accepted throughput:** only audio passing the strict word/acoustic checks
  contributes to the numerator; time spent on failed attempts stays in the
  denominator. It is withheld until verification completes. Warmups are excluded.
  Duration includes natural pauses. Inspect unusually long trailing silence
  separately: the renderer checks internal stalls and total silence fraction,
  but does not impose a dedicated trailing-silence limit.
- **Peak generation GB:** peak MLX allocation, including warmups, in decimal GB.
  This is not process RSS or whole-system memory; model-load peak is separate.
- **Speech checks:** raw float WAVs are retained, with PCM16 copies in `checked/`
  for the renderer's acoustic checks and audio-only Whisper transcription. The
  expected text is never passed as an ASR prompt. Normalized WER must be zero.
  Hitting the 600-frame cap fails even if the recognized words match. MLX-Audio
  does not expose an EOS reason here, so EOS below the cap is explicitly inferred
  from its generation loop and token count. No automatic retry hides a failure.

Listen to the saved clips for stable identity, speaker changes, natural pacing,
artifacts and expression. Transcription cannot judge those properties, and an
ASR mismatch can be a recognition error rather than incorrect speech.

Both backends use temperature 1.0, top-p 0.9, top-k 30, the same reference WAVs,
and measured seeds 1729–1731. Different implementations do not produce paired
waveforms from the same seed. This measures the practical runtime/checkpoint
combination, not the isolated causal effect of changing quantization. Run models
sequentially with other GPU work stopped before treating small speed differences
as meaningful. Neither local RAM savings nor speed measures API-dollar savings.

Offline checks for the harness:

```sh
.venv-quantized/bin/python -m unittest discover -s tests -p test_quantized_benchmark.py -v
```

The WAV-conversion regression uses FFmpeg when available; arithmetic and
reference checks do not require MLX or model downloads.
