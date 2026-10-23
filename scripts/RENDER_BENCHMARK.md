# Comparing local and remote rendering

`benchmark_rendering.py` freezes 24 actual book passages and up to eight current
voice references, then measures each engine with those exact inputs. It does
not alter voice selections, chapter caches, or the book's script. Preparation
uses verified lossless FLAC reference copies suitable for transferring to a
worker. The fixed workload contains 13 short, seven medium and four long
requests; long grouped paragraphs are included when available. These proportions
approximate Book 5, rather than selecting only the easiest requests.

Prepare the corpus once, while reading the book's current saved voices:

```sh
python scripts/benchmark_rendering.py prepare /path/to/book \
  --config epub-tts-narrator.toml --output /path/to/benchmark/corpus
```

When the local renderer is idle, generate the M5 baseline:

```sh
python scripts/benchmark_rendering.py generate /path/to/benchmark/corpus/corpus.json \
  --config epub-tts-narrator.toml --worker local --concurrency 1 \
  --output /path/to/benchmark/m5-first
```

Use the same corpus on the GPU worker. Its renderer config must contain the
configured remote engine in `[render_pool]`; select that worker by its existing
name. Run each concurrency separately with a fresh output directory:

```sh
python scripts/benchmark_rendering.py generate /path/to/benchmark/corpus/corpus.json \
  --config /path/to/worker.toml --worker sglang --concurrency 4 \
  --output /path/to/benchmark/sglang-c4-first
```

Repeat at concurrency 1 and 8. The native serial Fish backend intentionally
rejects concurrency above one; use the batching-capable adapter for those runs.
Warmup initializes every selected reference separately before the measured
requests. Run at least two repetitions, preferably reversing the concurrency
order for the second repetition. Stock SGLang does not support per-request
seeds, so its results remain stochastic even though requested seeds are fixed
and retained. Identical requested seeds also do not promise identical samples
across runtimes or model quantizations.

To measure synthesis and verification together, use `pipeline` with the same
corpus and a new output directory:

```sh
python scripts/benchmark_rendering.py pipeline /path/to/benchmark/corpus/corpus.json \
  --config /path/to/worker.toml --worker drama3-cuda --concurrency 8 \
  --output /path/to/benchmark/sglang-c8-pipeline
```

The selected config controls the verifier device and compute type. A CUDA
comparison uses `type = "faster_whisper"`, `device = "cuda"` and a compatible
`compute_type`, such as `"float16"`, with locally available Whisper weights.
Keep the verifier model and quality policy fixed when comparing CPU and CUDA.
The benchmark creates one verifier and uses the production pool's wrapper to
run recognition on its persistent runtime thread. It does not load a verifier
for each paragraph. Remote synthesis can continue while recognition or text
review is in progress.

The pool's `max_inflight` bounds the number of complete passage transactions;
its worker concurrency separately limits simultaneous synthesis requests. Set
`render_pool.max_inflight` explicitly when comparing admission windows. Voice
warmups and one ASR warmup complete before the measured pipeline starts. Each
measured passage generates once, uses production audio normalization and
quality validation, and retains the generated audio unchanged. The executor
drains before the shared verifier and engine are closed.

Pipeline manifests report **verified audio seconds / pipeline wall second**,
alongside pass rate and generated audio duration. Per-request and per-method
times overlap. Verifier method timings include time waiting for the shared
runtime or text-review queue; their sum can exceed wall time. They are not the
isolated method costs reported by sequential verification. This measures the
first-pass passage pipeline: retries, chapter encoding, publication and book
assembly still require a separate production run. Do not add sequential
generation and verification timings to estimate the measured pipeline result.

Verify all runs with the same verifier model, configuration, and quality policy:

```sh
python scripts/benchmark_rendering.py verify /path/to/benchmark/m5-first/manifest.json \
  --config /path/to/comparison-verifier.toml
```

In `generate`/`verify` mode, verification is a separate phase so engine throughput
and verification cost remain distinguishable. The ordinary acoustic and transcript checks, including
configured secondary recognition and JEV, still apply. There is one generation
per case; rejected audio stays rejected, and there is no automatic regeneration
or pronunciation repair in this first-pass comparison. To compare end-to-end
production retries, use a separate chapter run after this controlled test.

Each manifest records generation makespan, per-request latency, failure counts,
generated audio duration, verified audio duration, engine metadata and optional
local-host NVIDIA memory/utilization/power samples. A sampler running on the Mac
cannot measure a remote GPU; run the generator on the GPU host for those samples.
GPU warmup/model loading and ASR warmup are recorded separately. The M5's unified
memory is not an NVIDIA VRAM measurement; absent samples are unavailable, not zero.

Each verified case also records `verification_stage_seconds`: normalization and
each recognizer or text-review method actually called, including failed calls.
The summary sums these timings across measured cases; ASR warmup has its own
`verification_warmup_stage_seconds`. Missing stages were not called. Stage totals
exclude acoustic checks, comparison bookkeeping, and file handling, so they need
not sum to the complete verification wall time. Instrumentation forwards the
same inputs and results and does not add retries or change verification policy.

The main comparison is **verified audio seconds / generation wall second**,
alongside pass rate and generated audio seconds / generation wall second. Failed
requests cost time and earn no verified audio. **Verified audio seconds / total
second** additionally includes the subsequent verification phase; it is a serial
two-phase measurement, not a claim about production overlap. Parallel request
latencies are never summed as the throughput denominator. Corpus hashes must
match, and engine/model/quantization differences must be reported. Listen to
representative accepted samples: ASR does not measure voice fidelity or acting.

Output directories are deliberately new and separate from production. Retain
the audio and manifests for review; transfer them in a compressed archive rather
than transferring uncompressed WAV files. Remove the benchmark directory when
the comparison is finished. Do not run the local model alongside another active
local book render. This harness never creates, starts or stops cloud resources.
