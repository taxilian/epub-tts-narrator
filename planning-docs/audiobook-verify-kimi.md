## Recommended solution

Build an **evidence-driven verification and recovery controller**, not a larger collection of transcript-matching exceptions.

The key change is to stop treating a mismatch as a rejection. Instead, ask:

1. What exactly disagrees?
2. Does the evidence implicate generation, recognition, or neither conclusively?
3. Can the existing audio be vindicated without resynthesis?
4. If not, what is the cheapest materially different recovery?
5. What durable state is needed to resume if anything stops?

No fully automatic system can guarantee correctness on every disputed or nonverbal clip. A small, durable human-review path is still necessary—but it should receive only the residual cases, with precise crops and evidence.

## 1. Create an immutable spoken-content contract

Verification should compare audio against a prepared **spoken expectation**, not directly against either the book text or the TTS input.

Each passage should store:

- Immutable source text and source ranges.
- Actual synthesis text and delivery controls.
- Ordered spoken atoms, with source mappings.
- Accepted pronunciation variants for names and invented words.
- Numeric values, negation, and other protected spans.
- Nonlexical events such as cries, laughs, or `arrgh`.
- Explicit distinctions between lexical content and performance instructions.
- Voice and cloning-reference identities.
- A hash covering the complete preparation revision.

For example, reviewed preparation could represent:

```text
Source:       “Wheerrre issss sheeee?”
Spoken atoms: where / is / she
Delivery:     prolonged vowels, distressed
```

`B-but` might become `but` plus a stutter directive. Whether that stutter is required is a preparation decision; the verifier must not silently decide that it is disposable.

Double Metaphone can remain a cheap way to propose possible equivalences, but it should never authorize acceptance by itself.

## 2. Use complete, ordered, two-way alignment

For every recognizer, align its complete transcript against the complete spoken expectation.

The alignment must account for:

- Every expected atom, in order.
- Every recognized atom, in order.
- Omissions, insertions, substitutions, repetitions, and reordering.
- Speech-bearing audio regions not explained by either sequence.
- Protected discrepancies involving numbers, negation, quantities, and similar high-risk content.

This replaces both substring acceptance and passage-wide WER tolerances. A small total error rate must not excuse an omitted clause or a changed number.

Word timestamps should support cropping and diagnosis, but missing or unusable timestamps should remain a warning—not automatic evidence of bad speech.

### Handling the representative cases

| Case | Appropriate treatment |
|---|---|
| `halfblade` / `half blade` | Safe token-boundary normalization when the resulting character and phonetic sequence is unchanged and all other coverage passes. |
| `seventh` / `7th`, `see see` / `CC` | Context-aware spoken-form expansion, not unrestricted string replacement. |
| `Derrins`, `Vandire`, `Qarm` | Compare against reviewed pronunciation variants or phonetic evidence; unresolved names go to targeted review. |
| `weapons` / `weapon`, `body’s` / `body` | Never apply blanket suffix deletion. Inspect the disputed ending with alternate recognition and an audio crop. |
| `B-but`, `Wheerrre` | Resolve during preparation by separating spoken atoms from delivery instructions. |
| `arrgh` | Route through nonlexical-event verification. A text-only judge cannot establish whether a cry occurred. |
| Omitted clause, repetition, unrelated `thank…` | Localize through full alignment. Recheck the original audio before declaring a generation failure. |

## 3. Combine blind ASR with acoustic evidence

A conservative starting acceptance rule would be:

```text
Accept only when:
    no hard acoustic failure
    AND complete ordered two-way coverage
    AND no unexplained speech
    AND no unresolved protected discrepancy
    AND (
        two architecture-diverse blind recognizers pass
        OR
        one blind recognizer passes
        AND calibrated acoustic verification supports it
    )
```

This is deliberately conservative. Relax it only when your human-reviewed benchmark demonstrates that a cheaper path meets your false-acceptance target.

### Recognizer selection

On Apple Silicon, `whisper.cpp` is a practical primary runtime because it supports Metal and Core ML acceleration. On NVIDIA workers, faster-whisper is a natural deployment option.

However, **two Whisper runtimes or repeated Whisper decodes are not independent observations**. Parakeet TDT provides a useful architectural contrast because it is a Token-and-Duration Transducer rather than another Whisper decoder. Architectural diversity reduces shared failure modes but does not guarantee independence; measure that on your recordings.

Both recognizers should initially run without the expected passage, vocabulary hints, or previous-transcript conditioning. A prompted recognizer can be used later as a separate, explicitly biased diagnostic—not as blind evidence.

### Add contrastive acoustic verification

This is the most promising technical addition beyond secondary ASR.

NeMo Forced Aligner can accept supplied reference text and produce token-, word-, and segment-level timestamps using CTC models. WhisperX also demonstrates the combination of VAD and forced phoneme alignment for word-level timing.

But merely fitting the expected script to the audio is insufficient. Instead:

1. Obtain free, unscripted recognition.
2. Construct competing hypotheses:
   - Expected passage.
   - Each complete blind transcript.
   - Specific omission, insertion, or substitution hypotheses indicated by alignment.
3. Score those hypotheses against the acoustic emissions.
4. Inspect per-span alignment support, unexplained speech intervals, and the expected-versus-alternative score margin.
5. Calibrate those features by model, language, and passage length.

A deleted clause should not be accepted merely because a forced aligner found some path through it. Conversely, a harmless spelling difference should not fail when the expected pronunciation explains the audio as well as the recognized spelling.

Raw acoustic scores are not universally comparable across different sequence lengths. Length normalization and calibration are essential. This component should begin as an experiment, not a new unconditional gate.

## 4. Replace JEV voting with an explicit decision policy

Model confidence is not automatically a calibrated probability. Research on Whisper, for example, has found overconfident incorrect predictions under noisy conditions.

Use the following states:

| State | Meaning | Required next action |
|---|---|---|
| `verified` | Acceptance policy satisfied | Reuse or assemble |
| `accepted_recognition_variation` | Complete coverage using a reviewed spoken-form variation | Record the exact rule and evidence; reuse |
| `verification_unavailable` | Required verifier, model, or service unavailable | Retain audio; retry verification with backoff |
| `recognition_dispute` | Recognizers or evidence families disagree | Recheck existing audio using another method or crop |
| `unresolved_uncertainty` | Available checks cannot establish correctness | Audio-aware review or durable human queue |
| `generation_failure` | Localized defect supported by the validated generation-failure policy | Enter budgeted recovery |
| `acoustic_failure` | Objective corruption, digital silence, severe clipping, or another calibrated hard failure | Repair or regenerate; models cannot override |
| `deferred_needs_attention` | Strategy or budget exhausted | Continue independent work; expose resumable operator handoff |

Important invariants:

- An outage never becomes a rejection.
- Uncertainty never becomes acceptance.
- A saved rejection never later becomes success without new evidence and a new decision record.
- Human acceptance is recorded as evidence, with reviewer, time, reason, and policy version.
- Cloning references use a separate strict policy; relaxed chapter settings do not apply.

### Where audio-aware models fit

Audio-input models such as Qwen3-Omni are worth testing because they can inspect audio directly rather than judge two text strings.

Use them initially only for disputed spans and nonlexical events:

1. First request a blind transcription or vocalization description.
2. Then, if needed, present shuffled competing hypotheses with `none` and `uncertain` options.
3. Do not reveal which hypothesis is the expected one.
4. Do not count repeated calls to the same model as independent votes.
5. Promote the method to automatic acceptance only if it demonstrates additional value on held-out recordings.

Until then, its best use is producing a better human-review crop and diagnosis.

## 5. Make recovery verification-first and budget-aware

The recovery ladder should be:

1. **Recheck existing audio**
   - Retry unavailable services.
   - Run a different recognizer architecture.
   - Re-window around boundaries or long pauses.
   - Inspect only the disputed span.
   - Run acoustic contrastive verification.

2. **Ordinary generation retry**
   - Appropriate when generation is implicated.
   - Change the seed and record the exact strategy identity.
   - Do not burn a generation start merely because ASR disagreed.

3. **Change the likely cause**
   - Reviewed pronunciation respelling.
   - Plain-delivery controls for expressive stalls or vocalization failures.
   - Different chunking or context handling.
   - A failed applied repair escalates immediately.

4. **Split and join**
   - Admit the plan only if its complete estimated cost fits the remaining budget.
   - Reuse accepted children.
   - Verify the actual joined audio again.
   - A rejected join invalidates only that join—not the accepted children or every future whole-passage strategy.

5. **Alternate approved engine**
   - Preserve the selected voice/reference and content contract.
   - Verify the resulting take from scratch.

6. **Defer with an actionable handoff**
   - Continue unrelated passages.
   - Preserve all candidates and evidence.
   - Resume at the failed passage, not at the beginning of the book.

### Improve the 12-start safeguard

Use a shared atomic counter across parent passages, children, and replans. Before every generation start, require:

```text
spent
+ estimated_remaining_cost_of_current_plan
+ reserve_for_a_viable_alternative
<= generation_start_budget
```

The reserve should be computed from the available alternatives rather than hard-coded. This prevents whole-passage retries from making splitting impossible, while still allowing a return to a whole-passage performance.

Give every strategy a durable fingerprint containing its action, target span, cause category, normalized input changes, engine, and voice contract. This prevents superficially different cycles such as repeated `arrgh` respellings that leave the actual delivery cause untouched.

Escalate after three distinct failed takes as your current policy does, but treat the count as a planning signal—not proof that the audio is bad or the passage is irrecoverable.

## 6. Separate JEV’s responsibilities

JEV should not be a hidden dependency for otherwise local operations. Split the current coupling into independent capabilities:

- Secondary-recognition dispatch.
- Pronunciation-variant review.
- Audio-aware review.
- Recovery planning.
- Report summarization.

In particular:

- Parakeet must run when JEV is disabled.
- Local alignment and acoustic checks must run when JEV is disabled.
- Recovery can fall back to a deterministic strategy ladder.
- If an optional judgment service is unavailable, only cases genuinely requiring it should wait.

A text-only JEV may still summarize evidence or propose a recovery plan, but it should never supply the decisive acoustic observation.

## 7. Use durable, append-only state

A local SQLite-backed implementation with immutable content-addressed artifacts is sufficient initially; use the same interfaces for pooled or remote storage.

Persist at least:

- Passage and source-mapping hashes.
- Audio-byte hash and storage location.
- Canonical generation-input hash.
- Engine, model, seed, voice, reference, and settings.
- Every transcript and aligned discrepancy.
- Acoustic measurements.
- Verifier and policy identities.
- Append-only decisions and evidence references.
- Recovery episodes, strategy fingerprints, budgets, and reservations.
- Worker leases, idempotency keys, and checkpoints.

Write audio and evidence atomically, then commit their metadata. Workers should use leases and idempotent operations so a crash cannot duplicate a completed generation or corrupt its state.

For publication, `complete` must mean that every required passage is accepted and every required join or assembly check has passed. `waiting`, `needs_attention`, and `complete` must remain visibly different.

## 8. Build the evaluation before tuning thresholds

### Human-reviewed data

Use saved real recordings from all relevant engines, voices, passage lengths, and join patterns. Add controlled failures for:

- Beginning, middle, and end deletions.
- Clause omission and phrase duplication.
- Reordering and unrelated insertion.
- Number, negation, name, plural, and possessive changes.
- Quiet speech, dramatic pauses, stalls, clipping, and digital silence.
- Boundary truncation and context-dependent join failures.
- Nonlexical vocalizations.
- Cloning-reference transcript mismatches.

Reviewers should first establish what is actually audible, then compare it with the expectation. Keep separate labels for audio truth and each recognizer’s correctness. Preserve genuinely uncertain labels.

Split calibration, development, and final test data by book and preferably by voice or engine, so memorized names and passages do not inflate results.

### Measure both accuracy and completion

Report at least:

- **Defect escape rate:** defective audio automatically accepted ÷ all defective audio.
- **Accepted-error rate:** defective audio among everything automatically accepted.
- **False generation-rejection rate:** good audio diagnosed as generation failure.
- **Avoidable regeneration rate:** starts spent on later-vindicated audio or repeated exhausted strategies.
- **Automatic recovery rate.**
- **Correct book-completion rate.**
- **Terminal failures and manual interventions per book.**
- **Model calls and cost per verified audio hour.**
- **Paid GPU idle time and combined TTS/ASR utilization.**
- **End-to-end completion time and resume time after interruption.**

Define throughput as:

```text
unique finally accepted audio seconds
────────────────────────────────────
total wall-clock seconds, including recovery and paid idle time
```

Count accepted audio only once, regardless of how many times it was verified.

Also plot **coverage versus accepted-error risk**. Include confidence intervals, grouped by book; zero observed escapes in a small sample is not evidence of zero risk.

### Required comparisons

1. Current system.
2. Spoken contract plus complete ordered alignment.
3. Independent secondary ASR.
4. Contrastive acoustic verification.
5. Audio-aware disputed-span review.
6. Calibrated evidence routing.
7. Full recovery and scheduling stack.

For verification comparisons, rerun every policy on exactly the same saved audio and explicitly recompute verdicts—never reuse cached acceptance. For recovery comparisons, run complete recovery, assembly, and publication flows while injecting worker termination, ASR/JEV outages, remote-worker preemption, and storage failures. Sweep several generation-start caps, including 12, rather than assuming it is optimal.

## Best implementation order

1. **Durable evidence model, explicit states, and benchmark replay.**
2. **Spoken-content contract and complete ordered alignment.**
3. **Verification-before-regeneration and JEV dependency separation.**
4. **Budget-aware recovery, join handling, and operator repair queue.**
5. **Independent ASR and acoustic contrastive verification.**
6. **Calibration and selective automation.**
7. **Audio-aware review and scheduling optimizations, only where measurements justify them.**

The highest-value first milestone is not a new model: it is a system where **every nonaccepted candidate has a precise next action, every action is resumable, and every claimed improvement is measured by correctly completed books**.
