# Verifying and Recovering Generated Audiobook Speech: Evidence-Backed Options for a Fish Audio Drama 3 Pipeline

The approach most likely to cut both incorrect acceptances and soundalike rejections is to stop asking any single transcript, or any text-only judge, for a verdict. Instead, collect independent per-span evidence: two different free-decoding ASR systems, a free-decoding phoneme recognizer, and a script-conditioned CTC likelihood-ratio check. Align all of it in phoneme space against the reviewed expected speech. Each span then gets one of five explicit states, and a failure-type-specific recovery action follows. Audio LLMs, more blind LLM votes, and blanket WER tolerances do not resolve the central "TTS or ASR?" uncertainty and should be advisory at most. The second big lever is architectural. Make joins verifiable by construction, so that a whole-join recognition mismatch inside byte-identical, already-verified audio triggers a recheck rather than a regeneration. Bind every verdict to audio hashes and policy versions in MongoDB.

## TL;DR

- **Verification:** Replace Double Metaphone plus text-only JEV with phoneme-space, in-order coverage alignment. Feed it free ASR (Whisper plus Parakeet), a free phoneme recognizer (wav2vec2 XLSR-53 eSpeak CTC), and a script-vs-free CTC likelihood ratio, a goodness-of-pronunciation (GOP)-style check that, unlike plain forced alignment, can detect deletions and insertions. Protect numbers and negation lexically. Verify vocalizations acoustically, not lexically. Treat outages as "unavailable", never as "bad audio".
- **Recovery:** Classify the failure first (stall/silence, local omission, repetition, name pronunciation, missing vocalization, recognition-side), then change the input that matches it. Drama 3 specifically shows mid-line stalls at Fish's default sampling, so stall handling deserves its own cheap path. Replace "whole join must pass" with "parts verified by byte identity, plus boundary windows freshly verified, plus whole-join ASR as extra evidence".
- **Measurement:** Build a span-labeled, human-reviewed set that keeps explicit "uncertain" labels. Report false acceptance with finite-sample upper bounds (not naive thresholds), per-failure-class recovery curves to evaluate the 12-start budget, and verified audio seconds per wall-clock second, including paid GPU idle time. Run every candidate verifier against the same saved bytes.

## Key Findings

1. **WER-style checks are known to miss exactly the failures you care about.** In SP-MCQA (arXiv 2510.26190), FishSpeech V1.4 had the lowest WER of the four TTS models tested (5.739%) but the worst SP-MCQA accuracy (81.194%). The authors attribute this to "mid-sentence word drops and cut-offs due to normalization issues". INTP's human study found that WER-selected "positive" and "negative" samples often disagreed with human reading-accuracy labels. For example, 45.7% of WER-negative ARS samples were judged accurate by humans.
2. **Autoregressive TTS failures are local at onset and then propagate.** Hallucination research on LM-based TTS (GOAT, EMNLP 2025; Experience-Calibrated Contrastive Decoding, 2026) describes repetition/deletion errors, mid-generation collapse, and a "localized onset pattern" that autoregression turns into "a persistent content error". This supports span-local detection and split-around-the-span recovery, not whole-passage retries.
3. **Drama 3 has a documented, cheap-to-detect failure mode: stalls.** In Sogni's MLX integration on an M5 Max, only 14 of 24 takes of Fish's own example line were complete at Fish's SDK defaults (temperature 0.7, top-p 0.7). They switched to temperature 1.0 / top-p 0.9 and re-sampled any chunk containing 2 s or more of silence, then got 16 of 16 complete requests (one stall re-sampled). This is a single small community report and should be treated as a hypothesis to test, not a setting to adopt.
4. **Fish's own runtime ignores `repetition_penalty`.** Fish's docs say S2-Pro and Drama 3 "both ignore it and use built-in repetition-aware sampling". Tuning it is wasted effort. Fish also notes that `--seed` "does not fix the timbre", and MLX seeds are not the same as CUDA seeds. So seeds are not portable between the M5 Max and Runpod, and cache keys must include backend and quantization.
5. **Plain forced alignment cannot detect wrong speech; CTC GOP with a free-path denominator can.** Cao et al. (Interspeech 2024) state that traditional alignment-based GOP "preclude[s] the possibility of deletion and insertion errors". Their alignment-free CTC variant, which compares the canonical path against "any phoneme sequence" between left and right context, reached AUC 0.915 on real child-speech errors, against 0.796 for the GOP-DNN baseline.
6. **Audio LLMs are weak, text-biased judges of "does this audio match this text".** The ALME benchmark ("When Audio-LLMs Don't Listen", arXiv 2602.11488; 57,602 conflict stimuli in 8 languages) found that Gemini 2.0 Flash followed conflicting text 16.6% of the time and GPT-4o 23.2%, even when told to trust the audio. When the audio was replaced by its transcript, those rates were only 1.6% and 0.9%. Forcing explicit transcription raised the rate by 14%. On SpeechJudge (ICLR 2026), the best zero-shot judge, Gemini-2.5-Flash, agreed with humans on 69.1% of 1,000 pairs, below a fine-tuned Bradley-Terry reward model (72.7%) and SpeechJudge-GRM (77.2%). On non-verbal vocalization quality, the best LLM judge (Gemini 3 Flash) correlated with experts at only 0.468.
7. **Whisper's silence hallucinations are well characterized and filterable, but only partly.** Barański et al. (ICASSP 2025) found that Whisper large-v3 produced hallucinations on 121,378 of 301,317 non-speech clips (40.3%), with "thank you" making up 24.76% of hallucinations. Careless Whisper (Koenecke et al., FAccT 2024) found that 1.4% of audio segments sent to the 2023 Whisper API produced a hallucination. Of 312 hallucinated transcriptions, 38% contained harms. Hallucinations were associated with longer non-vocal durations, which is exactly the dramatic-pause case.

## Details

### 1. Verification evidence sources

**Priority A: Two architecturally different free-decoding ASR systems, plus VAD gating.**
- *Why it addresses the spec:* Limitation 4 says secondary recognizers "share biases". Whisper (an attention encoder-decoder with an LM-like decoder) and Parakeet TDT (a transducer) fail differently. Whisper invents fluent text over silence. Transducers tend to drop rather than invent. Disagreement between them is informative evidence, whereas repeated deterministic decoding of the same model adds almost nothing.
- *Tools:* `parakeet-mlx` (default `mlx-community/parakeet-tdt-0.6b-v3`, word timestamps, Python CLI/API) and `mlx-whisper` on the M5 Max. In one third-party M4 benchmark, parakeet-mlx returned a short utterance in 0.4995 s against 1.0230 s for mlx-whisper large-v3-turbo. Treat that as indicative only. VoiceStudio's notes put Parakeet MLX at about 2 GB of unified memory.
- *Whisper hallucination controls:* gate on Silero VAD before Whisper. Defaults are threshold 0.5, `min_silence_duration_ms` 100, `speech_pad_ms` 30; tune them on your quiet-speech clips. Use `no_speech_prob` and a Bag-of-Hallucinations filter, the frequent non-speech outputs published by Barański et al. A BoH match is only a flag ("possible recognizer insertion"), never a deletion of evidence: a character might really say "thank you".
- *Failure modes:* Parakeet v3 has no prompt or bias list, so invented names will be spelled arbitrarily. That is fine if matching is phonemic (Section 2). Both systems normalize disfluencies, so "B-but" becomes "But" in each.
- *Test:* on the labeled set, measure per-span recall of generation defects when (a) Whisper alone, (b) Parakeet alone, or (c) either one flags the span. Measure soundalike false rejections separately.

**Priority B: A free-decoding phoneme recognizer as a third, lexicon-free observer.**
- *Model:* `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (Xu, Baevski & Auli, Interspeech 2022), a CTC model that outputs eSpeak-style IPA. An ONNX int8 export exists. Its MatMuls are quantized and convolutions stay fp32, because quantizing everything "cost 2 PER points". That means it can run directly in Node via onnxruntime-node with no Python sidecar.
- *Why:* it has no vocabulary, so "Qarm" cannot be "corrected" into "calm". It directly addresses the invented-names problem and the lossy-code problem (limitation 3).
- *Evidence and caveats:* the int8 copy scored PER 50.6 (narrow scoring) across 95 languages on VoxAngeles. That is a hard multilingual benchmark, not English audiobook speech, but it shows that raw phoneme output is noisy. Use it as evidence inside alignment, never as a transcript. It "leans towards dictionary pronunciations" because it was trained on eSpeak labels.

**Priority C: Acoustic checks that tolerate drama.**
- Use `ffmpeg -af ebur128=peak=true` for integrated loudness, LRA and true peak. EBU R128 gating ignores audio below −70 LUFS and 10 LU below the running integrated level, so long pauses do not drag the measurement down. The production ceiling is −1 dBTP.
- Detect clipping as runs of consecutive full-scale samples (for example via `astats`), not from peak level alone.
- Define a stall as VAD-silence exceeding an expected-pause model conditioned on punctuation and delivery cues. An ellipsis or em-dash allows more; mid-clause allows less. Do not use a fixed seconds threshold.
- Estimate expected duration from phoneme count times the voice's measured speaking rate. Flag only gross deviations, such as more than 2× or less than 0.5×. Duration is evidence for runaways and truncations, not proof of correct words.
- *Test:* the false-rejection rate on human-labeled "intentional long pause" and "whispered/quiet" clips is the key metric.

### 2. Matching and alignment

**Priority A: Phoneme-space, in-order coverage alignment (replaces Double Metaphone).**
- *How:*
  - Run G2P on the reviewed expected speech with eSpeak-ng. Add a reviewed pronunciation lexicon that can hold multiple variants per word, for names like Derrins, Vandire, Qarm and Jas.
  - Run G2P on each recognizer's text output as well, so "Darin's", "Van Dyer", "Khorm", "Jazz", "CC" and "7th" are converted to phonemes after number and letter-name normalization.
  - Globally align the expected phonemes against each observer's phonemes with a weighted edit distance whose substitution cost is articulatory-feature distance. PanPhon maps over 5,000 IPA segments to 21 articulatory features and provides `weighted_feature_edit_distance`. Its feature table is a data file you can port to TypeScript.
  - Ignore word boundaries during alignment. "halfblade" and "half blade" then have identical phoneme strings.
  - The self-refining ASR-data paper used PER on phonemized text for the same reason: it "discount[s] orthographic variations, such as homophones".
- *Why it beats current approaches:* the alignment must consume every expected phoneme in order, so complete coverage is structural (limitation 2). Costs are graded instead of being an equal/not-equal code collision (limitation 3). Thresholds are **local**: score each word span and each window of roughly 5 to 8 phonemes, never a passage-level WER. A dropped clause in a long paragraph is then a long run of deletions that cannot be averaged away.
- *Protected tokens:* numbers, ordinals, negation (not, n't, no, never, nor, without), and reviewed "key" words require a lexical match after normalization, or a strong phoneme match from two observers. "Can" versus "can't" is a single /t/ at the phoneme level, so phoneme similarity alone must never excuse it.
- *Weak endings ("weapons" vs "weapon"):* make final-consonant deletions, plural -s and possessive -'s a reported, low-cost "weak ending" class. Accept them only if at least one observer, or the CTC likelihood ratio (below), supports the ending. Otherwise mark the span UNRESOLVED and send it to the listening queue. Decide per book whether that class is tolerable.
- *Failure modes:* eSpeak's guesses for fantasy names may differ from Drama 3's actual rendering. The lexicon fixes this once per name, and the evaluation set should include each recurring name. Different G2P paths for recognized and expected text can introduce spurious differences, so use the same G2P for both.

**Priority B: Script-conditioned CTC likelihood ratio (GOP-style) to localize mismatches.**
- *How:*
  - Compute CTC emissions from a CTC acoustic model: the phoneme model above, or MMS_FA through `torchaudio.pipelines` for character-level alignment.
  - For each expected word span, compare the log-probability of the canonical path against the log-probability of a path that allows any phoneme sequence between the same left and right context. This is Cao et al.'s GOP-CTC-AF-SD/SDI formulation. torchaudio's `<star>` token gives a cheap approximation: align expected text with `<star>` wildcards between segments, and measure how much audio the wildcards absorb.
  - Unexplained audio after the last aligned word, which LocalText2Voice also measures, is a direct test for runaway continuations and appended "thank…" hallucinations.
- *Why it is not "merely fitting the script":* a large negative ratio means the audio prefers some other content. A wildcard that absorbs seconds of voiced audio indicates an insertion. A near-zero-duration span with low posterior indicates a deletion. This is the only listed method that is conditioned on the script and still detects wrong speech.
- *Evidence:* Cao et al.'s alignment-free GOP reached AUC 0.915 against 0.796 for the baseline on real errors. That was child read speech, so treat it as transfer evidence rather than proof for TTS.
- *Caveats:* one community port notes that `torchaudio.functional.forced_align` was deprecated from torchaudio 2.6 onward with removal planned. Pin versions, or use the standalone `ctc-forced-aligner`. CTC "peaky" outputs make boundaries imprecise, which is fine for span localization but not for word timestamps.
- *Test:* ROC/AUC per error type (omission, insertion, substitution, repetition) on controlled failures, plus false rejections on the soundalike cases.

**Priority C: ROVER-style combination, but only as alignment scaffolding.**
- ROVER builds a word transition network from multiple hypotheses and votes. Classic ROVER is designed to output a combined transcript, which is exactly what the spec forbids ("never combine fragments to manufacture a complete transcript").
- Use the alignment idea only: per expected span, record which observers support it and which report competing content. The decision uses the support pattern, not a voted transcript.

**Disagreement as evidence (decision rule sketch):**
- A span is **verified** when (i) at least one free observer covers it within the local phoneme budget, (ii) no observer reports confident competing content there, and (iii) the CTC ratio is not strongly negative.
- If the two ASR systems disagree and only one supports the expected text, the span is **accepted recognition variation** only when the CTC ratio also supports it and the span is not protected.
- If no observer and no CTC ratio supports it, the span is a **generation failure** (localized).
- Everything else is **unresolved**.
- *Trade-off to measure:* this rule trusts a single supporting recognizer more than the current approach does. That reduces soundalike rejections but needs FA measurement, especially for Whisper, whose decoder can produce fluent expected-looking text.

### 3. Audio-aware judging and vocalizations

**Audio LLMs: advisory only, never as approvers, and never shown the target text first.**
- *Evidence against using them as gates:*
  - Text dominance: in ALME (arXiv 2602.11488), Gemini 2.0 Flash followed conflicting text 16.6% of the time and GPT-4o 23.2% under audio-text conflict, against 1.6% and 0.9% when the audio was replaced by its transcript. Supplying the expected text invites the model to "hear" it, which is the exact failure behind accepted added or changed wording.
  - SpeechJudge (ICLR 2026, arXiv 2511.07931): the best zero-shot model, Gemini-2.5-Flash, scored 69.1% agreement with humans on 1,000 TTS naturalness pairs, against 72.7% for a Bradley-Terry reward model and 77.2% for SpeechJudge-GRM. Qwen2.5-Omni-7B reached 60.6%.
  - "Beyond Naturalness" (2026), on injected TTS errors: "neither MOS predictors nor Audio-LLMs judges reliably detecting errors". Gemini 3.5 Flash reached Kendall τ of only 0.41 to 0.51 on phonetic accuracy, and Qwen3-Omni roughly 0 to 0.19.
  - Shortcut audits: judges given a wrong reference label followed it 95 to 99% of the time for attributes they could not hear.
- *Where they might still help:* blind description of a specific gap ("What non-speech sounds, if any, occur between 3.2 s and 4.1 s?") fed into the evidence ledger for the human queue. Even this is unreliable. On NVMOS, the best LLM judge correlated with expert ratings of non-verbal vocalizations at 0.468, Qwen-Omni 30B at −0.049, and judges "confuse whether an NV is present". Fish's S2 report found Gemini 3 Pro agreed with humans on acoustic-event tags 76.2% of the time (κ = 0.47).
- *Local option:* Qwen3-Omni-30B-A3B is open and might run quantized on the M5 Max. It would compete with Drama 3 (about 28 to 44 GB) for unified memory, and its measured judging quality is the weakest of the group. **Recommendation: do not put an audio LLM in the acceptance path.** Correlated calls to one model do not add acoustic evidence, which is the same flaw as the three blind reviews.

**Vocalizations ("arrgh") and performance cues ("B-but", "Wheerrre"): verify presence acoustically and route content to humans.**
- In reviewed preparation, mark each non-lexical span with an expected class (cry, groan, gasp, stutter) and an expected duration range. Lexical verification then skips it explicitly instead of silently.
- *Verification:* the CTC alignment of the neighbouring lexical words (with a `<star>` wildcard for the vocalization) must leave a voiced, non-silent gap of plausible duration. Optionally, an audio-event classifier scores it. NonverbalTTS used BEATs to detect breathing, laughter, sighing, groaning and grunting, but at a deliberately low 0.1 threshold followed by human validation, which tells you its precision is not gate-grade.
- *States:* if the gap is present and the classifier agrees, the span is "vocalization present (acoustic)". A missing or near-zero gap means "generation failure: vocalization missing". Anything ambiguous goes to the listening queue with a pre-cropped clip.
- *ASR-side fine-tuning evidence:* adding non-verbal tokens to ASR works with data. In "Beyond Words" (2026), cry F1 rose from 32.1 to 69.0 using 400 cry samples plus laugh and breath transfer. That is a later option once your human labels accumulate, not a day-one tool.
- *Stutters and elongations* ("B-but" → "But", "Wheerrre" → "Where"): phoneme alignment naturally treats the extra /b/ or lengthened vowel as a cheap insertion. Accept them as a "performance cue" class only when the lexical word itself is covered.

### 4. Decision policy and evaluation methodology

**Five explicit states, each with a next action:**

| State | Meaning | Next action |
|---|---|---|
| VERIFIED | Every expected span supported; no unexplained audio; acoustics pass | Bind verdict to audio hash plus inputs plus policy; publishable |
| ACCEPTED_VARIATION | Covered, with a reviewed variation class (boundary, soundalike name, weak ending allowed by policy, performance cue) | Same as verified; record class and supporting observers |
| UNRESOLVED | Evidence conflicts (for example one ASR supports, CTC ratio ambiguous) or protected token uncertain | Recheck existing audio first (crop re-recognition, other observers); then the human listening queue; never accept |
| GENERATION_FAILURE | Localized defect with type (omission, insertion, repetition, stall, clipping, missing vocalization) | Failure-type-specific recovery (Section 5) |
| VERIFICATION_UNAVAILABLE | An observer, sidecar or model is down or timed out | Keep the take; retry verification with backoff; never resynthesize, never accept |

- Cloning and audition reference transcripts use the same pipeline but with **no** variation classes beyond boundary and normalization, regardless of chapter settings. Fish's API requires the reference text to match the reference audio.

**Building the evaluation set:**
- Label per span, not per clip. Record (a) what was audible, (b) generation defect type, (c) whether each recognizer erred, and (d) a confidence level including "uncertain".
- Use two listeners for disputed spans, and keep disagreements as "uncertain" rather than forcing a label.
- Include the representative cases, clip boundaries, long paragraphs, and joined parts. Add controlled failures created by editing saved audio: delete a clause, duplicate a phrase, splice in an unrelated "thank you", swap a number word, insert silence. Edited audio has known ground truth and costs no GPU time.
- Stratify by voice, engine/backend and passage length.

**Metrics and thresholds:**
- False acceptance = a GENERATION_FAILURE-labeled span ending in VERIFIED or ACCEPTED_VARIATION. False rejection = a correct span ending in GENERATION_FAILURE. UNRESOLVED counts as cost (listening time), not error.
- Use cost-sensitive thresholds (Chow's rule generalizes: accept only if expected error cost is below the review cost). Report risk-coverage curves.
- Do not trust naive empirical thresholds. An audit of selective classifiers found that uncertified empirical thresholding exceeded its declared error budget in up to 73% of trials. Under grouped shift (unseen generators), even certified rules overran in 9 to 30% of trials. In practice, report a Clopper-Pearson upper bound on FA among accepted spans per stratum (voice and engine), and recalibrate when a new voice or engine arrives.
- Sample size reality: with zero observed false acceptances in n accepted clips, the 95% upper bound is about 3/n. Claiming FA below 1% requires roughly 300 accepted human-checked clips per stratum you care about.

**Separating TTS errors from ASR errors (operational tests):**
- (i) Re-recognize a ±1.5 s crop around the discrepancy. Context-dependent ASR errors often vanish on crops; generation defects do not.
- (ii) Check whether the observers agree with each other against the script (shared error points to the audio) or disagree (points to recognition).
- (iii) Check the CTC ratio, which has no LM prior.
- (iv) For joins, check byte identity with verified parts (Section 6).
- Record which test changed the state. That gives a measured "recognition-side rescue rate" per test.

**JEV decoupling:**
- Split JEV's current roles into separate capabilities: (1) trigger secondary recognition, (2) trigger pronunciation review, (3) text plausibility.
- Make (1) unconditional and local: always run both ASR systems plus the phoneme observer.
- Replace (2) with the phoneme and CTC evidence.
- Demote (3) to an advisory feature that can move a span only toward UNRESOLVED, never to acceptance, because JEV "receives no audio".
- An outage of any one capability should then produce VERIFICATION_UNAVAILABLE for that capability only, and should not block the others.
- Log the actual rejecting stage as a first-class field, since the spec notes that local coverage checks sometimes reject before JEV sees a case.

### 5. Recovery strategies

**Principle:** diagnose first, then change the input that the diagnosis implicates. Seed changes are only appropriate for stochastic failures such as stalls and one-off repetitions.

| Diagnosed failure | First action | Escalation |
|---|---|---|
| Stall / long silence mid-line | Re-sample the same text (new seed); test temperature 1.0 / top-p 0.9 versus Fish's 0.7/0.7 defaults | Split at the stall point's sentence boundary |
| Local omission or repetition at a span | Split so the span sits near the start of a short child | Respell or simplify the span's text; plain delivery |
| Runaway / appended speech | Shorter chunk; explicit end punctuation | Split |
| Name mispronounced (consistent across observers) | Respelling from the reviewed lexicon (affects TTS input only) | Operator lexicon entry |
| Vocalization missing | Generate the vocalization as its own short take with the same voice, or plain delivery for the surrounding line | Operator decides between expression and intelligibility |
| UNRESOLVED / recognition-suspect | No resynthesis; recheck existing audio | Listening queue |
| Hard acoustic (clipping, unusably quiet) | Regenerate or adjust gain (gain only if no clipping) | Alternate approved engine with the same voice contract |

- **Drama 3 specifics to test:**
  - Fish's runtime splits long text automatically into chunks of about 300 UTF-8 bytes and feeds the previous chunk back as context. Each Drama 3 chunk is capped at 1024 frames (about 47 s) by default. A long "single" passage therefore already contains engine-internal joins that you cannot see or verify. Controlling splits yourself, keeping passages under about 300 bytes or using `single_chunk`, would make every join visible. This is my inference from Fish's docs, not a published finding.
  - Sampling: the only evidence is Sogni's 14/24 versus 16/16 report. Run an A/B on 100+ saved hard passages measuring stall, omission and repetition rates and the human expressiveness rating, since higher temperature may trade stalls for content errors.
  - `repetition_penalty` is ignored by Fish's Drama 3 runtime, so do not spend budget on it.
- **The 12-start safeguard:** keep it as a per-episode cost ceiling, but evaluate it with data. From attempt logs, plot the cumulative recovery probability by attempt number for each failure class. If stalls usually recover within 2 to 3 re-samples but omissions rarely recover without a split, reserve budget by class. Examples: at most 3 seed-only starts before a structural change, and at least 4 starts reserved for split children (the current reservation is a good instinct). The planner's `arrgh → aargh → argh` loop is a "changed something irrelevant" failure. Encode a rule that a repair must change an input implicated by the diagnosis, and that it counts as exhausted after N failures *of that diagnosis*.
- **Open-source precedent:**
  - `tts-audiobook-tool` keeps "the generation with the least number of errors" across retries, so it preserves best candidates rather than discarding them.
  - AudiobookMaker re-synthesizes chunks that are "cut short, silent or runaway". It also noted that a Whisper prompt for Chinese "was echoed into transcripts", a reminder to keep recognition unprompted.
  - None of these tools publish FA/FR measurements, so they are pattern sources, not evidence of accuracy.

### 6. Durable architecture and throughput

**Joins verifiable by construction (highest-value architectural change):**
- Assemble joins deterministically: sample-exact concatenation with a short, recorded crossfade, storing each part's audio hash and sample offsets in the join record.
- The join's verification then becomes three things. (1) Each part region is byte-identical, outside the crossfade, to an audio blob that already has a VERIFIED verdict. (2) Fresh verification of each boundary window (about ±2 s around each seam) with all observers. (3) Whole-join ASR as additional, non-veto evidence.
- If whole-join ASR disagrees *inside* a byte-identical verified region, the state becomes UNRESOLVED (recognition-suspect), followed by crop re-recognition, not regeneration.
- This is not "concatenating child transcripts as proof". It is proof of audio identity plus fresh evidence at the only places where new audio exists, which directly targets the reported context-dependent failures on joined audio.

**State and caching (fit to MongoDB, minimal new infrastructure):**
- Collections: `takes` (keyed by SHA-256 of audio bytes, plus synthesis inputs hash covering text sent, controls, voice/reference hash, engine, backend, quantization and seed), `verifications` (keyed by audio hash plus expected-speech hash plus observer versions plus policy version; append-only; a rejection is never overwritten by success, and a re-verification creates a new record), `episodes` (budgets, strategy history, exhausted strategies), `repairQueue` (operator items with cropped evidence and a resumable token).
- Use leases with heartbeats for worker claims.
- **Engine choice:**
  - DBOS is attractive because it is a library with durable queues, but it requires Postgres. Its docs say "the only point of failure in DBOS is Postgres". Adopting it means adding Postgres next to MongoDB.
  - Temporal adds a server cluster.
  - BullMQ adds Redis.
  - Restate is a single binary.
  - Given that you dislike complexity, and that the durable value here is in content-addressed artifacts and append-only verdicts rather than workflow replay, a Mongo-backed state machine is the simpler fit. Revisit Restate or DBOS only if cross-machine orchestration logic becomes the bottleneck.
- **Python integration cost:** one long-lived sidecar per machine (stdio JSON-RPC or local HTTP) holding Parakeet-MLX, mlx-whisper and the CTC/GOP code warm. The phoneme CTC can run in Node via the ONNX export. G2P via the eSpeak-ng CLI. The PanPhon table can be ported to TS.

**Throughput and resource contention:**
- The 4-bit Drama 3 MLX load peaks at 28.2 GB (q8 at 43.7 GB) on an M5 Max, according to drama3-mlx. MLX and PyTorch-MPS share unified memory. Parakeet at about 2 GB is the cheapest observer to keep warm locally.
- Published Drama 3 MLX speeds conflict. Sogni reports 3 to 5 s per request for 5 to 9 s of speech on an M5 Max. A BF16/4-bit model card reports RTF 3.45 to 5.24 on an M3 Max (slower than real time). Measure on your own hardware.
- On Runpod, run verification on the same pod as synthesis only if the GPU has idle gaps. Otherwise ship audio back to the Mac or a CPU worker, because GPU time is the scarcest resource.
- Track, per invocation: verified audio seconds ÷ wall-clock seconds (including recovery, deferral waits and assembly), paid GPU seconds idle ÷ paid GPU seconds, starts per verified second, and recognition-side rescues (regenerations avoided).

## Recommendations: Experiment Plan

Run every experiment by re-verifying the **same saved audio bytes**, never by regenerating or reusing cached verdicts. Use the human-labeled span set and the controlled-failure set.

1. **Week 1: Build the instrument.**
   - Label 300 to 500 saved clips at span level, including all representative cases and 50+ joins, and keep "uncertain" labels.
   - Generate 200+ controlled failures by editing audio.
   - Record the current pipeline's verdicts and the actual rejecting stage as the baseline.
2. **Soundalike rejections (biggest FR source):**
   - Add phoneme-space coverage alignment (eSpeak-ng G2P plus feature-weighted edit distance, local windows, protected tokens) over the existing Whisper and Parakeet outputs.
   - Compare FR and FA against Double Metaphone on identical bytes. Expect the largest gain on names and word boundaries.
   - Pass criterion: FR falls with no rise in the FA upper bound on omissions, numbers or negation.
3. **Accepted added/changed wording (biggest FA source):**
   - Add the free phoneme observer and the CTC GOP likelihood ratio with `<star>` wildcards and trailing-audio detection.
   - Measure per-error-type AUC on controlled insertions, substitutions and appended "thank…".
   - Remove JEV's ability to approve. Measure how many current JEV acceptances the new evidence would flag, then have humans listen to those.
4. **Context-dependent failures on joined audio:**
   - Implement deterministic, offset-recorded joins and boundary-window verification.
   - Replay every saved rejected join and count how many were byte-identical verified parts with recognition-only disagreement. Each one is an avoided regeneration.
5. **JEV coupling and outages:**
   - Make both ASR systems and the phoneme observer unconditional.
   - Run a chaos test that kills each observer and sidecar mid-run and kills the CLI mid-episode.
   - Pass criterion: no resynthesis of already-generated takes, no promotion of rejections, and resumption from the checkpoint with identical budgets.
6. **Unverifiable vocalizations:**
   - Add non-lexical span markup plus the acoustic-gap check, and measure presence detection against human labels.
   - Test an audio LLM only as a blind gap describer, and only if the acoustic check leaves many UNRESOLVED cases.
7. **Recovery and throughput:**
   - Using attempt logs, fit recovery curves per failure class.
   - A/B Drama 3 sampling (0.7/0.7 versus 1.0/0.9) and self-controlled splitting (≤300 bytes) on 100+ hard passages.
   - Then run full-book recovery runs comparing completion rate, terminal failures, interventions, starts per verified second, and verified audio seconds per wall-clock second, including GPU idle time.

## Caveats

- Most quantitative evidence here comes from adjacent domains: L2 and child pronunciation assessment, generic ASR hallucination, and TTS naturalness judging. None of it measures FA/FR for audiobook TTS verification. Your labeled set is the only valid proof.
- The Drama 3 stall statistics (14/24 versus 16/16) come from one community integration's PR. The MLX ports are unofficial, and Fish's official Drama 3 runtime is NVIDIA-only ("CPU, Apple MPS and AMD ROCm are not supported"). Behaviour may differ between your MLX build and Runpod CUDA workers, so stratify all metrics by backend.
- Several cited audio-LLM and selective-prediction papers are 2026 preprints that have not been peer-reviewed.
- Phoneme recognizers are noisy (PER of about 50 on a hard multilingual benchmark), and eSpeak G2P will mis-guess some invented names. The design relies on multi-observer agreement and local budgets, not on any one phoneme string, but the thresholds must be calibrated on your data.
- Trusting a single supporting recognizer will reduce rejections but could raise FA if Whisper's decoder produces expected-looking text. Measure that specific risk before rollout.
