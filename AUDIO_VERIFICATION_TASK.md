# Task: Reliable verification and recovery for generated audiobook speech

Build a resumable system that verifies the **complete intended spoken passage, in order**, and recovers defective or disputed recordings so the audiobook can finish. **Rejection is an intermediate diagnosis, not a satisfactory end result.**

Prevent **missing, extra, repeated, reordered, or unrelated speech**, while usually tolerating minor pronunciation, dialect, and harmless transcription differences. Reject silence, prolonged stalls, excessive clipping, and unusably quiet audio. Character casting, voice fidelity, and artistic performance are separate concerns.

Integrate with the CLI pipeline: primarily Fish Audio Drama 3 on Apple Silicon and remote workers, including paid Runpod GPUs. Support other engines and voice audition/cloning samples. Inputs are audio, intended speech/source mapping, selected voice, actual synthesis text/controls, generation settings, and attempt history.

## The central uncertainty

**Both TTS and ASR can be wrong.** A mismatch does not identify which failed. Regenerating correct audio wastes the slowest resource; accepting incorrect audio damages the book. Neither failure rate can be ignored.

Terminating a long-running job can be near-catastrophic. Optimize for **correct, completed books**, not just accept/reject decisions: recover difficult clips while independent work continues, and retain recordings through verification outages.

Verify spoken content, not semantic similarity: synonyms are not acceptable replacements, while different spellings can sound identical. A plausible name variation cannot excuse an omitted clause elsewhere; numbers and negation require particular protection.

### Representative cases

These are evaluation cases, not blanket equivalence rules. Disputed recordings need human listening to establish what was actually audible.

| Intended text | Recognized text | Question to resolve |
|---|---|---|
| `halfblade`, `apeman` | `half blade`, `ape man` | Harmless word-boundary differences? |
| `Derrins`, `Vandire`, `Qarm`, `Jas` | `Darin’s`, `Van Dyer`, `Khorm`, `Jazz` | Plausible representations of the same sounds? |
| `seventh`, `what’cha`, `ya`, `see see` | `7th`, `what you`, `you`, `CC` | Equivalent spoken forms with complete coverage? |
| `weapons`, `body’s` | `weapon`, `body` | Weak endings, actual pronunciation changes, or ASR mistakes? |
| `B-but`, `Wheerrre` | `But`, `Where` | Printed performance cues absent from the transcript, or missing audible content? |
| `arrgh` | `oh`, `are`, or nothing | An audible vocalization, misrecognition, or genuinely missing speech? |
| A complete paragraph | An omitted clause, repeated phrase, or unrelated “thank…” | Serious generation failure or recognition error? |

Reviewed preparation can turn `Wheerrre issss sheeee?` into `Where is she?` plus delivery instructions, preserving lexical content, source text, attribution, and synchronization mappings. Verification must not silently rewrite the book to match ASR.

## Existing approaches and their limitations

1. **Acoustic checks, then independent ASR.** Measure duration, level, silence, stalls, and clipping; transcribe with Whisper without prompting it with expected text. This limits confirmation bias, but normal energy/duration do not prove correct words. Dramatic pauses and quiet speech can resemble defects.

2. **Normalized exact matching and word-edit alignment.** Record insertions, deletions, and substitutions. Exact matching initially rejected too much acceptable audio; blanket edit-distance/WER tolerances can hide omissions in long paragraphs or permit changed numbers/negation. Substring matching cannot establish complete coverage, especially for middle-of-passage errors.

3. **Bounded representation and phonetic matching.** Local rules handle compounds, numbers/ordinals, colloquialisms, and letter names. Double Metaphone resolves many soundalikes without network calls. Alignments must consume the entire passage in order. Codes are lossy, however: different words can share a code without matching pronunciation or syllables. Continually adding exceptions is not a general solution.

4. **Secondary recognition, including Parakeet.** It can rescue Whisper errors, but recognizers can disagree or share biases. Repeated deterministic decoding usually adds little evidence. Never combine fragments to manufacture a complete transcript no recognizer produced.

5. **JEV text-based plausibility judgment.** Prompts moved from “equivalence” toward plausible ASR transcription. There have still been soundalike rejections and user-reported acceptances of added/changed wording. JEV receives **no audio**; its score is not a calibrated probability of correctness and cannot establish whether an untranscribed cry occurred. Local coverage checks may reject before JEV sees a case; identify the actual rejecting stage.

6. **Three blind LLM pronunciation reviews.** Require at least two approvals per disputed eligible span. Calls to the same model are not independent acoustic observations: correlated consensus cannot supply missing evidence. They add latency, cost, and another service dependency.

Word timestamps were previously too strict a gate. Unusable optional alignment is now omitted with a warning, not treated as bad speech. Any proposed alignment method must detect incorrect speech, not merely fit a supplied script to audio.

## Recovery: what has helped and what has failed

Ordinary retries use different seeds. Analysis-model recovery normally escalates after **three distinct failed takes**, including nonconsecutive or mixed acoustic/text failures; a failed applied repair can escalate immediately. The planner receives source and synthesis text/controls, transcripts, acoustic measurements, and repair history. It can recommend reviewed recognition-variation acceptance, respelling, plain delivery, splitting, retrying, or stopping.

Reported project failures and subsequent corrections:

- A planner cycled through `arrgh → aargh → argh` while retaining “pained, breaking into a cry.” Plain-delivery recovery now changes that likely cause, but trades expression for intelligibility and does not generally solve vocalization verification.
- Whole-passage retries exhausted the budget before splitting could generate anything. Recovery now reserves attempts within a default **12 generation starts per passage/recovery episode per invocation**, shared across children and replans.
- A saved rejected join permanently blocked later runs. Recovery can now reuse accepted parts and return to a different whole-passage performance without repeating the exhausted split strategy.
- **Accepted parts can fail after concatenation.** Parts recognized correctly alone were recognized as different words when joined. That may be context-dependent ASR failure. Currently the whole join must pass verification; concatenating child transcripts is not proof.

Respellings affect TTS input, not book text or source references. New takes still require verification, and model approval cannot override hard acoustic failures or excuse missing/extra content.

### Completion-preserving escalation

Recheck existing audio when recognition is suspect; change relevant synthesis inputs when generation is suspect. Escalate through reviewed pronunciation/delivery changes, safe split/join, or another approved engine/verification method. Alternate generation must preserve the selected voice and content contract, not silently substitute a generic voice.

When a strategy is exhausted, defer the passage, continue independent work, and escalate to another method or a durable operator repair queue. Preserve candidates, evidence, budgets, and history so intervention resumes locally. Evaluate the current 12-start safeguard rather than treating it as proof of irrecoverability or replacing it with unbounded blind retries.

If recovery cannot establish valid audio, provide a precise, resumable handoff instead of an unexplained whole-job exception. Distinguish waiting/needs-attention from completion. Never omit speech, weaken validation, or mislabel incomplete output to avoid failure. Identify genuine job-wide blockers, such as invalid configuration or storage failure, and checkpoint before stopping where possible.

### An architectural dependency to evaluate

JEV currently enables some ordinary-speech secondary-recognition and pronunciation-review paths. Disabling it changes more than one judgment step; an outage can prevent later checks. Separate and evaluate these capabilities rather than assuming JEV is an isolated optional vote.

## Required outcomes

### 1. Evidence and explicit decision policy

Create a **human-reviewed evaluation set** of saved real recordings and controlled failures covering the examples, clip boundaries, long paragraphs, and joined parts. Separate generation defects from recognition mistakes; retain genuinely uncertain labels.

Distinguish verified content, accepted recognition variation, unresolved uncertainty, generation failure, and unavailable verification, with a next action for every nonaccepted state. Outages do not establish bad audio; uncertainty does not authorize acceptance. Cloning-reference transcripts must match the recording’s actual words, regardless of permissive chapter settings.

### 2. Measured comparison, not just passing tests

Measure false acceptance/rejection, automatic recovery and book-completion rates, terminal failures, manual interventions, avoidable regenerations, model calls/cost, and end-to-end time. Use **verified audio seconds per wall-clock second**, including recovery and paid GPU idle time. Test interruption and outage recovery as well as eventual acceptance.

Compare on the same saved audio, explicitly reverified rather than regenerated or accepted from an old cached verdict. The existing benchmark measures first-pass synthesis/verification overlap, not full recovery, assembly, and publication. Audio-aware models, alignment, confidence scores, and sampling changes are candidates to test, not assumed solutions.

### 3. Useful, bounded, shared recovery

Recovery must learn from previous attempts, change something relevant, and avoid repeating exhausted strategies. Share the algorithm across local, pooled, and remote execution and engines where applicable. Keep models warm and independent synthesis moving during verification, repair, and deferred-passage waits, within bounded resource limits. GPU ASR may help or compete with synthesis; measure the combined pipeline.

### 4. Durable state and actionable diagnostics

Preserve accepted work, rejected evidence, and pending recordings across interruptions. Retry unavailable verification without unnecessary resynthesis; never promote a saved rejection to success. Bind reusable results to the audio bytes and relevant synthesis inputs, retaining the policy and evidence that originally accepted them.

Private attempt reports must contain expected speech, actual synthesis inputs and controls, transcripts, audio measurements, aligned discrepancies, model/seed/policy identities, decisions, repair history, and reasons. Routine logs should identify the failing stage and inspection path without dumping book passages or credentials. Fatal errors must provide enough safe context to troubleshoot and resume.

**Success means correctly completed books with fewer terminal failures, incorrect acceptances, unnecessary regenerations, and interventions. Demonstrate it on human-checked recordings and recovery runs: detecting defects while routinely leaving books unfinished is not enough. Existing examples establish specific repairs, not overall accuracy or completion reliability.**
