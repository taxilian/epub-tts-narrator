# Local audiobook renderer

`audiobook-render` consumes a version-2 analyzer bundle, creates or reuses persistent character voices, verifies each generated chunk, and produces chapter audio plus a chaptered `.m4b`. The analyzer and renderer read the same global `epub-audiobook.toml`, while the book directory and optional path overrides are command arguments. They remain separate commands.

Fish Audio Drama 3 creates character reference voices and renders the book locally through the **unofficial Sogni MLX port** on Apple Silicon. The renderer uses Drama 3's scene prompt to cast one character from a description, saves the recording, and clones it for later lines and books. Qwen3 VoiceDesign remains an optional alternative. The port reports 48–55 GB peak memory for `q8` on an M5 Max; Drama 3's download is about 41.5 GB. Allow additional disk space for Whisper, dependencies, voice references, and retained attempts. These are upstream memory figures, not combined-memory guarantees for this renderer. [Sogni port](https://github.com/Sogni-AI/drama3-mlx), [scene casting format](https://arxiv.org/html/2610.18010v1#A2)

**Validation status:** automated tests cover real FFmpeg encoding, synchronization, quality gates, adapters, interactive casting, and voice replacement across books. A live local Drama 3 / MLX Whisper pilot produced a 17.38-second chapter and M4B with two persisted voices and nine verified speech clips; every accepted clip had zero normalized word errors. During voice creation, the acoustic checks rejected one real attempt for excessive silence and a long internal stall, and a subsequent attempt passed. A separate live Whisper check rejected an extra spoken sentence. The pilot outputs are in `audiobooks/last-lantern-demo/`, with accepted transcripts in `verification.json` and retained attempts in `render-work/last-lantern-demo/`.

The staged workflow was also exercised with nine newly generated Drama 3 passages using copies of the two saved demonstration voices. All nine passed strict Whisper verification on their first attempts. After closing the models, separate assembly produced a 16.27-second M4B with chapter metadata, an embedded PNG test cover, source-linked synchronization, and a normalized M4A chapter. Its chapter measured -19.81 LUFS against the -20 LUFS target; original reference files were unchanged. Evidence and playable outputs are in `output/staged-render-pilot-20261021/`. Automated integration tests separately cover multi-chapter assembly, chapters generated in separate runs, interruption recovery, recasting, and corrupted checkpoints. This small pilot is not a full-book or perceptual-quality validation.

A live interactive casting session also generated two verified alternatives after editing the description and sample text, accepted a replacement, restored the first sample, and retained the edits after reopening. Native `afplay` start/stop was exercised with volume zero. Its results are in `render-work/casting-pilot/live-validation.json`. Full-book rendering and human audition remain untested.

Earlier automatic-startup testing with the first Soulship book produced 26 samples with zero normalized word errors, but used unconditioned Drama 3 voices. Those results verified the spoken words and reuse behavior, **not the requested character voice**. In particular, the description was not applied during voice creation. The historical record in `render-work/soulship-01/startup-validation.json` must not be treated as evidence that those voices were cast correctly.

An earlier workaround regenerated all 24 distinct saved Soulship voices with Qwen3 VoiceDesign, covering 26 cast entries. Each replacement passed acoustic and exact-transcript checks; every original recording remains intact. Jasper's narration, dialogue, and Soulscape share one reference. New Drama 3 speech conditioned on the replacement Nova and Jasper references also passed verification, and Jasper's shared reference was checked against all seven bundles. Evidence is saved in `render-work/soulship-01/designed-cast-regeneration.json`, `designed-clone-validation.json`, and `shared-voice-series-validation.json`. These checks establish prompt application, text fidelity, and reference reuse; listening still determines whether the performance suits the character.

Three later whispered chorus takes were recovered without resynthesis after a word-alignment failure and a one-word “in”/“and” discrepancy. Live Whisper and Parakeet checks agreed on the corrected transcript for two takes; Parakeet matched the requested script for the third. All original audio and rejection reports remain unchanged, all three takes are saved, and the first is selected. Full evidence is in `render-work/soulship-01/recovered-whispering-voices.json`.

Native Drama 3 casting is now implemented and exercised locally. The investigation retained 27 controlled takes across prompt formats and sampling settings. All nine scene takes at the renderer's normal temperature 1.0/top-p 0.9 matched their scripts, including three readings of Nova's actual passage. In a separate full casting pilot, Nova passed on the first attempt and Jasper on the third (a wording discrepancy and silence stall were rejected). Both saved references then produced verified new dialogue, and Jasper's three roles shared one asset. Existing Soulship selections were preserved. See `render-work/drama3-design-investigation/report.html` and `production-validation/validation.json` beneath that directory. The 747-test suite passes; perceptual suitability still needs listening.

## Install

Run these commands from the project directory. They create a dedicated Python 3.12 environment without activating it or setting environment variables:

```sh
cd /path/to/epub-audiobook
uv venv --python 3.12 .venv-renderer
uv pip install --python .venv-renderer/bin/python -r renderer.requirements.txt
uv pip install --python .venv-renderer/bin/python -e .
```

FFmpeg and FFprobe must also be installed. Their executable names or absolute paths belong in the configuration file. The environment commands follow [uv's environment documentation](https://docs.astral.sh/uv/pip/environments/).

The requirements pin the Drama 3 port to commit `3b76bf84714d3766d7872f3409639a438e296cc8` and MLX Whisper to version `0.4.3`. Download the corresponding local model files:

```sh
.venv-renderer/bin/hf download fishaudio/drama-3 \
  --revision b2b5ee998034b2975720906c815e7fb0e9f007bd \
  --local-dir "./models/drama-3"

.venv-renderer/bin/hf download mlx-community/whisper-large-v3-turbo \
  --revision a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb \
  --local-dir "./models/whisper-large-v3-turbo"
```

The renderer uses local paths and does not download missing models during inference. If those exact files already exist elsewhere, point the configuration at their existing directories instead of making duplicate downloads. `hf download` stores download metadata in the chosen directory. [Hugging Face download documentation](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download)

## Configure once

Both analysis and rendering use one global `epub-audiobook.toml`. Once configured, run the commands below from the project directory and omit `--config`. From another working directory, pass `--config /path/to/epub-audiobook.toml` to use that same file. Per-book and per-series configuration files are not needed.

For a new setup only, add the sections from [audiobook-render.example.toml](audiobook-render.example.toml) to the existing global `epub-audiobook.toml` alongside the analyzer sections. If you have not added renderer settings yet:

```sh
cat audiobook-render.example.toml >> epub-audiobook.toml
```

Do this once: TOML does not allow duplicate sections. Both commands load `epub-audiobook.toml` from the current working directory; `--config PATH` selects the same global file from elsewhere. Each command validates the recognized sections for the other command, then uses its own settings. No book or series needs a separate configuration file.

Edit the model paths before running. Configured paths are literal strings, resolved relative to the configuration file; shell variables and environment variables are not substituted. `~` is expanded for configured filesystem paths. The main settings are:

```toml
[engine]
type = "drama3_mlx"
checkpoint = "./models/drama-3"
experts = "q8"
temperature = 1.0
top_p = 0.9
top_k = 30
max_new_tokens = 2048

[voice_design]
type = "engine"

[verification]
type = "mlx_whisper"
model = "./models/whisper-large-v3-turbo"
language = "en"

[render]
chapter_format = "m4a"
aac_bitrate = "128k"
mp3_bitrate = "128k"
normalize_loudness = true
target_lufs = -20.0
true_peak_db = -1.5
chapter_numbers = []
max_chunk_chars = 320
max_attempts = 3
seed = 1729
sample_rate = 44100
paragraph_pause_ms = 250
blank_pause_ms = 500
bracket_pause_ms = 120
allow_review = false
ffmpeg = "ffmpeg"
ffprobe = "ffprobe"

[quality]
max_wer = 0.0
```

The example file includes the remaining quality thresholds and the fallback reference passage used to create voices. Start with those defaults.

For a directory produced by `epub-audiobook analyze`, the renderer obtains the series ID and continuation link from the book itself. Voices are stored in `<book-directory>/voices/<series-hash>/`; work files default to `<book-directory>/render-work/`, and the finished audiobook defaults to `<book-directory>/audiobook/`. Use `--work-dir PATH`, `-o PATH` (or `--output PATH`), and `--epub PATH` for per-run overrides.

Voice lookup checks the current book first, then its parent from `--continue-from`, then that parent's parent, continuing until it finds the voice. This lookup happens when casting or rendering, so analyzing all books before generating voices works. The saved relative parent links include identity and integrity checks; a missing or changed parent fails explicitly instead of silently casting a replacement. Moving the whole series together preserves the relative links.

Missing voices and all new samples, edits, imports, and selections are written only in the current book. Inherited assets remain in their original directories. Regenerating a voice in book 2 creates a local override: book 2 and descendants that inherit it use the new voice, while book 1 and sibling branches keep theirs. A closer override takes precedence. Already generated audio is not changed.

Legacy exported bundles without directory-analysis metadata can still use explicit `voices.directory`, `voices.series_id`, `render.work_dir`, `render.output_dir`, and `render.epub` settings. Their shared-library replacement behavior remains unchanged. Manual aliases must move into the relevant book's root `voice-aliases.json`; cached alias files inside a voice store no longer supply manual book policy. New analysis directories reject those legacy location settings with instructions to remove them; use the book defaults and command arguments instead.

Roles belonging to the same explicit character ID share that character's saved reference. For example, Jasper's narration and spoken dialogue use one voice even when the analyzer assigned separate role voice IDs. This binding uses character identity, not a similar name. Regenerating that character updates every linked role.

Additional deliberate sharing belongs to a book, using exact voice IDs. Inspect the effective mappings or set one with:

```sh
.venv-renderer/bin/audiobook-render voice-aliases runs/book-01
.venv-renderer/bin/audiobook-render voice-aliases runs/book-01 \
  --set SOURCE_VOICE_ID TARGET_VOICE_ID
.venv-renderer/bin/audiobook-render voice-aliases runs/book-02 \
  --remove SOURCE_VOICE_ID
```

`SOURCE_VOICE_ID` uses `TARGET_VOICE_ID`'s saved reference. The command saves manual mappings in `<book-directory>/voice-aliases.json`, bound to that book and series. Repeat `--set` or `--remove` for several edits. `--remove` deletes the current book's override, allowing an inherited or automatic mapping to apply again. It does not edit an ancestor.

Aliases follow the same continuation chain as recordings. A mapping in book 1 is available to later books even when added after their analyses. A manual override in book 2 applies to book 2 and its descendants; book 1 and sibling branches keep their mappings. Earlier mappings are read from their original book instead of copied into each child. Invalid IDs and cycles are rejected. Global `[voices.aliases]` settings are rejected: move their mappings into the relevant book with this command. Aliases affect rendering and do not merge the analyzer's character records.

Choose chapters per book with `--chapters 3` or `--chapters 3-5,7` on `plan`, `render`, or `assemble`. Numbers use the **one-based order of script entries in the bundle manifest**, including front matter. Ranges are inclusive, duplicate numbers are ignored, and chapters always stay in source order. Omit the option to use all entries. `plan` lists the corresponding titles. The legacy global `chapter_numbers` setting remains supported, but book-specific selections belong on the command line. A selected subset produces a partial audiobook and records `partial_book: true`.

The current Soulship bundles remain drafts with review flags and some unassigned voices. `--allow-review` permits acknowledged review issues for that command; it **does not permit spoken text with a null voice**. Correct those attributions in the analyzer and re-export, or select a pilot chapter without missing voices. There is no implicit narrator substitution. The global `allow_review` setting remains supported, but no configuration edit is needed to acknowledge review items for one book.

## Inspect, cast, and render

Check installation and model files:

```sh
.venv-renderer/bin/audiobook-render doctor
```

`doctor` inspects packages and files. It does not load the model, test Metal, or prove that speech generation works.

Inspect the bundle, chapter selection, voice count, and blockers without inference:

```sh
.venv-renderer/bin/audiobook-render plan \
  "runs/book-01"
```

List voice IDs, descriptions, and saved reference paths:

```sh
.venv-renderer/bin/audiobook-render voices \
  "runs/book-01"
```

Open the interactive casting tool:

```sh
.venv-renderer/bin/audiobook-render cast \
  "runs/book-01"
```

Startup automatically prepares all voices: it reuses existing selections, selects previously saved verified samples, and generates missing ones. New voices and successful regenerations are selected automatically. There is no acceptance step. Missing samples show generation progress; an individual failure leaves the other voices available.

Voice playback matches perceived loudness automatically, with a default target of -20 LUFS and a -1.5 dB true-peak ceiling. Startup prepares cached playback copies of the selected samples before showing the menu; regenerated samples are prepared before playback. Existing, inherited, imported, and newly generated voices all use the same settings. No reanalysis or voice regeneration is needed.

These are playback copies in the current book's `render-work/voice-previews` directory. The saved cloning references, their transcripts and checksums, voice selections, parent books, and already rendered audio stay unchanged. This setting controls auditions; it does not normalize the final audiobook. To prepare all saved candidates as well as selected voices without loading TTS or transcription models:

```sh
.venv-renderer/bin/audiobook-render cast runs/book-01 \
  --action normalize-samples
```

There is one menu. Type a number or command and press Enter:

| Input | Action |
| --- | --- |
| A voice number | Play immediately. Any key stops playback and returns to the menu; no Enter is needed to stop. The menu also returns when the sample finishes. |
| `G` | Edit and regenerate the last voice played. If none has been played, asks for a voice number. |
| `G 5` or `G5` | Edit and regenerate voice 5 directly. |
| `Q` | Quit. |

Regeneration offers two fields: the voice prompt and the spoken sample script. Press Enter to keep either value. The new voice is verified, saved, selected, and played immediately. If generation fails, the previous voice stays selected. Inputs are case-insensitive.

Menu and text entry use the terminal's standard line editor: arrow keys,
Ctrl+A/E, Ctrl+K/U, and session history follow your readline/libedit bindings.
While a sample is playing, any key stops playback and returns to the menu.

Sample scripts now default to each character's actual words from the analyzed book. Casting selects a clearly attributed, contiguous passage, prefers ordinary dialogue of roughly 20–60 words, and records its chapter, segment and exact source offsets. Narrators use narration where no character dialogue is available. Shared roles such as Jasper's narration and dialogue use one sample script and one saved voice.

For a new voice, the default prompt includes both its description and structured voice traits, such as pitch, timbre, and delivery. Book quotations used as evidence are excluded. Existing edited prompts and established voice prompts are preserved. This applies to both the casting menu and automatic voice creation during rendering. Speech verification checks the recording and transcript; listening determines whether the performance suits the character.

Opening the casting tool updates legacy generic script drafts automatically, preserving your edited voice descriptions and custom scripts. It does not regenerate saved audio: the new script appears under **G** and is used when you regenerate. Number playback continues to play the current saved recording. A later series book keeps an established excerpt or custom script; a fallback may acquire a suitable passage when one becomes available.

Only verified source mappings are used. Ambiguous or review-required attribution, extreme delivery, control syntax, and very short cries are excluded. The fallback is `voices.reference_text`; the menu explains when it is used. In Soulship book 1, 20 of 24 distinct voices receive book excerpts, while the chorus and three Whisperers retain the fallback because their available lines are too short or unsuitable. Nestor and Hallway Eaterling Three have shorter usable excerpts, flagged in the provenance.

No new analysis or API calls are needed. To prepare the drafts and print their scripts and provenance without loading speech models:

```sh
.venv-renderer/bin/audiobook-render cast runs/book-01 \
  --action prepare-samples
```

Edits, samples, and selections persist across sessions in the current book and are available to descendants through the continuation chain. Earlier samples remain on disk for recovery, but history controls are outside the interactive menu. Already published audiobooks are unchanged. To rebuild a finished book, choose a new output directory; retained work caches reuse unaffected characters' clips. A render in progress keeps the voices it loaded at the start.

Playback uses macOS `afplay` by default. Configure another player in the TOML file if needed. The sample path is appended to this argument array without invoking a shell:

```toml
[casting]
player = ["/usr/bin/afplay"]
normalize_samples = true
sample_loudness_lufs = -20.0
sample_true_peak_db = -1.5
# Example alternative:
# player = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
```

Set `normalize_samples = false` to audition the original levels. Loudness targets accept -40 through -10 LUFS; the true-peak ceiling accepts -9 through 0 dB. Changing either setting creates new cached playback copies automatically. Normalization uses the configured `render.ffmpeg` executable and never starts a speech model.

With `[voice_design] type = "engine"`, the character description conditions Drama 3's scene system prompt and the sample script supplies its single named line. Several sentences may occupy that line. No reference recording is required to create the initial voice; later synthesis clones the saved recording. State vocal presentation, age range, accent, pitch, and a few stable traits clearly. Keep the reference at an ordinary speaking pace with one clean, dry voice, and avoid conflicting performances in one description. Word verification cannot judge whether every vocal trait sounds right; use the audition menu for that.

For a fresh analysis, the analyzer now supplies stable acoustic directions in both the voice description and traits, including voices added during identity reconciliation. Casting uses these automatically together with the selected book excerpt; no prompt-preparation or repair command is needed. Existing edited drafts and selected references retain precedence, so this change does not rewrite older casting work. Speech verification checks the words and audio quality, not whether vocal gender, character fit, or every requested texture sounds right.

To use Qwen3 VoiceDesign instead, download its local checkpoint and configure the separate designer:

```sh
.venv-renderer/bin/hf download mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 \
  --revision 7d3824abff87e49756bb0f83fb5411de75d160c4 \
  --local-dir "./models/qwen3-voice-design"
```

```toml
[voice_design]
type = "qwen3_mlx"
checkpoint = "./models/qwen3-voice-design"
language = "English"
temperature = 0.7
max_tokens = 2048
```

Changing the designer affects future creations and regenerations. Existing voice selections stay fixed. Both designers retain the same persistent reference format and shared-character bindings.

Description conditioning is not proof that every audible trait was followed. Whisper checks the spoken words; it does not assess gender presentation, age, accent, timbre, or the suitability of a performance. Use the menu to audition and regenerate a voice when needed.

For scripts, `cast --action list` prints the cast and sample history as JSON without loading a model. `generate` immediately selects its verified result; `generate-missing` prepares all unselected voices, reusing existing samples first. `regenerate-all` creates and selects a new reference for each distinct canonical voice, so shared narration and dialogue are generated once. Before generation, it prints a warning to stderr describing the scope and linked roles known within the bundle. In the directory workflow, replacements are local to the current book and may be inherited by its descendants; earlier books and sibling branches are unchanged. Legacy shared-library mode warns that replacements affect every book using that library and voice ID. Previous samples remain available, and existing audiobook audio is not rewritten. The command continues without a confirmation prompt, retains prior references after individual failures, and reports failure at the end. Run it in the book where you want the replacement voices to start applying. `edit` saves prompt changes without generation, and `play` plays the selected voice. Replace the voice ID below with an ID from the listing:

```sh
.venv-renderer/bin/audiobook-render cast runs/book-01 \
  --action generate --voice-id VOICE_ID_FROM_VOICES \
  --prompt-file "/path/to/voice-description.txt" \
  --text-file "/path/to/sample-text.txt"

.venv-renderer/bin/audiobook-render cast runs/book-01 \
  --action play --voice-id VOICE_ID_FROM_VOICES

.venv-renderer/bin/audiobook-render cast runs/book-02 \
  --action regenerate-all
```

The two UTF-8 files are optional for `generate`; omit them to reuse the saved description and sample text. Browsing, editing, and playback do not initialize TTS or transcription models. For advanced recovery, `--action accept --voice-id VOICE_ID --candidate-id CANDIDATE_ID` restores a previous saved sample listed by `--action list`; normal casting never requires this command. `--action play --candidate-id CANDIDATE_ID` can audition a specific earlier sample.

For unattended initial casting, the existing command creates, verifies, and immediately selects missing voices for the selected chapters:

```sh
.venv-renderer/bin/audiobook-render create-voices \
  "runs/book-01"
```

Use `cast` to listen and regenerate voices. If older casting data contains samples without an active selection, opening `cast` initializes those selections automatically. Existing active voices are retained when a newer unselected sample also exists.

To start with a chosen reference recording instead, import it before that voice ID has a local saved asset. An inherited reference can be overridden by an import in the current book; an existing local asset is not silently replaced. Replace `VOICE_ID_FROM_VOICES` with an actual ID from the listing. The transcript file must contain exactly what the recording says:

```sh
.venv-renderer/bin/audiobook-render import-voice \
  "runs/book-01" \
  --voice-id "VOICE_ID_FROM_VOICES" \
  --audio "/path/to/reference recording.wav" \
  --transcript "/path/to/reference transcript.txt"
```

References are normalized and independently transcribed before publication. Each saved asset contains its reference WAV, exact transcript, cast profile, model identity, verification report, and content hash. The nearest selected voice in the continuation chain is reused until the current book introduces an override or that selection is changed. Importing an inherited voice creates a current-book override without changing the ancestor. An existing local asset is never silently replaced.

Generate all chapters and assemble the complete book:

```sh
.venv-renderer/bin/audiobook-render render \
  "runs/book-01" \
  --epub "/path/to/book-one.epub"
```

To choose separate work and publication directories for one run, add `--work-dir /path/to/work -o /path/to/audiobook`. Omit `--epub` when no original EPUB cover is needed.

M4A chapters with AAC audio at 128 kb/s are the default. They are directly playable and use the same codec as the final M4B. `--chapter-format mp3` on `render` or `assemble` produces MP3 chapter copies when needed; the final book remains AAC in an M4B container. Bitrates are global settings in `[render]`.

To generate and listen to one chapter before committing to the whole book, inspect the chapter list and save a chapter checkpoint:

```sh
.venv-renderer/bin/audiobook-render plan runs/book-01
.venv-renderer/bin/audiobook-render render runs/book-01 \
  --chapters 3 --chapters-only
```

For a draft with review items, add `--allow-review` after inspecting them. Unassigned spoken passages still block the selected chapter. `--chapters-only` saves lossless chapter audio, its encoded playback copy, timing, and provenance beneath `<book-directory>/render-work/chapters/`; it does not publish a final audiobook directory. The CLI prints each playable file's path. `render-work/completed-chapters/chapter-NNNN.json` records the current checkpoint for each manifest chapter, pointing to its files under `render-work/chapters/<cache-key>/`. A completed chapter remains available if a later chapter fails.

To generate the remaining chapters and package later:

```sh
.venv-renderer/bin/audiobook-render render runs/book-01 --chapters-only
.venv-renderer/bin/audiobook-render assemble runs/book-01 \
  --epub /path/to/book-one.epub
```

`assemble` uses the verified chapter checkpoints and loads neither TTS nor transcription models. Missing, changed, or incompatible checkpoints stop assembly with an explanation; use `render` to prepare them. Changing only chapter format or bitrate can re-encode from saved lossless PCM during assembly. Changes to text, voices, verification, pauses, or loudness require `render` again; unchanged accepted speech is reused where possible. Use the same `--work-dir` if it was overridden. To package a subset, pass the same `--chapters` selection to `assemble` and choose a separate `-o` directory, preserving the final output location for the full book. Existing finished audiobooks are never overwritten.

By default, completed chapters receive two-pass loudness normalization to -20 LUFS with a -1.5 dB true-peak ceiling. Normalization preserves the PCM frame count and timing data, and operates on the assembled chapter rather than flattening every short utterance to the same level. Voice references and verified speech clips remain unchanged. These controls are separate from casting playback: use `[render] normalize_loudness`, `target_lufs`, and `true_peak_db` for the audiobook. Empty chapters and chapters shorter than the loudness measurement window are explicitly recorded as skipped for normalization.

`render` also creates missing voices if no casting edits or audition candidates exist. Use `cast` first to choose your voices before a long run. Generation is sequential to keep one shared model instance and bounded GPU memory use. Book text and generated audio are processed locally.

## Verification and recovery

Each chunk is normalized to mono PCM16 at 44.1 kHz. Before acceptance, the renderer checks duration, overall level, excessive silence, long internal silence, and clipping. Local Whisper then transcribes the audio independently: it receives no expected transcript as a prompt. The comparison records added, missing, and substituted words, with conservative punctuation and number normalization.

The default `max_wer = 0.0` requires the normalized words to match. Unknown names, unusual pronunciations, or recognition errors can reject otherwise acceptable speech. A passing check is useful evidence, not a guarantee that pronunciation, performance, or character identity sounds right. Audition a pilot chapter.

Reusable voice references have a separate check: they need intelligible audio and a matching reference transcript, but do not need word timestamps. Casting therefore disables Whisper's word alignment, whose timing failures can otherwise reject a useful recording.

An optional local Parakeet recognizer can recheck a reference when Whisper reports up to four small variations, with at most 10% word error. These may be substitutions or a single alphabetic word split into two or three words (or the reverse), with exactly the same letters and at least two letters per component. For example, `Sourcepalace` transcribed as `Source Palace` can reach this independent check. Each split or join counts toward the four-variation limit; number words, contractions, genuine added or omitted words, and larger discrepancies do not qualify through this rule. A potential word boundary is not treated as proof that the pronunciations or meanings match.

Explicit alphabetic compounds in the requested script, such as `pack-mate`, can also be compared with joined or spaced transcriptions such as `packmate` or `pack mate`. Only source-written hyphens authorize this equivalence; inferred splits do not change the final comparison between recognizers. Original word-error metrics remain in the report alongside the reference comparison and the bounded-variation alignment.

A separate pattern check recognizes expressive consonant spelling, such as `sshe iss ssweet` transcribed as “she is sweet.” This can reach Parakeet even when literal word error exceeds 10%, but only when every mismatch shortens repetitions of the same consonant, word counts and order match, and at least three distinct words show the pattern. An initial `ss` or a run of at least three consonants must also support the detection. Vowels, numbers, contractions, added or missing words, and unrelated substitutions cannot use this exception. It never applies a general doubled-letter correction or changes how the two recognizers' transcripts are compared.

The original sample script, raw mismatch measurements, and each proposed spelling reduction are retained. A recovered reference saves the independently agreed transcript alongside its unchanged audio. This verifies the recording's words; it does not assess the strength of a hiss, stutter, accent, or other performance trait. Those still require listening in the casting menu. Book text, source offsets, and audiobook passage checks are unchanged.

Parakeet receives the audio without the requested text. Even a discrepancy consisting solely of compound spelling requires this independent check. If Parakeet matches the requested words under that comparison, casting retains the requested transcript and records Whisper's disagreement. If both recognizers agree on different words, casting saves their agreed transcript alongside the recording, retaining the original requested script in the metadata and draft. Other disagreements, actual added or omitted words, larger changes, and acoustic failures remain rejected. These rules apply only to reusable references; audiobook passages keep the configured strict text and timing checks. Raising the speech `max_wer` does not relax reference checks.

To enable this recheck, download the model once and set `reference_model` under `[verification]`:

```sh
.venv-renderer/bin/hf download mlx-community/parakeet-tdt-0.6b-v3 \
  --revision ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15 \
  --local-dir "./models/parakeet-tdt-0.6b-v3"
```

```toml
[verification]
type = "mlx_whisper"
model = "./models/whisper-large-v3-turbo"
language = "en"
reference_model = "./models/parakeet-tdt-0.6b-v3"
```

This adds about 2.5 GB of model weights. The current Soulship configuration already enables it. The adapter uses MLX Audio's local Parakeet implementation. [MLX Audio Parakeet documentation](https://github.com/blaizzy/mlx-audio/blob/main/docs/models/stt/parakeet.md)

To recheck a retained casting attempt without generating another voice:

```sh
.venv-renderer/bin/audiobook-render cast runs/book-01 \
  --action recover \
  --voice-id VOICE_ID --attempt-file "/path/to/attempt-001.json"
```

Recovery preserves the original recording and rejection report and creates a saved candidate with the new verification evidence. By default it selects the recovered voice, requiring that the current prompt and script still match the original request. Add `--keep-unselected` to retain an alternative without changing the active voice. The interactive menu remains number, G, and Q.

Each failed chunk gets at most `max_attempts` attempts with distinct deterministic seeds. Failed audio and reports remain in the work directory; a rejected last attempt is never used as a successful result. Exhausted retries stop the run before audiobook publication. Inspect the failed transcript and audio before changing settings.

Rejection output includes a short `expected → heard` summary and the reason a secondary recognition check could not resolve it. Full transcripts and alignment details remain in the attempt JSON.

Run the same command again to resume. Accepted audio is reused only when its bytes, synthesis request, verifier identity, and quality policy still match. A changed verifier or quality policy triggers revalidation. Completed chapter checkpoints are also retained and reused when their inputs and assembly settings match. Progress output identifies reused clips and completed chapters; failure output identifies the work directory containing recovery data. Completed rejected attempts remain recorded rather than being retried indefinitely. Increasing `max_attempts` permits additional numbered attempts.

A normal interruption retains the cache and releases locks. After a hard kill or system crash:

1. Confirm the renderer process has stopped before touching locks.
2. Remove only the empty `.render.lock` directory inside that book's work directory and, if present, the empty `.lock` inside its series voice directory. A kill during final publication can also leave `.<output-directory-name>.publish.lock` beside the output directory; remove that empty lock after confirming publication has stopped. `rmdir` is appropriate because it refuses nonempty directories. Retain the rest of the cache and voice store.
3. Current runs publish the completed output directory atomically after validation; a failure retains the work cache. If an older renderer left an incomplete output directory without `render.json`, move that older output aside or choose a fresh `-o` directory. Existing output is never overwritten automatically.
4. Rerun the command. An incomplete voice asset or repeatedly interrupted raw attempt may require inspection; the renderer stops rather than silently replacing an established voice.

Voice, work, and output directories are separate children of the analyzed book by default. The renderer checks unsafe overlaps, including overrides that could write into a parent book or model files.

## Output and synchronization

The completed output directory (`<book-directory>/audiobook/` by default) contains:

| File | Purpose |
| --- | --- |
| `chapters/chapter-0001.m4a` or `.mp3`, etc. | Chapter audio using original manifest numbers, including when a subset is selected; empty entries are marked skipped. |
| `book.m4b` | AAC audiobook with chapter titles and timing. |
| `artwork/cover.jpg` or `.png` | Original declared EPUB cover, when available. |
| `sync.json` | Exact PCM frame ranges linking audio chunks to source text. |
| `source.json`, `cast.json` | Portable canonical text and voice assignments. |
| `verification.json` | Accepted chunk reports, transcripts, and hashes. |
| `audio-manifest.json` | Audio inventory, durations, chapter ranges, and checksums. |
| `render.json` | Completion marker, provenance, engine/verifier identities, voice hashes, and file checksums. |

The EPUB's byte hash must match the analyzed book before its cover is extracted. Declared JPEG and PNG covers are copied without changing the artwork and embedded in the M4B. Missing or unsupported cover formats produce no artwork; an unsafe or mismatched EPUB stops the run. All seven current Soulship EPUBs have supported JPEG covers.

Chapters and the final M4B are encoded independently from the same assembled lossless PCM. The final book is never built by re-encoding the lossy chapter files. FFprobe checks the resulting duration, chapter tags, and embedded cover. Completed chapter PCM, encoded chapter copies, accepted chunk PCM, and failed attempts remain in the work cache for recovery; the published audiobook does not duplicate the PCM files.

`sync.json` records chapter-relative and book-relative integer frame offsets, chunk IDs, segment and block IDs, and exact source ranges. Frame ranges exclude inserted pauses and have exclusive ends. Bracket delimiters stay in the source text but are omitted from spoken text; mappings retain that distinction. A configurable pause at the outer edges of a bracketed passage separates its delivery while preserving the character voice. Paragraph and blank-line pauses use the larger overlapping cue.

Synchronization currently has **chunk precision**. ASR word times are retained in verification reports, but are not yet published as validated word-to-source alignment. Audio players must account for codec delay and padding; container chapter times may round to milliseconds. Exact highlighting inside the original EPUB still needs DOM/CFI mapping; the current ranges address the included canonical `source.json`.

## Adding another TTS backend

The `TTSEngine` protocol separates engine identity, capabilities, voice creation, and synthesis from planning, verification, caching, and packaging. A separate `[voice_design]` configuration supplies reference creation when the speech engine lacks descriptive voice design. Setting its type to `engine` delegates creation to the speech engine only if that engine advertises voice-design support; random unconditioned generation is not a fallback. Saved voice bindings use a `kind` discriminator; the implemented kind is `reference`. Actual model content, runtime version, sampling settings, designer identity, and voice-asset hashes participate in cache identity.

Installed adapters can register a Python entry point in the `audiobook_renderer.tts` group. Set `engine.type` to that entry point's name and put adapter-specific values in `[engine.options]`; the factory receives that options dictionary and must return a `TTSEngine`. Its identity must include stable `engine` and `model` strings plus every setting affecting synthesis. Unknown or invalid adapters fail explicitly, with no fallback.

A future backend with predefined voices can add a `preset` binding and explicit character-to-provider-voice mapping. That mapping and backend are not implemented yet. Adding them does not require changing the analyzer bundle or audio assembly format. See [ENGINE_NOTES.md](audiobook_renderer/ENGINE_NOTES.md) for the current Drama 3 adapter's limits.
