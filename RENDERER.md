# epub-tts-narrator: audiobook rendering

`audiobook-render` consumes a version-2 analyzer bundle, creates or reuses persistent character voices, verifies each generated chunk, and produces chapter audio plus a chaptered `.m4b`. The analyzer and renderer read the same global `epub-tts-narrator.toml`, while the book directory and optional path overrides are command arguments. They remain separate commands.

Voice creation prefers Gemini when its API key is configured. Fish Audio Drama 3 renders the book locally on Apple Silicon, cloning the saved character recordings. The default model uses the **unofficial Sogni MLX port**; prequantized checkpoints use MLX-Audio, selected through `engine.model`. Sogni can also create references locally from scene prompts, and Qwen3 VoiceDesign remains an optional alternative. The Sogni port reports 48–55 GB peak memory for `q8` on an M5 Max; its model download is about 41.5 GB. Allow additional disk space for Whisper, dependencies, voice references, and retained attempts. These are upstream memory figures, not combined-memory guarantees for this renderer. [Sogni port](https://github.com/Sogni-AI/drama3-mlx), [scene casting format](https://arxiv.org/html/2610.18010v1#A2)

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
cd /path/to/epub-tts-narrator
uv venv --python 3.12 .venv-renderer
uv pip install --python .venv-renderer/bin/python -r renderer.requirements.txt
uv pip install --python .venv-renderer/bin/python -e .
```

FFmpeg and FFprobe must also be installed. Their executable names or absolute paths belong in the configuration file. The environment commands follow [uv's environment documentation](https://docs.astral.sh/uv/pip/environments/).

Unknown configuration keys, such as `engine.old_option`, warn and are ignored. Engine selection uses `engine.type`, `engine.model`, and `engine.checkpoint`; plugin-specific settings belong in `engine.options`. Invalid recognized values and unknown configuration sections still stop with an error. Ignored fields do not participate in audio cache identity.

Both original and prequantized Drama 3 models use this environment, with stable MLX-Audio `>=0.6.1,<0.7` and MLX Whisper `0.4.3`. The vendored Sogni port retains the inference code from commit `3b76bf84714d3766d7872f3409639a438e296cc8`; version `0.1.1+epub.1` updates its dependency metadata for the shared runtime. See [provenance](vendor/drama3-mlx/PROVENANCE.md). Upgrade an existing environment by rerunning the requirements install command above; there is no need to recreate it.

A local MLX-Audio 0.6.1 compatibility pilot exercised Sogni reference cloning and scene voice design, MLX-Audio 4-bit cloning, Qwen3 voice design, Whisper, and Parakeet. Every accepted clip had zero normalized word errors; three first attempts were rejected and retried by the normal quality checks. The usual renderer environment also passed a first-attempt 4-bit generation using the current config. Audio and verification records are in `output/shared-runtime-pilot/`. These short checks establish runtime compatibility, not full-book or perceptual quality.

When casting or rendering first needs a missing supported model, it shows the model and destination and asks before downloading the pinned files. Complete local models are reused. `doctor`, `plan`, listing voices, playback, and assembly do not download models. If a download is interrupted, rerun the same command to reuse its saved download progress.

Existing Hugging Face cache snapshot paths are also recognized, such as `~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/<commit>`. Missing files or broken links are downloaded through that cache, keeping the repository and commit identified by the path. You do not need to rename the directory or change the config to a project-local model folder.

To authenticate model downloads, add this optional table to your existing global configuration file, or edit it if already present:

```toml
[huggingface]
api_key = "hf_YOUR_TOKEN"
```

Keep the real token in your ignored local config. An omitted or empty key downloads anonymously; the tool does not pick up environment tokens or a saved Hugging Face login. Model access restrictions still apply to the configured account. The key is masked in download errors.

You can also download the local model files manually:

```sh
.venv-renderer/bin/hf download fishaudio/drama-3 \
  --revision b2b5ee998034b2975720906c815e7fb0e9f007bd \
  --local-dir "./models/drama-3"

.venv-renderer/bin/hf download mlx-community/whisper-large-v3-turbo \
  --revision a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb \
  --local-dir "./models/whisper-large-v3-turbo"
```

If those exact files already exist elsewhere, point the configuration at their existing directories instead of making duplicate downloads. Downloads use the configured paths, resolved relative to your config file, so remove paths copied from the old computer. `hf download` stores download metadata in the chosen directory. The standalone `hf` command uses its own authentication settings, not the application's TOML file. [Hugging Face download documentation](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download)

## Configure once

Both analysis and rendering use one global `epub-tts-narrator.toml`. Once configured, run the commands below from the project directory and omit `--config`. From another working directory, pass `--config /path/to/epub-tts-narrator.toml` to use that same file. Per-book and per-series configuration files are not needed.

Existing `epub-audiobook.toml` files still work when `epub-tts-narrator.toml` is absent. Keep using that file or rename it; do not create a second partial config, because the new filename takes precedence and the two files are not merged. If retaining the old filename, use it in the append command below.

For a new setup only, add the sections from [audiobook-render.example.toml](audiobook-render.example.toml) to the existing global `epub-tts-narrator.toml` alongside the analyzer sections. If you have not added renderer settings yet:

```sh
cat audiobook-render.example.toml >> epub-tts-narrator.toml
```

Do this once: TOML does not allow duplicate sections. Both commands load `epub-tts-narrator.toml` from the current working directory; `--config PATH` selects the same global file from elsewhere. Each command validates the recognized sections for the other command, then uses its own settings. No book or series needs a separate configuration file.

Edit the model paths before running. Configured paths are literal strings, resolved relative to the configuration file; shell variables and environment variables are not substituted. `~` is expanded for configured filesystem paths. The main settings are:

```toml
[engine]
type = "drama3_mlx"
model = "fishaudio/drama-3"
checkpoint = "./models/drama-3"
experts = "q8"
temperature = 1.0
top_p = 0.9
top_k = 30
max_new_tokens = 2048

[voice_design]
type = "auto" # Gemini when its key is configured; otherwise the local engine

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
paragraph_synthesis = true
max_paragraph_chars = 1000
max_attempts = 3
seed = 1729
sample_rate = 44100
normalize_pauses = true
sentence_pause_ms = 220
clause_pause_ms = 100
paragraph_pause_ms = 250
blank_pause_ms = 500
bracket_pause_ms = 120
allow_review = false
ffmpeg = "ffmpeg"
ffprobe = "ffprobe"

[quality]
max_wer = 0.0
```

The example file includes the remaining quality thresholds. Character auditions use prepared source excerpts or character-specific creative scripts; the legacy `voices.reference_text` setting is not used as a generic fallback by the book workflow.

For a directory produced by `epub-tts-narrator analyze`, the renderer obtains the series ID and continuation link from the book itself. Voices are stored in `<book-directory>/voices/<series-hash>/`; work files default to `<book-directory>/render-work/`, and the finished audiobook defaults to `<book-directory>/audiobook/`. Use `--work-dir PATH`, `-o PATH` (or `--output PATH`), and `--epub PATH` for per-run overrides.

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

Analysis notes are advisory and do not require a review or acceptance step. `--allow-review` and the global `allow_review` setting remain accepted for compatibility but are unnecessary. Rendering requires a valid script and assigned voices for every spoken passage in the selected chapters; missing speakers still block those chapters. Rerun `epub-tts-narrator analyze BOOK_DIRECTORY` to complete remaining assignments using the saved book. There is no implicit narrator substitution.

## Choose the speech model

Set `engine.model` to one of these model IDs. Omitting it preserves the existing
Sogni configuration. `checkpoint` is the local directory containing that model's
files, resolved relative to the config file; give each model its own directory.

| `engine.model` | Runtime | Download |
| --- | --- | --- |
| `fishaudio/drama-3` | Sogni | About 41.5 GB |
| `mlx-community/fish-audio-drama-3-8bit` | MLX-Audio | About 44 GB |
| `mlx-community/fish-audio-drama-3-4bit` | MLX-Audio | About 23.8 GB |
| `fcmeyer/drama-3-MLX-3bit-HQ` | MLX-Audio | About 21.3 GB |

For example, change the existing engine table to:

```toml
[engine]
type = "drama3_mlx"
model = "mlx-community/fish-audio-drama-3-4bit"
checkpoint = "./models/fish-audio-drama-3-4bit"
temperature = 1.0
top_p = 0.9
top_k = 30
max_new_tokens = 2048
```

`experts` controls only Sogni's runtime quantization. It is ignored for MLX-Audio,
where the checkpoint already determines quantization. Sampling settings still
apply. Missing files use the selected model's pinned download; `doctor` checks
the matching files and runtime. A directory identified as another model is
rejected rather than filled with an incompatible download.

Use the same Python 3.12 renderer environment for every model. To upgrade an
existing installation, run from the project directory:

```sh
uv pip install --python .venv-renderer/bin/python -r renderer.requirements.txt
.venv-renderer/bin/audiobook-render doctor
.venv-renderer/bin/audiobook-render render runs/book-01
```

Existing saved Drama 3 references work with either runtime. MLX-Audio does not
create voices from descriptions: use saved voices, import recordings, or configure
Gemini, ElevenLabs, or Qwen3 VoiceDesign for missing voices. The normal `auto`
setting uses Gemini when its key is present. Switching models invalidates generated
speech caches while retaining the saved voices.

The 4-bit checkpoint passed a short local narration/dialogue benchmark with lower
time and memory than Sogni q8. The 8-bit and 3-bit HQ options have config and adapter
coverage but have not been run locally. See [measurements and listening samples](scripts/QUANTIZED_TTS.md);
these results do not establish full-book reliability.

The config-based 4-bit adapter also rendered the Last Lantern demo: all six blocks
passed exact-transcript and acoustic checks across eight attempts (two silence
stalls were rejected). Resume reused the completed chapter, and assembly produced
an M4B. A separate 529-byte passage crossed two MLX-Audio text chunks and produced
32.69 seconds of verified speech with zero normalized word errors. Saved reference
recordings remained unchanged. Local evidence and audio are in
[`output/config-model-pilot/`](output/config-model-pilot/), including
[`validation.json`](output/config-model-pilot/validation.json) and
[`long-text.wav`](output/config-model-pilot/long-text.wav).

For a render pool, configure every worker with the same model. Quantized HTTP
workers inherit `engine.model` when their `model` field is omitted; an explicit
value must match it. The server checks the requested model and reports its selected
ID in `/health`. The pool defaults `model_revision` to the selected model's pinned
revision; keep an explicit override consistent with the files on every worker.

When using `[remote_chapters]`, set `engine.model` on the chapter-server host.
The client uses the server's advertised renderer identity; its local engine
settings do not select the remote server's model.

## Render across multiple machines

The main machine can distribute speech generation across its local Drama 3 model and multiple HTTP workers. Each request includes the selected immutable voice recording and its exact transcript; you do not copy the book or voice library to the workers. The main machine verifies every returned take, retains resumable clips, and assembles the chapters in source order. Casting still uses the configured voice designer.

On the second Apple Silicon Mac, install this project and `renderer.requirements.txt` as above. In that Mac's global `epub-tts-narrator.toml`, configure its local model and server:

```toml
[engine]
type = "drama3_mlx"
checkpoint = "./models/drama-3"
experts = "q8"

[server]
host = "0.0.0.0"
port = 8080
api_key = "YOUR_SHARED_SERVER_KEY"
```

Use a long random key and put the same value in the client's worker entry. These settings belong in each machine's existing global config; no book-specific config is needed. The server does not need Whisper, analyzer credentials, or a book directory. Start it from the project directory:

```sh
.venv-renderer/bin/audiobook-render serve
```

If the Drama 3 files are missing, startup offers the normal model download. Startup then loads the model before printing the listening address; cold loading and quantization can take several minutes. Wait for that ready message before starting rendering. The model stays loaded and one persistent inference thread owns its GPU work. Keep the command running while the main machine renders. The default host is `127.0.0.1`; binding to a LAN interface requires a configured key. For internet access, put the service behind HTTPS or a private network; the built-in server itself serves HTTP.

On the main Mac, add these entries to its existing global config, replacing the hostname and key:

```toml
[[render_pool.workers]]
name = "local"
type = "local"
concurrency = 1

[[render_pool.workers]]
name = "second-mac"
type = "fish_http"
url = "http://second-mac.local:8080/v1/tts"
api_key = "YOUR_SHARED_SERVER_KEY"
model = "drama-3"
concurrency = 1
timeout_seconds = 300
```

Then use the usual command:

```sh
.venv-renderer/bin/audiobook-render render out/longchills
```

The configured workers replace the default single renderer. Omit the `local` entry to generate all speech remotely; local Whisper verification is still required, but the coordinator will not download or load Drama 3. Missing voices can still be created with configured Gemini or ElevenLabs, or prepared with `cast` on a machine with the local model. There is at most one local inference slot; set each remote's concurrency to its server's capacity. One is appropriate for another Mac. Faster workers take more jobs as slots become free. Local inference and verification share a persistent execution thread to protect MLX; remote generation can continue while verification runs.

For a compatible hosted deployment, add another `fish_http` worker with its full HTTPS TTS URL and key. For example, Fish's hosted API uses `https://api.fish.audio/v1/tts` with `model = "drama-3"`. The adapter sends the Fish MessagePack request format, bearer authentication, a `model` header, inline reference audio, and `format = "wav"`. It also supports Drama 3's tagged multi-speaker paragraphs. Set `multi_speaker = false` on a worker if its deployment only supports single-speaker requests; the planner then uses single-speaker chunks throughout the pool. [Fish HTTP API](https://github.com/fishaudio/fish-audio-python/blob/main/_autodocs/endpoints.md), [Drama 3 speaker tags](https://huggingface.co/fishaudio/drama-3)

The built-in server implements this reference-based WAV rendering subset, including JSON/base64 requests. It does not implement Fish's voice-management API, saved remote `reference_id` lookup, or streaming PCM. A cloud deployment must accept this same wire format; merely having a route named `/v1/tts` is insufficient. Some SGLang/vLLM deployments use a different reference-audio schema and need an adapter. Saved character references remain local to the coordinator, and only the voice recordings and requested passages are sent to configured workers.

Within pool mode, changing worker addresses, credentials, order, or concurrency retains compatible accepted clips. `render_pool.model_revision` identifies the deployed model version; change it when replacing the model behind an endpoint. Sampling or voice changes invalidate the affected requests. Pool mode has a distinct synthesis identity from the original single-local mode, so first enabling it changes the cache identity; start it at a new book or finish an existing single-local run first if you need to avoid regeneration. Remote endpoint models and settings are recorded as execution provenance, while acceptance remains attached to the returned audio bytes. A transport failure stops after bounded retries; rerunning retries incomplete requests and retains accepted work. Verification thresholds are unchanged.

Validation includes two real HTTP requests to the local Drama 3 model using the existing demonstration voices: a 2.60-second single-voice passage and a 4.04-second two-character passage. Both passed strict Whisper text and acoustic checks, and the original reference checksums were unchanged. The report is `output/distributed-render-pilot-paoj0g29/report.json`. Automated tests additionally exercise two HTTP workers through chapter/M4B assembly and offline resume. The real-model pilot used loopback on one Mac; a second machine and hosted Fish service have not yet been exercised.

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

Startup automatically prepares all voices: it reuses existing selections, selects previously saved verified samples, and generates missing ones. By default, creation uses Gemini when `gemini.api_key` is configured, otherwise the local speech engine. This preference also applies to `generate`, `generate-missing`, `regenerate-all`, `create-voices`, and any missing references needed during rendering. Audiobook speech continues to use the configured local engine and its saved reference recordings. New voices and successful regenerations are selected automatically. There is no acceptance step. Missing samples show generation progress; an individual failure leaves the other voices available.

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
| `G` | Edit and regenerate the last voice played locally. If none has been played, asks for a voice number. |
| `G 5` or `G5` | Edit and regenerate voice 5 with the local speech engine. |
| `GG 5` or `GG5` | Edit and regenerate voice 5 with Gemini. No command-line flag is needed. |
| `GG` | Use Gemini for the last voice played, or ask for a voice number. |
| `G11 5` or `G115` | Edit and regenerate voice 5 with ElevenLabs. |
| `G11` | Use ElevenLabs for the last voice played, or ask for a voice number. |
| `Q` | Quit. |

All regeneration commands offer two fields: the voice prompt and the spoken sample script. Press Enter to keep either value. The new voice is verified, saved, selected, and played immediately. If generation fails, the previous voice stays selected. Inputs are case-insensitive. `G` chooses the local speech engine, `GG` chooses Gemini, and `G11` chooses ElevenLabs for that action; none changes the session's startup setting or global configuration. Use `G 11` with a space to regenerate voice 11 locally, since `G11` is the ElevenLabs command.

Menu and text entry use the terminal's standard line editor: arrow keys,
Ctrl+A/E, Ctrl+K/U, and session history follow your readline/libedit bindings.
While a sample is playing, any key stops playback and returns to the menu.

Sample scripts contain 25–60 words and at most 650 characters, normally aiming for 35–50 words. Casting prefers clearly attributed ordinary speech and can join several actual statements by the same canonical voice to provide a useful audition. Balanced outer dialogue quotation marks are removed before joining excerpts; embedded quotations and apostrophes are retained. Every excerpt keeps its own chapter, segment, exact source ranges, and position in the joined script; a montage is not represented as one contiguous quotation. Narrators can use narration, and identified participants in simultaneous dialogue can each audition those words alone. Shared roles use one script and one saved reference voice.

Every new voice-design request starts with `audiobook narrator portraying a character in a book:` followed by the voice description. This context applies to all designers, including edited and older saved drafts, and is added only once. It is not part of the spoken sample script. Custom draft wording, selected recordings, and inherited voices are preserved; new request records contain the actual prefixed prompt. This applies to both the casting menu and automatic voice creation during rendering.

For a creation prompt, describe the voice's lasting identity in one or two concise sentences using a few compatible audible traits: typical pitch, resonance, vocal weight, texture, accent, articulation, and natural cadence. Include age, gender presentation, or fictional context when they help establish the sound. Translate personality into vocal choices: precise consonants and measured phrasing for a meticulous character, or a thin, reedy tone and dry rasp for an abrasive creature. Preserve essential harsh or unusual qualities. Put temporary emotions, urgency, whispering, shouting, and changes of pace in individual passages' delivery directions. The voice generator knows nothing about the book: make the prompt self-contained, translating character names, fictional species, and lore into explicit audible characteristics.

Fresh analysis is instructed to make the description complete. Default voice creation uses only that description plus the audiobook context. Structured trait fields remain evidence metadata; neither those fields nor their source quotations are appended to the prompt. An older profile whose important vocal details appear only in trait fields may need its creation prompt edited before casting. Existing saved and custom prompts are retained as written, including older prompts with trait lists; the tool does not rewrite them automatically. An automatic prompt over 4,000 characters, including the context prefix, fails with a request to shorten it instead of silently truncating vocal identity. The sentence count and coherence are model guidance, not a guarantee. Speech verification checks the recording and transcript; listening determines whether the performance suits the character.

The adapters place this prose differently: Gemini receives it as the voice-design input, following its [one- or two-sentence prompting guidance](https://ai.google.dev/gemini-api/docs/generate-content/voice-design#prompting-best-practices-for-voice-design), while Drama 3 receives it as a character description in the [scene prompt's system message](https://arxiv.org/html/2610.18010v1#A2). Neither adapter sends our trait array as separate model controls. This does not establish that a trait list could never help a particular model; the generated voice still needs listening and evaluation.

If a benign prompt is rejected, edit it down to the essential vocal characteristics, then reintroduce details individually to investigate. These are clarity improvements and troubleshooting steps: no wording guarantees acceptance, and a rejection alone does not identify the word or characteristic responsible. The tool preserves the provider's error details and does not automatically rewrite your prompt after a rejection.

Fresh CLI analysis prepares the audition text automatically. If a character has too little suitable speech, the configured analysis model writes an original audition fitting the character, setting, and speech style. Generated text is explicitly marked as a casting script, never a source quotation or audiobook content. The book's `provenance.voice_samples` retains the prepared scripts and selection provenance.

Opening casting for an older book automatically prepares missing scripts using the same global configuration. Creative scripts are saved in `audition-scripts.json`, bound to the book's source and cast fingerprints, without editing its analysis, chapter scripts, character identities, or continuation history. Preparation happens before loading speech models. Source selections require no API calls; missing creative scripts use the configured text model and its cached responses. Prepared scripts are reused on later runs. Ambiguous or review-required attribution, extreme delivery, control syntax, and repeated short cries are excluded from source selection. A missing prepared script blocks voice generation instead of silently substituting generic prose.

Legacy generic and short automatic drafts are upgraded while edited voice descriptions, explicit user scripts, and useful established auditions are retained. Saved recordings and their cloning transcripts remain unchanged. The revised draft appears under **G** and is used when you regenerate; numbered playback continues to play the current recording. Listing voices, playing existing samples, and normalizing their loudness require no new script or text-model call.

To prepare drafts and print their scripts and provenance without loading speech models (the configured text model is called only for missing creative scripts):

```sh
.venv-renderer/bin/audiobook-render cast runs/book-01 \
  --action prepare-samples
```

To rewrite existing voice descriptions and rebuild sample scripts using the latest guidance:

```sh
.venv-renderer/bin/audiobook-render cast runs/book-01 \
  --action refresh-prompts
```

Add `--voice-id ID` to refresh one voice. This uses the configured analysis model to rewrite descriptions, retaining useful edited casting directions as context. It selects fresh book excerpts or prepares creative fallback scripts as needed. It replaces local casting drafts, including custom wording, and saves previous drafts and refresh details under the book's voice directory in `casting-refreshes/`. Shared roles receive one draft. Recorded voices and ancestor books stay unchanged; use `G`/`GG`/`G11` or `regenerate-all` afterward to create recordings from the refreshed drafts. Inherited voices still require `--include-inherited` with `regenerate-all` to replace their recordings locally.

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

To configure Gemini voice creation, put the following in the same global
`epub-tts-narrator.toml` file. The default `[voice_design] type = "auto"` prefers Gemini
when its API key is present; with no key, it uses the local engine. Existing files
that set `type = "engine"` explicitly keep local creation until changed to `auto`.
These settings also supply credentials for `GG` in the casting menu.

```toml
[gemini]
api_key = "YOUR_GOOGLE_API_KEY"
model = "gemini-3.8-flash-tts"
language_code = "en-US"
timeout_seconds = 180.0
```

```sh
# Open casting; missing voices prefer Gemini when configured.
.venv-renderer/bin/audiobook-render cast runs/book-01

# Optional: require Gemini even if voice_design.type selects another designer.
.venv-renderer/bin/audiobook-render cast runs/book-01 --voice-designer gemini

# Replace local or missing voices with Gemini-created voices; keep inherited ones.
.venv-renderer/bin/audiobook-render cast runs/book-01 \
  --voice-designer gemini --action regenerate-all

# Explicitly use the local speech engine for new voices.
.venv-renderer/bin/audiobook-render cast runs/book-01 --voice-designer local
```

An API error or failed verification is reported. Individual `G`/`GG`/`G11`
replacements preserve any existing selection on failure; `regenerate-all` clears
its target selections first, so failed replacements have no active voice.
Automatic selection does not silently switch providers after a failure.
Use `G` in the menu for an explicit local replacement, `GG` for Gemini, or `G11` for ElevenLabs.

To enable `G11`, add an ElevenLabs key to the same ignored global
`epub-tts-narrator.toml` file. ElevenLabs requires a paid plan for voice design
through its API; a free account returns `feature_not_available`:

```toml
[elevenlabs]
api_key = "YOUR_ELEVENLABS_API_KEY"
model = "eleven_ttv_v3"
timeout_seconds = 180.0
guidance_scale = 5.0
loudness = 0.0
```

Open casting normally and enter `G11 5` to replace voice 5, or `G11` to
replace the last voice played. To use ElevenLabs for missing voices in this
session as well:

```sh
.venv-renderer/bin/audiobook-render cast runs/book-01 --voice-designer elevenlabs
```

For a persistent default, change the existing `[voice_design]` table to
`type = "elevenlabs"`. Adding the ElevenLabs key alone leaves `auto` selection
unchanged: Gemini when configured, otherwise local. Credentials are read from
the file, never environment variables. Missing credentials fail before casting
can replace a selected recording.

ElevenLabs receives the voice description and sample script to design preview
audio. The preview passes the same transcription and audio verification as
other designers, then is saved locally for Drama 3 to clone. This does not add
a permanent voice to the ElevenLabs account or require ElevenLabs during book
rendering. Shared roles and later books reuse the saved reference as usual.
Returned alternatives are cached for verification retries. If casting is
interrupted while preparing a missing ElevenLabs voice, reopening casting
resumes the newest unfinished request with matching text, prompt, book, and
voice settings, reusing downloaded previews. Explicit `G11` always starts a
fresh design. Corrupt request records stop recovery with an inspection path.
The description sent to ElevenLabs must contain 20–1000 characters including
the audiobook context prefix; the sample script must contain 100–1000
characters. Unsupported lengths produce an actionable error without truncating
or rewriting your script. `eleven_multilingual_ttv_v2` is also supported;
`guidance_scale` accepts 0–100 and `loudness` accepts -1–1.

Casting prints the API error message and response details, including the
HTTP status, model, and any field validation errors. Network errors
include their underlying reason. These diagnostics are also saved in each
`attempt-*.json` report. Messages are not shortened; configured credentials are
masked and terminal control sequences are removed. The final failure includes
the last generation or verification exception as well as the inspection path.

Gemini casting sends the voice description and selected sample script to Google's
API. It creates a persistent custom voice, synthesizes the exact script, and runs
the same independent transcription and audio checks as local creation. Existing
selections are reused unless you regenerate them. Shared character roles still
use one canonical reference, and later books inherit it through `--continue-from`.
Drama 3 renders the audiobook locally by cloning the verified recording; Gemini
is not needed to render that saved voice. Normalized audition copies work for
both sources.

The remote voice ID is recorded in candidate metadata and reused when the same
sample needs a synthesis or verification retry. Explicit `GG` regeneration and
bulk regeneration request a fresh Gemini voice, even when an earlier creation
failed or is awaiting verification. Prior pending records are preserved and do
not block explicit regeneration. After success, casting prints the selected
Gemini `voice_...` ID; the character's `series_voice_...` ID remains stable.
Startup recovery still protects an uncertain creation outcome: follow its
pending-file instructions to recover that voice, or use `GG` to deliberately
create a new one. Credentials are read only
from the configuration file and are excluded from candidate metadata and cache
identities. No Google SDK is required.

Google currently allows up to 200 stored custom voices per project, with expiration
after one year of inactivity. Each regeneration creates another custom voice;
manage unused remote voices in AI Studio. Local saved recordings remain usable
after a remote voice expires. See [Google's voice design documentation](https://ai.google.dev/gemini-api/docs/voice-design).

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

Changing the designer affects future creations and regenerations. Existing voice selections stay fixed. All designers retain the same persistent reference format and shared-character bindings.

Description conditioning is not proof that every audible trait was followed. Whisper checks the spoken words; it does not assess gender presentation, age, accent, timbre, or the suitability of a performance. Use the menu to audition and regenerate a voice when needed.

For scripts, `cast --action list` prints the cast and sample history as JSON without loading a model. `generate` immediately selects its verified result; `generate-missing` prepares unselected voices, reusing existing samples first, except voices explicitly cleared by bulk regeneration. `edit` saves prompt changes without generation, and `play` plays the selected voice.

`regenerate-all` clears all targeted active selections before generating replacements. It targets each distinct canonical voice that is local or missing, skipping voices inherited from earlier books. Add `--include-inherited` to explicitly replace inherited voices too. Shared narration and dialogue are generated once. Skipped voices need no new sample scripts or generation calls; if every voice is inherited, the default command does nothing.

Failed replacements stay unassigned and appear as `FAILED — no active voice` in the casting menu, with their saved error details. Targets not attempted before an interruption appear as `needs regeneration — no active voice`. Reopening casting preserves these states without adopting old samples or retrying automatically. Use `G number`, `GG number`, or `G11 number` to fix them; a successful replacement clears the failure. Rendering cannot use an old reference or fall back to an ancestor for a cleared voice. Original recordings remain as explicit recovery history only, and existing audiobook audio is not rewritten.

Before generation, the command warns about the targeted voices and linked roles, then continues without confirmation. In the directory workflow, replacements and failure states belong to the current book and may be inherited by its descendants; earlier books and sibling branches are unchanged, including with `--include-inherited`. Legacy shared-library mode warns that changes affect every book using that library and voice ID. The command continues after individual generation failures and reports them at the end. Run it in the book where you want replacement voices to start applying. Replace the voice ID below with an ID from the listing:

```sh
.venv-renderer/bin/audiobook-render cast runs/book-01 \
  --action generate --voice-id VOICE_ID_FROM_VOICES \
  --prompt-file "/path/to/voice-description.txt" \
  --text-file "/path/to/sample-text.txt"

.venv-renderer/bin/audiobook-render cast runs/book-01 \
  --action play --voice-id VOICE_ID_FROM_VOICES

.venv-renderer/bin/audiobook-render cast runs/book-02 \
  --action regenerate-all

# Also create local replacements for voices inherited from earlier books:
.venv-renderer/bin/audiobook-render cast runs/book-02 \
  --action regenerate-all --include-inherited
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

Rerun the same `render` command after updating the scripts or selected voices. It reuses matching verified clips and unchanged completed chapters, generates the changed passages, and rebuilds the affected chapter files, synchronization data, and M4B in the existing output directory. Export sequence renumbering after splitting a passage does not regenerate later unchanged speech. An unchanged local render also skips final encoding. Older published outputs without the new publication key are repackaged once from their compatible cached audio.

The previous audiobook remains available while its replacement is prepared. Failed speech or encoding leaves it intact; a failed final rename restores it. Replacement requires a complete, unmodified renderer output for the same source book, with no extra files or symlinks. Unrelated or manually modified output directories are refused before generation. Keep personal files outside the generated output directory. A partial chapter selection cannot replace an audiobook covering different chapters: use `--chapters-only`, followed by a full `render` or `assemble`, or publish the subset to another directory.

M4A chapters with AAC audio at 128 kb/s are the default. They are directly playable and use the same codec as the final M4B. `--chapter-format mp3` on `render` or `assemble` produces MP3 chapter copies when needed; the final book remains AAC in an M4B container. Bitrates are global settings in `[render]`.

Drama 3 joins adjacent fragments within a paragraph only when they have the same role and voice. Character/narrator/character spans produce separate synthesis requests, including when the narrator and character share a saved voice. This prevents a missed model speaker switch from assigning a narrative tag to the character. Long performances split at existing safe boundaries under `max_paragraph_chars` (default 1000). Bracketed passages, simultaneous speech, and passages whose source mapping cannot be joined exactly retain their individual treatment. Other engines without paragraph support retain the existing clip workflow. Set `paragraph_synthesis = false` to disable grouping altogether.

Accepted individual clips and paragraph performances within one role/voice remain reusable. Older cached paragraphs combining different roles or voices are no longer reused: their original passages are rendered separately, and affected chapter checkpoints are rebuilt. Word recognition checks the requested text, not the speaker's identity or timbre; a passing transcript cannot prove that an internal voice switch happened. Synchronization records each generated interval and its original text ranges and voice assignments in `source_segments`. Grouped entries have `precision = "paragraph"`; exact word timestamps are not inferred. If an older script itself assigns a narration tag to a character, rerun `epub-audiobook analyze BOOK_DIRECTORY` to repair those boundaries before rendering. No voice regeneration is needed.

If a paragraph exhausts its generation attempts because of missing speech, silence, or another audio quality failure, rendering automatically retries its original shorter passages individually, using the same assigned voices and verification rules. The recovery decision is saved in `render-work/paragraph-fallbacks` before any individual passage is generated, so interruptions and later invocations keep the smaller boundaries. Previously verified clips and completed chapters are reused; failed paragraph takes remain available for inspection. Each recovered passage has its own source mapping and audio interval. An API or verifier outage does not trigger splitting, and a shorter passage that still fails its own bounded attempts stops the run normally.

To generate and listen to one chapter before committing to the whole book, inspect the chapter list and save a chapter checkpoint:

```sh
.venv-renderer/bin/audiobook-render plan runs/book-01
.venv-renderer/bin/audiobook-render render runs/book-01 \
  --chapters 3 --chapters-only
```

Advisory notes do not require an override. Unassigned spoken passages still block the selected chapter. `--chapters-only` saves lossless chapter audio, its encoded playback copy, timing, and provenance beneath `<book-directory>/render-work/chapters/`; it does not publish a final audiobook directory. The CLI prints each playable file's path. `render-work/completed-chapters/chapter-NNNN.json` records the current checkpoint for each manifest chapter, pointing to its files under `render-work/chapters/<cache-key>/`. A completed chapter remains available if a later chapter fails.

Rendering, assembly, encoding, and cache reuse show the chapter's position in the full book, such as `chapter 6 / 35`, even when rendering only selected chapters. During speech generation, a live terminal line shows completed blocks, elapsed time, and the active block's generation or verification phase. A block is usually a paragraph; standalone passages and split paragraphs count separately. The count advances only when the entire block succeeds, including verification and any simultaneous speakers.

After at least three newly generated blocks, the display estimates the remaining speech work in that chapter from the text length and observed speed. This is an approximate estimate for generation and verification, excluding chapter encoding and final packaging. Reused clips do not inflate the measured generation speed. Drama 3 does not expose a within-paragraph percentage, so active generation shows elapsed time rather than an invented percentage. Redirected output receives periodic plain-text updates instead of terminal control sequences. Each finished chapter also reports its elapsed time, total run time, and the number of selected chapters ready. These display changes do not alter audio cache keys or regenerate accepted speech.

To generate the remaining chapters and package later:

```sh
.venv-renderer/bin/audiobook-render render runs/book-01 --chapters-only
.venv-renderer/bin/audiobook-render assemble runs/book-01 \
  --epub /path/to/book-one.epub
```

`assemble` uses the verified chapter checkpoints and loads neither TTS nor transcription models. Missing, changed, or incompatible checkpoints stop assembly with an explanation; use `render` to prepare them. Changing only chapter format or bitrate can re-encode from saved lossless PCM during assembly. Changes to text, voices, pauses, or loudness require `render` again; unchanged accepted speech is reused. Verification-policy changes alone do not invalidate accepted clips or chapters. Use the same `--work-dir` if it was overridden. To package a subset, pass the same `--chapters` selection to `assemble` and choose a separate `-o` directory, preserving the final output location for the full book. Assembly can update the same book's existing output using the same replacement checks as rendering.

All pauses added during assembly are controlled in the global config's existing `[render]` table. Each value is an integer from 0 through 5000 milliseconds:

| Setting | Default | Where it applies |
| --- | ---: | --- |
| `sentence_pause_ms` | 220 | After `.`, `!`, `?`, or an ellipsis at an audio clip boundary |
| `clause_pause_ms` | 100 | After a comma, semicolon, colon, or dash at an audio clip boundary |
| `paragraph_pause_ms` | 250 | At a source paragraph boundary, including the chapter's final paragraph |
| `blank_pause_ms` | 500 | Before a paragraph marked as following a blank line |
| `bracket_pause_ms` | 120 | On entering or leaving bracketed text |

Pause normalization is enabled by default with `[render] normalize_pauses = true`. During chapter assembly it conservatively removes excess silence at the outer edges of each clip, retaining protective margins around speech. The settings are total gap targets: retained quiet tails and heads count toward the pause, and overlapping cues use the largest target. For example, a sentence ending a paragraph uses the larger of the sentence and paragraph settings. Zero disables a particular added pause; another applicable cue or protected audio may still leave a gap. Protected audio is never removed just to meet a target.

Pauses inside a clip are unchanged. When adjacent passages with the same role and voice are generated as one continuous performance, their internal sentence pauses remain under the TTS engine's control. Role and voice changes use separate clips. There is no separate added speaker-change pause, and chapter concatenation adds no extra silence of its own. Set `normalize_pauses = false` to retain original clip padding and disable sentence/clause targets; paragraph, blank-line, and bracket pauses are then added on top of the original padding.

Changing pause settings rebuilds chapter audio from the accepted clips; it does not regenerate speech or modify voice references. Rerun `render BOOK_DIRECTORY --chapters N --chapters-only` using the same work directory, then listen to the newly printed chapter file. Use `assemble` afterward to package it, choosing a new output directory if a finished audiobook already exists. Timing records include `source_audio_start_frame` and `source_audio_end_frame` so each assembled clip can be traced to the retained range of its original accepted WAV. Source text ranges remain unchanged.

By default, completed chapters receive two-pass loudness normalization to -20 LUFS with a -1.5 dB true-peak ceiling. Normalization preserves the PCM frame count and timing data, and operates on the assembled chapter rather than flattening every short utterance to the same level. Voice references and verified speech clips remain unchanged. These controls are separate from casting playback: use `[render] normalize_loudness`, `target_lufs`, and `true_peak_db` for the audiobook. Empty chapters and chapters shorter than the loudness measurement window are explicitly recorded as skipped for normalization.

`render` also creates missing voices if no casting edits or audition candidates exist. It prepares any missing character audition scripts before loading speech models, using the global analysis provider when creative text is needed. Use `cast` first to choose your voices before a long run. Without a worker pool, speech generation is sequential on the configured local model. With a pool, independent requests run across available workers; each local model and the coordinator's Whisper verifier remain serialized. Configured HTTP workers receive the requested passages and selected reference recordings.

## Audio storage and cleanup

By default, `[render] keep_intermediate_audio = false` removes redundant raw speech WAVs after the normalized take is accepted and its checksum is verified. The accepted WAV and verification records remain available for resume, changing pauses, and rebuilding chapters without synthesizing speech again. Raw files from unsuccessful or pending requests remain available for recovery. Set the option to `true` to keep raw engine output for debugging. This setting does not affect voice creation or any voice samples.

For files created by earlier runs, preview and then run cleanup for one book:

```sh
.venv-renderer/bin/audiobook-render cleanup runs/book-01 --dry-run
.venv-renderer/bin/audiobook-render cleanup runs/book-01
```

The default cleanup removes redundant raw speech files and complete chapter cache versions that are no longer referenced by any saved chapter checkpoint. It preserves accepted speech, normalized rejected takes, current chapter checkpoints, and finished audiobook outputs. Cleanup loads no TTS or transcription model and refuses to run alongside a render or assembly for the same book or work directory. Use the same `--work-dir` as the render if you overrode it. Dry runs list the files and bytes eligible for removal.

To remove the generated speech and chapter caches as well, use the explicit discard option:

```sh
.venv-renderer/bin/audiobook-render cleanup runs/book-01 --discard-render-cache --dry-run
.venv-renderer/bin/audiobook-render cleanup runs/book-01 --discard-render-cache
```

This deletes the managed `clips`, `mixes`, `chapters`, and `completed-chapters` stores under that book's render work directory, including the playable checkpoint files created by `--chapters-only`. **Speech resumability is lost:** rendering again will generate new speech using the saved voices. Published M4A/MP3 chapters and the M4B in the output directory remain intact.

Both modes always preserve voice samples, cloning references, casting candidates (including failed attempts), audition previews, and ancestor books. They do not traverse or delete the voice stores. Remaining accepted speech and current lossless chapter WAVs are intentional: they avoid lossy re-encoding and retain exact synchronization when reassembling. Voice-sample compression is not performed; their existing identity includes the original WAV bytes, so converting them in place would invalidate voice identities and speech caches.

## Verification and recovery

Before synthesis, the renderer trims surrounding whitespace and quotation marks (straight, curly, and guillemets) from each speech string. Interior quotation marks, apostrophes, and sentence punctuation remain intact. This prevents a closing quote at the runtime's byte limit from becoming a separate generation target that can repeat the preceding passage. Dialogue turns are trimmed before adding speaker or delivery controls; their word boundaries are preserved when joined. Source scripts, synchronization ranges, saved cloning references, and verification text remain unchanged. Previously verified clips and chapters remain reusable; newly generated takes record the actual `synthesis_text` in their engine metadata.

Parentheses wrapping an entire target, or left dangling at its edges by a speaker split, are also removed from the synthesis input. For example, `(Lie,` becomes `Lie,`, and `I thought fiercely).` becomes `I thought fiercely.`; the same words are still required by verification. This avoids reference-sample repetition triggered by an incomplete parenthetical target. Balanced parentheticals inside a sentence remain intact, and a thought and its narration remain separate performances.

Each chunk is normalized to mono PCM16 at 44.1 kHz. Before acceptance, the renderer checks duration, overall level, excessive silence, long internal silence, and clipping. Local Whisper then transcribes the audio independently: it receives no expected transcript as a prompt. The comparison records added, missing, and substituted words, with conservative punctuation and number normalization.

Exact normalized text and complete transcripts differing only by bounded word spacing pass directly. For example, `over taking` / `overtaking` and `sourcepalace` / `source palace` need no JEV or second recognizer. Every letter must remain in order, and each change must join or split one alphabetic word and two or three whitespace-separated pieces of at least two letters. Numbers, apostrophes, punctuation boundaries such as `re-sign`, and any additional missing or changed content do not qualify. The report preserves the original word-error metrics and records each spacing change. Acoustic checks still apply.

Next, a local [Double Metaphone](https://pypi.org/project/Metaphone/) check compares full phonetic codes. It accepts spelling and pronunciation differences such as `Derrins` / `Darin's`, `Qarm` / `Khorm`, and `Vandire` / `Van Dyer` without calling JEV. Unlike four-character Soundex, these keys are not truncated. The ordered alignment must consume every token, matching one word to one word or one word to two or three whitespace-separated words. Multiword matches concatenate each word's full code, retain punctuation boundaries, and exclude function words and unchanged words with extra neighbors. Numbers, ordinals, and negations must remain unchanged. Empty codes cannot match. The original transcript and word-error metrics remain in the report, and `phonetic_comparison` records the keys and aligned spans. Acceptance is labeled `double_metaphone` or `secondary_double_metaphone`.

This prioritizes missing or extra speech over pronunciation fidelity. Phonetic codes can collide: `Qarm`, `Quorum`, and `Karim` share a code, for example. The check is a pronunciation heuristic, not proof of word identity or syllable count. It never removes unmatched tokens to make a passage pass. Acoustic checks still apply. The small `Metaphone==0.6` package is included in `renderer.requirements.txt` and the CUDA verification extra; existing environments can install it with `uv pip install --python .venv-renderer/bin/python 'Metaphone==0.6'`. Verifier startup reports a missing dependency before synthesis. Analysis does not require this package.

With JEV enabled, remaining nonexact transcripts reach JEV for sound-and-coverage judgment. Without JEV, the configured `max_wer` applies to chapter audio and the existing independent-recognizer rules apply to references. Neither speech recognizer receives the expected text as a prompt.

Whisper word timestamps are optional synchronization metadata. If they overlap, leave their segment, or are otherwise unusable, the report retains an alignment warning and omits word times while still checking the actual transcript and audio. This does not trigger speech regeneration. Chapter assembly and source synchronization retain accurate chunk boundaries from the audio duration; no word timestamps are invented. Voice samples do not request word timestamps.

An optional Parakeet recognizer supplies a second audio-only transcript. With JEV, the primary transcript is judged first. A second recognizer is used only if that judgment does not pass; a good primary result cannot be vetoed by a bad secondary transcription. Without JEV, reference recovery retains its bounded substitution, compound-boundary, and expressive-consonant checks and requires independent transcript agreement or an exact secondary match to the request.

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

### Optional JEV confirmation

JEV compares two written representations of speech and judges **sound and coverage only**. Spelling, recognition mistakes, homophones, and minor pronunciation differences can pass even when the written meanings differ. Missing, extra, repeated, or replaced spoken content without a plausible sound match fails. A local soundalike cannot excuse missing speech elsewhere. This applies to voice samples and rendered chapter clips; the acceptance threshold is unchanged.

The complete requested chunk or sample and complete recognized transcript are sent to JEV. The concise prompt lives in `QUESTIONS` in `epub_tts_narrator/jev.py`; saved judgments include the exact prompt and its policy identity. JEV receives no audio and can still make classification mistakes. A prompt-policy change causes previously rejected takes to be rechecked on resume; already accepted audio retains its original evidence.

The exact prompt is `QUESTIONS["roughly_same"]` in `epub_tts_narrator/jev.py`, including its `instructions` and `criteria`. New attempt reports also save the exact question under `report.text_judge.questions`, with requested/recognized text under `report.text_judge.comparisons` and the score under `report.text_judge.probabilities`. Inspect `attempt-001.json` (or another attempt) in the directory printed after a verification failure. Historical reports retain their old policy version and prompt hash; reports created before this addition do not contain the full question text. The internal verdict name `equivalent` means accepted as plausible speech recognition, not semantic equivalence.

The primary transcript is judged first. If it passes, verification finishes without a second recognizer. Otherwise an optional independent recognizer can resolve the discrepancy, either by matching the requested words exactly or receiving its own passing JEV judgment. A negative first judgment is retained in the report when a second transcript is checked. An unavailable JEV service preserves the audio for a retry.

When JEV is uncertain, `verification.phonetic_fallback = "analysis"` (the default) uses the existing global `[provider]` analysis model to review a small number of word substitutions. This is inactive without a configured analysis model; set it to `"off"` to disable it. It sends the differing word pairs with short text context, never audio. Three separate calls receive no prior votes, and at least two must approve **each** pair. These votes can still share model errors; they are not independent acoustic evidence. The local alignment must retain every word in order; insertions, deletions, reordered words, changed numbers, and changed negation cannot use this fallback. JEV's acceptance threshold stays unchanged.

The fallback prompt lives in `epub_tts_narrator/renderer/phonetic_review.py`. Reports retain JEV's original uncertain judgment plus the fallback's exact prompt, provider identity, word pairs, and votes under `report.phonetic_reviews`. Verification records the actual accepting method as `phonetic_primary` or `phonetic_secondary`. Enabling the fallback changes verification identity so rerunning rechecks saved failed audio before synthesizing another take. A fallback service error pauses for a retry and retains the recording. Remote chapter servers use their own global `[provider]` configuration; client credentials are not forwarded.

For a live synthetic regression check, run `.venv-renderer/bin/python scripts/evaluate_pronunciation_review.py --output output/phonetic-evaluation.json`. This sends only the script's synthetic cases to the configured analysis provider and saves full prompts and votes. It checks unfamiliar names and homophones alongside missing speech, changed numbers, synonyms, and missing syllables; it exits unsuccessfully if a case fails.

There are no character-distance, word-error-percentage, substring, or short-name rejection gates before JEV. Every usable transcript beyond exact normalized text, complete word-spacing matches, or complete local phonetic matches reaches it. Acoustic failures and malformed/empty recognition still fail directly. Whisper uses scalar temperature zero and greedy decoding, so repeatedly decoding identical audio with unchanged settings normally repeats the result; independent recognition is the fallback instead.

Enable it once in the global `epub-tts-narrator.toml`; no book or series settings are needed:

```toml
[jev]
enabled = false
api_key = "YOUR_JEV_API_KEY"
model = "jev-latest"
timeout_seconds = 30
threshold = 0.75
```

Replace the placeholder and set `enabled = true` to use it. The key is read only from this file. Keep one `[jev]` table when combining templates. `timeout_seconds` accepts 1–120; `threshold` must be greater than 0.5 and at most 1. The default 0.75 is an acceptance threshold, not a measured accuracy guarantee. Pin the model version for repeatable evaluation.

If JEV is unavailable, the run stops and retains the generated WAV. Rerun the same command to retry verification without generating it again. A different or uncertain judgment rejects the take under the normal retry policy. JEV receives text, so listening still determines whether the voice, accent, or performance suits the character.

Already accepted audio keeps its original verification evidence and is reused across changes to the verifier or its policy. Text, voice, synthesis settings, and audio integrity still determine whether a clip can be reused. Failed takes are rechecked under changed verification rules. Completed chapter checkpoints likewise retain their original acceptance evidence. API keys never enter reports or cache identities.

To recheck a retained casting attempt without generating another voice:

```sh
.venv-renderer/bin/audiobook-render cast runs/book-01 \
  --action recover \
  --voice-id VOICE_ID --attempt-file "/path/to/attempt-001.json"
```

Recovery preserves the original recording and rejection report and creates a saved candidate with the new verification evidence. By default it selects the recovered voice, requiring that the current prompt and script still match the original request. Add `--keep-unselected` to retain an alternative without changing the active voice. The interactive menu remains number, G, and Q.

Each individual speech request gets at most `max_attempts` new generation attempts per render invocation (three by default). Rerunning the same command gives failed passages a fresh allowance with new deterministic seeds. Take numbers continue across runs (`attempt-004`, `attempt-005`, etc.), preserving previous rejected normalized audio and verification reports. Abandoned raw files without a normalized take are deleted when a new generation starts. The allowance is shared across chapter recovery and duplicate requests within that invocation, so internal retries cannot reset it. A rejected last attempt is never used as a successful result. Exhausted retries stop the run before audiobook publication. Inspect the failed transcript and audio before changing settings.

Rejection output includes a short `expected → heard` summary and the reason a secondary recognition check could not resolve it. Full transcripts and alignment details remain in the attempt JSON.

Run the same command again to resume; no cache deletion or extra retry flag is needed. Accepted audio is reused when its bytes and synthesis request still match; changing the verifier or quality policy does not revalidate or regenerate it. The original successful report remains attached to those bytes. Completed chapter checkpoints are also retained and reused when their synthesis inputs and assembly settings match. Saved unsuccessful audio is rechecked first when verification settings change or JEV confirmation was interrupted. Otherwise the renderer generates new takes for failed individual passages. Failed paragraph groups retain their exhausted budget and recover by splitting into their original passages as described above. This automatic retry allowance does not regenerate voice references or change casting selections. Removing a cached audio file regenerates that clip; modifying an existing file fails its integrity check. Progress output identifies reused clips and completed chapters; failure output identifies the work directory containing recovery data.

A normal interruption retains the cache and releases locks. After a hard kill or system crash:

1. Confirm the renderer process has stopped before touching locks.
2. Remove only the empty `.render.lock` directory inside that book's work directory and, if present, the empty `.lock` inside its series voice directory. A kill during final publication can also leave `.<output-directory-name>.publish.lock` beside the output directory; remove that empty lock after confirming publication has stopped. `rmdir` is appropriate because it refuses nonempty directories. Retain the rest of the cache and voice store.
3. New outputs are published by directory rename after validation; replacement briefly moves the previous complete output to a sibling `.<output-directory-name>.previous-*` directory. Normal exceptions restore it. If a hard kill occurred between those renames and the output is absent, restore that previous directory before rerunning. If both directories exist after a hard kill, keep the previous copy until the current output has been verified. If an older renderer left an incomplete output directory without `render.json`, move that incomplete output aside or choose a fresh `-o` directory. Complete, intact outputs for the same book update automatically.
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

Installed adapters can register a Python entry point in the `epub_tts_narrator.tts` group. Existing registrations in `audiobook_renderer.tts` remain supported when the configured name has no registration in the new group. A matching registration in the new group takes precedence; duplicate matches in the selected group are rejected. Both rendering and `doctor` use this lookup. Set `engine.type` to that entry point's name and put adapter-specific values in `[engine.options]`; the factory receives that options dictionary and must return a `TTSEngine`. Its identity must include stable `engine` and `model` strings plus every setting affecting synthesis. Unknown or invalid adapters fail explicitly, with no fallback to a different engine.

A future backend with predefined voices can add a `preset` binding and explicit character-to-provider-voice mapping. That mapping and backend are not implemented yet. Adding them does not require changing the analyzer bundle or audio assembly format. See [ENGINE_NOTES.md](epub_tts_narrator/renderer/ENGINE_NOTES.md) for the current Drama 3 adapter's limits.
