# epub-tts-narrator: audiobook rendering

`audiobook-render` consumes a version-2 analyzer bundle, creates or reuses persistent character voices, verifies each generated chunk, and produces chapter audio plus a chaptered `.m4b`. The analyzer and renderer read the same global `epub-tts-narrator.toml`, while the book directory and optional path overrides are command arguments. They remain separate commands.

Voice creation prefers Gemini when its API key is configured. Fish Audio Drama 3 renders the book locally on Apple Silicon, cloning the saved character recordings. The default model uses the **unofficial Sogni MLX port**; prequantized checkpoints use MLX-Audio, selected through `engine.model`. Sogni can also create references locally from scene prompts, and Qwen3 VoiceDesign remains an optional alternative. The Sogni port reports 48–55 GB peak memory for `q8` on an M5 Max; its model download is about 41.5 GB. Allow additional disk space for Whisper, dependencies, voice references, and retained attempts. These are upstream memory figures, not combined-memory guarantees for this renderer. [Sogni port](https://github.com/Sogni-AI/drama3-mlx), [scene casting format](https://arxiv.org/html/2610.18010v1#A2)

**Validation status:** automated tests cover real FFmpeg encoding, synchronization, quality gates, adapters, interactive casting, and voice replacement across books, and live pilot runs have produced verified chapters and complete M4B output. Full-book rendering at scale and perceptual suitability still need exercise and listening; structural validation does not prove a voice fits its character.

## Install

Run these commands from the project directory. They create a dedicated Python 3.12 environment without activating it or setting environment variables:

```sh
cd /path/to/epub-tts-narrator
uv venv --python 3.12 .venv-renderer
uv pip install --python .venv-renderer/bin/python -r renderer.requirements.txt
uv pip install --python .venv-renderer/bin/python -e .
```

The requirements include Textual for the full-screen casting screen and run shell. FFmpeg and FFprobe must also be installed. Their executable names or absolute paths belong in the configuration file. The environment commands follow [uv's environment documentation](https://docs.astral.sh/uv/pip/environments/).

For development and the full offline test suite, install `test.requirements.txt` instead of only the runtime requirements. It includes the runtime plus server/storage and worker-decoder test dependencies, including `boto3` and `faster-whisper`:

```sh
uv pip install --python .venv-renderer/bin/python -r test.requirements.txt
.venv-renderer/bin/python tests/run_detached.py
```

Rerun the install command when test dependencies change; do not recreate an existing environment. Tests require no model downloads, cloud credentials, or previously generated book bundles.

Unknown configuration keys, such as `engine.old_option`, warn and are ignored. Engine selection uses `engine.type`, `engine.model`, and `engine.checkpoint`; plugin-specific settings belong in `engine.options`. Invalid recognized values and unknown configuration sections still stop with an error. Ignored fields do not participate in audio cache identity.

Both original and prequantized Drama 3 models use this environment, with stable MLX-Audio `>=0.6.1,<0.7` and MLX Whisper `0.4.3`. The vendored Sogni port retains the inference code from commit `3b76bf84714d3766d7872f3409639a438e296cc8`; version `0.1.1+epub.1` updates its dependency metadata for the shared runtime. See [provenance](vendor/drama3-mlx/PROVENANCE.md). Upgrade an existing environment by rerunning the requirements install command above; there is no need to recreate it.

A local MLX-Audio 0.6.1 compatibility pilot exercised Sogni reference cloning and scene voice design, MLX-Audio 4-bit cloning, Qwen3 voice design, Whisper, and Parakeet. Every accepted clip had zero normalized word errors; three first attempts were rejected and retried by the normal quality checks. The usual renderer environment also passed a first-attempt 4-bit generation using the current config. Audio and verification records are in `output/shared-runtime-pilot/`. These short checks establish runtime compatibility, not full-book or perceptual quality.

When casting or rendering first needs a missing supported model, it shows the model and destination and asks before downloading the pinned files. Complete local models are reused. `doctor`, `plan`, listing voices, playback, and ordinary assembly do not download models. Assembly with censorship can need the configured Whisper model to align saved speech when reliable word timing is unavailable; it follows the same model-download prompt. If a download is interrupted, rerun the same command to reuse its saved download progress.

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
temperature = 0.7
top_p = 0.7
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
normalize_take_pauses = true
level_takes = true
take_level_tolerance_db = 3.0
sentence_pause_ms = 500
clause_pause_ms = 250
paragraph_pause_ms = 700
blank_pause_ms = 1200
bracket_pause_ms = 200
allow_review = false
ffmpeg = "ffmpeg"
ffprobe = "ffprobe"

[quality]
max_wer = 0.0
max_close_mismatch_fraction = 0.25
```

Takes of one chapter can differ by several dB in speech level for no reason in the text, and `normalize_loudness` moves the whole chapter, not one take against another. With `level_takes`, assembly measures each take's speech level and compares it with where it should sit: the chapter's typical take, 6 dB lower for a line directed whispered, soft, quiet or murmured, and 4 dB higher for one directed shouted or yelled (its delivery direction, or a whisper or other delivery control covering most of it). A take within `take_level_tolerance_db` of that is left as it is, so ordinary contrast stays; one further off is brought back to the edge of that range, by at most 12 dB and never past -1 dBFS peak. Saved takes are never changed: only the assembled chapter is, and its record lists every take's level and gain. Changing either setting assembles every chapter again without generating any speech.

`max_new_tokens` is the most Drama 3 codec frames any take may use (2048 is about 95 seconds). Each take also gets a budget from its own text: 10 seconds plus half a second per spoken letter (three per digit; speaker tags and bracketed directions do not count), never more than `max_new_tokens`. Drama 3 sometimes stops speaking without ending the take and keeps generating near-silence; the budget ends such a take long before the full limit, and it is rejected and retried as an unfinished take. No accepted take of a measured 12,734-passage book exceeded its budget, so takes that end on their own are unchanged and the budget is not part of any cache identity.

On `fish_speech_cuda` workers and in local MLX rendering (`drama3_mlx`) each take is also watched while it decodes: every two seconds (every half second once it has gone quiet) its newest half second is decoded and measured the way verification measures silence (25 ms windows below -60 dBFS). After 8 seconds of continuous silence, plus any `[pause for N seconds]` the text requests, the take ends at once. In chapter rendering its audio is kept: it is cut a quarter second after its last speech (the codec frames are cut at the same point) and verified like any take. A take that had said every word and only failed to end is used; one that stopped mid-line fails, counts as a Drama 3 truncation, and so leads to sentence-split recovery (below) after two such takes. A take that went silent is never chapter history and never a best-effort choice. Outside chapters (casting, previews, auditions) it still fails as an unfinished take. A take that pauses once for up to 7.5 seconds (three times verification's 2.5-second internal-silence limit) and then finishes is still accepted, because assembly shortens the pause (stall recovery), so the watch waits longer than that; at 3 seconds it cut off takes that would have resumed and passed, and they failed for their missing words. Long passages go silent far more often: in one week of `serve` logs, 0.5% of takes under 100 characters, 3.3% at 100–299, 12% at 300–599 and 17% from 600, while takes with and without chapter history went silent equally often (2.4% with two history turns, 3.9% with none), which is why recovery splits a passage rather than dropping its history. Takes that end on their own are unchanged, frame for frame. On a measured book the watch would have saved 5.8% of all decoding; the watch itself costs about 1.4% of decode time on an RTX PRO 6000 and about 3% on Apple silicon, where each codec decode is relatively more expensive.

A take that said the start of its passage correctly and then stopped (it ended early, or the silence watch stopped it) is finished instead of discarded, once per passage per render. Its transcript must be the start of the passage: it ends on the two words just before the first missing one, near the same position, mostly matching in order, after at least three words, and its only rejection reason is its words. The take is cut just after its last heard word (the recognizer's word end, moved forward to the next quiet window), and the next attempt generates only the missing words in the same voice, delivery, accent cue and pronunciation controls. The kept start is the latest chapter history turn for that generation (its codec frames cut at the same point), so the missing words continue the same reading. The two are joined and the joined take is verified in full like any take; one that fails goes on to ordinary recovery. Finishing a take does not use up one of the passage's fresh attempts. A completed take records `engine_metadata.completion` (the take it finished, the seconds kept, the words generated and their seed) and never becomes chapter history. Whole lines, sentence parts and one-voice paragraph groups are completed (a paragraph's missing words are the rest of the turn where it stopped plus every later turn); a line with scoped controls only when the cut does not split a control, and a paragraph that mixes voices never. The kept start takes the latest history turn's place, so a completion still hears Fish's first turn and at most two in all. A completion the worker refuses falls back to ordinary takes instead of failing the chapter.

In the run shell's takes list, `t` marks a take **truncated**: what it says is right, but it stops before the passage ends. Such a take is never used as it is (marking the used take truncated retires it, like `b`), and the next render of its passage finishes it first: it keeps everything the take said, without checking those words again against verification, and generates only the rest, with the same history and joining. The joined take is still verified in full. If the words where the take stopped cannot be found in the passage, the render says so and makes ordinary takes.

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

Choose chapters per book with `--chapters 3` or `--chapters 3-5,7` on `plan`, `cache-report`, `render`, or `assemble`. Numbers use the **one-based order of script entries in the bundle manifest**, including front matter. Ranges are inclusive, duplicate numbers are ignored, and chapters always stay in source order. Omit the option to use all entries. `plan` lists the corresponding titles. The legacy global `chapter_numbers` setting remains supported, but book-specific selections belong on the command line. A selected subset produces a partial audiobook and records `partial_book: true`.

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
temperature = 0.7
top_p = 0.7
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
speech caches by default while retaining the saved voices. Updating the runtime or
adapter does not invalidate verified speech or completed chapters; their versions
remain recorded as provenance, and no alias is needed for a runtime update.

To keep already verified speech when changing models, add an optional
`cache_model_alias` to the existing engine table. For example, when moving from
the original model to the 4-bit checkpoint:

```toml
[engine]
type = "drama3_mlx"
model = "mlx-community/fish-audio-drama-3-4bit"
checkpoint = "./models/fish-audio-drama-3-4bit"
cache_model_alias = "fishaudio/drama-3"
```

Keep your existing sampling settings in this table. The alias explicitly declares
the models, revisions, and quantization settings interchangeable for
successful cached speech. Text, selected voice references, delivery, shared sampling
settings, seed, and output sample rate must still match. New or changed passages
use `engine.model`; the alias does not change the model being loaded or reported
in generation provenance. It applies to the built-in Drama 3 engine, including
quantized models and render pools.

Use the previous model ID as the alias to reuse recordings made before this setting
existed. Compatible saved successes are matched automatically on the next render;
no migration command or reanalysis is needed. You can instead choose a new group
name and keep it consistent across future model changes, but an arbitrary new name
cannot match older recordings from unrelated model IDs. Without this option, the
normal model, revision, and quantization safeguards remain in force.

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
host = "0.0.0.0" # Default; binds every interface, so a generated api_key is required.
port = 8080
api_key = "" # serve generates and saves a random key if missing or empty.
```

These settings belong in each machine's existing global config; no book-specific config is needed. The server does not need analyzer credentials or a book directory, and needs Whisper only to act as a [recognition worker](#recognition-workers). Start it from the project directory:

```sh
.venv-renderer/bin/audiobook-render serve --host 0.0.0.0 --port 8080
```

`--host` and `--port` override the configuration for this invocation without saving those overrides. Omitted flags use `[server]` defaults (`0.0.0.0:8080`; use `server.host = "127.0.0.1"` or `--host 127.0.0.1` for loopback-only listening). The auto-serve worker that small jobs start themselves always binds loopback regardless of `server.host`.

When `server.api_key` is missing or empty, `serve` generates a random key and saves it in the config it loaded (including a file selected with `--config`). Copy that saved value into the client's worker `api_key` below. Subsequent starts reuse it; existing keys are never rotated automatically. The key is not printed. The update preserves other settings and comments, writes the file with owner-only permissions, and must succeed before the server starts.

If the Drama 3 files are missing, startup offers the normal model download. Startup then loads the model before printing the listening address; cold loading and quantization can take several minutes. Wait for that ready message before starting rendering. The model stays loaded and one persistent inference thread owns its GPU work. Keep the command running while the main machine renders. Binding to a LAN interface requires a configured key. For internet access, put the service behind HTTPS or a private network; the built-in server itself serves HTTP.

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
roles = ["tts", "recognition"]  # optional; default ["tts"]
recognition_concurrency = 1     # recognition slots on this worker
```

When rendering starts, each TTS worker is asked once over `/health` whether it carries chapter history. A worker that gives no answer at all (switched off, asleep, or off the network) is set aside: the log says so, it is sent no speech, and the render goes on with the other workers. The same happens to a worker that stops answering partway through: when a take gets no answer (a refused connection, no route, or running out of time), a short `/health` request asks whether the worker is still there. If that gets no answer either, the worker is set aside and the take is sent to another worker at once, without counting as a failed attempt; a take that runs out of time on a worker that has gone is not sent there again with a longer limit. A worker that still answers is only slow or refusing, so its failure stays an ordinary failed attempt, and the last worker left is never set aside. Every 30 seconds each set-aside worker is asked again (the same chapter-history check as at the start); once it answers rightly the log says so and it takes speech again, in its configured place. A worker that answers wrongly (an older protocol, a refused key) at the start still stops the render, and so does having no worker that answers.

### Recognition workers

Speech recognition (Whisper, plus the reference recognizer used for second opinions) can also run on `serve` machines. The controlling machine always recognizes on its own recognition thread; each worker whose `roles` include `"recognition"` adds `recognition_concurrency` more slots, and every take's recognition call joins one FIFO queue and claims the first free slot, so several clips are recognized at once across machines. A worker with `roles = ["recognition"]` only recognizes and adds no TTS slots. Everything else in verification (acoustic checks, word comparison, JEV and phonetic review, acceptance) stays on the controller, and censorship alignment and casting always use the local recognizer.

A `serve` process offers recognition when its own config sets `[verification] model` (and `reference_model` to provide second opinions); its `/health` then lists the `recognition` role and the recognizers' identity: model file hashes, package versions, language and options, never local paths. When rendering starts, the controller requires each recognition worker's identity to equal its own recognizers' exactly; a worker with different models or package versions stops the render with the names of the differing fields (install the same model files and versions there, or remove `recognition` from its roles). Because a matching worker runs exactly the controller's recognizers, saved verification identities do not change and existing accepted takes stay valid. Each response is also checked against the identity verified at startup, so a worker restarted with other models is refused. With remote recognizers configured, each attempt report records `recognized_by` (`local` or the worker name). A Runpod CUDA worker uses faster-whisper, which cannot match an MLX Whisper controller, so it stays TTS-only unless the controller also uses `faster_whisper`.

Then use the usual command:

```sh
.venv-renderer/bin/audiobook-render render out/longchills
```

The configured workers replace the default single renderer. `render --no-local` (with one book or several) leaves the `local` entry out for that run only, without editing the config: speech is made only on the remote workers and this Mac loads no Drama 3, so it is free for other work (recognition and verification still run where configured). It needs at least one remote TTS worker, and a configured `[remote_chapters]` service is already remote, so the flag changes nothing there. Omit the `local` entry to generate all speech remotely every time; local Whisper verification is still required, but the coordinator will not download or load Drama 3. Missing voices can still be created with configured Gemini or ElevenLabs, or prepared with `cast` on a machine with the local model. There is at most one local renderer. With `drama3_mlx`, its `concurrency` above 1 (at most 16) turns on batched decoding: the one loaded model decodes that many passages together, one frame per step, and the pool gives it that many slots (see "Batched MLX decoding" below). Otherwise it is one slot. Set each remote's concurrency to its server's capacity: one for another Mac, or that Mac's `server.batch_rows` when it batches. Faster workers take more jobs as slots become free. Local generation and local recognition each own a persistent MLX thread with its own streams, so verification runs beside generation instead of pausing it (the GPU interleaves both; same-seed Drama 3 audio is not bit-reproducible even alone, so this changes no guarantee). `serve` likewise recognizes on a separate thread from its synthesis.

On an interactive terminal, `render` keeps one live row per active chapter at the bottom of the screen, plus a scheduler row, for example `Ch 39/92 | 12/110 blocks (11%) | 02:01 | #13 second-mac gen 2/6 00:37 | ~05:00 left`. A chapter is always called by its own number, the one its log lines, failures, `--chapters` and the run shell use; when chapters with no speech are hidden, the row adds its place among the chapters being rendered (`Ch 16, 10/40`). Rows show which worker is generating, a generic worker wait when all slots are busy, and local normalization or verification with the audio's worker name. Synthesis takes start in FIFO submission order and each claims the first free worker; chapters have no worker affinity, and a queued take holds no lane thread or worker slot. Everything else scrolls above the rows, tagged with its chapter: chapters starting, reused or finished, rejections (`Ch 39: Chunk 2 (audio from second-mac): Attempt 1/6 rejected: …`), retries and recovery. Routine per-chunk completions appear only in the rows. Redirected output (a log file or pipe) keeps plain status lines every 30 seconds. Cache reuse does not report new worker activity.

On an interactive terminal, `render` (one book or a multi-book queue), `assemble`, `import-voices`, `create-voices` and bulk casting (`cast --action generate-missing`, `prepare-samples`, `normalize-samples` or `regenerate-all`) run inside a full-screen run shell instead (Textual, installed by `renderer.requirements.txt`). `render` gets the dashboard below; the other commands get the plainer task layout described in [ANALYSIS.md](ANALYSIS.md#long-books-resuming-and-cost-controls) (health, step, model requests, warnings, errors, latest output; log; errors). `--plain`, redirected input or output, `TERM=dumb`, or an environment without Textual keep the line output above unchanged. The shell has five views:

- **Summary** (`1`, the first view): at a glance, the run's health (running normally, a stall with no block accepted for 15 minutes, failures, waiting for retries, stopping, or how it ended), how many books are rendering, assembling, queued, done or failed, how many blocks are generating (and waiting for a worker) and verifying right now, blocks accepted with the rate over the last five minutes, rejected takes, failures, and the estimated time left. Under the tiles, one line per render node shows its busy speech and verify slots (`second-mac: speech 6/8 · verify 1/1`). Below that, one row per book with its percentage done (finished chapters plus the accepted share of active ones), state, chapters ready out of the total, active chapters, blocks and failures. `Enter` on a failed book shows why it failed (the error that stopped it, and its failed chapters with theirs), a suggested fix, and how to resume it: while the run is going, put it back in the queue now (see below); after the run, resume it. It also offers to show the book's chapters. `Enter` on any other book drills into it: the other views show only that book.
- **Chapters** (`2`): every chapter of the shown books (one book after a book filter, otherwise every book in the queue), in queue and chapter order, whether it has started or not. A queued book lists its chapters from its manifest until it starts; once it starts, its render's plan (the selected chapters with their block counts) replaces them. Each row shows the book, chapter, title, state, blocks accepted out of the total, rejected takes this run, elapsed and estimated time left, and what it is doing now: its active blocks with their worker and phase, or a failed chapter's error. The states are `rendering`, `assembling` (its speech is done and the chapter audio is being built), `retrying`, `queued` (waiting inside a rendering book), `not started` (its book has not started), `saved earlier` (a checkpoint from an earlier run, checked when the chapter is reached), `done` (`done · earlier run` when restored from that checkpoint), `failed` (no take of a passage passed verification), `blocked` (something must be fixed first) and `stopped`. Above the table, a breakdown counts the chapters in each state, per book and for all books; the tab's label shows chapters done out of the total. `s` shows only chapters in one state. Rows update in place, so the highlighted chapter and the scroll position stay put. `Enter` on a chapter shows its state, blocks, rejected takes, error, active blocks (phase, worker, voices) and recent messages. The scheduler's `sched` gauges appear underneath.
- **Nodes** (`3`): first the chapter workers: each renders one chapter and sends its blocks to whichever node has a free slot, so they belong to no node (`Chapter workers: 5 of 16 busy`, with each one's book, chapter, blocks and active blocks, and idle ones listed). Then what each speech node reports doing now: its batch rows, how many are decoding, how many requests are having their prompt read (prefilling) and how many wait on the node for a free row, its requests in progress from every client, and how many this render has sent it. Batched decoding is continuous: a passage joins a free row when its prompt has been read and leaves when it finishes, so rows start and end at different times. Remote workers are asked over `/health` every two seconds and a local batched engine is read in memory; a node that cannot be reached, or an older `serve`, shows `—`. Then this render's own requests on every node, one row per slot: each `[[render_pool.workers]]` entry's speech (TTS) slots (its `concurrency`, for example 8 on a batched Mac) and the verify (transcription) slots of this machine's recognizer and of any worker with the `recognition` role, each with the book, chapter and block it is working on and how long ago it was sent (which includes waiting on the node), or `idle`. These slots are this render's requests in the order they were sent, not the decoder's physical rows. Last, the render pipeline's stages (generating speech, waiting for a slot, transcribing, reviewing, preparing audio) with their active and queued counts.
- **Log** (`4`): every message the line output prints, tagged with its book and chapter and colored by level.
- **Rejected** (`6`): every take verification rejected during the run, with its book, chapter, block, your verdict (`good`, `bad` or `not judged`) and reasons; the tab's label counts the takes still to judge. A failed request that made no audio (for example a refused connection to the speech server) is not a take and is not listed. `Enter` opens all takes of that passage side by side (see the takes dialog below), where you can judge each one and change which take is used. `e` writes your judgements for later detection work (see below).
- **Errors** (`5`): every chapter or book failure. For a chapter that failed on one passage, `Rejected takes` shows how many takes of it verification rejected and how many you have still to judge (`5, 2 to judge`, or `5, all judged`). The table shows each error's first line, wider on a wider screen; a longer error ends in `… (Enter: full error)`. A chapter that failed again after a retry is listed once, with its newest failure; a chapter that rendered after a retry shows `retried`. `Enter` or `d` opens the failure (`d` also opens the highlighted chapter, book or rejected take from its own view) (so does `Enter` on a failed chapter in Chapters, and each failed chapter listed in a failed book's dialog): first, in plain words, what its attempts did (for example "Drama 3 went silent partway through this passage and stopped, on all 3 attempts; no audio was kept"), then the whole error, never cut short (`PageUp`/`PageDown` scroll it); the failed passage (shown even when no attempt made audio) (chapter, block, speakers, its text and direction); every attempt of it in order, whether it made audio, its rejection reasons and its full error, including attempts that stopped with an engine error and made no audio; a suggested fix; the saved record (chapter number, error type, block, speakers, and the failed take's `segment_ids`, `chunk_id` and `request_hash` when they exist, as saved in `render-work/status.json`); and the repairs that apply. Every failure offers something to do; when a repair is impossible, the dialog says why and what to do instead.

Chapters with no passages are hidden from chapter lists, render progress and completion summaries, as they are from the M4B's chapter navigation. Visible chapter totals exclude them. Original chapter numbers remain unchanged for chapter selection, cached audio and failure links, so the displayed numbers can have gaps. A chapter with passages stays visible even when it has not rendered or its render failed. Saved source, plan and skipped-chapter records retain the empty entries for compatibility and resume.

`review-ui` can also ignore a nonempty chapter: select it, press `I` and confirm. `H` shows ignored chapters so `I` can restore one. The choice is local to this book and takes effect on the next render or assembly; it does not change a running render's selection. Ignored chapters remain in the plan under their original numbers but contribute no speech or audiobook chapter marker. This is a full book with chosen omissions, distinct from an explicit `--chapters` partial render. Assembly can omit an ignored chapter even when it has no completed checkpoint, and an existing checkpoint is retained for restoration. Title-only repairs preserve ignore choices; a changed partition must be reviewed again rather than moving the choice to another chapter. Source text and cached takes are not deleted by ignoring a chapter; ordinary pruning can still remove audio the current plan does not need.

`b`, `c` and `w` filter the log and errors by book, chapter and worker, `s` filters chapters by state, `V` shows only chapters with an active block spoken by one voice; `v` cycles the level (information, warnings, errors only, everything), `/` searches text, and `x` clears the filters. `?` lists every key. `r`, `t` and `d` act on the highlighted row of the view you are in (a chapter in Chapters, a book in Summary, a failure in Errors, a take's chapter in Rejected), never on another view's highlight, and notices name the chapter by the number its row shows, with its title.

Chapter failures do not stop the other chapters, as in the line interface. The repairs:

- **Retry** (`r`, or Retry in the failure's detail): a retry always tries again. The failed passage gets a new allowance of new takes; a used-up best-effort choice and clip repair of that passage are set aside first (renamed under a `fresh-retry` suffix beside a `fresh-retry-*.json` record, never deleted), so they run again from the new takes. Every verified take is reused, a take you chose is used as chosen (even when the passage had gone on to sentence-split recovery), and takes you marked bad or set aside stay unused. Whatever recovery state the failure left, the retry makes at least one new take of the failed passage before it can fail again: a sentence split whose joined take was rejected (by verification, or because the best-effort chooser could publish none of the takes, e.g. every one misread a chosen name) goes back to new whole takes, with or without a speech resolver. When a retried chapter fails again, a notice says so and how many new takes its retry made, and its row in Errors reads `failed after retry N`.
- **Retry at any time.** A failed chapter can always be retried while the run is alive: by its book's render while that render runs or waits for retries; through its book when that render has ended in a multi-book queue (the book goes back in its command-line place, finished chapters are restored and the failed ones render with fresh takes); and, when no render can take it yet, as soon as one can: when its book's render ends in a queue, or when the run ends, which then renders that book again by itself. After the run, `r` renders the book again. A retry that itself hits an unexpected error fails only its chapter, which can be retried again.
- **Where a retry goes**: back at the front of the queue while the others keep rendering. Its clips are dispatched ahead of every waiting clip (still in their own order), several retried chapters run at once, and an engine that renders one chapter at a time renders it next. After the run has ended, `r` renders the book again instead (failed passages get new takes, completed work is reused); a rerun or `R` gives every failed passage new takes the same way.
- **Change the passage's direction**: edits how the line is performed (at most 240 characters) and saves it to the book as your choice, exactly as `review-ui` does (backups, later books rebound; a live render's lease does not refuse it). The running render keeps the script it loaded, so the new direction is used when the book renders again: after the run (`F` ends the wait for retries), `R` renders the passage with it and new takes. A paragraph passage joining several lines is edited line by line in `review-ui`. The spoken words cannot be changed here: they are the book's own text (for a word said wrong, use `pronounce`). Splitting a passage into shorter parts is not a manual repair: speech recovery splits on its own after repeated Drama 3 truncation, or after recoverable rejections when the passage has shorter parts to fall back on.
- **Show the chapter's messages**: the Log filtered to the book and chapter.
- **When a name keeps coming out wrong** (`renderer/history_review.py`): Drama 3 says a name the way the passage's history says it, so the cause is usually something the passage hears. Three views of the failed passage, each listing takes with where they are (book, chapter), what part they play, how the name check heard the problem word and whether the take is in the book; `Enter` plays the take around the word (an excerpt written to `render-work/name-excerpts/`), `f` the whole take, `g`/`b` mark it good or bad (a bad take in the book is retired and its chapter renders again):
  - **What it hears**: the takes its attempts heard: the voice's first line in the chapter, its previous line, and any example of the name taken from another chapter or an earlier book. `a` keeps this passage from hearing the highlighted take again (`A`: every listed take); only this passage changes (`render-work/history-avoid.json`), nothing else renders again, and the chapter is retried.
  - **Other takes of the word**: every take of the voice saying the problem word, this book then earlier books, in book order: a sweep for takes saying it wrong that are teaching it.
  - **Stream**: the voice's chain of takes in the chapter that led to the passage, each heard by the next. `r` renders the stream again from the highlighted take (by default the first line): it and every later take of the voice in the chapter (in the chain and the chapter's saved checkpoint) are set aside after you confirm the count, then the chapter is retried. `h` changes only this passage's immediate history (its first and latest turns), like `A` in the first view. Set-aside takes carry a `redo` verdict in `take-feedback.json`: never used or heard again and their acceptance retired (renamed, never deleted), but not counted as a verification disagreement in the feedback export.
- **Put the book back in the queue** (`Enter` on a failed book in Summary, or `r` on it in Errors, in a multi-book render), for example after choosing its missing voices with `cast` or confirming words with `pronounce` in another terminal. The order of the books on the command line is the queue's priority, and a retried book takes its own place in it. A book listed before one that is rendering is admitted at once: the chapters already rendering finish, then its chapters take the next free chapter slots, ahead of the later books' chapters. A book listed after every rendering book waits for room as a queued book does, still ahead of books listed after it. When only failed books are left, the queue waits for retries the same way.
- **Listen to the takes** (`t`), for a chapter that failed because no take of one passage passed verification, opens the takes dialog for that passage. When no attempt made audio (each stopped with an engine error, such as Drama 3's silence watch), there is no take to hear; the dialog says so and the retries above make new ones.

The takes dialog lists every retained take of one passage, including the one used (●) even when verification passed it or it is a repaired copy: whether verification passed or rejected it, its length, why it failed, your verdict, and whether it can be used, with the passage and what the recognizer heard. `Enter` plays a take and `Space` stops, so you can compare them; `>` switches between 1x and 2x (the take playing starts again at the new speed; see Listening at 2x). `g` marks a take good (verification wrongly rejected it) and `b` marks it bad; `c` clears the mark. A take marked bad is never used again. Marking the take in use bad (●) retires it at once (its acceptance is renamed, never deleted) and its chapter goes back to the front of the queue to render that passage again, as using a take does; after the run, `R` or the next render does it. No recheck, resumed render or best-effort choice picks a take marked bad, so a passage whose takes are all bad gets new ones. Any other mark does not change the take in use right away, but the next time the passage is rendered (a chapter retry, a rebuild or a rerun) a take you marked good is used instead of making new ones: the one you judged most recently, if its audio is unchanged and usable. `u` uses the highlighted take for the passage at once: if the passage is still rendering, it stops there (no new takes, no repair, no best-effort choice; a take already being made is not used, even if it passes). A failed take is saved as an audited manual selection (never as a pass; the failed report is kept), and its chapter goes back to the front of the queue: a failed chapter renders again, and a chapter that already rendered is rebuilt from its verified clips (after it finishes, if it is still rendering), before its book is assembled. After the book has been assembled, `R` or the next render rebuilds it. When a take failed only because words were heard differently, marking it good or using it can also teach the recognizer: those words become person-approved heard variants (`heard-variants.json`), the same as confirming a `pronounce` test sentence, so later takes heard the same way pass. A missing or extra word, a number or a negation is never learned. A history echo or damaged audio cannot be used.

Your verdicts are saved in the book's `render-work/take-feedback.json` (using a take counts as judging it good). To study where detection goes wrong, write every take you judged differently from verification (rejected but good, or passed but bad) with the passage, what the recognizer heard, the reasons and metrics, and where its audio is:

```sh
.venv-renderer/bin/audiobook-render take-feedback runs/book-01
```

This writes `render-work/take-feedback-review.json` (`--file PATH` elsewhere, `--all` to include the takes where you agreed); `e` in the Rejected view writes the same file for each book in the run. The file contains book text and transcripts, like the rest of `render-work`.
- **Resume the book** (`R`): after the run, renders again only the books that did not finish (or only the highlighted book from the errors view), reusing every verified clip and completed chapter.

How a render can fail, and what to do: a passage no take could pass (listen and accept, or retry); a chapter-local recovery or cache problem such as changed audio (retry); a book that cannot start, such as missing voices, unsure pronunciations or another render holding its lock (fix it, then put the book back); and run-wide stops (an unavailable verifier or model service, a configuration or setup error, or an interruption), which end the run so it can be resumed with `R` once the cause is fixed.

When the other chapters are done and failed chapters remain, the render waits ("waiting for retries") instead of ending; finish it with `F`, which reports the remaining failures exactly as the line interface does (`status.json`, no M4B). When the run ends with a failure or is interrupted, the shell stays open on the errors. A run that rendered everything but left rejected takes you have not judged (in Rejected, or of a failed passage) stays open on the Rejected view; `q` closes it whenever you are done. `R` is not offered after `cast --action regenerate-all`, because running it again would replace every voice again. A clean run with every rejected take judged closes the shell by itself.

`p` pauses the render, for moving a laptop between networks: no new speech request (to any worker, local or remote) and no model or JEV review request starts, while local work on takes already made carries on. Requests already running finish. One that loses its connection while paused (no answer at all: refused, unreachable or timed out, never an answer from the server) is sent again unchanged after resume, so it never counts as a failed take or an unavailable verifier. For two minutes after resuming, a request that still cannot connect is tried again every five seconds while the network comes back. `p` again resumes. Summary shows "Paused" meanwhile, and the paused time never counts as a stall. The pause covers every book of a render queue and stays on across `R`. Stopping still works while paused.

Nothing stops a render without asking. `q`, `Ctrl-Q`, `Ctrl-C` and `F` each open a confirmation whose default is to keep running, so `Enter` or any number of `Escape` presses keep it going; press `y` to confirm. A confirmed stop behaves like `Ctrl-C` in the line interface: active tasks finish verification and are saved first. Books and chapters that had not finished are shown as stopped, never as failed, and `R` resumes them from where they stopped. Asking again while it is stopping offers to stop immediately. Model download questions appear as a dialog. When the shell closes, the run's last messages and every error are printed to the terminal. Exit codes are unchanged.

The server prints timestamped request IDs as requests queue, start inference, complete, fail, or disconnect. Completion includes elapsed time and returned audio duration. A failure names its cause without quoting any text, for example `failed (EngineError: the take hit its frame budget without ending)`, an ordinary rejected take the controller retries. A chapter take that went silent completes instead, with `; the take went silent and was cut after its last speech`: its response carries `finish_reason: "silence"` and the controller verifies it (an older controller, or a preview, still treats it as unfinished); an unexpected error shows only its type and its cause's type, and its full message goes to the controller. Queue wait is separate from actual inference; logs include character/reference counts but never book passages, reference transcripts, or API keys. Output is flushed immediately even when redirected. Update and restart `serve` on the remote Mac to enable these logs; client progress uses the coordinator's code.

A `serve` process whose `[engine]` is `fish_speech_cuda` or `drama3_mlx` can set `[server] batch_rows` (default 1; at most 32 for CUDA and 16 for MLX; no more than `max_pending_requests`) to decode that many requests together on one loaded model; give that worker the same `concurrency` in the controller's `[[render_pool.workers]]`. Each request keeps its own prompt, history, sampling, seed and returned codes, and batched takes have their own engine identity. Other engines reject `batch_rows` above 1.

#### Batched MLX decoding

EXPERIMENTAL and off by default (`concurrency = 1`, `batch_rows = 1`). A local `drama3_mlx` renderer with `concurrency = N`, or a `drama3_mlx` `serve` with `batch_rows = N`, decodes up to N passages per step on one loaded model (`mlx_drama3_batch.py`): each passage's prompt is prefilled alone exactly as the vendored runtime does, then joins the batch with its own caches at its own position, and leaves when it finishes. Each passage keeps its own seed (counter-hash noise, the same stream as the CUDA batched decoder), sampling, history and silence stop. The start of a prompt that a voice's later takes repeat (its references, then the chapter's pinned history turn) is processed once and reused from a bounded in-memory cache, with results identical to processing it again; this matters because each new passage's prompt otherwise stalls every row while it is read (measured: 43% of the decode thread at 8 rows). A `serve` worker with `batch_rows` set (and `max_pending_requests` at least as large) does all of this for the requests it receives; give its controller worker entry the same `concurrency`. Its `GET /health` reports `concurrent_requests` (the batch rows it decodes at once, 1 when unbatched), so you can check a worker before raising its `concurrency`, and `work`: the requests it is handling now from every client (not counting the health check itself) and, when batching, its rows: `{"requests": 9, "rows": 8, "decoding": 6, "prefilling": 1, "waiting": 2}`. Each batched request's `completed` log line ends with how it shared the batch, for example `; batch: joined 5 running of 8 rows after 0.40s, 6.5 rows on average (peak 8) over 120 steps, prompt 731 rows (512 cached), prefill 0.21s`: rows already decoding when it joined, its wait for a free row, the average and peak rows during its own decode steps, and how much of its prompt came from the prefix cache. While a batched request decodes, `serve` watches its connection: if the client goes away (a restarted render, a timeout), the request leaves the batch at the next step and its row goes to the next request (`client disconnected; its batched decode was stopped`). A controller that gets `503` (queue full, nothing started) waits the server's `Retry-After` and sends again, for up to the worker's `timeout_seconds`, without spending one of the take's attempts. A worker's `timeout_seconds` (and `[server] request_timeout_seconds` for the auto-started local server, default 300) is the time limit for a short passage: each character of the passage adds half a second, up to an hour, because a long passage in a full batch decodes for minutes (measured on 8-row MLX batches: about a quarter second per character, with a long tail). A take that gets no answer in time is sent once more with twice the limit and the same seed, so it is the same take and spends no attempt; a second timeout fails that attempt (`timed out after Ns`). To choose a Mac's row count, run `.venv-renderer/bin/python scripts/benchmark_mlx_batch_rows.py --checkpoint models/drama-3 --rows 1,2,4,8` on it (realistic prompt sizes, synthetic content, uses the GPU) and use the count past which frames per second stops rising. Batched takes sample differently from the vendored single-request loop, so they have their own engine identity (`decoder = "batched"`) and earlier unbatched takes are not reused for them. Measured on an Apple-silicon Mac (128 GB) with drama-3 q8. Generation only (32 synthetic passages): 45 frames/s unbatched, 167 with 8 rows, 206 with 12 and 221 with 16. Through the real pool with references, chapter history and MLX Whisper verification on the 24-passage benchmark corpus, over four seed variants: 1 row accepted 92/96 first attempts at 1.39x realtime verified audio per second; 8 rows accepted 87/96 at 2.77x (difference in acceptance not significant, p = 0.25; 8 vs 4 takes stopped as silent); 16 rows gave 2.27x and 2.81x on two variants, limited by this short corpus. Start with `concurrency = 8`; book renders run up to 16 chapters at once and keep more rows busy than this corpus can. With one row the decoder reproduces the vendored loop exactly; with several rows, bfloat16 GPU kernels compute a multi-row step slightly differently from a one-row step, so a batched take is a slightly different sample than the same request decoded alone (worst per-step KL 0.15 at temperature 0.7, the same for padded rows of different lengths and for identical unpadded rows), not a bit-exact replay. `scripts/check_drama3_mlx_batching.py --tiny` checks the decoder against the vendored loop exactly on the CPU with a tiny random model (no model files or GPU); `--checkpoint` checks the real model on the GPU (one row exact, padded rows no worse than identical rows). `scripts/benchmark_rendering.py pipeline CORPUS --worker local --concurrency N [--seed-offset K]` measures first-pass acceptance and throughput through the real pool and verifier. Every worker must be an `audiobook-render serve` process (protocol `audiobook-history-wav-v2`): each request carries the chapter's history turns with their codec tokens and each response returns the WAV with the generated codes. The controlling machine owns sampling: every request carries its `[engine]` temperature, top_p, top_k, max_new_tokens and seed, the worker applies exactly those (its own config's sampling never applies), and the response echoes the applied values, which the client checks before accepting the take. Worker-table sampling keys are ignored in favour of the coordinator's `[engine]` values. Stock Fish servers, Fish's hosted API and SGLang-Omni accept neither and are rejected; the client checks each worker's `/health` before rendering. The adapter sends MessagePack with bearer authentication, a `model` header, speaker-tagged references and Drama 3's tagged multi-speaker paragraphs. Set `multi_speaker = false` on a worker only if it must receive single-speaker requests; the planner then uses single-speaker chunks throughout the pool. Saved character references remain local to the coordinator, and only the voice recordings and requested passages are sent to configured workers.

Between this client and the built-in `serve` endpoint, the first successful synthesis response advertises content-addressed reference reuse. Later takes send a per-reference SHA-256 ID and transcript instead of uploading the same audio again, without a separate capability request. The server retains validated reference bytes in a bounded in-memory cache (64 entries, 256 MiB). After eviction or restart, an unknown ID is rejected before synthesis and the client resends inline once. This optimization does not invalidate accepted speech. Chapter-service uploads already register compressed voices once, so Runpod's public chapter hop does not resend reference audio per take.

Local book rendering overlaps one chapter's CPU composition/encoding with the next chapter's synthesis, using the same chapter preparation and quality gates as before. At most one assembly is pending, checkpoints remain source-ordered, and active assembly drains before run locks are released. Progress from adjacent chapters can interleave. Within a chapter, `render_pool.max_inflight` bounds complete operations including verification and recovery; a small window or a persistently slower verifier can still leave synthesis slots idle. The chapter service separately overlaps two chapter coordinators and bounded assembly. These are bounded pipelines, not a guarantee of full GPU utilization for every workload; compare verified audio per wall second with the pipeline benchmark rather than counting generated takes.

### Scheduling modes

`[render_pool] scheduler` selects how clip work is scheduled. `"pipeline"` is the default: chapter continuity makes each chapter sequential, so several chapter coordinators share one FIFO logical scheduler and keep the model busy. `"legacy"` renders one chapter at a time. Both modes reuse every already accepted clip, report, and cache identity: scheduling never changes what is synthesized or accepted.

| Setting (`[render_pool]`) | Default | Meaning |
| --- | --- | --- |
| `scheduler` | `"pipeline"` | One FIFO logical scheduler shared by all chapters; `"legacy"` renders chapters one at a time. |
| `chapter_concurrency` | `0` | Chapter coordinators; `0` is automatic: one per TTS slot under pipeline (`max(4, slots)`; each chapter's per-voice queues keep more than one take ready, so one take verifies while the chapter's other voices generate), 2 effective in legacy. Hard ceiling 16. |
| `max_inflight` | `0` | `0` is automatic. Pipeline: a global transaction ceiling — 16 local, `min(256, max(64, 16 × slots))` remote. Legacy: a per-chapter window of total slots + 1 (an explicit value is shared across chapters). |
| `synthesis_backlog` | `0` | Queued synthesis bound; `0` is automatic: `min(256, 4 × total TTS slots)`. Also the exact attempt backlog bound. |
| `pending_audio_mb` | `256` | Soft high-water on materialized untranscribed audio. New generation claims pause at the mark and resume below half of it. Soft bytes, not RSS or a disk cap: already-running outputs may land and overshoot. |
| `review_concurrency` | `2` | Parallel text-review HTTP leaves; the blind-review coordination lane runs `2 × review_concurrency`. |

Fixed lane sizes complete the picture: one TTS lane worker per declared endpoint slot (a local MLX worker is exactly one slot and one model copy — no GPU model duplicates), two audio-preparation workers, one recognition lane worker per recognition slot (the controller's own recognizer plus each recognition worker's `recognition_concurrency`; calls reserve a free slot at dequeue), and one review coordination worker per review-HTTP worker times two. The pipeline is work-conserving until a bound is hit: synthesis continues whenever a slot, the recognition model, or a review slot is free, and a chapter blocked on review or recovery never prevents other ready chapters from using TTS slots. GPU idle while a downstream lane is full is intentional backpressure, not a defect; the honest performance metric is accepted (verified) source audio per wall second, which the pipeline benchmark measures — GPU-utilization percentage alone proves nothing. Diagnostics: the render status line appends a `sched` segment with live per-stage queued/active counts, the attempt backlog and its monotonic peaks, and the logical scheduler's gauges; the managed observer is described in [REMOTE_RENDERING.md](REMOTE_RENDERING.md). No throughput claim is made for pipeline mode here: it ships only after the recorded live A/B gate in `scripts/benchmark_chapter_pipeline.py` passes.

Fatal renderer output identifies the exception type and file/line/function locations without dumping source lines, locals, or exception-chain payloads. Configured credentials are redacted. Storage and idle-stop failures report safe exception types and available HTTP/provider codes, retaining saved audio for retry. Failed managed Runpod operations skip the optional stop delay; if shutdown also fails, both operation and shutdown diagnostics are retained alongside the billing warning.

Changing worker addresses, credentials, order, or concurrency retains compatible accepted clips. Switching between direct local generation and a pool also reuses accepted speech when the actual model revision and synthesis settings match; the transport itself does not make speech different. `render_pool.model_revision` identifies the deployed model version; change it when replacing the model behind an endpoint. An unknown model revision is not assumed compatible unless the concrete worker identity establishes the match. Sampling or voice changes invalidate the affected requests. Use `engine.cache_model_alias` as described above when deliberately sharing successful speech across different model versions or quantization. Remote endpoint models and settings remain execution provenance, while acceptance remains attached to the returned audio bytes. A transport failure stops after bounded retries; rerunning retries incomplete requests and retains accepted work. Verification thresholds are unchanged.

Validation includes two real HTTP requests to the local Drama 3 model using the existing demonstration voices: a 2.60-second single-voice passage and a 4.04-second two-character passage. Both passed strict Whisper text and acoustic checks, and the original reference checksums were unchanged. The report is `output/distributed-render-pilot-paoj0g29/report.json`. Automated tests additionally exercise two HTTP workers through chapter/M4B assembly and offline resume. The real-model pilot used loopback on one Mac; a second machine and hosted Fish service have not yet been exercised.

### Chapter continuity (always on)

Drama 3 sets a passage's pace, energy and intonation from what is already in
its prompt. Every chapter therefore carries its own performance history across
requests; there is no setting to turn it off.

- **References say who is speaking.** Each voice's saved recording and
  transcript are sent unchanged on every request and are never replaced by
  generated audio.
- **History says where that voice's performance is.** Following Fish's own
  long-text policy (`fish_omni/chunking.py`), each request carries up to two
  completed user/assistant turns from the **same voice** in this chapter: that
  voice's first accepted take (pinned; a heading never becomes the pin) and its
  latest accepted take. No other character's audio enters a line's prompt.
  Drama 3 documents history for continuing one voice's script or revising a
  take; another voice's recent audio competes with the reference for identity
  and accent. A voice's first line in a chapter has no history. "Voice" means
  the script's voice id, not the recording: a narrator linked to a character
  (`voice-aliases.json`) clones the same reference but keeps its own history,
  so narration and that character's dialogue stay separate performances.
- **A chosen pronunciation hears an example of itself.** Drama 3 says a name
  the way the history says it. Across one series' recorded takes of a name
  with a chosen reading, a narrator line whose latest history turn said the
  name correctly was right every time (52 takes); one whose history turn had
  it misread repeated the misreading every time (12 takes); with no example in
  history the narrator misread it about 4 times in 10, and half the time when
  the name opened the line. So when a line uses a person-chosen CMU reading
  (`pronounce`) and neither of its history turns was heard saying that reading
  in its own shape, the latest slot holds that voice's most recent earlier
  take in the chapter that was (the one saying the most of the line's chosen
  readings, when there are several). Takes heard with the name rearranged are
  rejected (see the pronunciation check under verification), so they never
  become examples. The policy name is unchanged, so under tolerant history
  reuse takes already accepted for these lines are kept. A chapter's first
  line to use a reading has no such take in its chapter, and its retries keep
  the same history, so it used to misread again and again; it now hears the
  same voice's take from another chapter that was heard saying the reading
  right (the nearest earlier one in the book, else the nearest later one;
  `renderer/reading_examples.py`). The index of such takes is read once per
  render, grows as takes are accepted, and is checked again before an example
  is used, so a take since retired (marked bad, misread) is never heard.
  The example a line hears depends only on what is saved, so an unchanged
  chapter composes the same history on every render and restores even under
  `history_reuse = "strict"`. An example line usually keeps several passing
  takes (one for each history it was rendered under), and which of them a
  render picked once depended on the order the index listed them, so a fresh
  render could pick another and strict reuse rendered every later line of
  that voice again. Now a line whose saved take passed keeps the history that
  take had (the take the chapter's saved checkpoint uses, else its newest):
  the example it heard, while that example's sound and history codes are still
  on disk, even after it was retired or a stricter name check judged it a
  misreading, since the line was made after exactly that audio and already
  said the name right; and no example when it passed without one (chapters
  render in any order, so a later chapter's take appearing must not change
  it). One you marked bad (or set aside) is never heard again, so the lines
  that heard it render again. A line that needs a new take hears only
  examples that say the name right, equally good ones ordered by request hash.
- **Each voice change is its own request.** The planner groups consecutive
  chunks only for the same performer (since 2026-10-21), because speaker tags
  cannot guarantee an audible switch and transcript verification cannot detect
  the wrong voice. A multi-voice request would only hear turns by its own voices.
- **Only accepted takes advance history.** Rejected attempts never do. Joined,
  repaired and best-effort takes stay accepted audio but leave the history as
  it was, because they are not one generation of the exact text. Simultaneous
  speaker lanes share the same history and do not advance it.
- **Prompt overflow** follows Fish's fallback: drop the latest turn, then the
  pinned one. Each take records how many turns it actually used
  (`history_turns`).
- **Chapters start fresh.** Silent controls do not reset history.

#### History echo

Fish's history layout is a take-revision conversation: Drama 3 was trained to
treat earlier takes as previous attempts and the next user turn as a note on
how to redo them. A short line, a speech tag ("Sal asked carefully.") or a
heading can read like such a note, and the model then speaks the previous
history turn again instead of the requested text. Another seed of the same
prompt rarely helps. In a replay of 16 recorded cases, the unchanged prompt
echoed 42 of 48 times and temperature 1.0 made no difference.

The renderer detects this from recognition evidence. A failed take whose
transcript lines up with a heard history turn far better than with its own
text is recorded as a `history_echo`, naming the turn and both match shares.
Its next attempt changes the prompt instead of re-rolling it, one rung at a
time:

1. **Role tag**: the same history, with `[narration]` or `[dialogue]` leading
   the new line (merged into an existing direction, for example
   `[dialogue; quietly]`). This keeps the usual layout and the most continuity.
2. **Streaming layout**: the same history audio in Drama 3's
   streaming-continuation layout (technical report, figure 10), with the
   reference recording as its first turn. This needs a single voice, no
   bracketed controls, and an engine that builds it: the local MLX adapter, or
   a `serve` worker whose `/health` lists it. Otherwise this rung is skipped.
3. **No history**: the reference recording alone. This cannot replay history.

Each rung gets a real attempt, up to one extra per rung beyond
`max_attempts`. The speech-recovery planner, clip repair and the exhaustion
chooser run only after the ladder. A take records the rung it used in
`engine_metadata.history_recovery` (`requested` and `applied`). Role-tagged
and streaming takes are ordinary generations of their text, so later lines
hear them as history. Every other take, and its cache identity, is unchanged.
Takes recorded before this check existed are reassessed from their saved
transcripts, so a rerun continues the ladder.

To find echoes already published in a book, run:

```bash
audiobook-render history-echoes BOOK
```

It reads only the saved takes and history codes: no models, no network, and
nothing changes. It lists each best-effort choice that speaks a heard history
line instead of its passage (only best-effort choices can be echoes, because
ordinary acceptance requires the passage's words), with its chapter, what was
heard and the line it repeats, and prints the chapters to re-render. `--json`
prints the full report. Add `--retire` to retire those choices under the
render lock. Their files are renamed with a `history-echo-…` suffix (never
deleted), and the next `render` regenerates just those lines through the
ladder and rebuilds their chapters from the other cached takes. Render also
rejects echo choices by itself, so `--retire` mainly makes the affected
chapters explicit before you render.

Within a chapter, takes form per-voice queues: a take starts once every
earlier take whose voices are all among its own has been accepted (it hears
only those), so the narrator's lines and a character's quoted lines render at
the same time, while a paragraph mixing both waits for each. Every take still
hears exactly the history a strictly source-order render gives it, picked by
source position rather than by which take finished first. On a measured book
this gives a median of 1.7 takes in flight per chapter. Throughput also comes
from overlapping chapters. Drama 3
always renders through a pool (a one-slot local pool when no workers are
configured), and the default `[render_pool] scheduler = "pipeline"` runs one
chapter coordinator per TTS slot (`chapter_concurrency`, minimum four), each
queueing its chapter's ready takes (one per independent voice queue) for the
shared workers while other takes verify. All synthesis takes join one FIFO queue and claim the first free
worker in configuration order. Each request carries its complete composed
history and references, so a chapter can continue on any worker without worker
affinity or changes to saved-audio identity.

Every engine returns the codec tokens it generated. The renderer saves them as
hash-validated `*.codes.json` artifacts beside each take and rebuilds history
from them on resume; codes are never reconstructed from WAV files. Each take's
cache identity includes the exact earlier takes it heard.

`[render] history_reuse` decides what happens to the rest of a chapter when one
take changes (a corrected line, a pronunciation fix, a changed speaker, an
edited paragraph from a serial update):

- `"tolerant"` (the default) regenerates only the changed take. A later take
  whose own text, voice, delivery, engine settings and continuity policy are
  unchanged is kept although an earlier take it heard is now different; only
  takes generated afterwards hear the current history. Reuse is tied to the
  take's own line (chunk ID): an identical line elsewhere is never moved into
  this position, and one take never fills two positions in a chapter. Only
  takes that passed ordinary verification are kept this way. A best-effort
  exhaustion choice regenerates when its history changes, since the new
  history may produce a take that passes. The cost is a possible small seam in
  delivery right after the changed take.
  `cache-report` counts these as hits and separately as
  `history_changed_hits`. A take reused this way is never published under the
  new history's cache identity, so `strict` still sees it as generated after
  the old history.
- `"strict"` regenerates from the changed take to the end of that voice's
  chapter, so every take was generated after exactly the history it now
  follows (`cache-report` shows these as `earlier_take_in_chapter_changed`).

A change to the continuity policy itself (which takes a line hears) refreshes
audio in both modes. Remote chapter services (`serve-chapters`) apply their own
configuration's setting.

Sampling defaults are Fish's own evaluation settings: `temperature = 0.7`,
`top_p = 0.7`, `top_k = 30`. Stalls and off-voice takes are caught by the
normal verification and retried.

Character speech can use its own sampling while narration keeps the `[engine]`
values:

```toml
[dialogue_sampling]
temperature = 0.95 # unset: engine.temperature
top_p = 0.7        # unset: engine.top_p
```

A request uses these values when every passage in it is a character's own
speech: segment kind `dialogue`, `thought` or `letter`. Narration, headings and
other text, and a paragraph that mixes narration with speech, use `[engine]`.
`top_k` and `max_new_tokens` are always shared. Only values that differ from
`[engine]` change anything. Such a take's cache identity records the sampling
it was generated with, so turning the setting on, off or to another value
regenerates character takes and keeps narrator takes. The setting applies to
local Drama 3 (MLX and MLX-Audio, batched or not) and to `serve` workers, which
must echo the exact values before a take is accepted. Remote chapter services
(`serve-chapters`) apply their own configuration's setting. It requires
`engine.type = "drama3_mlx"`.

Supported engines: the vendored Sogni runtime, the MLX-Audio quantized models,
`audiobook-render serve` workers (another Mac, or Runpod's inference service),
remote chapter services and Runpod. Plugins must advertise the
`chapter_continuity` capability, accept history through `continuity.compose`
and return `generated_codes`; otherwise rendering stops before synthesis. Stock
Fish and SGLang servers cannot accept history turns or return codes and are not
supported as render workers.

## Multi-book render queue

`render` accepts several books; they render through one client process, in order:

```sh
.venv-renderer/bin/audiobook-render render runs/book-01 runs/book-02 runs/book-03
just render runs/book-01 runs/book-02 runs/book-03
```

- **Remote when a service is available.** With `[remote_chapters].url` configured, the external service is used as-is. With `--remote`, the managed Runpod service is joined or bootstrapped through the same probe-and-ready flow as the managed commands, and the service's own idle monitor later stops the compute — the queue never stops it.
- **Tail backfill.** Remote books open as independent client sessions. Book k+1 is admitted once the service's live synthesis work stays below its total slot capacity on two consecutive scheduling reads (about five seconds apart; a single dip never opens a book), or — on servers without scheduling data — after an admitted book's first completed chapter. A book, once admitted, runs to completion; books only open, never pause or preempt. Books normally finish in order because start order and whole-book sizes dominate, but a failed book does not idle the others' paid slots.
- **Configured worker pool.** Without a chapter service, books share one engine, verifier and configured worker pool in this process. Admission opens the next book after two reads show spare synthesis capacity, including while an earlier book assembles its M4B. Voice setup finishes before that book can open the next admission. Models load once; the worker counts remain the pool's global limit across all admitted books. Engines without a parallel-capacity contract remain sequential.
- **One chapter coordinator budget.** The whole queue shares a chapter coordinator pool bounded by the configured synthesis slots. Pending chapters are ordered by book position, then chapter position: earlier books refill free coordinator slots before later books, and later books use only the remaining capacity. The live display therefore shows at most that many active chapter rows across all books.
- **Failures never stop the queue.** A run-fatal failure in one book is recorded with its cause and the queue continues with the next book; the exit code is 0 only when every book rendered, otherwise 1 with a per-book summary. Ctrl-C stops admissions, gives running books a brief cleanup window (their sessions are deleted server-side), prints each book's status, and exits 130; verified chapters are retained for resume.
- Duplicate book paths in one invocation are rejected up front, as is `--chapters` for a multi-book queue (run that book's selection as its own command). `--chapters-only`, `--censor`/`--no-censor`, and the chapter-format and performance-mode overrides apply to every book; `--output`, `--work-dir`, `--epub`, and `--performance-plan` remain single-book options.
- Each book's output lines are prefixed with its directory name (`[book-2] …`). An interactive queue owns one terminal board: active chapters from all admitted local books keep distinct, book-prefixed status rows below scrolling logs and share a scheduler footer. As soon as a book finishes, a persistent line pinned at the very bottom tallies the outcomes with color — for example `Books: 2 succeeded | 1 failed` (green succeeded count, red failed count; zero counts omitted) — and stays visible, with or without active chapter rows, until the queue exits. Redirected output remains plain text. One-off renders from other terminals can join the same running service as independent clients at any time.

A single book without `--remote` renders exactly as before. `--remote` also works for a one-off single-book render, for example to redo a failed chapter on the managed service:

```sh
.venv-renderer/bin/audiobook-render render runs/book-03 --remote --chapters-only
```

## Local auto-serve

Configured local render workers and interactive pronunciation jobs reuse a
local `serve` worker on `server.port` by default (`server.auto_start = true`).
They share the TTS model rather than loading competing copies.

`just server` (`audiobook-render local-server`) says whether that server is
running: its process, when it started, its batch rows, what it is working on
or how long it has been idle, whether its code is still this checkout's, and
its log. `just server stop` stops it (refused while it is working; `--force`
stops it anyway, and a running render starts it again) and `just server
restart` replaces it. The server outlives the commands that start it, so a
code change does not reach it until it restarts: its `/health` lists the
renderer files it loaded that changed on disk after it started
(`code_changed`), and render and pronounce replace a server running older
code once it has been idle for a minute. One that is in use keeps running, and
they say so.

- The job probes `http://127.0.0.1:{server.port}/health` and, when a worker
  with the expected protocol and model answers, sends its synthesis there as a
  client. Saved voices keep working: the client reports the Drama 3 binding
  the pronunciation tools check.
- When nothing is listening, the job starts `audiobook-render serve` itself as
  a detached child (under an exclusive lock, so concurrent jobs spawn one
  worker), waits up to 300 seconds for readiness, and connects. A worker that
  is still loading its model is waited for, never duplicated.
- The spawned worker binds loopback and uses the local worker's configured
  batch width. It exits after `server.idle_timeout_seconds` without requests,
  using 600 seconds when that setting is zero. Explicit `serve` invocations
  retain their configured bind and idle settings.
- A port owned by any other process fails closed with a message; nothing is
  ever killed. Engine identity is not advertised on `/health`, so protocol +
  model is the accepted compatibility check.
- If that worker disappears during a render (macOS stops a process when the
  Mac runs out of memory; its log then simply ends), a request that gets no
  answer at all checks the port. When nothing listens there, the render starts
  the worker again, once for all of its slots, and sends the same requests
  again; those retries spend no take attempts. A worker that is running, busy
  or still loading is never restarted, and an HTTP error from it stays an
  ordinary failed attempt. After three restarts in one render, or when the
  worker cannot be started, the render stops as an outage (like a verification
  outage) with a message instead of failing every remaining take; rerun it
  once memory is free, and finished chapters are kept.

`scripts/benchmark_rendering.py generate|pipeline --worker local` routes
its slots the same way (and can start the worker), recording each slot's
transport in the manifest's `slots`.

Set `auto_start = false` to explicitly retain direct in-process inference.
Diagnostics live in `~/.cache/epub-tts-narrator/`:
`serve-{port}.log` (the spawned worker's output — check it when startup fails),
`auto-serve-{port}.json` (pid, port, started-at of the last spawn) and
`auto-serve-{port}.lock` (spawn serialization). `auto_start` requires
`engine.type = "drama3_mlx"` with an existing `engine.checkpoint`.
Voice-creation jobs retain their separate designer path.

## Compare legacy and expressive chapter generation

Expressive generation is opt-in. Existing render commands retain their legacy conditioning and cache identities. To compare a chapter, keep the same saved voices and global config and choose a separate output directory:

```sh
audiobook-render compare-performance book.audiobook \
  --chapter 4 --directory comparisons/book-chapter-1 --local
```

Chapter numbers refer to the exported manifest, including front matter. Run `audiobook-render plan book.audiobook` to see the numbered titles. For the current Soulship book 4 export, manifest chapter **4** is the story's **Chapter 1**; manifest chapter 1 is its cover. A chapter containing no spoken text cannot be compared.

The command freezes the validated source, cast, selected voice references, effective aliases, sampling settings, and base seed before planning or synthesis. It follows inherited voice selections while taking the snapshot, then copies only the selected immutable references. Missing voices stop the comparison before model loading; this command never casts or regenerates voices. Original books, voice stores, and render caches are read-only inputs. The comparison directory must be outside those locations.

The analysis model proposes source-supported delivery spans, then a separate review can remove weak suggestions. Every control is tied to exact spoken text; the analyzer's canonical text and attribution remain unchanged. Existing spelling-normalization directions are scoped to their corresponding replacements by the expressive renderer. Hand-authored plans can also request validated CMU Arpabet pronunciation, inline vocalizations, and pauses. Automatic planning is deliberately conservative: it does not invent laughs, breaths, pronunciation, or other events merely because a nearby sentence mentions them.

One engine and verifier instance renders both halves with the normal audio and speech checks. `--local` ignores remote chapter and render-pool workers for this invocation, without editing the config. Without it, configured local/render-pool engines are used; remote chapter servers do not yet transport expressive plans. Each participating adapter must advertise structured-performance support. Model downloads, if required, follow the normal setup behavior.

Results are saved as:

```text
comparisons/book-chapter-1/
  comparison.json          # input/voice hashes, runtime identity, durable status, statistics
  performance-plan.json    # reviewed exact-text-bound controls
  index.html               # local players, changed-control links, elapsed time and duration
  snapshot/                # frozen bundle and selected reference voices
  legacy/work/             # independent resumable speech and chapter caches
  legacy/output/           # chapter.m4a (or configured MP3) and sync.json
  expressive/work/
  expressive/output/
```

Open `index.html` to listen side by side. Links beside changed controls start the containing passage in each version; grouped passages share paragraph-level timing rather than fabricated word alignment. Each half has independently measured source synchronization. Chapters use the same configured loudness and encoding settings. The same **base** seed is retained, but individual take seeds also depend on request identity, so different controls do not share identical sampling noise. Timing includes verification, retries, and assembly; a resumed experiment may reuse caches and is not a clean throughput benchmark.

Rerun the identical command to resume. Completed halves and accepted clips remain reusable. A changed source, voice selection, config, or performance plan requires a new comparison directory, so old and new inputs cannot silently mix. Audio in an existing original publication is not imported as a baseline without a matching experiment binding.

To prepare the snapshot and reviewed plan before any audio generation, add `--prepare-only`. To reuse a separately prepared one-chapter plan, pass `--plan-file PATH`; that input remains unchanged. Performance planning alone does not require selected voices or TTS/ASR models:

```sh
audiobook-render prepare-performance book.audiobook \
  --chapters 4 --plan-file comparisons/chapter-1-performance.json
```

To use an approved plan for an ordinary render, pass `--performance-plan PATH` to `plan`, `render`, or `assemble`; it implies expressive mode. `--performance-mode legacy` explicitly restores the old conditioning. If you eventually select `performance_mode = "expressive"` in the global `[render]` config, the plan defaults to each book's `performance-plan.json`; a per-book filename is not needed in the global config. Prepare all chapters you intend to render. The default remains `legacy` while evaluating the comparison.

Both commands read analysis-provider and renderer settings from the same global TOML file; use `--config /path/to/epub-tts-narrator.toml` from another directory. Plans, request caches, and audio are resumable local artifacts and contain book text; keep them out of Git. Transcript verification does not certify accent, identity, emphasis, or emotion, so use the comparison to judge the performance by listening. Native pause duration and freeform style resets are approximate; unsupported controls fail explicitly instead of disappearing.

## Pronunciation review

Initial analysis can carry reviewed, source-linked spoken-spelling edits and inline `performance_controls`. These are consumed during default rendering without `prepare-performance`; a whisper, emotion, or laughing delivery may cover only part of a segment. Point vocalizations and pauses remain separate from lexical text. Optional external records merge only when compatible; conflicting overlaps fail rather than silently replacing an instruction. Actual adherence to a cue still depends on the selected model.

Pronunciation work is optional and manual. Nothing runs automatically: the renderer never discovers candidates, creates CMU suggestions or estimates, or generates name annotations on its own. The `pronunciations` commands need no analysis-provider configuration or network access. The one model-assisted tool is `pronounce` (formerly `pronounce-name`), which you invoke for one name or invented word at a time (below). Existing confirmed and keep-default choices remain valid, including inherited choices and saved literal respellings.

### Names and invented words (`pronounce`)

Use this when the engine reads a name or invented word inconsistently or wrongly (for example "Salvatore" coming out sometimes as "sal-vuh-tore" and sometimes as "sal-vuh-torie", or a place, people or term such as "Yarran" or "Kestran"). A hyphenated name such as "Tor-Vell" is one word: give it with its hyphen, and only the whole name gets the reading, never "Cho" or "Hag" alone, whichever hyphen character the book uses. `pronounce-name` is the same command under its older name.

```sh
just pronounce runs/quest-academy-08 Salvatore
just pronounce runs/quest-academy-08 Salvatore "sal-vuh-TORE, no ee at the end"
just pronounce runs/book Ally --as "proper noun"
just pronounce runs/book Milekey --say-as mile-key
just pronounce runs/book Milekey --phonemes "M AY1 L | K IY1"
# or: audiobook-render pronounce BOOK WORD ["HOW IT SOUNDS" | --phonemes "S AE2 L V AH0 T AO1 R" | --say-as SPELLING] [--as PARTS] [--voice-id VOICE]
```

A reading is one of three things:
- **Phonemes:** for example `S AE2 L V AH0 T AO1 R`.
- **Phoneme groups:** phonemes split with ` | `, for example `M AY1 L | K IY1`. Each group becomes its own engine phoneme tag, so a compound like "milekey" is read as "mile key" rather than blended into "mily-key" or "milky". The primary stress may be in any group.
- **A say-as spelling:** for example `mile-key`. The engine reads that spelling in place of the word, with possessives keeping their ending (`Milekey's` → `mile-key's`) and capitals for a word written in capitals. Parts typed with spaces are joined with hyphens.

Each choice says which parts of speech get the reading: proper noun, common noun, adjective, verb, adverb, or interjection. This keeps homonyms apart. A character called Ally can be read "AL-ee" while "an ally" keeps its ordinary reading. "Kestran" can cover "the Kestran King" (proper noun), "a Kestran" (proper noun), and "Kestran kings" (adjective). `--as` sets the parts up front, for example `--as "proper noun, adjective"` or `--as all`. `S` in the menu changes them. When you don't choose:

- A spelling that isn't in the CMU dictionary (Yarran, Kestran, Valdoria) applies to every part of speech. No ordinary word can share it, so every use is tagged without asking the model.
- A spelling that is also an ordinary English word (Rose, Grant, Ally, Riva) applies only to proper nouns, and the tool says so.

1. The analysis model proposes CMUdict-style ARPAbet with stress digits (the Fish/Drama 3 phoneme-tag convention). It sees the word, the bundled CMUdict readings, the matching cast entries, the chosen parts of speech, and a few book excerpts as untrusted data. A final argument describing how the word should sound, in your own words, is passed to the model as your instruction for the first guess. It takes precedence over a saved draft or an existing choice. `--phonemes` starts from your own ARPAbet instead. Without either, a saved draft resumes, and otherwise a confirmed choice from this book or a series ancestor is offered first.
2. When the model's help is needed (an ordinary-word spelling, or only some parts of speech), it first labels each use's part of speech in context, in checkpointed batches. The tool shows the counts, for example "41 proper noun, 3 common noun", and how many uses get the reading. The labels don't depend on your choice of parts, so changing the parts later needs no new requests.
3. The tool renders and plays up to three short sentences from the book with uses that get the reading, preferring different forms and parts of speech. It uses the voice that says the word most often (normally the narrator). A sentence that also contains a use that keeps the default reading is skipped. Every audition ends with the word's derived plural, for example "The Salvatores are here.", so you confirm how a later series book's first plural use will sound even when this book never uses the plural. When no book sentence is short enough, a built-in one is used instead, such as "The name is Salvatore. It is Salvatore's name. Give it to Salvatore. The Salvatores are here." The audio file path is printed. These are lossless listening previews under `render-work/pronunciations/`, not book audio. If Drama 3 stops a take before the end of the sentence (for example when it runs into silence), the tool tries up to three fresh seeds. If all three stop early, the next `P` continues with new seeds and never repeats one that already failed.
4. On a terminal this all happens on one full-screen screen (`renderer/pronounce_tui.py`; `--plain` keeps the line menu): the current reading with its possessive and plural, the test sentence with every use that gets the reading highlighted, a menu of the choices below, and a log of what the tool is doing. Each choice is one key (or the menu); describing the sound, editing phonemes and say-as spellings open a text box, and parts of speech open a checklist. Space stops the audio and `>` switches it between 1x and 2x. Unsure uses are then decided one at a time beside their excerpt (Y, N, Enter keeps, Q stops). On the line menu, choose one:
   - `A` accepts.
   - `I more like sal-vuh-TORE-ee` asks the model to revise from your description.
   - `N` asks for a different guess.
   - `E S AE2 L V AH0 T AO1 R` sets the phonemes yourself. Input is case-insensitive; every vowel needs a 0/1/2 stress digit. `E M AY1 L | K IY1` sets phoneme groups. ARPAbet's punctuation marks `.` `-` `!` `+` `/` `#` `:` may stand between symbols as their own words (`E S EH2 . N EH1 D R AH0`, or typed `EH2.N`); they go to Drama 3 inside the tag as written, are never counted as sounds (a possessive or plural still follows the last sound), and whether the engine does anything with them is for listening to show. Each group needs at least one sound. In the full-screen tool a reading it would refuse keeps the edit box open with the reason, so you can correct it instead of starting over (the listening queue's spelling and phoneme edits too).
   - `R mile-key` says the word as that spelling instead.
   - `S proper noun, adjective` (or `S all`) changes which parts of speech get the reading.
   - `B` plays the engine's default reading for comparison.
   - `P` replays.
   - `Q` quits.

   Every listened step is checkpointed as a draft in `name-pronunciations.json`. Drafts never affect rendering, and rerunning resumes the draft.

   On `A`, the configured speech recognizer transcribes the test sentence you just accepted, once. If it spells the word differently (for example `Marla` for `Mara`) and every other word of the sentence is heard as written, that spelling is saved as a trusted heard variant for the word and its possessive and plural (see "Learned heard variants"), so renders with the same recognizer accept it. If anything else in the sentence is heard differently, nothing is learned. A missing or failing recognizer only skips this step; the confirmed choice stays.
5. After acceptance, the tool asks you about any use the model was unsure of and lists the uses that keep the default reading, with their labels. `--review-tags` lets you check or flip every use the model labelled. Your answers ("should this use get the reading?") always win over labels.

Possessive and plural readings are derived from the base by the final sound: S, Z, or IH0 Z, so `Salvatore's` → `… R IY0 Z`. Other continuations (`Salvatore'll`, `Salvatore-like`) tag only the word part. Matching ignores case, so shouted names are included. Related words with a different ending, such as Kestra and Kestran, are separate choices.

At render, every tagged use becomes a structured CMU pronunciation control, compiled into Drama 3 `<|phoneme_start|>…<|phoneme_end|>` tags only by the performance compiler. Uses labelled outside the chosen parts of speech keep the default reading, and tagged spans take precedence over `pronunciations.json` choices for the same word. Only affected speech regenerates on the next `render`.

Confirmed choices, including their parts of speech, inherit through `--continue-from` continuation links, nearest book first. Each book labels and tags its own uses. Labels are bound to each paragraph's exact text and to the version of the labelling prompt. After a book update, only changed paragraphs need labelling. After a prompt improvement, the model's older labels are redone; your own decisions are kept. `analyze` does this for you: whenever it publishes, reruns, or updates a book (including a new book with `--continue-from`), it finishes by labelling every new, changed or outdated use of each word confirmed in this book or inherited from an earlier one. `render` also checks before planning, for both single books and queues, and automatically labels any missing or outdated uses through the configured analysis provider. It preserves your readings and decisions, checkpoints each batch, and reuses completed labels on rerun. Current labels and words that need no labels require no provider call or provider configuration. If the model is unsure, rendering stays blocked and names the `pronounce` command for your decision; a queue records that book's failure and continues. `plan` remains read-only and reports missing or unsure tags without labelling them. To tag every pending word yourself (for example after defining a word in an earlier book), run:

```sh
just pronounce BOOK
```

Use `--list` to show each word's reading, parts of speech, and status, and `--tag-only WORD` to tag without listening again. Older `name-pronunciations.json` files (version 1) are read as they are. Their choices get the default parts of speech, your yes/no decisions are kept, and the model's older "name / not the name" verdicts are replaced by labels the next time the word is tagged. The file is saved in the new format the next time it changes.

To undo a choice, for example one defined in the wrong book, remove it from the book that defines it:

```sh
just pronounce runs/quest-academy-08 Salvatore --remove
```

This deletes that book's confirmed reading and any draft for the word, and prints the removed phonemes and parts of speech so you can enter them elsewhere with `--phonemes` and `--as`. Books continued from it stop inheriting the choice. They fall back to a choice from an earlier book if one exists, or to the engine's default reading. Removing an inherited choice from a later book is refused, and the message names the book that owns it. Each book keeps its labels and decisions, because they record what each use is, not how it sounds. If the word is defined again, they are reused without new requests. Re-render the affected chapters to hear the change.

To move your choices to another book (for example a re-analysis from a cleaner EPUB, which has new paragraphs but the same names), export them to a YAML file and import it there:

```sh
just export-pronunciations runs/series-book-2 series-names.yaml
just import-pronunciations runs/series-book-1-v2 series-names.yaml
# or: audiobook-render pronounce BOOK --export FILE [--local-only]
#     audiobook-render pronounce BOOK --import FILE [--replace]
```

The export holds every confirmed choice the book uses, inherited ones included (`--local-only` writes only the book's own). Each entry has the word, its `phonemes` or `say_as` reading, any respelling hint, and its parts of speech (`[all]` or labels such as `[proper noun, adjective]`), so it can be edited by hand. Drafts and per-use decisions are not exported, because uses belong to one book's paragraphs. Import confirms the file's choices in the target book in one write, with an `imported` history step. A word whose choice there is already the same is left alone. A different choice the book made itself is kept unless you pass `--replace`, and one it only inherits is overridden in that book. Import then tags every confirmed word's uses in the book, as `just pronounce BOOK` does. The file is a small, strict subset of YAML (plain or quoted values, `[a, b]` or one `- item` per line, `#` comments), and anything else is refused with its line number.

### Manual commands

```sh
audiobook-render pronunciations BOOK --action list

# Audition an unsaved literal respelling or exact CMU sequence.
audiobook-render pronunciations BOOK --action play --word Qarm --say-as Quarm
audiobook-render pronunciations BOOK --action play --word Qarm --phonemes "K W AO1 R M"

# Explicitly save a manual choice without audition or model conversion.
audiobook-render pronunciations BOOK --action set --word Qarm --say-as Quarm
audiobook-render pronunciations BOOK --action set --word Qarm --phonemes "K W AO1 R M"

# Keep the engine default, or remove the local choice to restore inheritance.
audiobook-render pronunciations BOOK --action ignore --word Qarm
audiobook-render pronunciations BOOK --action reset --word Qarm
```

A readable hint is one literal respelling, optionally hyphenated; it is passed directly to the existing synthesis compiler and never converted to CMU. Both `--say-as` and `--phonemes` with `--action set` directly save the supplied manual choice. `--action play` does not change the registry. Listing, saving, ignoring, and resetting need no synthesis model. Playback still requires an existing saved character voice and the configured synthesis engine; entering review alone generates no audio.

The former `discover` action is retired. Add or change a word explicitly with `--word` instead of running a scan.

### Listening review

`audiobook-render pronunciations BOOK` opens the saved-entry listening queue. On a terminal it is a full-screen screen (`renderer/pronounce_tui.PronunciationQueueApp`; `--plain` keeps the line menu): every entry is listed with its current choice and whether it awaits listening, the selected entry's details beside it, and the choices below as keys or a menu (S listens from the selected entry, Enter plays it, E/P edit with a text box, C/I/R, O and Shift+O mark for listening again). While listening, any key interrupts exactly as on the terminal, and the selection moves to where listening resumes. It does not prepare proposals, estimates, or previews before the menu. `S [N]` starts listening at entry N (default: the saved resume position or first queued word), `L` lists entries, `O <all|N|N-M>` marks entries for re-listening, a bare number plays that entry's excerpt, and `Q` quits. `B` plays the default, `E` edits a literal respelling, `P` edits CMU directly, `C` confirms the played draft, `I` keeps the default, and `R` resets a local override.

The queue speaks each queued word's current manual choice or default in isolation, twice, with interruptible gaps. An uninterrupted pair accepts that choice against the exact hash-bound playback receipt; a heard default is recorded as keep-default. Failed or interrupted playback never accepts. Queue progress and drafts survive quitting, but drafts affect rendering only after confirmation. Auto-acceptance requires an interactive terminal.

A saved legacy machine proposal remains readable and can be auditioned explicitly, but the ordinary queue skips it rather than auto-playing or accepting it. Select its entry explicitly with `S N`, or use numbered playback followed by `C`, to accept that exact saved proposal. No replacement proposal is generated.

### Existing scoped annotations and legacy choices

Books with saved `name-script.json` annotations retain entity-scoped review. Use `--class name --name-id N1` for one named entity or `--class special` for ordinary occurrences. Scoped commands read the stored annotations only; if required annotations are absent, they fail clearly instead of generating them or widening a rule. `--name-id` requires `--class name`.

```sh
audiobook-render pronunciations BOOK --class name --name-id N1 \
  --action set --word Ally --phonemes "AE1 L IY0"
audiobook-render pronunciations BOOK --class special \
  --action set --word zorb --say-as zawrb

# Explicitly migrate an existing confirmed/ignored legacy choice.
audiobook-render pronunciations BOOK --action migrate --word Ally \
  --class name --name-id N1
```

Name rules apply only to occurrences marked with that entity; Special rules exclude marked or unresolved names. Drafts, previews, receipts, and queue state remain scope-isolated. Migration leaves `pronunciations.json` byte-identical and saves scoped intent in `scoped-pronunciations.json`; resetting it does not revive an unsafe global rule. Existing scoped reads and rendering do not require name-script generation.

### Saved-choice behavior

Default review rows retain the existing offline dictionary filter for legacy pending candidates. Cast names, explicit drafts or decisions, conflicts, and `--word` selections remain reviewable; filtering never deletes registry records. Descriptions show only the actual default, literal respelling, or supplied CMU sequence—never a newly estimated pronunciation.

Explicit previews are cached lossless listening aids, not accepted audiobook clips or voice references. Confirmation requires a played preview for the current choice and saved voice. Excerpt correction previews omit only analyzer spelling edits overlapping the target word on a row copy; canonical rows and production ambiguity guards stay unchanged.

Ordinary capitalization variants share review and confirmed runtime choices, with conservative capitonym/initialism exceptions. Existing exact preview identities and playback receipts are preserved. Choices live in the book's `pronunciations.json` or scoped sidecar and inherit through continuation links, nearest explicit decision first. Pending candidates and proposals never override an ancestor's confirmed choice.

Confirmed pronunciations apply in legacy and expressive rendering without reanalysis. Only affected speech needs regeneration on the next `render`; unchanged effective speech remains reusable, and already assembled audio is not rewritten. Backends that do not support structured pronunciation controls fail explicitly.

This experimental mode keeps the existing paragraph and speaker boundaries. It supports the documented paired delivery controls, scoped freeform styles, CMU Arpabet, inline vocalizations, and approximate pauses through the Drama 3 adapters. Standalone sound effects, language switching, audio infilling, and history-conditioned continuation are not implemented. Spoken-word checks remain active; an inline laugh or sigh is not yet acoustically certified. Explicit pauses are allowed through silence checks only when their measured position and duration match the requested ASR word boundary.

Whole-take retries and transcript review remain available. Sentence-split recovery works for a single-speaker expressive line: each part is synthesized and verified with exactly its own slice of the line's controls (`performance_plan.slice_performance`), and a split that would cut a chosen pronunciation in two is not offered. Text-only spelling rewrites stay disabled for expressive leaf clips because they cannot rebind scoped controls to changed words. Failed paragraph groups can still fall back to their original independently bound turns. This limitation is confined to expressive mode; the legacy recovery path is unchanged.

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

On an interactive terminal, `cast` opens a full-screen casting screen (Textual, installed by `renderer.requirements.txt`; an older environment gets it by rerunning `uv pip install --python .venv-renderer/bin/python -r renderer.requirements.txt`).

Without Textual, with `--plain`, or when input or output is not a terminal, `cast` keeps the numbered line menu described below, unchanged. Every full-screen screen (casting, accents, pronunciation, the run shell and `review-ui`) also takes vim movement keys: `j`/`k` down/up, `h`/`l` left/right, `Ctrl+D`/`Ctrl+U` page down/up, `g`/`G` first/last, each doing what the arrow, page and Home/End keys do in the focused list or pane; text boxes still type the letters, and where a screen already uses one of these letters it keeps that meaning (the cast's `g` generates; a takes list's `g` marks a take good). The screen shows:

- **Voices:** every voice with its status (`accepted`, `missing`, `pending` (takes but none accepted), `inherited`, `derived`, `shared`, `failed` with its saved error, `needs regeneration`, and `queued`/`generating` while a job runs). `/` filters by name, voice ID or description; `f` cycles a status filter; `o` sorts what needs attention first.
- **Detail:** your notes about the character, the voice description (the clone direction for a derived voice), audition script and its source, speech traits, sharing, and reference clip count, each with the key that edits it. When your notes give a gender and the description reads as the other one, a **Check** line says so.
- **Takes:** every saved take of the voice, with the accepted one marked. `Enter`/`p` plays the highlighted take, or the accepted take from the voice list; `Space` stops. `m` marks a take as A and `x` alternates between A and the highlighted take for comparison; `a` accepts the highlighted take after confirmation.
- **Log and jobs:** generation, acceptance and every save run one at a time on a background worker, so the screen stays responsive. A failed job stays listed with its error (`Enter` shows the full detail) and `R` retries it while other jobs keep running.

Keys: `n` edits your notes about the character (see [Your notes about a character](#your-notes-about-a-character)), `v` opens the book's accent plan (the `accents BOOK` review, see [Accent cues](#accent-cues); `q` comes back), `g` generates a new voice (choose local, Gemini or ElevenLabs; replacing an accepted, shared or inherited voice asks first), `e`/`s`/`t` edit the description, script and traits and `A` the accent cue (pre-filled with the cue the description suggests; empty means no cue), all validated before saving, `d` shares another voice's recording and `D` removes this book's own sharing link, `c` sets this voice's reference clip count (see [Reference clips](#reference-clips)), `r` rewrites the description and script with the analysis model after confirmation, `i` revises descriptions as you instruct (choose this voice, every voice without an accepted take, or every voice, then type what should change, e.g. "remove every mention of accent or regional speech; the accent cue adds it"; the analysis model receives each voice's name, current description and the accent cue the application adds when the voice is designed, and returns only the descriptions that change; a review lists them with added words green and removed words struck through, Space unmarks one, and Enter saves the marked ones as drafts, which apply the next time those voices are generated; recordings are unchanged; asking the model waits for no other job), `B` generates many voices at once (every voice with no voice yet, every voice without an accepted take with saved verified takes used first, or voices you mark with Space; then the designer and a confirmation; it refuses until those voices' accents are decided, so fix accents with `v` first), `w` chooses how one of the character's names is said (see below), `X` cancels a queued job (the one highlighted in the Jobs table, else this voice's queued jobs; a running job finishes), and `?` lists every key. A voice with no recording at all shows a yellow `missing` badge with "no voice yet; g generates one", and the summary line counts them (`f` shows only them). Playing a voice that has nothing to play, or any playback failure, is reported in the log and a notice; it never closes the screen. Opening the screen does the same preparation as the line menu (default drafts, voices without an accepted take, playback copies), shown as jobs. A model download question appears as a dialog. Gemini generations run up to three at once (at most `casting.gemini_concurrency`): press `g` on several voices and they start together, sharing one verifier, and the next starts as one finishes; local and ElevenLabs generations, cloned (changed) voices, and every other job still run one at a time, never beside them. Saving a description, script, traits, accent cue or clip count happens at once, even while other voices generate; only a save for a voice that has its own work queued or running waits for that work, which would otherwise write its older draft over the edit. Quitting while a job runs waits for that job and cancels queued ones. The screen acts only on your keys: it never accepts a take or regenerates a voice on its own.

The casting screen generates nothing when it opens: it saves the default drafts, prepares playback copies and says how many voices still have no accepted take. Voices are designed with their accents, so settle the accents first (`v`), then generate one voice with `g` or many with `B`.

The detail lists **Pronunciations**: every invented capitalized word of the character's names and aliases, and of the other names in your notes (ordinary dictionary words such as "Princess" or "Old" only when they already have a reading), with how it is said (a confirmed reading, one inherited from an earlier book, an unconfirmed draft, or the engine default). `w` picks one (or types another word) and opens [`pronounce`](#names-and-invented-words-pronounce) for it, with the test sentence in this character's voice when it has one; the casting screen is suspended while it runs and returns, reread, when it closes.

The line menu (`--plain`, or without Textual) has the same features and also generates nothing when it opens: settle accents with `A`, then generate one voice with `G` or many with `B` (see the table below). By default, creation uses Gemini when `gemini.api_key` is configured, otherwise the local speech engine. This preference also applies to `generate`, `generate-missing`, `regenerate-all`, `create-voices`, and any missing references needed during rendering. Audiobook speech continues to use the configured local engine and its saved reference recordings. New voices and successful regenerations are selected automatically. There is no acceptance step. Missing samples show generation progress; an individual failure leaves the other voices available.

When casting uses Gemini, startup, `--action generate-missing`, and `--action regenerate-all` overlap independent voice jobs, with up to four jobs in flight by default. Local transcription and verification stay serialized on one model thread while other Gemini requests continue. Each shared canonical voice is generated only once. Local and ElevenLabs casting remain sequential; `[render_pool]` does not control casting.

To tune Gemini casting, merge this setting into your existing global configuration:

```toml
[casting]
gemini_concurrency = 4 # Integer 1–8; 1 disables parallel casting.
```

The concurrency limit bounds whole voice jobs, including generation, verification, and hosted-voice cleanup; it is not a request-rate quota. Lower it if the Gemini project encounters rate limits. Completed voices are saved as they finish and reused on rerun. Ctrl-C during generation stops new jobs, waits for admitted jobs and their cleanup to finish, then exits; it does not discard their completed recordings. Bulk regeneration still invalidates every targeted selection before creating replacements, and unstarted targets remain explicitly pending.

Voice playback matches perceived loudness automatically, with a default target of -20 LUFS and a -1.5 dB true-peak ceiling. Startup prepares cached playback copies of the selected samples before showing the menu; regenerated samples are prepared before playback. Existing, inherited, imported, and newly generated voices all use the same settings. No reanalysis or voice regeneration is needed.

These are playback copies in the current book's `render-work/voice-previews` directory. The saved cloning references, their transcripts and checksums, voice selections, parent books, and already rendered audio stay unchanged. This setting controls auditions; it does not normalize the final audiobook. To prepare all saved candidates as well as selected voices without loading TTS or transcription models:

```sh
.venv-renderer/bin/audiobook-render cast runs/book-01 \
  --action normalize-samples
```

There is one menu, arranged in as many columns as the terminal width allows. Long names wrap instead of disappearing, and numbering stays stable. Type a number or command and press Enter:

| Input | Action |
| --- | --- |
| A voice number | Play immediately. Any key stops playback and returns to the menu; no Enter is needed to stop. The menu also returns when the sample finishes. |
| `E 5` or `E5` | Edit voice 5's description, sample script, and vocal traits without generating audio. `E` uses the last selected voice. |
| `N 5` or `N5` | Your notes about voice 5's character (other or later names, gender, origin, free notes; see [Your notes about a character](#your-notes-about-a-character)): each field shows its current value; Enter keeps it, `-` clears it, `B` cancels. After saving, `y` asks the analysis model for a description that agrees with them, which you see beside the current one and save only with `y`. |
| `W 5` or `W5` | List voice 5's names with how each is said, then open `pronounce --plain` for one (its number) or for a typed word, with the test sentence in that voice; the menu returns when it closes. |
| `A` | Review this book's accent plan as `accents BOOK --plain` does: a saved plan is shown as it is and a book without one gets a proposal; then `A` saves, `I` revises with your instructions, `N` asks for a new plan, `L` leaves. It says whose notes changed since their accents were assigned. |
| `B` | Generate many voices: `M` every voice with no voice yet, `U` every voice without an accepted take (saved verified takes are used first), or voice numbers such as `3 5 8-10`; then, for `M` and numbers, the designer (`L`/`G`/`E`, Enter for the configured one; `U` uses the configured one), and `y` to confirm. Voices are generated one after another; a model or verification outage stops the rest. It refuses until those voices' accents are decided. |
| `D 5 2` | Mark voice 5 as using voice 2's saved voice. `D 5` asks for the target; `D` uses the last selected source. |
| `G` | Edit and regenerate the last voice played locally. If none has been played, asks for a voice number. |
| `G 5` or `G5` | Edit and regenerate voice 5 with the local speech engine. |
| `GG 5` or `GG5` | Edit and regenerate voice 5 with Gemini. No command-line flag is needed. |
| `GG` | Use Gemini for the last voice played, or ask for a voice number. |
| `G11 5` or `G115` | Edit and regenerate voice 5 with ElevenLabs. |
| `G11` | Use ElevenLabs for the last voice played, or ask for a voice number. |
| `>` | Switch playback between 1x and 2x (pitch unchanged); the menu shows the current speed. |
| `Q` | Quit. |

All edit and regeneration commands offer four fields: the voice description, the spoken sample script, vocal traits (accent, articulation, cadence, and other stable speech behavior), and the accent cue (see below). Press Enter to keep a value, or enter `-` in the traits or accent field to clear it. `E` saves without generating. Description/script edits take effect when a sample is regenerated; traits condition the next render and do not change the saved sample. Keep the accent in the description and vocal traits consistent when editing both.

Regeneration verifies, saves, selects, and plays the new voice immediately. If generation fails, the previous voice stays selected. Inputs are case-insensitive. `G` chooses the local speech engine, `GG` chooses Gemini, and `G11` chooses ElevenLabs for that action; none changes the session's startup setting or global configuration. Use `G 11` with a space to regenerate voice 11 locally, since `G11` is the ElevenLabs command.

If you choose the wrong voice, enter `B` or `/cancel` at any description, sample-script, or vocal-traits prompt to return to the menu. Ctrl-C during these edit prompts also cancels only that edit. All fields are staged until the edit completes: cancellation discards the staged changes, preserves the selected recording, and makes no generation request. Enter still keeps the current field and continues.

`D` saves a book-local `voice-aliases.json` mapping. Character identities remain distinct; no recordings are deleted or generated. Existing roles linked to the source also follow its new target. Description/script edits and recasting operate on the shared voice; vocal traits remain separate for each logical voice/role. Descendants inherit aliases at lookup time, while earlier books and sibling branches stay unchanged. To remove a local duplicate choice, use `voice-aliases BOOK --remove SOURCE_VOICE_ID`; the original recordings remain available, and any inherited mapping resumes.

Menu and text entry use the terminal's standard line editor: arrow keys,
Ctrl+A/E, Ctrl+K/U, and session history follow your readline/libedit bindings.
While a sample is playing, any key stops playback and returns to the menu.

Sample scripts contain 25–60 words and at most 650 characters, normally aiming for 35–50 words. Casting prefers clearly attributed ordinary speech and can join several actual statements by the same canonical voice to provide a useful audition. Balanced outer dialogue quotation marks are removed before joining excerpts; embedded quotations and apostrophes are retained. Every excerpt keeps its own chapter, segment, exact source ranges, and position in the joined script; a montage is not represented as one contiguous quotation. Narrators can use narration, and identified participants in simultaneous dialogue can each audition those words alone. Shared roles use one script and one saved reference voice.

Voice-design requests use the complete description without an automatic prefix. Analysis and prompt refresh author any useful adult-performer, age, or fictional context naturally within the description, once; youthful portrayals can name an adult actor without stacking generic introductions. The same rule applies to every designer, the casting menu, and automatic voice creation during rendering. Known prefixes from older application versions are removed when preparing a new request, while the description's own performer context and custom wording are preserved. Character ages and source evidence remain in their metadata. The limit is 4,000 characters for the actual description, with no space reserved for application boilerplate. Selected recordings and inherited voices are reused unchanged. Recovery of a recorded Gemini sample awaiting verification retains that sample's original request and provenance; removing a prefix does not require creating another voice. Remote voices left by an interrupted operation retain the normal cleanup/recovery behavior.

For a creation prompt, describe the voice's lasting identity in one or two concise sentences using a few compatible audible traits: typical pitch, resonance, vocal weight, texture, accent, articulation, and natural cadence. Include age, gender presentation, or fictional context when they help establish the sound. Translate personality into vocal choices: precise consonants and measured phrasing for a meticulous character, or a thin, reedy tone and dry rasp for an abrasive creature. Preserve essential harsh or unusual qualities. Put temporary emotions, urgency, whispering, shouting, and changes of pace in individual passages' delivery directions. The voice generator knows nothing about the book: make the prompt self-contained, translating character names, fictional species, and lore into explicit audible characteristics.

Fresh analysis is instructed to make the description complete. Default voice creation uses only that description. Structured trait fields remain evidence metadata; neither those fields nor their source quotations are appended to the prompt. An older profile whose important vocal details appear only in trait fields may need its creation prompt edited before casting. Custom description wording and older trait lists are preserved; only recognized obsolete application prefixes are removed when preparing a request. A description over 4,000 characters fails with a request to shorten it instead of silently truncating vocal identity. The sentence count and coherence are model guidance, not a guarantee. Speech verification checks the recording and transcript; listening determines whether the performance suits the character.

The optional voice-profile `speechTraits` field records a voice's stable speech behavior: an explicit accent and habitual articulation, rhythm, pace or verbal habits, never emotions or personality. Rendering does **not** send it with each line. Drama 3 clones identity, accent included, from the saved reference recording and from that voice's own earlier takes in the chapter history (see "Chapter continuity"); a per-line voice description competes with the reference and dilutes each line's own delivery direction, which is all a rendered line now carries in its style control. Keep accents in the reference recording itself. Pronunciation previews still apply the traits. The field accepts at most 400 characters of nonempty, single-line plain text without engine markup.

Use `E` in casting to override these traits, or `cast BOOK --action edit --voice-id ID --speech-traits "Light Mexican Spanish-influenced English; relaxed cadence"`. An empty CLI value clears traits. These explicit edits live in the book's `voice-traits.json`, separate from checksum-bound analysis. Lookup uses the current book, then its parent recursively; a local choice (including an explicit clear) overrides an earlier choice. No global config edits or reanalysis are needed. Each render snapshots the choices for recovery and sends them with remote chapter jobs. Pronunciation previews and frozen comparison snapshots use the same choices.

Changing speech traits does not change rendered speech (an accent cue does; see "Accent cues"). Existing valid reference voices do not need to be regenerated. Structured analysis `traits[]` and their evidence are not rewritten by the casting editor; changing those records still uses the validated analysis/corrections workflow.

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

#### Listening at 2x

`>` switches playback between 1x and 2x everywhere audio is played to you: the run shell's takes dialog (from the Errors and Rejected tabs), `review-ui` (passages and their takes, full-screen and `--plain`), the casting screen and menu, `pronounce`, and the pronunciation listening queue. The speed is shared, so switching it in one place applies to every later playback in that command; each screen shows the current speed in its help or status line, and a line menu takes `>` as its command. On a full-screen screen a take that is playing starts again at the new speed; elsewhere the new speed applies from the next play. In the listening queue `>` does not interrupt the walk. Only the tempo changes, never the pitch: `afplay` plays at its own time-pitch rate (`-r 2 -q 1`), and any other configured player gets a temporary FFmpeg `atempo` copy (FFmpeg from `PATH`), named after the take and deleted when playback ends. Saved takes are never changed. A command starts at 1x.

### Accent cues

Drama 3 keeps the basic sound of a voice from its reference recording, but it does not reliably reproduce an accent from the recording alone. In blind listening tests on six accented voices, the take that was told the accent was preferred in 9 of 11 sets, and the plain take in none; with each line's real delivery direction as well, the cued take was preferred in 11 of 12. Accents are also how a listener tells a large cast apart. Every voice therefore gets a short **accent cue**, such as `heavy Brooklyn accent`, that opens every line spoken in that voice, ahead of the line's own delivery direction:

```text
[heavy Brooklyn accent][raspy][threatening] Mr. Derrins,
```

The cue and each part of the line's direction (split at `;` and `,`) are separate tags, the way Fish layers cues (`[sad][whispering]`): inside one bracket, the line's feeling tended to crowd out the accent. A one-word whisper, sigh or laugh keeps its exact control. Lines with inline performance controls get the cue as their own leading tag, and a styled span returns to it afterwards. The controller builds this text for every engine, remote `serve` workers included; the recorded direction stays `cue; direction`. A clause written as sentences (a clip repair's "Flat, matter-of-fact note. Begin on the first word.") is an instruction rather than a list, so it stays one tag. Lines whose direction has more than one part render again on the next render. Takes recorded in the earlier one-bracket form stay valid records, so they never block a render, but they render again in the separate-tag form: ordinary takes because their engine text changed, and repaired, model-chosen and recovered takes because their identity notes any part recorded in the earlier form (`speech_identity.recorded_in_earlier_form`, looking through every take a repair or recovery is made of). A take you picked yourself in `review-ui` is kept.

**Accent assignment.** `accents BOOK` (below) assigns the book's accents for a person to review; before a render plans, every voice that already has a recording but no accent yet also gets one from a single book-wide request to the analysis text model (`provider` in the TOML file). For each speaking voice the request carries the character's name and every alias (titles such as "Prince of Drasnia" often say where someone is from), the analysis's summary of the character (and the one from the earliest book they appear in), the narration where the series first mentions them (taken from the earliest book in the series that names them, which is where a story usually says who they are and where they come from), one of their own lines, the description that designed their voice, their speech traits (the ones a person set in casting, else the analysis's), and your own notes about the character when you wrote some (see [Your notes about a character](#your-notes-about-a-character)). A description or traits often already name the right accent, and the model keeps it when it fits the story. The model groups the cast by where each character comes from (nation, region, city, people, culture or class) and gives each group a distinct real-world English accent from Earth: characters from the same city share an accent, different places get clearly different ones, a fictional people gets the real accent that best fits how the story presents them, and General American is for at most one group. An accent must name a real place or people and never the book's own (a "Valdorian accent" is refused, and the model is asked again), in the plain words a voice actor would use: a phonetics term such as "Received Pronunciation" or "RP" is refused too, and one saved earlier, or named in a voice description, is read as "upper-class English" wherever cues are made. Descriptions and traits are evidence, not decisions: they were written one voice at a time, so when nearly every one says "General American", or one names the book's own people, the story wins. Cues a person chose, an accent named at the start of vocal traits a person set in casting ("General American; cool measured pace", "Heavy Russian accent, slow"; it then counts as that person's choice), and accents already decided are sent as fixed, so a continued book keeps its inherited characters' accents and new characters are placed around them. The model also says why each group gets its accent. Each voice's accent is saved in the book's `voice-accent-assignments.json` (inherited by continued books, nearest book first), and each group's origin, accent, reason and members in `voice-accent-groups.json`; a voice the model leaves out gets no assigned accent. A later book's request carries every origin the series already decided ("Valdoria: Mid-Atlantic accent", its reason and a few of its characters): a new character from a known origin joins it with the same accent, a group that reuses an origin's name with a different accent is refused and asked again, and a new origin gets an accent different from all of them. The report marks groups that continue an earlier book's origin. The request goes through the book's request cache, so a rerun asks nothing new. Set `casting.assign_accents = false` to stop the automatic assignment; without `provider.model` it is skipped with a notice.

**The accent list.** The model chooses each group's accent from a fixed list of 108 short names (`renderer/accent_catalog.py`), so a series keeps using the same few words for the same accent and every request stays small (about 2,000 characters). The list is grouped into families of similar accents by region: England south (upper-class English, Estuary, Cockney, West Country, …), England Midlands and north (Brummie, Yorkshire, Scouse, Geordie, …), Scotland, Ireland, Wales and islands, three US families, Canada, the Caribbean, Oceania, Africa, South Asia, East and Southeast Asia, the Middle East and North Africa, Western, Southern, Nordic and Eastern Europe, and Latin America. A few links join close accents of different families (Mid-Atlantic ≈ upper-class English, Northern Irish ≈ Scottish, New Zealand ≈ South African). Places far apart in the story get accents from different families; places that are close or related (provinces of one kingdom, rival cities, a colony and its homeland) get different accents of one family, so they sound related yet distinct. Fish publishes no list of the accents Drama 3 knows, so a star marks an estimate, not a measurement: an accent seldom heard in recorded English (Doric, Black Country, Kerry, …), which the engine may blur into its family's common accents. The model is told to prefer unstarred accents. A reply naming an accent off the list is refused and asked again, except one already decided (a fixed voice's, a series origin's, a kept accent) or one the person's guidance or instructions name. Other words for a listed accent are saved under its listed name ("Liverpool" → Scouse, "Transatlantic" → Mid-Atlantic); a strength word stays ("broad Scouse").

To redo a book's accents, or to steer them, run:

```bash
audiobook-render accents BOOK --instructions "The northern clans sound Scandinavian, the imperial court Italian, the river traders Indian English"
```

The first time, it asks the model for a plan covering every voice in the book except cues a person chose and accents inherited from earlier books, as a fresh request (an interrupted run finishes with `--resume RUN_ID`). Once a plan is saved, running it again opens that saved plan for editing and asks the model nothing until you ask (`i`, `n`, or `--instructions`, which revises the saved plan); characters added to the book since appear under "no accent", and a book saved before groups were recorded shows one group per accent with its origin marked "not recorded" (`--keep` asks the model for the origins and reasons). It shows the plan: each group's accent, origin, reason and characters, and how many voices would change. On a terminal the plan opens in a full-screen review (`--plain` for line output): groups on the left, the highlighted group's reason and characters on the right, and the model requests and log below. Its keys:

| Key | Does |
| --- | --- |
| `i` | Type what to change ("the imperial court should sound Italian", "keep Marn with his uncle"); the model revises the whole plan and only what the instructions imply |
| `a` / `o` / `w` | Change the highlighted group's accent, origin or reason by hand. `a` opens the accent list under its family headings, the current accent highlighted; typing filters by name, family or close accent ("Scottish" lists Scottish first, then every accent close to it). Accents another group already uses are yellow with that group's origin. The last row types an accent of your own, such as one with a strength word or one not on the list (checked like the model's: a real accent, not the book's own) |
| `m` | Move a character (the highlighted one in the characters list, else choose) to another group or a new group with its own origin and accent |
| `n` | Ask for a fresh plan, replacing the current one |
| `s` | Save, after confirming; nothing is saved before this |
| `q` | Quit; with unsaved changes it asks first |

The same review opens inside the casting screen: press `v` there to see and edit the book's accent plan without leaving casting, and `q` (or `Ctrl+Q`) goes back to the cast, asking first when the plan has unsaved changes. While it is open, the casting keys do nothing; after you save, the cast shows the new accents.

The highlighted group's details show which other groups have the same accent ("Same accent as", in yellow) or a close one ("Close to"), and the similar accents to choose from to tell related places apart; in the groups list a close neighbour is marked `≈` and a shared accent is yellow. When the clues for some characters are too thin to tell where they come from, the model may first ask, once, to read more about up to 12 of them: a second request then sends further book passages naming each (up to six beyond the two introductions always sent, earlier series books first) and up to three more of their own lines, and the plan must follow. The log says whom it asked about. While a model request runs, a banner across the top of the screen says what is being asked and for how long, and the groups list shows that it is loading until the first plan arrives. Accents are named tersely, in one or two words ("Scottish", "heavy Russian", "upper-class English"); the screen shows them without the word "accent", which the cue keeps. The model is asked for that style and new replies are normalized to it (`accent_assignment.terse_accent`: a leading strength word in lower case, "German-accented English" → "German"); an accent decided earlier (in the series, kept with `--keep`, or chosen by a person) keeps its exact words. A group continuing an origin from an earlier book is marked "(series)", and its accent cannot change unless the origin is renamed. Characters the model left out appear as "no accent". Every character with lines is listed: those whose accent was decided elsewhere (a cue chosen in casting, an accent leading vocal traits set there, or an earlier book's assignment, shown with that book's origin) appear dimmed as "(kept)" groups, which the model works around and does not change. `m` moves any character, kept ones included, to a group, a new group, or "No accent"; saving makes it this book's own choice (for a cue chosen in casting, the move replaces that cue). If every character is kept, the screen still opens so you can move one. Without the screen, a terminal is offered `[A]` save, `[I]` instructions for the model to revise the plan (typed after `I` or on the next line), `[N]` a new plan, or `[L]` leave; `--apply` saves a new or revised plan without asking (a saved plan with no `--instructions` has nothing to save), and anywhere else the command only reports. Instructions typed while reviewing are saved as the book's accent guidance too, so later assignments (new characters, later books) follow them. The guidance is saved with the accents (`voice-accent-guidance.json`, inherited by continued books) and reused for voices assigned later; run it again without `--instructions` to reroll with the same guidance. Assign earlier books of a series first, since later books keep their accents and origins. A book assigned before reasons were saved can record its groups and reasons without changing any accent: `audiobook-render accents BOOK --keep` (the model must return every current accent unchanged, or it is asked again).

A voice with neither a chosen cue nor an assignment (assignment off, no model, or the model gave it none) uses the accent its description names, if any: the description that designed its saved recording, or, before the voice is made, the one that will, taken from the last "… accent" phrase after its last joining word ("voice with a thick Welsh accent" gives `thick Welsh accent`), with the same real-place and book-term rules. A derived voice (a character's later state) keeps its main voice's cue. Casting shows the cue a render will use and where it came from. Casting offers only voices with lines in the book. The analysis casts every named character, so a series keeps its identities, including ones a book only names; those never appear in casting (the cast screen, the line menu, `--action list`), get no draft, audition script or description refresh, and are never generated in bulk. `--voice-id` naming one is refused. They need no accent either, and are cast in a book where they speak. A bundle loaded without chapter rows cannot tell, so every voice is offered.

A voice description written before accents were assigned may still describe a different accent in its own words ("broad warm Irish" with a West Country cue, or "General American" with a Yorkshire one), and the designer then hears both; the cue is added, but only an accent phrase ending in "accent" is replaced. Press `i` in casting with an instruction such as "remove every mention of accent or regional speech" to clean those descriptions before regenerating. Accents are decided before voices are made, and a person decides them. Every new voice is designed with the accent its lines will be cued with, so casting (`g` in the menu, `generate`, `generate-missing`, `regenerate-all`), `create-voices`, and a render that would create missing voices stop before designing a voice whose accent nobody has decided, naming the voices and the command: review and save the book's plan with `accents BOOK` first. A decision is a cue chosen in casting (an empty one included), an accent leading vocal traits set in casting, or a saved assignment (an inherited one, or the model's "no accent", included). A voice with no lines in the book needs none, and a derived voice follows its main voice. Browsing, listening, accepting and editing in casting need no decision. A render still assigns accents by itself, but only for voices that already have recordings, so their lines get cues; it never decides the accent a new voice is designed with. Without `provider.model`, or with `casting.assign_accents = false`, nothing is required and voices are designed as their descriptions say. An accent its description already names is replaced by the cue ("with a neutral General American accent" becomes "with a broad Yorkshire accent"), otherwise "Speaks with a … accent." is added; the saved voice records that description. A description already at the 4,000-character limit is designed as written. A voice recorded before its accent was assigned keeps its recording: it keeps its timbre and takes the cued accent as far as Drama 3 follows it, and regenerating it in casting makes the two agree.

To choose a different cue, use `E` in casting: the fourth field shows the current cue and, when the description suggests a different one, that suggestion (`=` uses it). Without the menu, use:

```bash
audiobook-render cast BOOK --action edit --voice-id VOICE_ID --accent "heavy Brooklyn accent"
```

Keep the cue to a few words naming the accent (at most 80 characters, plain text). A chosen cue wins over an accent in vocal traits, the assignment and the description, and an empty cue (`-` in the menu, `--accent ""`) means no cue for that voice at all. Chosen cues are saved per logical voice in the book's `voice-accents.json` and inherited by continued books, nearest book first; a later book can clear an inherited cue for itself. No recording is regenerated. A cue (automatic or chosen) is part of the synthesis input of that voice's lines only, so adding, changing or clearing one re-renders those lines on the next render and reuses every other voice's takes; voices without a cue render exactly as before. A render freezes every voice's cue when it starts. Vocal traits are a separate, longer description and are still not sent with each line. The cue is added at render time, so `refresh-directions` neither needs rerunning for it nor changes it.

### Your notes about a character

You often know more about a character than the book has revealed so far: that a "cowled shadow" is later revealed as Zandramas, is a woman, and is a Mallorean with a more exotic accent than other Malloreans. Press `n` on a voice in the casting screen to write it down. The notes open as lines you edit:

```text
names: Zandramas
gender: female
origin: Mallorea
notes: Revealed later as Zandramas; a more exotic accent than other Malloreans.
```

`names` are other or later names (comma-separated, at most 8), `gender` and `origin` a few words each, and any other line is a free note (at most 1,500 characters; plain words, no brackets). Every field is optional. The notes are your choice, saved per voice in the book's `character-notes.json` and inherited by continued books, nearest book first; saving them empty in a later book clears inherited notes there, and no recording is changed. A derived voice (a character's later state) shares its main voice's notes. `/` finds a voice by the names and words in its notes.

Your notes reach the two places that decide how the character sounds. The analysis text model receives them as a description of the character, never as instructions, and the voice designer never receives them directly:

- **Accent.** Accent assignment ([Accent cues](#accent-cues)) sends them with the character as what the listener knows, which may go beyond the book so far, and follows them over the book's clues; their extra names also find the passages that name the character (eight instead of one for the cowled shadow in *Guardians of the West*). Each saved assignment records the notes it was made with (`voice-accent-assignment-notes.json`), so changing a character's notes makes its accent undecided again: `accents BOOK` lists whose notes changed, keeps them in their groups for review, and saving the plan decides them (with a revision, or as they are). A voice that is not recorded yet cannot be designed until then; a render reassigns a recorded voice's accent by itself, as for any voice without one. An inherited accent counts too, so notes added in a later book reassign the accent there only. A cue you chose in casting still wins over everything.
- **Voice description.** After you save notes, the screen offers to ask the model to revise the description to agree with them (you review the proposal, as for `i`; recordings are unchanged until you regenerate). `i` revisions and `r` rewrites also receive the notes, and the description they write must agree with them in audible terms only, without names or plot secrets. A voice generated while its description contradicts the notes' gender is noted in the log.

Requests without notes are byte-for-byte the same as before, so their cached answers and saved plans are unchanged.

### Reference clips

Drama 3 clones a voice more closely from more reference speech: its technical report measures speaker similarity 0.605 from one clip and 0.633 from three, and Fish recommends two or three clips of 15–20 seconds. With the Gemini designer, a new voice (from `cast`, `G`/`GG`/`G11`, bulk casting, or automatic creation during `render`) is therefore recorded as up to three clips of the same hosted voice:

```toml
[casting]
reference_clips = 3  # 1 through 3; 1 records a single clip as before
```

The first clip speaks the voice's audition script (the one `E` edits). Each further clip speaks other book speech by the same character, chosen by the same audition rules and never repeating a passage already used. When the book has too little suitable speech by that character, the configured text model writes the remaining scripts. It sees the character's description, a few of their lines and the scripts already chosen, and writes new speech in the same style, following the audition script rules. These invented scripts are prepared along with the audition scripts, before any speech model loads, and saved under `extra_scripts` in `audition-scripts.json`, so later runs reuse them. Like an invented audition, they never enter the audiobook. Every clip is verified like any reference. The clips are joined in order with half-second pauses and saved as the voice's reference, with the clips' transcripts joined in the same order. The joined recording passes the same audio checks as any take (length, silences, level, clipping). It is not transcribed again: its words are the verified clips' words, and recognizing the longer recording mostly adds recognizer errors, such as an invented "thank you" at the end or another spelling of a name. Its verification record says `joined_verified_clips` and lists each clip's audio checksum and take request. Later checks of that recording, when you accept it after verification settings changed or `import-voices` copies it to another book, also check its audio and keep the clips' evidence instead of transcribing it whole. After `G`/`GG`/`G11` and `--action generate`, casting says how many clips the new voice kept, and says so when a Gemini voice kept fewer than `reference_clips`. Its creation metadata lists each clip (`reference_clips`: transcript, audio checksum, take request and book source).

One voice can record a different number of clips than the default. The full-screen casting screen (`c` on a voice) saves that choice in the book's `voice-clips.json`, keyed by the shared voice; a continued book inherits it unless it saves its own, and clearing it returns the voice to `casting.reference_clips`. The choice applies the next time that voice is generated and never changes an existing recording. Books without the file prepare exactly the same audition and extra scripts as before. Only the Gemini designer records more than one clip, so the choice has no effect with other designers.

Rendering needs nothing new: the joined recording is one longer reference, so every engine, `serve` worker, chapter service and Runpod worker uses it as it is. Existing voices keep their single clip until you regenerate them, and each regenerated voice re-renders only its own lines.

Fewer clips are kept in these cases:
- **No invented scripts:** a character without enough other clean speech gets fewer clips when the text model's request fails. The next run asks again.
- **A clip fails verification:** that clip is left out.
- **The joined recording fails an audio check:** the voice keeps its first clip alone.

Only Gemini keeps one voice across several scripts within one creation. ElevenLabs and local designers would design a different voice for each clip, so their voices keep one clip. A creation interrupted after its hosted voice was deleted cannot add matching clips, so the next run starts that voice again with a fresh hosted voice and records every clip from it. Derived voices (age or condition changes) are cloned from their main voice's saved reference, so they hear all of its clips.

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
The description sent to ElevenLabs must contain 20–1000 characters; the
sample script must contain 100–1000 characters. Unsupported lengths produce
an actionable error without truncating or rewriting your script.
`eleven_multilingual_ttv_v2` is also supported;
`guidance_scale` accepts 0–100 and `loudness` accepts -1–1.

Casting prints the API error message and response details, including the
HTTP status, model, and any field validation errors. Network errors
include their underlying reason. These diagnostics are also saved in each
`attempt-*.json` report. Messages are not shortened; configured credentials are
masked and terminal control sequences are removed. The final failure includes
the last generation or verification exception as well as the inspection path.

Gemini casting sends the voice description and selected sample script to Google's
API. It creates a temporary hosted custom voice, synthesizes the exact script, and runs
the same independent transcription and audio checks as local creation. Existing
selections are reused unless you regenerate them. Shared character roles still
use one canonical reference, and later books inherit it through `--continue-from`.
Drama 3 renders the audiobook locally by cloning the verified recording; Gemini
is not needed to render that saved voice. Normalized audition copies work for
both sources.

The remote voice ID is recorded in candidate metadata and reused across synthesis
retries within one voice-creation operation, whether started through casting or
automatic creation during rendering. When the operation succeeds, exhausts its
attempts, or is interrupted, the hosted voice is deleted. Startup also cleans up
voices left by a crash. Local reference WAVs, selected character voices, and
their metadata remain available. Explicit `GG` regeneration and
bulk regeneration request a fresh Gemini voice. If an earlier creation timed out
before returning an ID, its exact saved request name must first be reconciled and
cleaned up; an unresolved outcome blocks another creation and explains the problem.
The historical Gemini `voice_...` ID records where the local reference came from;
the character's `series_voice_...` ID remains stable.
Saved audio awaiting verification can still be checked after the hosted voice
is removed. If it fails verification, another take uses a fresh hosted voice.
Failed deletions are recorded and retried when the same command is rerun,
without discarding a saved local sample. Credentials are read only from the
configuration file and are excluded from candidate metadata and cache
identities. No Google SDK is required.

Google currently allows up to 200 stored custom voices per project, with expiration
after one year of inactivity. Automatic cleanup releases this quota after voice
creation; local saved recordings remain usable after remote deletion or expiration.
See [Google's voice design documentation](https://ai.google.dev/gemini-api/docs/voice-design).

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

To reuse the voices you chose for another book, for example after re-analyzing a book from a cleaner EPUB (which gives it new voice IDs), import them by character:

```sh
just import-voices runs/series-book-1-v2 --from runs/series-book-1 --dry-run
just import-voices runs/series-book-1-v2 --from runs/series-book-1
# or: audiobook-render import-voices BOOK --from OTHER_BOOK [--dry-run]
```

Voices are matched by character, never by voice ID. The narration voice goes to the narration voice. A character's voice goes to the other book's voice whose character has the same name, or else to the one voice sharing one of its names or aliases. Each match is imported like `import-voice`, with the other book's selected reference recording and its verified transcript, so it is normalized and transcribed again. Its book-local speech traits come too, unless this book set its own. A voice this book already chose, or already inherits with the same recording, is kept. Derived voices (aging, conditions) are cloned from their main voice and are skipped. Ambiguous and unmatched characters are listed for `cast`, and the command then exits 2. `--dry-run` shows the matches without loading models.

### Merging two cast entries into one character

When the analysis made two cast entries for one person (a stranger later named, a nickname read as someone else), say they are the same character:

```sh
just merge-characters runs/book-01 "the old man" Arlen --dry-run
just merge-characters runs/book-01 "the old man" Arlen
# or: audiobook-render merge-characters BOOK CHARACTER OTHER [--keep CHARACTER] [--name NAME] [--yes] [--dry-run]
```

Name each entry by its name, an alias, its character ID or its voice ID. This is not voice sharing: `cast` `d` and `voice-aliases` only reuse another voice's recording, while a merge changes the book's identities. One entry is kept, with its character ID, voice ID and recorded voice, its description, and its accent and other per-voice choices. When one of the two comes from an earlier book in the series and the other is new to this book, the earlier book's character is always kept and the new entry is merged into it (`--keep` naming the new one is refused, and nothing offers to swap). Otherwise the default is the entry that already has its own accepted voice, else the one with more lines, and `--keep` chooses. The absorbed entry's names and aliases become the kept entry's aliases (`--name` picks the merged name from either entry's names), and its evidence joins the kept entry's. Every line it spoke, narrated or was a candidate for becomes the kept entry's line. Its roles merge into the kept entry's role of the same kind; a role the kept entry lacks, a changed-voice state (now cloned from the kept voice) or a separate narration voice moves to the kept entry. A passage the two spoke together becomes the kept entry's line alone, and a larger group keeps its other speakers. Review issues about the absorbed entry now point at the kept one. The absorbed character and its voice leave the cast, so casting, accents and rendering see one character. Its own per-voice choices in this book are dropped: speech traits, accent cue and assigned accent, accent-group membership, reference clip count, voice sharing from it, and its audition script. Another voice that shared its recording now shares the kept voice. When the kept entry used the absorbed entry's recording, that link is dropped and it goes back to its own voice; the command says so, and says when an accepted recording of the absorbed entry will no longer be used.

The command shows what will change, including each entry's description and first line so two entries with the same name can be told apart, and asks: `y` saves, `s` swaps which entry is kept (not offered when an earlier book's character must be kept), `c` chooses the merged name, `n` changes nothing. `--yes` saves without asking, `--dry-run` only shows the change, and without a terminal nothing is saved unless `--yes` is given. In `cast`, `J` does the same for the highlighted character, which is the easier way when the two share a name: choose the other character (each choice shows its status, line count and description), then which entry is kept (skipped when an earlier book's character must be kept), then confirm; the cast reopens on the merged book.

A merge changes the named book and every later book in the series that carries the absorbed entry (series character IDs are shared across books). A later book that has only the absorbed entry gains the kept character in its place, with the kept voice inherited as series casting. Earlier books keep the two entries apart; the command lists them, and merging in the earliest book where both appear changes the whole series. Publication goes through the same receipt-bound revision as speaker decisions: each changed book's `provenance.character_merges` records the merge, and the saved analysis must equal that record applied to the original. All books, their series histories and continuation links, and their per-voice files change together, with originals and a journal in `.analysis-revisions/<id>/`. It refuses while a render uses any of the books. An unfinished speaker-decision journal is carried over to the merged cast, so `analyze BOOK` resumes it. Pronunciation choices are bound to words, not characters, and are unchanged. A prepared `performance-plan.json` binds the cast and scripts, so expressive rendering needs `prepare-performance` again, as after any attribution change.

Lines that moved to the kept voice get new chunk IDs: their current speakers become part of their chunk identity (`revoiced_segments` in the merge record), so the next `render` renders them, and their chapters, again with the kept voice. Every other line keeps its chunk ID and audio.

Generate all chapters and assemble the complete book:

```sh
.venv-renderer/bin/audiobook-render render \
  "runs/book-01" \
  --epub "/path/to/book-one.epub"
```

To choose separate work and publication directories for one run, add `--work-dir /path/to/work -o /path/to/audiobook`. Omit `--epub` when no original EPUB cover is needed.

Rerun the same `render` command after updating the scripts or selected voices. It reuses matching verified clips and unchanged completed chapters, generates the changed passages, and rebuilds the affected chapter files, synchronization data, and M4B in the existing output directory. Moving speech, renumbering segments, changing source IDs or offsets, and editing attribution explanations do not regenerate an otherwise identical performance. Chapter assembly and synchronization use the current locations. An unchanged local render also skips final encoding. Older published outputs without the new publication key are repackaged once from their compatible cached audio.

After `analyze` appends newly published chapters to a serial (see [ANALYSIS.md](ANALYSIS.md#updating-a-serially-published-book)), the same `render` command designs or casts only the new voices, renders only the new chapters, and reuses every earlier chapter: chunk IDs and chapter checkpoints bind `provenance.chunk_identity_source_fingerprint`, the first publication's source identity, so earlier chapters' plans are byte-identical. Their only change is the chapter count in each chapter file's track tag (`3/95`), which `render` and `assemble` apply by re-encoding the saved, verified chapter PCM; no speech is regenerated, re-verified or re-composed. Either the original EPUB, the update's EPUB, or a rebuilt or typo-corrected edition of the same book supplies the cover and metadata. This retag-only reuse covers local rendering; remote chapter services (`serve-chapters`) are not yet specialized for it.

When the update also edited earlier chapters, their unchanged lines keep their block and segment IDs and so their chunk IDs. Those chapters are recomposed: lines before the first edit match their saved takes exactly, the edited lines are generated, and the lines after them are kept under the default `history_reuse = "tolerant"` (or regenerated for that voice under `"strict"`). An edit that adds or removes segments renumbers later chapters' script `sequence` values; that bookkeeping does not regenerate or recompose their speech.

To investigate unexpected regeneration, run a read-only report on the computer
that owns the book's render cache:

```sh
.venv-renderer/bin/audiobook-render cache-report out/my-book > /tmp/audiobook-cache-report.json
```

It compares the current plan with accepted recordings and reports reusable clips,
cache misses, and the fields that differ from nearby saved passages. Use
`--chapters 7` to inspect one chapter or `--max-examples 30` for more miss examples.
It reads model fingerprints and verifies cached audio checksums, but does not load
inference models, download files, generate speech, or modify the book/cache/model
directories. No verifier is needed. The report omits API keys and full source text.
Counts are a snapshot; stopping a render before inspection makes the result easier
to compare. Built-in local engines and rendering pools are supported; for chapter
server mode, run it against the server's local book and cache.

Successful speech is indexed in `render-work/speech-cache/<content-key>.json`. For the built-in Drama/Fish engines, this key binds the prepared text and applied delivery controls, reference audio checksum and transcript, actual model identity and sampling settings, base seed, and output sample rate. Paragraph performances also bind their ordered text and speaker-reference mapping. Runtime names, runtime and adapter versions, character labels, profile/designer metadata, verification settings, source positions, installation paths, and download requirements are excluded. Unknown plugins retain their engine/voice inputs conservatively because they may interpret additional fields. Original requests, take numbers, seeds, and reports remain under `render-work/clips/<original-request-hash>/` as provenance; those directory names are not the content lookup key. New generation seeds also use the content identity.

The optional `engine.cache_model_alias` replaces the strict model component
for compatibility matching, allowing accepted speech from the named model or alias
to survive a deliberate model change. It never makes rejected or unverified audio
eligible, and the actual generation identity remains in the original request.

Only successful, checksum-verified recordings enter this index. Compatible legacy accepted takes are indexed automatically when their exact saved voices can be resolved; unresolved historical voice bindings are never guessed. Index entries bind both the acceptance file and waveform checksums. Failed takes remain diagnostics, not cache hits. A passage moved to another chapter can reuse its waveform while retaining separate current source mappings and chapter timing.

Completed chapters also check the prepared speech inputs against their saved performances. If an engine-input change makes some clips obsolete, `render` reports the affected block count, rebuilds that chapter, and reuses its matching verified clips. For example, older Drama 3 takes may record a freeform delivery direction without actually applying it; those passages need a new take when the direction is now sent to the engine. An unchanged script or request hash alone cannot prove that the performance matches. Corrupted audio, checkpoints, and verification records still stop rendering. `assemble` does not generate speech and reports that `render` is needed when a chapter contains obsolete performances.

The previous audiobook remains available while its replacement is prepared. Failed speech or encoding leaves it intact; a failed final rename restores it. Replacement requires a complete, unmodified renderer output for the same source book, with no extra files and no symlinks outside the renderer-owned `chapter-links/` convenience directory. Unrelated or manually modified output directories are refused before generation. Keep personal files outside the generated output directory. A partial chapter selection cannot replace an audiobook covering different chapters: use `--chapters-only`, followed by a full `render` or `assemble`, or publish the subset to another directory.

A chapter that exhausts its bounded speech verification attempts, or hits another ordinary chapter-local render failure such as a permanent generation error, fails as a chapter, not the whole book: unrelated running and pending chapters continue through their checkpoints, and the run ends without assembling the final M4B. `render-work/status.json` records `status: "blocked"`, the aggregate error, and each failed chapter's number, error type, and reason; when the failure is tied to a speech request, the entry also carries the failed passage's segment IDs, chunk ID, and request hash (the failed take's directory name) so `review-ui` can mark and jump to the exact passage. Rerun the same command to reuse every completed chapter and give failed chapters a fresh bounded allowance. Interruption, provider or verification outages, setup/configuration failures, and checkpoint or book-level integrity errors still stop admission; those failures are never reported as completed chapters.

M4A chapters with AAC audio at 128 kb/s are the default. They are directly playable and use the same codec as the final M4B. `--chapter-format mp3` on `render` or `assemble` produces MP3 chapter copies when needed; the final book remains AAC in an M4B container. Bitrates are global settings in `[render]`.

Drama 3 joins adjacent fragments within a paragraph only when they have the same role and voice. Character/narrator/character spans produce separate synthesis requests, including when the narrator and character share a saved voice. This prevents a missed model speaker switch from assigning a narrative tag to the character. A line longer than `max_chunk_chars` is first cut after its last sentence within that length (else at a semicolon or colon, else a comma or dash, else between words). Long performances split at existing safe boundaries under `max_paragraph_chars` (default 1000), ending after the last full sentence that fits (else a semicolon or colon); saved groupings are kept as they are. Bracketed passages, simultaneous speech, and passages whose source mapping cannot be joined exactly retain their individual treatment. Other engines without paragraph support retain the existing clip workflow. Set `paragraph_synthesis = false` to disable grouping altogether.

Accepted individual clips and paragraph performances within one role/voice remain reusable. Older cached paragraphs combining different roles or voices are no longer reused: their original passages are rendered separately, and affected chapter checkpoints are rebuilt. Word recognition checks the requested text, not the speaker's identity or timbre; a passing transcript cannot prove that an internal voice switch happened. Synchronization records each generated interval and its original text ranges and voice assignments in `source_segments`. Grouped entries have `precision = "paragraph"`; exact word timestamps are not inferred. If an older script itself assigns a narration tag to a character, rerun `epub-audiobook analyze BOOK_DIRECTORY` to repair those boundaries before rendering. No voice regeneration is needed.

If a paragraph exhausts its generation attempts because of missing speech, silence, or another audio quality failure, rendering automatically retries its original shorter passages individually, using the same assigned voices and verification rules. The recovery decision is saved in `render-work/paragraph-fallbacks` before any individual passage is generated, so interruptions and later invocations keep the smaller boundaries. Previously verified clips and completed chapters are reused; failed paragraph takes remain available for inspection. Each recovered passage has its own source mapping and audio interval. An API or verifier outage does not trigger splitting. Individual passages can also use the bounded sentence recovery described below; unresolved failures still stop publication.

`--chapters` accepts comma-separated numbers and inclusive ranges. `start` means the first manifest chapter and `end` means the last; for example, `just render BOOK --chapters 39-end --chapters-only`, `--chapters start-5`, or `--chapters start,10-12,end`. These aliases are case-insensitive and work wherever `--chapters` is supported. Reversed and out-of-bounds ranges are rejected.

To generate and listen to one chapter before committing to the whole book, inspect the chapter list and save a chapter checkpoint:

```sh
.venv-renderer/bin/audiobook-render plan runs/book-01
.venv-renderer/bin/audiobook-render render runs/book-01 \
  --chapters 3 --chapters-only
```

Advisory notes do not require an override. Unassigned spoken passages still block the selected chapter. `--chapters-only` saves lossless chapter audio, its encoded playback copy, timing, and provenance beneath `<book-directory>/render-work/chapters/`; it does not publish a final audiobook directory. The CLI prints each playable file's path. Local completed files use the manifest number: `render-work/chapters/<cache-key>/chapter-0039.flac` (the lossless chapter; `chapter.json` names it `chapter-0039.wav`, see [Kept audio is lossless FLAC](#kept-audio-is-lossless-flac); books rendered before that keep `chapter-0039.wav`), `chapter-0039.m4a` (or `.mp3`), and `chapter.json`. `render-work/chapter-links/chapter-39.m4a` is a relative symlink to that encoded chapter, created when it completes and restored on resume, so completed audio is easy to find during an in-progress render. `render-work/completed-chapters/chapter-NNNN.json` records the current checkpoint for each manifest chapter, pointing to its files under `render-work/chapters/<cache-key>/`. Older caches that store generic `chapter.wav` and `chapter.m4a` remain readable; their `chapter.json` metadata records filenames when present. A completed chapter remains available if a later chapter fails.

Rendering, assembly, encoding, and cache reuse show the chapter's position in the full book, such as `chapter 6 / 35`, even when rendering only selected chapters. During speech generation, a live terminal line shows completed blocks, elapsed time, and the active block's generation or verification phase. A block is usually a paragraph; standalone passages and split paragraphs count separately. The count advances only when the entire block succeeds, including verification and any simultaneous speakers.

After at least three newly generated blocks, the display estimates the remaining speech work in that chapter from the text length and observed speed. This is an approximate estimate for generation and verification, excluding chapter encoding and final packaging. Reused clips do not inflate the measured generation speed. Drama 3 does not expose a within-paragraph percentage, so active generation shows elapsed time rather than an invented percentage. Redirected output receives periodic plain-text updates instead of terminal control sequences. Each finished chapter also reports its elapsed time, total run time, and the number of selected chapters ready. These display changes do not alter audio cache keys or regenerate accepted speech.

To generate the remaining chapters and package later:

```sh
.venv-renderer/bin/audiobook-render render runs/book-01 --chapters-only
.venv-renderer/bin/audiobook-render assemble runs/book-01 \
  --epub /path/to/book-one.epub
```

`assemble` uses the verified chapter checkpoints and never loads a TTS model. Ordinary assembly needs no transcription model; changing censorship can require a separate alignment pass using the configured Whisper model. Missing, changed, or incompatible checkpoints stop assembly with an explanation; use `render` to prepare them. Changing only chapter format or bitrate can re-encode from saved lossless PCM during assembly. Censorship changes rebuild from saved original clips, without synthesizing speech. Changes to text, voices, pauses, or loudness require `render` again; unchanged accepted speech is reused. Verification-policy changes alone do not invalidate accepted clips or chapters. Use the same `--work-dir` if it was overridden. To package a subset, pass the same `--chapters` selection to `assemble` and choose a separate `-o` directory, preserving the final output location for the full book. Assembly can update the same book's existing output using the same replacement checks as rendering.

Chapter-title-only repairs also reuse saved lossless PCM. First run `just analyze "path/to/original.epub" -o runs/book-01` to refresh the bundle's TOC labels, then `just assemble runs/book-01 --epub "path/to/original.epub"`. The title refresh performs no model requests when source text and chapter layout are unchanged; assembly checks the old-title checkpoint against every other current audio input before retagging the chapter and M4B. Accepted speech, verification evidence, PCM samples, and synchronization stay unchanged. See [repairing saved chapter names](ANALYSIS.md#repairing-saved-chapter-names) for series and performance-plan details. `refresh-metadata` refreshes artwork and book tags; it does not replace this chapter-title repair.

All pauses added during assembly are controlled in the global config's existing `[render]` table. Each value is an integer from 0 through 5000 milliseconds:

| Setting | Default | Where it applies |
| --- | ---: | --- |
| `sentence_pause_ms` | 500 | After `.`, `!`, `?`, or an ellipsis, between clips and inside a take |
| `clause_pause_ms` | 250 | After a comma, semicolon, colon, or dash, between clips and inside a take |
| `paragraph_pause_ms` | 700 | At a source paragraph boundary, including the chapter's final paragraph |
| `blank_pause_ms` | 1200 | Before a paragraph marked as following a blank line (a scene break) |
| `bracket_pause_ms` | 200 | On entering or leaving bracketed text |

The defaults follow measured human audiobook reading. In LibriTTS, which is cut from LibriVox audiobooks, pauses between sentences average 370 ms, with 1 s at the 90th percentile ([Owoicho et al., Speech Prosody 2024](https://www.isca-archive.org/speechprosody_2024/owoicho24_speechprosody.pdf)). A professional narrator's audiobook reading showed comma, sentence and paragraph pauses rising in that order, with paragraph-final pauses about 1.6 times sentence-final ones ([Bailly et al., Interspeech 2012](https://www.isca-archive.org/interspeech_2012/bailly12_interspeech.pdf)). A storytelling-speech corpus puts long pauses at about 410–450 ms, and listeners do not hear anything under 50 ms as a pause ([Analysis and Modeling Pauses for Synthesis of Storytelling Speech, 2015](https://parakrant.github.io/pdf/pubs/IC3_2015.pdf)). The defaults sit at the unhurried end of that range, since an average of 370 ms sounded rushed in listening: a sentence pause of 500 ms, a clause pause half that, a paragraph pause of 700 ms, and a scene break of 1200 ms.

Pause normalization is enabled by default with `[render] normalize_pauses = true`. During chapter assembly it conservatively removes excess silence at the outer edges of each clip, retaining protective margins around speech. The settings are total gap targets: retained quiet tails and heads count toward the pause, and overlapping cues use the largest target. For example, a sentence ending a paragraph uses the larger of the sentence and paragraph settings. Zero disables a particular added pause; another applicable cue or protected audio may still leave a gap. Protected audio is never removed just to meet a target.

With `normalize_take_pauses = true` (the default, effective only while `normalize_pauses` is on), the same sentence and clause targets also apply inside each take. A long take, a paragraph performance, or a sentence-split recovery would otherwise keep the engine's own pauses: often well over a second at one sentence and almost none at a recovery join. Assembly measures the quiet stretch actually present between the two words at each interior sentence or clause boundary, using the same quiet definition as clip edges, and cuts or extends its middle so the whole stretch equals the target. Who made the silence does not matter, and the result is a total, never the engine's pause plus an added one. Boundaries are located from the take's verification word timings, matched word for word on both sides of the punctuation. Takes accepted without primary word timings are first recognized once more, audio only, by the local MLX Whisper verifier during `render` (cached under `render-work/pause-word-timings/`); only words that match the passage are used. Any other quiet stretch inside a take is only ever shortened: a stall in the middle of a phrase is capped at the clause target, or at the longer target when a take has no timings. Requested pauses (`[pause]` performance controls), simultaneous speech, and censored clips keep the engine's timing. Accepted WAVs never change; each edited clip's timing records its `pause_edits` in original-WAV frames, so `source_audio_start_frame`/`source_audio_end_frame` still trace it. Set `normalize_take_pauses = false` to leave pauses inside takes as generated. Role and voice changes use separate clips. There is no separate added speaker-change pause, and chapter concatenation adds no extra silence of its own. Set `normalize_pauses = false` to retain original clip padding and disable sentence/clause targets; paragraph, blank-line, and bracket pauses are then added on top of the original padding.

Because assembly shortens every such stall, a take rejected only by the `max_internal_silence_seconds` gate is not remade at once. Stall recovery (`renderer/stall_recovery.py`) verifies it again with that limit raised to twice the setting (5 seconds by default): recognition, word comparison and judges run as for any take, and every other acoustic gate is unchanged. A take that passes is used, and its report keeps the stalled report under `silence_stall_recovery`; one that fails now fails on its words, which heard-variant and skipped-word checks can still resolve; a longer pause stays a stall (`decision: "too_long"`). Recovery runs only while assembly will shorten the pause: `normalize_pauses` and `normalize_take_pauses` on, a nonzero sentence or clause target, censorship off, no simultaneous lane, and no requested interior pause. A recovered take never becomes chapter history, so later takes never hear its long silence. On resume, saved stall-only takes that recovery never assessed are rechecked on their audio before new synthesis. Verification still does not judge pronunciation, so a recovered take can sound wrong in the same ways an ordinary accepted take can.

Changing pause settings rebuilds chapter audio from the accepted clips; it does not regenerate speech or modify voice references. Rerun `render BOOK_DIRECTORY --chapters N --chapters-only` using the same work directory, then listen to the newly printed chapter file. Use `assemble` afterward to package it, choosing a new output directory if a finished audiobook already exists. Timing records include `source_audio_start_frame` and `source_audio_end_frame` so each assembled clip can be traced to the retained range of its original accepted WAV. Source text ranges remain unchanged.

By default, completed chapters receive two-pass loudness normalization to -20 LUFS with a -1.5 dB true-peak ceiling. Normalization preserves the PCM frame count and timing data, and operates on the assembled chapter rather than flattening every short utterance to the same level. Voice references and verified speech clips remain unchanged. These controls are separate from casting playback: use `[render] normalize_loudness`, `target_lufs`, and `true_peak_db` for the audiobook. Empty chapters and chapters shorter than the loudness measurement window are explicitly recorded as skipped for normalization.

`render` also creates missing voices if no casting edits or audition candidates exist. It prepares any missing character audition scripts before loading speech models, using the global analysis provider when creative text is needed. Use `cast` first to choose your voices before a long run. Without a worker pool, speech generation is sequential on the configured local model. With a pool, independent requests run across available workers; each local model and the coordinator's Whisper verifier remain serialized. Configured HTTP workers receive the requested passages and selected reference recordings.

## Changing voices and equipment effects

Analysis can mark two kinds of voice change (see [FORMAT.md](FORMAT.md#changing-voices)); neither needs a casting step of its own.

- **A derived voice** (the same character at another age, or changed by a lasting condition) has `derived_from` its main voice. It is never designed afresh and never aliased to the main voice: the book engine **clones** it from the main voice's selected reference recording, speaking the sample script with the change as its direction ("Boyish, mid-high, clear, earnest, quick"), and the take is verified like any reference. `render` creates the main voice first if needed, then the clone; in the casting menu the clone is listed as `Name (age variant, cloned from Name's voice: …)`, and bulk casting does main voices before their clones. In the menu, `E` on a clone edits its **clone direction** (not a design prompt), and the next `G` clones with it; refreshing descriptions leaves clone directions alone. A clone without its own audition script uses its main voice's. If the main voice is later re-cast, the old clone stays in use, the menu marks it "cloned from an older recording of its main voice; G re-clones", and `render` says the same.
- **Young characters and hosted design.** Hosted Gemini voice design refuses a minor's voice and reads some wording as sexual. Every prompt sent to Gemini passes through `voice_prompts.designer_prompt`: minor-age wording becomes sound while keeping its gender ("a young boy's voice" → "a light, bright male voice", "teen boy" → "male", "schoolgirl" → "female", "boyish" → "light, bright", ungendered "child" → "character"), ages under 18 and sexual-reading words are removed ("adult performer" or "adult actor" → "voice actor"), adult ages and acoustic qualities are kept, and no prefix is added. A retired prefix on an older saved prompt is kept verbatim so that request still resumes. The voice's `creation.requested_description` records exactly what Gemini received, and your casting edit keeps your own words. Clone directions use the same wording, and young versions of a character are cloned locally, never designed by Gemini.
- **An equipment effect** (`helmet`, `radio`, `vocoder`, `pa`, `phone`, `underwater`) is applied with FFmpeg to the line's accepted audio while its chapter is composed (`renderer/voice_effects.py`). Verification, censorship, silence trimming and chapter continuity use the dry take, the effected copy keeps the same length, and effected lines never join a paragraph performance. Only chapters containing an effect carry its recipe in their identity; changing an effect re-composes those chapters from cached speech.

## Optional audio censorship

Censorship is off by default. When enabled, the renderer generates and verifies the original words, then silences most of each selected word while composing chapter audio. It adds no replacement words or bleeps and removes no audio frames. Dialogue timing, source synchronization, saved reference voices, and original accepted clips remain unchanged. Simultaneous speakers are masked independently before their recordings are mixed.

Use one global table in your existing configuration file:

```toml
[censorship]
enabled = false
preserve_ms = 45
max_preserved_fraction = 0.2
fade_ms = 5
padding_ms = 15
# Optional replacement for the complete built-in word list:
# words = ["fuck", "fucks", "fucked", "fucking", "shit", "shits", "shitting", "bullshit"]
```

The built-in list contains explicit `fuck` and `shit` forms, including common inflections and compounds such as `motherfucker`, `bullshit`, and `shithead`. It is a limited list, not a general detector of offensive language. Supplying `words` replaces that entire list; spelling variants must be listed explicitly. Matching ignores case and joins internal apostrophes or hyphens, but requires a complete lexical token: `mother-fucker` matches `motherfucker`, while an innocent word containing similar letters does not match. A larger hyphenated compound such as `shit-eating` needs its own list entry; listing `shit` alone does not match it. It does not guess vulgarity from fuzzy spelling. Selection uses the original script and its approved source-to-spoken mapping, so a pronunciation respelling cannot silently hide an originally configured word. Ambiguous mappings stop instead of muting unrelated dialogue.

`preserve_ms` is the maximum opening duration, also limited by `max_preserved_fraction` of the timed word. The preserved portion includes its fade to silence, so the fully audible opening is slightly shorter. Defaults retain at most 45 milliseconds or 20% of the word. Very short words can be muted completely. `padding_ms` extends the silence after the word only into the available gap, and the return fade stays outside the timed word. It never intentionally extends into the following timed word. Allowed ranges are 0–250 milliseconds for `preserve_ms`, 0–0.5 for `max_preserved_fraction`, 0–20 milliseconds for `fade_ms`, and 0–100 milliseconds for `padding_ms`. Set `preserve_ms = 0` for whole-word muting.

Both `render` and `assemble` accept `--censor` and `--no-censor` as per-run overrides. For an already rendered chapter:

```sh
# Rebuild one playable chapter using saved verified speech, without TTS.
audiobook-render assemble BOOK --censor --chapters 13 --chapters-only

# Restore the uncensored chapter from those same original clips.
audiobook-render assemble BOOK --no-censor --chapters 13 --chapters-only

# Rebuild and package all saved chapters with censorship enabled.
audiobook-render assemble BOOK --censor
```

Numbers select manifest entries, including front matter; use `plan BOOK` to find the desired chapter. The command prints the encoded chapter path. Reuse the original work directory: changed censorship needs the retained individual clips, including every original simultaneous speaker lane. Deleting the render cache removes this ability. `--chapters-only` updates chapter checkpoints without replacing a published full book. To keep separate published editions, use different `-o` output directories when packaging them.

Reliable saved word timestamps are reused. If they are missing, ambiguous, overlapping, shorter than the supported 20-millisecond timing resolution, or cannot locate the target safely, a separate alignment pass reads the existing audio and known text. The current fallback uses local MLX Whisper and caches its results; it does not regenerate or re-accept speech. Before forcing the known text onto audio, independent ASR must confirm complete lexical coverage, allowing case, punctuation, and word-spacing differences. If the saved transcript cannot establish that, one audio-only recognition pass is saved and checked. Remaining missing or different words stop censorship instead of forcing an absent word onto neighboring speech. This can conservatively stop an accepted pronunciation variant without changing its original speech acceptance.

Alignment may load the configured Whisper model even during `assemble`. If it cannot establish usable boundaries, the operation stops with the original audio intact. Rerunning can resume completed alignment work and reuse saved recognition evidence. Timing remains an estimate of word boundaries, not a phoneme guarantee; listen to the result when choosing how much opening sound to preserve.

Censorship affects chapter composition and publication identities, not speech-generation or voice identities. `sync.json` records the policy, original audio checksum, matched text ranges, alignment method, and original/chapter/book frame intervals. Original verification reports still describe the uncensored accepted speech.

Updated remote chapter workers can apply the same policy while assembling on the server. Changing the policy submits new chapter work, which can reuse the server's retained accepted speech. A client holding only downloaded chapter audio cannot change censorship locally: `assemble` packages matching downloaded chapters, and a changed policy requires rendering on the server again or access to the original individual clips. The MLX Whisper fallback is currently local to Apple Silicon; other workers can use reliable existing word timestamps but stop if additional unsupported alignment is needed.

## Audio storage and cleanup

### Kept audio is lossless FLAC

The renderer keeps speech takes (accepted and rejected, including selections, recovery joins and parts, and clip-repair parts) under `render-work/clips/`, simultaneous-speaker mixes under `render-work/mixes/`, lossless chapter audio under `render-work/chapters/<cache-key>/`, and loudness-levelled listening previews under `render-work/voice-previews/` as FLAC, not uncompressed WAV. They take roughly a third of the space: measured on one book, about 38% of the WAV size for a take and 34% for a chapter.

Each FLAC holds the WAV's 16-bit samples together with the WAV's own header bytes (in a FLAC `APPLICATION` metadata block), so decoding it restores the original WAV byte for byte. Records keep naming the WAV (`audio_file: "…-take.wav"`, `wav_file: "chapter-0042.wav"`). On disk that name is either the WAV itself or `<same name>.flac` beside it. Every checksum a record or identity binds (`audio_sha256`, continuity `raw_audio_sha256`/`accepted_audio_sha256`, chapter `pcm_sha256` and `clips[].audio_sha256`, speech-cache entries, pause and censorship alignment bindings) is still the SHA-256 of the WAV bytes. Storing audio as FLAC therefore changes no request, cache, chapter or voice identity, and nothing is regenerated. A file is written as a WAV first, its record is saved, and the WAV is replaced only after the finished FLAC has been decoded back to the identical bytes; if that fails, the WAV simply stays. While both forms exist (an interruption between the two steps), the WAV is used. Verification, recognition and alignment receive an exact temporary WAV copy restored from the FLAC (deleted afterwards, in the system temporary directory), so recognizers see the same input as before. FFmpeg steps and audio players read the FLAC directly, since it decodes to the same samples. The final M4B is encoded from these lossless chapters. A book whose chapters are partly WAV and partly FLAC joins exact restored WAV copies, because FFmpeg's concat demuxer needs one codec throughout.

Two kinds of audio stay WAV. The engines' raw output is 32-bit float, which FLAC cannot hold exactly; it is removed after acceptance unless `keep_intermediate_audio` is on. Casting takes (`render-work/casting-candidates/`, `voice-candidates/`) and voice reference recordings in voice stores also stay WAV, because engines, engine plugins and remote workers read a voice's reference by its path; together they are about 1% of a book's audio. Pronunciation previews were already FLAC.

Books rendered by earlier versions keep working unchanged: their WAVs are read, resumed and assembled as before, and a book can hold both forms at once. To convert an existing book's WAVs explicitly:

```sh
just convert-flac runs/book-01 --dry-run
just convert-flac runs/book-01
```

`audiobook-render convert-flac BOOK...` converts the takes, mixes, chapters and previews of each named book (`--only clips,chapters,mixes,previews` picks groups). Each file is converted on its own, verified byte for byte before its WAV is removed, and a rerun resumes an interrupted conversion. Records are never rewritten. WAVs FLAC cannot hold exactly are listed and kept, and a file that fails verification keeps its WAV and is reported (exit 2). `--dry-run` counts the WAV files and changes nothing. Otherwise the command asks before converting; `--yes` converts without asking, and without a terminal nothing is converted unless `--yes` is given. `--json` prints the full report. It takes the same locks as `prune`, so it refuses while a render, assembly, casting session or cleanup uses the book. Hidden staging files, the published output, casting takes and voice stores are never touched. Do not convert a book that a render started with an older version of the renderer will still reach (for example a later book in a running multi-book queue): that older process does not know the FLAC form, treats the takes as missing and generates them again.

### Raw speech and cleanup

By default, `[render] keep_intermediate_audio = false` removes redundant raw speech WAVs after the normalized take is accepted and its checksum is verified. The accepted take and verification records remain available for resume, changing pauses, and rebuilding chapters without synthesizing speech again. Raw files from unsuccessful or pending requests remain available for recovery. Set the option to `true` to keep raw engine output for debugging. This setting does not affect voice creation or any voice samples.

For files created by earlier runs, preview and then run cleanup for one book:

```sh
.venv-renderer/bin/audiobook-render cleanup runs/book-01 --dry-run
.venv-renderer/bin/audiobook-render cleanup runs/book-01
```

The default cleanup removes redundant raw speech files, complete chapter cache versions that are no longer referenced by any saved chapter checkpoint, and abandoned chapter-assembly staging with a recognized writer layout. Obsolete chapter versions include their WAV and encoded M4A/MP3 files; current checkpoint-referenced chapter files stay intact. Interrupted composition, loudness-normalization, and encoding files are removable only inside the renderer's `.chapter-…/ready` staging layout; unknown layouts and invalid staging records are retained. Cleanup preserves accepted speech, normalized rejected takes, current chapter checkpoints, saved voices, and finished audiobook outputs. It loads no TTS or transcription model and refuses to run alongside a render or assembly for the same book or work directory. Use the same `--work-dir` as the render if you overrode it. Dry runs list the exact files and bytes eligible for removal. The report also includes `estimated_bytes_freed`: allocated file space counted once per inode, excluding files with hard links retained elsewhere. This estimate can differ from the sum of removed file sizes and is not an exact prediction for every filesystem's shared storage.

To remove the generated speech and chapter caches as well, use the explicit discard option:

```sh
.venv-renderer/bin/audiobook-render cleanup runs/book-01 --discard-render-cache --dry-run
.venv-renderer/bin/audiobook-render cleanup runs/book-01 --discard-render-cache
```

This deletes the managed `clips`, `mixes`, `chapters`, and `completed-chapters` stores under that book's render work directory, including the playable checkpoint files created by `--chapters-only`. **Speech resumability is lost:** rendering again will generate new speech using the saved voices. Published M4A/MP3 chapters and the M4B in the output directory remain intact.

### Removing what a book no longer needs (`prune`, `just cleanup`)

`cleanup` keeps everything a render could resume from. To remove superseded voices, chapter versions and takes as well, use `prune` (or `just cleanup`), on one or several books:

```sh
just cleanup runs/book-01 runs/book-02 --dry-run
just cleanup runs/book-01                     # shows the summary, then asks before deleting
just cleanup runs/book-01 --only voices,clips --yes
```

It works in three groups, all on by default; `--only` takes a comma-separated subset:

- **voices** — each voice keeps its current choice. This book's other recordings of that voice (older revisions and the first recording a later choice replaced), the casting takes that made them under `render-work/casting-candidates`, and loudness previews of recordings that are no longer current are deleted. A voice whose recast is pending or failed, or that has no choice in this book, keeps everything. Earlier books' voice stores are never touched, so a continued book keeps every voice it inherits; a casting take whose hosted (Gemini) voice has not been released yet is kept.
- **chapters** — each chapter keeps its newest complete version, even when it is out of date, and the version its checkpoint names (normally the same one); older versions and abandoned assembly staging are deleted. The published book in the output directory, and every `.m4b`, are never touched.
- **clips** — a take folder stays when the next render reuses or resumes it (decided by the speech cache exactly as `cache-report` replays the current plan), when a kept chapter was built from it, or when you judged one of its takes or a failed chapter names it; recovery parts a kept take names stay too. Inside a kept, accepted take the rejected attempts' audio and continuity codes are deleted while its records stay; pending (still failing) takes stay whole. Every other take folder, its speech-index entries and pause timings, and old run plans are deleted. Cleaning clips needs this book's own render plan. With a remote chapter service, a non-built-in engine, or a plan with problems (most often lines with no speaker yet: choose them with `epub-tts-narrator resolve-speakers BOOK` or `just review-ui BOOK`), cleanup skips the clips group, says why in the summary (and in `skipped` with `--json`), and still cleans voices and chapters; `--only clips` refuses instead.

`--dry-run` prints the per-group counts and sizes and deletes nothing; otherwise the same summary is shown and you are asked before anything is deleted. `--yes` deletes without asking; without a terminal and without `--yes`, nothing is deleted. `--json` prints the full report with every file. Anything `prune` does not recognize, and anything holding a link, lock or lease, is kept and listed. It takes the book's render, book-audio and voice-store locks for the whole run, so it refuses while a render, assembly, casting session or cleanup uses the book, and any of those started meanwhile refuses until it finishes. Lock and lease files are never deleted. Run `cache-report` afterwards if you want to confirm the next render reuses what it did before.

Both `cleanup` modes always preserve voice samples, cloning references, casting candidates (including failed attempts), audition previews, and ancestor books. They do not traverse or delete the voice stores. Remaining accepted speech and current lossless chapter audio (FLAC, or WAV in books not yet converted) are intentional: they avoid lossy re-encoding and retain exact synchronization when reassembling. Both commands recognize a kept file in either form. Voice samples are not compressed; their identity includes the original WAV bytes, and engines read them by path.

## Verification and recovery

### Broad clip-local recovery and best-effort exhaustion

Repeated Drama 3 truncation of a grouped paragraph triggers a retry as its original source passages, preserving each passage's assigned voice and pronunciation/performance controls. A model-truncation error returned through HTTP still takes this path even when the transport marks the attempt retryable on resume. Genuine connection failures and timeouts do not count as model truncation. Individual passages retain the existing clip-local recovery and verification paths.

When analysis-model recovery is enabled, repeated failures can trigger a whole-clip rendering experiment. The model may change phonetic TTS wording, transient delivery, clip-local trait conditioning, and splitting or regrouping together. Two independent blind reviews run in parallel, and both must approve before generation. These are synthesis-only changes: source text, assignments, normalization records, global voice profiles, and the character's bound reference remain unchanged. Every generated part uses the existing shared attempt allowance and sibling reservations.

An approved repair part that hits Drama 3's unfinished-text failure retries with a fresh seed, up to `[render] max_attempts` starts per part in that invocation and within the existing shared recovery allowance. Each start reserves capacity for unfinished later parts and checkpoints its seed sequence before synthesis. Timeouts, connection failures, and invalid inputs do not enter this model-truncation retry loop. Completed parts and the approved proposal/reviews survive failure or interruption; rerunning `render` resumes them without asking for another plan, including recovery children whose new-resolution allowance is disabled. That allowance still prevents new proposals and reviews. Exhausted retries still require usable audio and the existing verification or best-effort selection policy.

The repaired performance first receives ordinary verification. When ordinary strategies are exhausted, the runtime nominates retained, hash-bound, physically usable takes covering the complete logical clip. It does not exclude them using strict word equality, protected-word/phonetic rules, or an earlier failed verification result. Approximate syllables, word-count differences, omission/addition clusters, numbers and negation, ASR uncertainty, and acoustic advisories are supplied to an LLM chooser as evidence. Short and expressive utterances are not rejected by a rigid syllable heuristic.

The chooser must select a supplied candidate, and that selection is decisive: the same lexical or acoustic-advisory checks do not veto it afterward. This is explicitly best-effort acceptance and can retain imperfect or different content; it does not certify that every original word was spoken. `verification.json` marks it with `text_confirmation_method: "llm_exhaustion_selection"`, records the accepting policy and `ordinary_verification_passed`, and preserves `original_failed_report`. Original failed attempts remain unchanged. Corrupt, empty, nonfinite, effectively silent, unrelated-span, and unrelated-reference audio cannot qualify; no audio is fabricated when none is usable. When the intended passage contains deliberately elongated spelling, the chooser payload carries a `stylized_spelling` advisory noting that a fluent collapsed reading may be the intended realization; the chooser prompt and policy identity are unchanged, and a pending choice saved before the advisory existed resumes without being asked again.

A history echo is different: a take that speaks an earlier line of the chapter's history (see "History echo") is not a rendition of this passage at all. Echo takes are never offered to the chooser, and a clip repair for a line that echoed generates its parts without chapter history. If only echoes remain, nothing is published and the line fails with a message naming the echo. A saved best-effort choice that was an echo is not reused. Its `accepted.json`, `exhaustion-selection.json`, `clip-repair.json` and `clip-repair-parts` are renamed with a `history-echo-…` suffix (never deleted), and a `history-echo-….json` record names the evidence. The line, and a completed chapter containing it, then render again. A pending choice that offered an echo is retired the same way and asked again without it.

`clip-repair.json` checkpoints the proposal, both reviews, and each generated part. `exhaustion-selection.json` checkpoints chooser evidence and its response. Rerunning resumes completed work without new synthesis or semantic reranking. Structural chooser-format corrections are bounded and retain all previous responses; a valid choice is not reconsidered. Provider outages leave pending work available for retry. Selected audio and audits remain hash-bound through cache reuse, child/join provenance, and model-free assembly. Existing verified recordings and historical recovery policies retain their original authority.

A sentence-recovery assembly that stopped before producing a full waveform resumes its children on rerun, including an approved child repair that previously failed synthesis. Completed child audio stays unchanged. A completed joined waveform that definitively failed verification remains rejected; it is not repeatedly concatenated to evade verification.

Before synthesis, the renderer trims surrounding whitespace and quotation marks (straight, curly, and guillemets) from each speech string. Interior quotation marks, apostrophes, and sentence punctuation remain intact. This prevents a closing quote at the runtime's byte limit from becoming a separate generation target that can repeat the preceding passage. Dialogue turns are trimmed before adding speaker or delivery controls; their word boundaries are preserved when joined. Source scripts, synchronization ranges, saved cloning references, and verification text remain unchanged. Previously verified clips and chapters remain reusable; newly generated takes record the actual `synthesis_text` in their engine metadata.

Parentheses wrapping an entire target, or left dangling at its edges by a speaker split, are also removed from the synthesis input. For example, `(Lie,` becomes `Lie,`, and `I thought fiercely).` becomes `I thought fiercely.`; the same words are still required by verification. This avoids reference-sample repetition triggered by an incomplete parenthetical target. Balanced parentheticals inside a sentence remain intact, and a thought and its narration remain separate performances.

Invisible Unicode format characters (category Cf: zero-width spaces, soft hyphens, joiners, byte-order and direction marks) are removed from the synthesis input wherever they occur; a game-UI bullet written `●` plus a zero-width space is sent as `●`. They are typesetting, never speech, and `serve` workers reject requests that contain them. Source text and verification are unchanged, and verification already treats them as word boundaries. A take recorded with one has a different request identity and is regenerated.

Each chunk is normalized to mono PCM16 at 44.1 kHz. Before acceptance, the renderer checks duration, overall level, excessive silence, long internal silence, and clipping. Local Whisper then transcribes the audio independently: it receives no expected transcript as a prompt. The comparison records added, missing, and substituted words, with conservative punctuation and number normalization.

Correctly suffixed ordinal digits below one million match the same spoken ordinal without JEV: `seventh` / `7th`, `fifth` / `5th`, and `twenty-first` / `21st` are equivalent. Changed values, cardinal/ordinal confusion (`seven` / `7th`), malformed suffixes, leading zeros, signed or decimal forms, and missing surrounding words do not qualify. This changes comparison only, not the book or synthesis input.

Complete written-content equality passes directly before phonetic heuristics, JEV, or a second recognizer. The shared comparator case-folds and applies the existing lexical/number normalization, ignores word boundaries and punctuation including apostrophes, and requires the entire remaining character sequence to match in order. Thus `ship woman's` / `shipwoman's` / `shipwomans`, `sun, flower` / `sunflower`, and arbitrarily split compounds need no model judgment; there is no two-letter minimum or three-piece limit. This is a whole-transcript equality rule, not an allowance for missing, extra, changed, or reordered characters. Meaningful numeric syntax, signs, and currency content remain subject to the existing lexical normalization: `12 34` / `1234`, changed decimals, and dropped minus signs or currency units are not accepted by stripping their syntax. Original transcripts and word-error metrics remain intact; `word_boundary_comparison` records the exact canonical alignment with method `written_content_v1`. Acoustic checks still apply.

Next, a local [Double Metaphone](https://pypi.org/project/Metaphone/) check compares full phonetic codes. It accepts spelling and pronunciation differences such as `Derrins` / `Darin's`, `Qarm` / `Khorm`, and `Vandire` / `Van Dyer` without calling JEV. Unlike four-character Soundex, these keys are not truncated. The ordered alignment must consume every token, matching one word to one word or one word to two or three whitespace-separated words. Multiword matches concatenate each word's full code, retain punctuation boundaries, and exclude function words and unchanged words with extra neighbors. Numbers, ordinals, and negations must remain unchanged. Empty codes cannot match. The original transcript and word-error metrics remain in the report, and `phonetic_comparison` records the keys and aligned spans. Acceptance is labeled `double_metaphone` or `secondary_double_metaphone`.

The same alignment also recognizes the explicit colloquial equivalents `what'cha`, `whatcha`, and `whatchu` ↔ `what you`, and `ya` ↔ `you`. These can combine with phonetic name matches elsewhere, such as `Dalley` / `dally`, without a JEV call. The report records a `colloquial` operation with the original token spans and the expansion. Expanded multiword forms must be separated by whitespace; the rule cannot supply implied words such as `are` or `did`, omit `you`, or excuse missing or added speech elsewhere. Source text, synthesis text, and raw word-error metrics are preserved.

The indefinite article forms `a` and `an` are grammatical allomorphs and can match one-to-one in either direction, including alongside ordinary phonetic spelling variants such as `armour` / `armor`. Each `article_allomorph` alignment consumes exactly one article on each side; it cannot omit or add an article or excuse any other unmatched speech.

Spelled-letter sequences also match locally: `See-see?` / `CC?`, `see see` / `C C`, and `bee bee` / `BB` retain the same letter names in the same order. The matcher recognizes one through six letters in bounded initialisms and explicit English letter names, including hyphenated repetitions and dotted initials. A singleton requires an unambiguous uppercase initialism on one side, so `Q-Card` / `cue Card` and `Q` / `cue` match, while `A`, `I`, and lowercase singletons remain lexical. Every repetition is preserved: `see see` cannot match `C` or `CCC`. Ordinary lowercase words are not freely expanded into letters. The `letter_names` alignment records the consumed tokens and letter sequence; original text and error metrics remain intact. This exact representation check needs no JEV call, and missing or extra surrounding speech still fails.

This prioritizes missing or extra speech over pronunciation fidelity. Phonetic codes can collide: `Qarm`, `Quorum`, and `Karim` share a code, for example. The check is a pronunciation heuristic, not proof of word identity or syllable count. It never removes unmatched tokens to make a passage pass. Acoustic checks still apply. The small `Metaphone==0.6` package is included in `renderer.requirements.txt` and the CUDA verification extra; existing environments can install it with `uv pip install --python .venv-renderer/bin/python 'Metaphone==0.6'`. Verifier startup reports a missing dependency before synthesis. Analysis does not require this package.

Chapter speech is accepted locally when its transcript is close to the passage (`renderer/close_match.py`, `[quality] max_close_mismatch_fraction`, default 0.25 in the renderer configuration; 0 restores strict matching). Most nonexact transcripts are the recognizer mishearing accented speech or invented names, not the engine saying something else. A word-level alignment counts a word as present when it matches exactly or as a near sound or spelling (Double Metaphone or spelling similarity), including a name heard across new word breaks (`Killashandra Ree` / `Kilachandrari`, `Kill a Chandra`). The take is accepted when no passage word is missing, no more than two consecutive words are wrong or added, at most that share of the passage differs, and every number and negation is exact (a different or added number, or any changed negation, rejects it). Vocal sounds and symbols without a fixed spoken form count either way. Passages under two words, where one word is the whole line, keep the stricter checks, and so do voice references, whose transcript conditions cloning. A history echo (speech from another passage) is a long run of added words and is never close. Reports record `text_confirmation_method: "close_match"` (or `secondary_close_match`) with every difference under `close_match`. Calibrated on one book's listener verdicts: 130 of 144 word-rejected takes judged good become close, 4 of 48 judged bad do (three disliked pronunciations, which verification never judges, and `©` read as a word), and none of its 114 history echoes. Changing the setting changes the validation key, so takes rejected for their words under the stricter rule are rechecked on resume rather than remade.

A word with a pronunciation you chose (`pronounce`) is also checked for its shape (`renderer/name_shape.py`). Drama 3 does not always follow the phonemes: a name read "muh-REN-druh" can come out "MARN-eh-druh". The recognizer hears the difference ("Marendra" against "Marnedra"), but the word checks accept both because Double Metaphone ignores vowels. So the heard word in the name's place is compared with the chosen reading by how many consonants sit before, between and after its vowels. A different spelling keeps that shape (`Marendra`, `Morendra`, `Marindra`); a consonant moved across a vowel (`Marnedra`, `Mornedra`) fails the take as `pronunciation_mismatch`, and another take is made. Only a heard word with the reading's number of vowel groups and consonants is judged; other consonants, a dropped syllable or a different word are left to the ordinary checks. A possessive is compared by the name alone when it was heard with its own `'s` (`Marnedra's`), and as a whole word, possessive sound included, when the recognizer wrote it as a name ending in s (`Marendras` for "Marendra's", which earlier versions let through). Only chapter speech with a chosen CMU reading is checked, so only those passages' validation keys change. Each render also checks the takes already in use from their saved transcripts (no recognition runs, only takes accepted since the last check, recorded in `render-work/name-shape-sweep.json`; a new version of the check reads every take again). Takes that fail are retired (renamed, never deleted) and rendered again. A take you chose or marked good, and a best-effort choice, are kept. On one series' renders, 113 of about 7,300 chosen readings failed, nearly all one name's misreadings, including the takes a listener marked bad.

Optionally, each chosen reading is also *heard*, not just spelled (`renderer/name_phonemes.py`). The recognizer's spelling cannot tell "seh-NED-ruh" from "seh-NEE-druh" or "seh-NAY-druh"; a phoneme recognizer can. With `[verification] name_phoneme_model` set to a local copy of `facebook/wav2vec2-lv-60-espeak-cv-ft` (Apache-2.0, about 1.3 GB, runs on Apple's GPU at well under a tenth of a second per name), every passed chapter-speech take with a chosen CMU reading has the name's own audio (from its word timings; with a word either side when that alone does not match, where a match counts only when most of the sounds it matched were heard while the name was said or where no other word was timed, so the neighbouring words never stand in for it ("city of RAY-vuh" once passed for "REE-vuh" on the "-ty of") and a late word timing does not hide it; the whole take when unplaced) transcribed as phonemes and aligned with the reading. The reading is said as chosen when its consonants are there in order and its stressed vowel is of the reading's kind; unstressed vowels may be any vowel (one with secondary stress may reduce to a neutral vowel but not become one from the other side of the mouth: "sih-" or "suh-" for "seh-", never "soo-"), and regular English variation is allowed (voicing partners such as "t" for D, vowels that merge before an R, a swallowed syllable-final L or cluster R, two neighbouring vowels gliding into one, an ER said without its R, a nasal or breath beside a vowel, a consonant clipped at the name's edge). A name the recognizer heard in another language's sounds is uncertain, never a misreading. The rules were set against a listener's verdicts on rejected takes. When tagged takes of a passage misread a chosen name twice, its next takes send the name as written instead (`renderer/spelling_recovery.py`; Drama 3 misreads some names only when tagged: one narrator's "Ce'Nedra replied." was wrong in 5 of 5 tagged takes and right in 5 of 5 plain ones), still judged against the chosen reading; when plain takes have misread as often, tags return. A plain-spelling take records which tags it left out (`spelling_recovery` in its metadata) and stands for the original request. Your ear decides over the model's: when you mark a take good (`t` in the run shell, or `review-ui`) that the check heard saying a name differently, exactly the sounds it heard for that name count as said right in that book from then on; any other way of saying it is judged as before. A take that says the reading differently fails as `pronunciation_mismatch`; its verdicts are kept in the report (`name_phonemes`) and decide which takes may be heard as a reading example in history. The acceptance gate hears every other path's take the same way (resolver, rechecks, clip repairs, the best-effort chooser, tone-filtered copies), so a take saying the name wrong is never published, and the render's sweep of takes already in use retires those that fail (once per version; `name-shape-sweep.json` records it). A take you chose yourself is kept. Verdicts heard from saved takes are appended to `render-work/name-phoneme-verdicts.jsonl`, so an interrupted first sweep (a book's every saved take with a chosen reading, about a tenth of a second each) resumes where it stopped; delete the file to hear them all again. Unset or missing, nothing changes. On the series this was tuned on, the check agreed with a listener on every Ce'Nedra clip they judged and flagged 2–13% of most names' uses, nearly all audibly different vowels.

With JEV enabled, remaining nonexact transcripts must first pass a local spoken-coverage check before JEV judges the pronunciation differences. Without JEV, chapter audio applies the configured `max_wer`, while reference recordings that differ beyond the shared equivalence gates above require independent-recognizer agreement. Neither speech recognizer receives the expected text as a prompt.

Whisper word timestamps are optional synchronization metadata. If they overlap, leave their segment, or are otherwise unusable, the report retains an alignment warning and omits word times while still checking the actual transcript and audio. This does not trigger speech regeneration. Chapter assembly and source synchronization retain accurate chunk boundaries from the audio duration; no word timestamps are invented. Voice samples do not request word timestamps.

An optional Parakeet recognizer supplies a second audio-only transcript. With JEV, the primary transcript is judged first. A second recognizer is used only if that judgment does not pass; a good primary result cannot be vetoed by a bad secondary transcription. Without JEV, a reference transcript that differs beyond the shared equivalence gates is first checked by bounded substitution, compound-boundary, and expressive-consonant rules, and can pass through independent transcript agreement or an exact secondary match to the request. If that does not resolve it, the reference-only sound review below can assess a bounded replacement. An accepted variation is stored with its reference text adjusted to the actual spoken words.

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

This adds about 2.5 GB of model weights. The adapter uses MLX Audio's local Parakeet implementation. [MLX Audio Parakeet documentation](https://github.com/blaizzy/mlx-audio/blob/main/docs/models/stt/parakeet.md)

### Learned heard variants

Recognizers often turn an invented name, or another word they do not know, into ordinary words: `Kvothe` becomes `quote`. When that is the only thing wrong with a take, the renderer records how the word was heard, and once the same spelling is confirmed it stops rejecting it. This is on by default and needs no model or network:

```toml
[verification]
heard_variants = "learn" # "off" keeps ordinary verification only
```

Each take that fails only with `text_mismatch` is compared word by word with its passage. Every remaining difference (one that the spacing and Double Metaphone checks above do not already accept) must be a small block of one to three words heard as one to three other words, at most four blocks and one per five words. A block may not contain numbers, negation, digits or non-word tokens; the passage side may not contain function words (`the`, `and`, `is`, ...), the heard side may not be only function words, and no word may appear on both sides. If any difference fails these rules, the take records nothing. Otherwise each block is saved as a *sighting* of a variant of that word (`kvothe` → `quote`), bound to the take's audio hash, request hash and transcript hash.

A variant becomes trusted in one of three ways:

- **Repeated:** three different takes (three different audio files, from any lines or seeds) were heard the same way, and the variant passes a sound-plausibility check: letter lengths within a factor of two, and Double Metaphone keys at least half similar, or sharing their first sound and at least 0.3 similar. `kvothe` / `quote` passes; `kvothe` / `banana` never does. Rechecking the same saved audio after an interruption is not a new take.
- **Review:** an accepted take whose differences three separate pronunciation reviews approved (`phonetic_primary` or `phonetic_secondary`, see "Optional JEV confirmation") trusts its blocks immediately.
- **Listened:** the variant came from a `pronounce-name` test sentence you accepted.

A failed take whose every remaining difference is a trusted variant is accepted. The take that brings a variant to its third sighting is accepted too. The report keeps the original transcript, word metrics and `previous_reasons: ["text_mismatch"]`, and records `text_confirmation_method: "heard_variant"` with a `heard_variants` audit (policy, recognizer, and each variant's basis, take count and sightings). It is never reported as a verbatim match. Takes that were heard differently once or twice still fail as before, with a `heard_variant_observations` note and a progress line such as `Heard variant noted: kvothe → quote (heard this way in 2 of 3 takes)`.

A learned variant never excuses a missing or extra whole word, a changed number or negation, a reordered word, a difference in any other word, a history echo, or any failed acoustic check (those takes are not `text_mismatch`-only and record nothing). A variant is bound to its exact word and to the primary recognizer's identity without its machine-local model path: changing the recognizer model, package or decoding options starts a fresh set, while JEV, sound-review and recovery settings do not matter.

Variants live in the analysis directory as `heard-variants.json`, beside `name-pronunciations.json`, and are checked against the book's ID on every read. A book continued with `--continue-from` inherits its ancestors' variants read-only, nearest book first: their sightings count toward trust, and new sightings are written only to the book being rendered. Writes hold `.heard-variants.lock` and replace the file atomically, so concurrent chunks and interrupted runs keep every completed sighting. Remote chapter workers have no book directory and use ordinary verification only. Delete `heard-variants.json` to forget what a book learned; existing accepted audio is unaffected, and variants never change synthesis, caches or chapter identities.

### Source typos every take skips

Source text sometimes has a stray or doubled word, such as "when it he eventually got to work". A good reading leaves it out, and ordinary verification never excuses a missing word, so every take used to fail. With JEV enabled, a failed take is accepted when all of these hold:

- removing exactly one source word makes the transcript match directly (exactly, by word spacing, or by complete Double Metaphone sound match, so `labour`/`labor` elsewhere in the passage still counts);
- that word is not a number, quantity or negation;
- at least two different takes of the same request left out that same word, and no take of that request read it (clip-repair and spelling-rewrite takes speak other text, so they count neither way);
- JEV, asked whether the word is a mistake in the source text so that leaving it out gives the intended sentence, answers yes at the configured `threshold`.

The first take that skips the word only notes it (`Left-out source word noted (… when [it] he …)`); the next take that skips it asks JEV. An accepted report keeps the original transcript, metrics and `previous_reasons: ["text_mismatch"]`, and records `text_confirmation_method: "skipped_source_word"` with a `skipped_source_word` audit (word, context, takes, judgment). It is never reported as a verbatim match. If JEV keeps the word, is unsure or unavailable, the take fails as before. The check runs after verification, like learned heard variants, so validation keys and caches are unchanged. Fixing the EPUB and running a book update is still the cleaner fix for a typo you notice.

### Optional JEV confirmation

JEV compares two written representations of speech and judges **sound and coverage only**. Spelling, recognition mistakes, homophones, and minor pronunciation differences can pass even when the written meanings differ. Missing, extra, repeated, or replaced spoken content without a plausible sound match fails. A local soundalike cannot excuse missing speech elsewhere. This applies to voice samples and rendered chapter clips; the acceptance threshold is unchanged.

JEV's score cannot override the local coverage check. The check requires a complete ordered alignment; the only one-sided operations are the bounded vocal sounds, restarts, unspoken symbols and joining hyphens described below. Each word must have an exact or bounded phonetic counterpart; supported word joins and splits can consume multiple words, such as `admitted` / `admit it`. Numbers and negation remain protected. An omitted narrator clause, repeated phrase, unrelated replacement, or paraphrase fails even with a high JEV score. Reports retain the alignment or failure under `speech_coverage`. If two recognizers are used, one complete transcript must pass both coverage and judgment; their fragments or scores cannot be combined to approve a take.

Small sound changes within a complete word, such as `minds` / `mines`, can reach JEV and the pronunciation reviewers even when their full phonetic codes differ. This eligibility rule allows one encoded sound edit in sufficiently long words, preserves both complete word spans, and excludes grammatical words, numbers, negation, and written initialisms. It does not accept audio by itself. Exact compound spacing can coexist with a separate pronunciation difference, so `ship-woman` / `shipwoman` does not prevent review of `minds` / `mines` elsewhere in the passage. Casting samples and audiobook speech share this rule.

A separate review-only rule handles non-numeric one-to-one homophones such as `know` / `no`, `knot` / `not`, and `wee` / `we`. It requires one complete alphabetic word on each side and a matching nonempty full phonetic code; short codes may nominate a candidate but never accept it automatically, because these codes can lose vowel distinctions. Written initialisms, numeric words, contractions containing `n't`, missing or extra words, and nonmatching codes remain excluded from this route. The normal JEV or independently voted pronunciation-review decision is still required; an unavailable or negative judgment cannot be replaced with the candidate's phonetic code. Acoustic gates and the original source text remain unchanged.

Honorific title abbreviations (`Mister` / `Mr.`, `Doctor` / `Dr.`, and similar standard forms), review-nominated colloquial expansions (`kinda` → `kind of`, `gonna` → `going to`, `wanna`, `d'you`, `'cause`, `lemme`, `gimme`, `gotcha`), and the single-letter `I` / `l` transcription confusion follow the same nomination pattern: they make the pair reviewable for JEV or the pronunciation voters without accepting audio by themselves. Digit-only tokens that differ only by leading zeros (`0800` / `800`) compare as canonically equal written content at the same position; changed, reordered, signed, or decimal numeric content remains fully protected.

Coverage versions 15 and 16 widen eligibility for ordinary spoken variation that listeners accepted but the earlier gate stopped before any judge saw it (in one book, 126 of 142 rejected takes a listener marked good never reached a judge). Each rule only nominates; JEV or the pronunciation voters still decide:

- a contraction of any word may be heard spelled out (`else's` / `else has`, `to've` / `to have`, `this'll` / `this will`), and an apostrophe marking one to three left-out letters may be read in full (`'em` / `them`, `g'night` / `good night`);
- different dictionary words sharing a CMU reading (`high` / `hi`, `aloud` / `allowed`; number words stay protected) and words alike once a non-rhotic accent drops an R before a consonant or pause (`flaw` / `floor`, an invented `Killa` / `killer`);
- an uppercase Roman numeral of two or more letters and its cardinal (`VI` / `six`), a degree unit (`35° C` / `35 degrees Celsius`), `/` read as `slash` or passed over, `*` passed over, and a hyphen joining two alphanumeric pieces read as nothing (`23-4-3308`; a spaced `-4` keeps its spoken minus);
- at most two vocal sounds (`oh`, `ah`, `uh`, `um`, `huh`, `ha` and similar) added, left out or changed, or a hyphen-joined restart (`T-Trag`); a written hum (`Mmm`) may not be left out. Sound votes never approve a passage with an added or omitted sound, restart or hyphen; only JEV's whole-passage judgment can;
- a capitalized name no dictionary knows, in one or two source words, heard across new word breaks (`Killashandra Ree` / `Kila Chandra Rhee`, `Kilachandrari`) when the Double Metaphone codes of the joined spellings differ by at most one edit per four sounds; heard negations, numbers and copies of neighbouring source words are never absorbed, nor are spelled single letters;
- the colloquial reductions `gotta`, `oughta`, `hafta`, `outta`, `lotta`, `sorta` and `helluva`.

Lost, added or replaced words still fail before any judge: none of the listener-rejected takes in that book became newly eligible, and no history echo did.

Differences confined to a final `s` or possessive `'s`, such as `weapons` / `weapon` or `body's` / `body`, are eligible for sound review. They are not automatically accepted: JEV or the pronunciation fallback must judge them plausible. This rule retains the complete base word and all surrounding words; it does not stem words or disregard grammar generally. Contractions, function words, protected numbers and negation, and uppercase initialisms cannot use this rule. Reports identify it as `terminal_s_variation` and retain both original spellings.

The complete requested chunk or sample and complete recognized transcript are sent to JEV. The concise prompt lives in `QUESTIONS` in `epub_tts_narrator/jev.py`; saved judgments include the exact prompt and its policy identity. JEV receives no audio and can still make classification mistakes. Reference casting can recheck saved candidates, and already accepted audio retains its original evidence. Recurring rendering failures can also use the explicit recovery decision described below.

The exact prompt is `QUESTIONS["roughly_same"]` in `epub_tts_narrator/jev.py`, including its `instructions` and `criteria`. New attempt reports also save the exact question under `report.text_judge.questions`, with requested/recognized text under `report.text_judge.comparisons` and the score under `report.text_judge.probabilities`. Inspect `attempt-001.json` (or another attempt) in the directory printed after a verification failure. Historical reports retain their old policy version and prompt hash; reports created before this addition do not contain the full question text. The internal verdict name `equivalent` means accepted as plausible speech recognition, not semantic equivalence.

The primary transcript is judged first. If it passes, verification finishes without a second recognizer. Otherwise an optional independent recognizer can resolve the discrepancy, either by matching the requested words exactly or receiving its own passing JEV judgment. A negative first judgment is retained in the report when a second transcript is checked. An unavailable JEV service preserves the audio for a retry.

When JEV rejects a transcript or is uncertain, `verification.phonetic_fallback = "analysis"` (the default) uses the existing global `[provider]` analysis model to review a small number of pronunciation differences. This is inactive without a configured analysis model; set it to `"off"` to disable it. The complete transcript must first pass the local coverage check. Reviewable spans can include one word transcribed as two or three, such as `Insang` / `in saying`. The model receives these spans with short text context, never audio. Three separate calls receive no prior votes, and at least two must approve **each** span. These votes can still share model errors; they are not independent acoustic evidence. Insertions, deletions, reordered words, unrelated substitutions such as `does` / `is`, changed numbers, and changed negation cannot use this fallback. JEV's acceptance threshold stays unchanged.

The fallback prompt lives in `epub_tts_narrator/renderer/phonetic_review.py`. Reports retain JEV's original judgment plus the fallback's exact prompt, provider identity, source and transcript spans, coverage rules, and votes under `report.phonetic_reviews`. Verification records the actual accepting method as `phonetic_primary` or `phonetic_secondary`. Enabling the fallback changes verification identity for new speech and reference-candidate recovery. A fallback service error pauses for a retry and retains the recording as pending verification, never as accepted audio. Remote chapter servers use their own global `[provider]` configuration; client credentials are not forwarded.

Reference samples have an additional nomination path when the local phonetic matcher cannot establish coverage. Bounded replacements can reach the configured analysis model without first satisfying the strict sound-code heuristic: an unfamiliar name transcribed as a familiar word, or a phrase such as `you're here` transcribed as `you hear`, is a review candidate. The structural alignment must account for both sides of every changed span; a missing or added whole-word span, reordering, or changed protected number or negation is ineligible. Three separate reviewers still assess the sound of every nominated span in context, and at least two must approve each one. Nomination itself never accepts a take, and an existing negative strict sound review is not retried under this broader nomination rule.

This reference-only review also works without JEV when `verification.phonetic_fallback = "analysis"` and an analysis provider is configured. Its identity and cache policy are separate from ordinary speech verification. Reports retain the original failed coverage evidence and the complete sound-review audit; the cloning transcript is the recognized text that passed review. Unavailable reviews retain audio for verification retry. Saved rejected reference takes with matching synthesis identities can be rechecked under the updated reference policy before generating another take; already accepted voice assets remain intact. Disabled or missing reviewers retain the existing rejection behavior.

For a live synthetic regression check, run `.venv-renderer/bin/python scripts/evaluate_pronunciation_review.py --output output/phonetic-evaluation.json`. This sends only the script's synthetic cases to the configured analysis provider and saves full prompts and votes. It checks unfamiliar names and homophones alongside missing speech, changed numbers, synonyms, and missing syllables; it exits unsuccessfully if a case fails.

Coverage is checked word by word, rather than against a passage-wide error percentage: a long matching paragraph cannot hide a short omission. Phonetic alignment remains a heuristic, not proof of the recorded words; conservative failures can still result from inaccurate recognition or unfamiliar pronunciations. An independent recognizer can recover a complete transcript when the first omits words. Acoustic failures and malformed/empty recognition still fail directly. Whisper uses scalar temperature zero and greedy decoding, so repeatedly decoding identical audio with unchanged settings normally repeats the result; independent recognition is the fallback instead.

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

If JEV is unavailable, the run stops and retains the generated WAV. Rerun the same command to retry verification without generating it again. A different or uncertain judgment rejects the take if neither pronunciation review nor independent recognition resolves it. JEV receives text, so listening still determines whether the voice, accent, or performance suits the character.

Already accepted audio keeps its original verification evidence and is reused across changes to the verifier or its policy. Text, voice, synthesis settings, and audio integrity still determine whether a clip can be reused. Failed speech normally gets a fresh take; recurring resolution can explicitly reconsider a saved candidate after rechecking its audio. A take rejected only for `text_mismatch` under a different validation key (an earlier coverage or comparison policy, verifier or judge) is first verified again in full on its unchanged audio: fresh recognition, comparison and judges, with no synthesis and no generation allowance spent. Its record keeps the earlier report under `policy_recheck`; one that is still rejected is not rechecked again under the same key and starts no new recovery decision. Acoustic failures and history echoes are never rechecked. Pending verification outages and reference-candidate recovery remain resumable. Completed chapter checkpoints likewise retain their original acceptance evidence. API keys never enter reports or cache identities.

To recheck a retained casting attempt without generating another voice:

```sh
.venv-renderer/bin/audiobook-render cast runs/book-01 \
  --action recover \
  --voice-id VOICE_ID --attempt-file "/path/to/attempt-001.json"
```

Recovery preserves the original recording and rejection report and creates a saved candidate with the new verification evidence. By default it selects the recovered voice, requiring that the current prompt and script still match the original request. Add `--keep-unselected` to retain an alternative without changing the active voice. The interactive menu remains number, G, and Q.

### Automatic recovery of recurring speech failures

A rewrite may shorten a visibly repeated ASCII consonant fragment to the same single lowercase consonant, preserving every distinct letter and the source-word boundary. It requires at least three repeated letters; all-capital initialisms, vowels, `y`, different letters, and deletion do not qualify. This is TTS-input repair eligibility only: shared ASR coverage rules are unchanged, all three sound reviews are still required, and a newly generated recording must pass full verification. It never accepts a silent, stalled, clipped, or incomplete recording.

On resume, a previously blocked `invalid_proposal` rewrite can be reconsidered if it now passes the local guards and has not already failed sound review or been tried in a failed take. The renderer uses the existing byte-bound failure evidence, skips another planner call and recurrence wait, and obtains fresh sound reviews before generating. The audit's `reconsidered` field binds the candidate to its original attempt in `previous_decisions`; historical decisions and approved plain-delivery settings remain intact. Review outages or malformed sound responses keep the same candidate pending without spending a new TTS take or growing its pending history on each retry. Generation allowances remain bounded; the maximum attempt count is not a requirement to spend every attempt when no useful repair remains.

With the global `[provider]` configured, rendering escalates once the number of distinct failed takes reaches `recurrence_threshold`. Every failed take counts once, no matter how many problems it has: mixed text and acoustic failures and recordings that tripped several acoustic checks each count as one take, and repeated decoding of a single saved take adds nothing. Acoustic failures—including recordings that were too long and therefore never transcribed—accumulate toward the same threshold, so a stretch of different failure kinds escalates together instead of waiting for one identical signature to repeat. Alternative recognized spellings and omissions of the same source span stay grouped in the evidence, with the original spellings retained.

A reviewed pronunciation spelling or plain-delivery change that fails its own new take reaches the analysis model again immediately, with the disproved repair visible in the evidence, instead of waiting for more blind generations. The configured round limit still bounds the total number of decisions. A proposal that fails local validation or sound review does not consume the evidence: the next failed take can ask the model again without waiting for the threshold to refill.

The model sees the original passage, failed transcripts, acoustic measurements, actual synthesis text and acting controls, and any pronunciation spelling already tried. Later recovery rounds also receive earlier proposals and their local validation errors or sound-review outcomes, including after resuming. It can accept a plausible recognition variation, propose a small phonetic spelling change for the next generation, request plain delivery, retry, split at existing safe boundaries, or stop. A respelling may also be motivated by the latest take failing an acoustic check alongside a persistent wording problem; the failed acoustic recording itself can never be accepted, and every rewrite still needs its sound reviews plus a fully passing new take. Acceptance and respelling require at least two of three separate sound reviews for every changed span. The reviewers receive no prior votes and judge whether recognition could plausibly produce that spelling, allowing ordinary pronunciation variation and weak endings. They receive text, not audio; their agreement is additional evidence, not proof.

For directed Drama 3 speech, `plain_delivery` lets the model remove an acting instruction such as crying or whispering from a new take while retaining the same words and voice. The decision is saved and reused after interruption; it does not edit the script or accept an unverified recording. This action is offered only when the renderer can reproduce and validate that exact change. Coupled pronunciation/delivery normalization edits and whole dialogue groups are excluded. Supported individual split parts can use it too. An absent interjection in a transcript is still unresolved: a text-only model cannot establish that a cry happened in the audio.

Recovery and pronunciation-review requests start with up to 8,192 output tokens, including any model reasoning. A truncated response retries the same decision with up to 32,768 tokens, then the configured `provider.max_output_tokens` ceiling, for at most three requests. A smaller configured ceiling is always respected; the default ceiling when unset is 32,768. These retries do not generate audio or repeat already completed sound votes. Progress messages show the increased budget. If the ceiling is still exhausted, the error identifies the output limit and leaves the candidate saved for resume.

Visibly stretched spellings such as `Wheerrre issss sheeee?` can reach sound review against `Where is she?`, including stretched function words and negation (`Noooo` / `No`). This eligibility requires the same letters and words in order, with at least one run of three repeated letters in each changed source word; it does not automatically accept audio. Recovery can instead request the ordinary spelling for the next take and verify the complete reviewed wording. Dropping `is` or `she`, changing a pronoun, or replacing `Noooo` with `Yes` remains invalid. The separate whole-book spelling-preparation stage is retired; targeted edits may accompany initial attribution, and explicit manual overrides remain available afterward.

Incomplete hyphenated stutters also reach sound review: `B-but Mother` may be recognized as `But Mother`, or respelled as `Buh-but Mother` for synthesis. The complete final word and all other words must remain intact. This exception does not accept `Mother` (missing `but`) or `But but Mother` (an extra complete `but`), and the sound reviewers must still agree before a candidate or respelling is approved.

A synthesis rewrite may spell one literal ASCII letter as its exact English letter name, such as `N` to `En` or `t` to `tee`, including repeated-letter spelling variants of that same name such as `Enn`. This uses the shared letter-name vocabulary, preserves each source span and surrounding punctuation, and still requires all three sound reviews with a majority approving every edit. It does not authorize deleting a fragment, changing its letter, expanding an incomplete initialism into a guessed word, or accepting a hum as the requested speech. The new recording must pass the normal complete-text and acoustic verification. A previously blocked letter-name proposal can use the existing reconsideration path instead of repeating the planner and failed takes.

When cutoff punctuation repeatedly swallows a word, a reviewed rewrite may change prosodic punctuation inside a bounded one-to-three-word source span while keeping exactly the same normalized words in the same order. For example, `The room is—dark` may become `The room is. Dark` in synthesis input only. Markup, acting instructions, added or missing words, and capitalization-only no-ops cannot use this exception. All three sound reviews and normal verification of the newly generated audio still apply; an omitted interjection never counts as verified speech. Previous prompt bytes remain pinned for historical audit replay, and the upgrade retains existing reviewed spelling and plain-delivery choices. Split plans, source text, accepted parts, and the shared generation cap do not change.

Local checks prevent a model acceptance from excusing missing or extra whole words, reordered content, changed numbers or negation, or failed acoustic checks. An unfamiliar word substitution can reach sound review even when the normal phonetic heuristic could not match it. Pronunciation edits are bounded to four short word spans and apply only to TTS input. Book scripts, source offsets, voice selection, and delivery remain unchanged. Every new take is still verified, and a respelled take must account for the complete reviewed wording. Split recordings are verified individually and again after assembly; their saved pronunciation decisions can explain spelling differences in the final complete transcript, but cannot excuse extra speech.

These defaults are built in; no book-specific configuration or extra command is required:

```toml
[verification]
recurring_resolution = "analysis" # "off" disables model recovery
recurrence_threshold = 3
max_resolution_attempts = 12
max_resolution_rounds = 2
```

Merge these keys into your existing `[verification]` table if changing them. A passage and its split descendants share at most 12 fresh TTS generations per render invocation, so changing errors cannot create an endless retry loop. When shorter-part recovery is available, whole-passage attempts initially use at most half that allowance, reserving the rest for recovery. A newly approved spelling or delivery change at that boundary gets one test if enough starts remain for the parts. Each part reserves one start for each later part; very small custom limits cannot guarantee enough starts for every part. Each resolution session allows two model decisions. Model-selected splitting may happen before the generation limit. Transport errors retain the ordinary `render.max_attempts` limit; they are not pronunciation experiments. Without an analysis provider, or with recovery disabled, the existing `max_attempts` behavior applies. Voice casting retains its existing retry policy.

The full prompts are in `epub_tts_narrator/renderer/verification_prompts.py`. Each affected attempt directory contains `speech-resolution.json`, recording how many failed takes a saved decision already covered, model proposals, all sound votes, and the input/output evidence. The attempt and acceptance reports retain the original transcript and mismatch metrics alongside the explicit accepting decision. Pronunciation rewrites also appear in `engine_metadata.speech_resolution`; audited plain performances appear in `engine_metadata.delivery_recovery`. State and audio hashes bind these records to the original request. A model outage preserves the candidate and pending decision; rerunning resumes verification before spending another TTS take. Successful audio enters the normal cache. The evaluation script `scripts/evaluate_speech_resolution.py --output output/speech-resolution-evaluation` exercises the configured provider with synthetic failures; `--attempt-directory PATH` also checks supplied failed-take reports without uploading audio or altering the originals.

Rerunning the same command gives failed passages a fresh bounded allowance with new deterministic seeds while retaining recurrence history and approved pronunciation spellings. Take numbers continue across runs (`attempt-004`, `attempt-005`, etc.), preserving previous rejected normalized audio and verification reports. Abandoned raw files without a normalized take are deleted when a new generation starts. Internal replanning and duplicate requests cannot reset the invocation's shared allowance. A rejected take becomes usable only after a successful recheck or an explicit validated acceptance decision.

After a saved recovery stop, if the latest take lacks complete reviewable wording, resume can nominate older retained candidates before generating more audio. It scans at most the latest 30 records, freshly recognizes at most three eligible WAVs, and stays within the configured model-round limit. Saved transcripts are hints only: request and audio hashes must match, fresh recognition must pass acoustic checks and complete acceptance eligibility, and any nonexact candidate still needs the normal model decision and three sound votes. Negative reviews are retained rather than repeatedly rerolled. Pending confirmations pause without spending a new synthesis start; original failed reports and waveforms remain unchanged.

The renderer can automatically try smaller pieces after a text mismatch or an explicit model split decision. It splits the exact original text at sentence or strong clause boundaries into two to four parts, each containing at least two words. If sentence/newline splitting produces no valid plan, em-dash clause boundaries are tried instead, excluding boundaries inside URLs or email addresses. Existing sentence plans and their cache identities stay unchanged. Each part starts with the same voice and delivery; the model can request an audited plain performance for a supported failing part. With recurring resolution enabled, parts share the remaining parent allowance; otherwise each gets its existing bounded allowance. The renderer verifies every part, joins the verified lossless audio in order, then verifies the complete recording against the original passage. Source text, chunk IDs, and synchronization ranges stay unchanged. This strategy does not generate a new voice or recursively split failed sentence parts. Service outages pause verification rather than trigger splitting. A chunk whose first two failed takes both carry Drama 3's unfinished-text truncation signature (the engine's unfinished-text error, or a kept take that went silent and failed verification) goes straight to this split recovery, skipping further whole-chunk attempts and the resolver round-trip; interruption artifacts recorded for resume never count toward that trigger, and a chunk whose split already failed keeps the ordinary ladder.

A planner `stop` does not discard an executable, untried split while the shared allowance still has at least one generation start for every part. The model's original stop remains in `speech-resolution.json`; the failed attempt report records `planner_resolution_action: "stop"` separately from the renderer's effective `resolution_action: "split"`. The fallback never accepts the failed take, raises the generation cap, or re-enables a split after its complete joined recording failed verification. With no executable split or insufficient allowance, stop remains terminal.

A bound single-speaker recovery part with a short final em-dash tail has one additional cutoff-tail strategy: the exact prefix and one-to-three-word tail are synthesized and verified separately, then the complete joined part is independently verified against its original text. Pieces cannot fragment again. Both pieces share the existing episode cap and preserve sibling reserves. An approved plain-delivery choice is inherited only through its validated parent audit. The strategy is persisted for resume, retains verified pieces, and never rebuilds a definitively rejected join. This path cannot excuse a missing interjection or change source text, voice, or existing sentence split plans.

Every new single-speaker join retains the accepted child records needed to prove actual words, voice, and delivery, including normalization-edited passages that cannot use plain delivery. For an older accepted join missing those embedded records, cache lookup reconstructs them in memory from the child acceptance and WAV files only when their saved hashes match. It then applies the normal generation-input validation. Existing acceptances and audio are not rewritten or regenerated; missing or changed evidence cannot authorize cache reuse.

The strategy is saved as `speech-recovery.json` beside the original failed takes before generating any part. With recurring resolution enabled, resuming first rechecks the latest saved whole take for a deterministic match under the current verification rules. If it passes, that recording is reused without generating split parts. Otherwise, the saved strategy resumes without another batch of full-passage attempts: accepted parts are reused, and failed parts get fresh attempts within the new invocation's budget. Pending verification retains its audio. The final joined recording must also pass. If it definitively fails and analysis-model recovery is enabled, rendering returns to a new whole performance using the remaining shared allowance. The planner receives the failed-join result, retains approved pronunciation and delivery changes, and cannot select the exhausted split again. This also works when resuming a saved rejected join. Successful parts and the failed join remain saved; unchanged parts are not repeatedly concatenated. Child failures and verifier outages retain their own recovery paths. Once accepted, the whole recording enters the normal speech cache. If no suitable boundary exists or recovery still fails, rendering stops before publication.

Rejection output includes a short `expected → heard` summary and the reason a secondary recognition check could not resolve it. When some differences are spellings that already match by sound or word spacing (`labour`/`labor`, a recognizer's spelling of an invented name) and others are not, the failing differences come first with one word of context each side (`when it he → when he` shows a dropped word), followed by `sound-alike spellings, accepted: …`. Full transcripts and alignment details remain in the attempt JSON.

Run the same command again to resume; no cache deletion or extra retry flag is needed. Accepted audio is reused when its bytes and synthesis inputs still match; changing the verifier or quality policy does not revalidate or regenerate it. The original successful report remains attached to those bytes. Completed chapter checkpoints are also retained and reused when their synthesis inputs and assembly settings match. Rejected audio needs fresh validation or an explicit recurring-resolution acceptance before entering the success cache. Audio awaiting interrupted verification or an unavailable confirmation service remains pending. Failed paragraph groups retain their exhausted budget and recover by splitting into their original passages as described above. This automatic retry allowance does not regenerate voice references or change casting selections. Removing a cached audio file regenerates that clip; modifying an existing file fails its integrity check. Progress output identifies reused clips and completed chapters; failure output identifies the work directory containing recovery data.

During `render` with local synthesis or a render worker pool, press **Ctrl-C once** to stop starting new work. Every active take finishes normalization, verification, and saving its accepted audio to the speech cache; all active workers drain before the command exits with status `130`. A rejected active take saves its diagnostic report without starting another retry. The terminal confirms the stop request. Press **Ctrl-C again** to force an immediate exit, sacrificing any unfinished work but retaining completed cache entries. Rerun the same command to resume.

A graceful stop releases locks. Work, book, and voice locks created by this version also recover automatically after a forced stop once the owning process has exited. Keep their sibling `.lease` files; these provide coordination, not evidence of an active process. If an older or unrecognized lock prevents recovery after a hard kill or system crash:

1. Confirm the renderer process has stopped before touching locks.
2. Remove only legacy empty `.render.lock`, `.audio-render.lock`, or voice-store `.lock` directories. A publication lock left by this version is reclaimed automatically on the next run; an empty `.<output-directory-name>.publish.lock` from an older renderer can be removed after confirming publication has stopped. `rmdir` is appropriate because it refuses nonempty directories. Do not delete ownership records or `.lease` files; inspect any unrecognized nonempty lock instead. Retain the rest of the cache and voice store.
3. New outputs are published by directory rename after validation; replacement briefly moves the previous complete output to a sibling `.<output-directory-name>.previous-*` directory. Normal exceptions restore it. If a hard kill occurred between those renames and the output is absent, restore that previous directory before rerunning. If both directories exist after a hard kill, keep the previous copy until the current output has been verified. If an older renderer left an incomplete output directory without `render.json`, move that incomplete output aside or choose a fresh `-o` directory. Complete, intact outputs for the same book update automatically.
4. Rerun the command. An incomplete voice asset or repeatedly interrupted raw attempt may require inspection; the renderer stops rather than silently replacing an established voice.

Voice, work, and output directories are separate children of the analyzed book by default. The renderer checks unsafe overlaps, including overrides that could write into a parent book or model files.

## Output and synchronization

The completed output directory (`<book-directory>/audiobook/` by default) contains:

| File | Purpose |
| --- | --- |
| `chapters/chapter-0001.m4a` or `.mp3`, etc. | Chapter audio using original manifest numbers, including when a subset is selected; empty entries are marked skipped. |
| `<book-title>.m4b` | AAC audiobook with chapter titles and timing, using a filesystem-safe form of the actual book title. |
| `chapter-links/chapter-01.m4a` etc. | Convenience symlinks to each finished chapter's audio inside `chapters/`; renderer-owned and ignored during replacement validation. Entries are always symlinks. |
| `artwork/cover.jpg` | Square 1600×1600 cover embedded in the M4B. |
| `artwork/original-cover.jpg` or `.png` | Original declared EPUB cover, retained byte-for-byte. |
| `sync.json` | Exact PCM frame ranges linking audio chunks to source text. |
| `source.json`, `cast.json` | Portable canonical text and voice assignments. |
| `verification.json` | Accepted chunk reports, transcripts, and hashes. |
| `audio-manifest.json` | Audio inventory, durations, chapter ranges, checksums, and the authoritative `audiobook.path` basename. |
| `render.json` | Completion marker, provenance, engine/verifier identities, voice hashes, and file checksums. |

Fresh books use a sanitized form of the actual title, for example `The Last Lantern.m4b`, never the generic `book.m4b`. Path separators, controls, reserved device names, and unsafe override values are rejected; Unicode is preserved where possible and long names are bounded. The exact chosen basename is persisted in `audio-manifest.json` as `audiobook.path`, repeated in `sync.json` as `audio_file`, and included in `render.json` `files_sha256`. For example:

```json
{
  "audiobook": {
    "path": "The Last Lantern.m4b",
    "sha256": "<sha256>",
    "codec": "aac"
  }
}
```

An existing per-book `audiobook.path` is authoritative across render, assemble, metadata refresh, publication, result paths, and checksums, including a custom name or an explicitly recorded `book.m4b`. Changing the book title does not silently rename that output. An older manifest without `audiobook.path` can use an intact existing `book.m4b`; that compatibility fallback is not a fresh-book default. No existing book is renamed, deleted, re-encoded, or regenerated merely because its name is legacy.

### Audio filenames and existing output compatibility

New output uses descriptive names, such as `BookTitle.m4b`, `chapter-0042.m4a`, `voice-id-Hero-reference.wav`, contextual attempt raw audio, and purpose-specific context or temporary stages. Filenames are not part of speech, ASR, or reference cache identity. They identify files for people and let saved records locate them reliably. A recorded `.wav` name for a take, mix, chapter or preview may be kept on disk as `<same name>.flac` holding the identical WAV bytes ([Kept audio is lossless FLAC](#kept-audio-is-lossless-flac)).

For existing output, the recorded JSON filename remains authoritative, even when it is a legacy generic name. The renderer reads and resumes using these fields:

| Record | Filename field(s) |
| --- | --- |
| `audio-manifest.json` | `audiobook.path` (also inventoried by `render.json`; mirrored as `sync.json.audio_file`) |
| `voice.json` | `reference_file` |
| `normalization.json` | `audio_file` |
| `preview.json` | `audio_file`, `raw_file`, `normalized_file` |
| `attempt-NNN.json` | `audio_file`, `raw_file`; `accepted.json` uses `audio_file` |
| Local `chapter.json` | `wav_file`, `encoded_file` |
| Remote `chapter.json` | `audio_file`, `lossless_file` (remote chapter body fields) |
| `mix.json` | `audio_file` |

Local and remote chapter records use different fields. New local and remote chapters use numbered names; remote exports require a coordinated controller and worker update. A new controller can still read legacy worker and cache metadata. The audited writers also cover final books, imported and generated voice references, loudness and pronunciation previews, speech and recovery attempts, selections, mixes, server scratch audio, provider atomic stages, downloads, comparison copies, and worker smoke artifacts. This describes current renderer-owned defaults, not unmerged branches or opaque vendor internals.

Older records without a filename field use only their existing legacy-reader fallback. Existing audio, checksums, and selected voices are not mass-renamed or regenerated. These filenames are linked to integrity records: don't edit one manifest or rename one file by hand, since the other checksums and linked state would no longer agree. Compatibility means the application honors the name already recorded; it is not a freeform filename setting or a new command. New defaults apply to new output only, and changing a title does not rename an existing recorded book.

New EPUB analyses retain a local `source-epub.json` location hint, bound to the analyzed book's SHA-256. Rendering uses that original automatically. The hint is separate from analysis fingerprints and voice/audio cache identities. For an older analysis or a moved original, pass `--epub PATH`; the EPUB's byte hash must match the analyzed book, the latest update's recorded EPUB, or be a rebuilt or typo-corrected edition of the same book (verified by re-extracting its text against the saved analysis; the published book identity never changes). Local and remote chapter rendering use the same final publication path for artwork and tags.

The full declared JPEG/PNG cover is scaled proportionally into a 1600×1600 square, over a blurred, darkened background derived from the same image. The sharp foreground is never cropped or stretched. The original image is retained, and the square JPEG is embedded as attached cover art. Required artwork is checked before synthesis. Missing or unsupported artwork stops full-book publication by default; `--chapters-only` can still generate speech. Set `render.require_cover = false` only when deliberately publishing without artwork. Unsafe, corrupt, or mismatched EPUB inputs still stop the run.

The EPUB supplies title, author/album artist, album title, publication date, description, rights, language, and series grouping when present. The container is marked as an audiobook. Additional EPUB metadata, including publisher, identifiers, subjects, contributors, and series position, is retained in `audio-manifest.json`; player support for these extra fields varies. FFprobe checks the standard embedded tags and cover before publication.

To add or refresh artwork and metadata on a completed book without generating speech or re-encoding audio:

```sh
audiobook-render refresh-metadata path/to/book-directory --epub path/to/original.epub
```

This command needs only FFmpeg/FFprobe and the existing finished output; it does not need models or render caches. It stream-copies the M4B's encoded audio under the same recorded `audiobook.path` basename, preserves chapter files and synchronization ranges, verifies that the encoded audio packets are identical, and updates all affected checksums. Refreshed EPUB metadata does not rename the file. The previous book remains intact until the replacement passes verification. The original EPUB location is saved for future commands.

Chapters and the final M4B are encoded independently from the same assembled lossless PCM. The final book is never built by re-encoding the lossy chapter files. FFprobe checks the resulting duration, chapter tags, and embedded cover. Completed chapter PCM, encoded chapter copies, accepted chunk PCM, and failed attempts remain in the work cache for recovery; the published audiobook does not duplicate the PCM files.

`sync.json` records chapter-relative and book-relative integer frame offsets, chunk IDs, segment and block IDs, and exact source ranges. Frame ranges exclude inserted pauses and have exclusive ends. Bracket delimiters stay in the source text but are omitted from spoken text; mappings retain that distinction. Approved spelling normalization also preserves original source ranges: `Wheerrre issss sheeee?` can render as `Where is she?`, and a replacement maps back to its entire original span. The verifier expects these normalized words. The renderer combines the existing delivery with spelling-derived style directions and sends Drama 3 a validated `[style] text` control prefix alongside the saved voice reference. Existing whisper and other recognized controls keep their previous syntax. Newly applied directions change the affected speech cache inputs; speech with identical text, voice, and actual generation settings remains reusable. Voice references are not regenerated by this preparation. A configurable pause at the outer edges of a bracketed passage separates its delivery while preserving the character voice. Paragraph and blank-line pauses use the larger overlapping cue.

Synchronization currently has **chunk precision**. ASR word times are retained in verification reports, but are not yet published as validated word-to-source alignment. Audio players must account for codec delay and padding; container chapter times may round to milliseconds. Exact highlighting inside the original EPUB still needs DOM/CFI mapping; the current ranges address the included canonical `source.json`.

## Adding another TTS backend

The `TTSEngine` protocol separates engine identity, capabilities, voice creation, and synthesis from planning, verification, caching, and packaging. A separate `[voice_design]` configuration supplies reference creation when the speech engine lacks descriptive voice design. Setting its type to `engine` delegates creation to the speech engine only if that engine advertises voice-design support; random unconditioned generation is not a fallback. Saved voice bindings use a `kind` discriminator; the implemented kind is `reference`. Built-in speech cache identity uses actual model content, sampling settings, and voice-reference contents. Runtime versions, designer identities, and full voice-asset records remain provenance; they do not independently invalidate an identical built-in speech performance.

Installed adapters can register a Python entry point in the `epub_tts_narrator.tts` group. Existing registrations in `audiobook_renderer.tts` remain supported when the configured name has no registration in the new group. A matching registration in the new group takes precedence; duplicate matches in the selected group are rejected. Both rendering and `doctor` use this lookup. Set `engine.type` to that entry point's name and put adapter-specific values in `[engine.options]`; the factory receives that options dictionary and must return a `TTSEngine`. Its identity must include stable `engine` and `model` strings plus every setting affecting synthesis. Unknown or invalid adapters fail explicitly, with no fallback to a different engine.

A future backend with predefined voices can add a `preset` binding and explicit character-to-provider-voice mapping. That mapping and backend are not implemented yet. Adding them does not require changing the analyzer bundle or audio assembly format. See [ENGINE_NOTES.md](epub_tts_narrator/renderer/ENGINE_NOTES.md) for the current Drama 3 adapter's limits.
