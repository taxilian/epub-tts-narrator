# Collective voices — design

Status: **approved design (2026-11-19), implemented** (see §10). The phases in §8 are the implementation plan; §9 lists the questions still open.

A *collective voice* is one cast role heard as two or more voices: a chorus, a pair of twins, a crowd, a hive mind. This document surveys how the analyzed books under `runs/` present such speakers, what the project does today, and proposes a representation, an analysis rule set, review tools, a renderer recipe, and guards.

## 1. What the books contain

A survey of the 16 analyzed books (`runs/*/analysis.json`, Eddings' Belgariad/Mallorean, *Birth of the Firebringer*, *Crystal Singer*, *Quest Academy* 7–8, *Long Chills*) found these patterns. Quotes are kept to short snippets.

| Pattern | Book example | How the text presents it | Today's result |
|---|---|---|---|
| **Pair in unison** | Beltira & Belkira (Belgariad 3, 5) | "the two said in unison", "the twins said in unison, bowing" | 12 `simultaneous_speakers` lines, both twins mixed. Works. |
| **Pair alternating half-lines** | same twins | "one of them said" … "the other finished": "it's been such—" / "—a terribly long time" | Each half given to Beltira or Belkira, though the text never says which twin is which. The split is invented. |
| **Named people in unison, ad hoc** | "Lift!" (Mallorean 2); "Sorry!", "Yes!" (Quest Academy 7) | "the three of them said in unison", "Every former Reaver answered instantly" | `simultaneous_speakers` across 2–4 existing characters. Works. |
| **Overlapping, not unison** | "Guild do what?" (Crystal Singer) | "Borton and Jezerey asked" (direction: "overlapping") | Mixed as unison; the format has no overlap mode. |
| **Unnamed group in unison, ritual** | seven masked farmhands, "We are the Gods" (Belgariad 1); Algar Bear-cultists "Hail, Cho-Hag" (Belgariad 4); younger Gods (Belgariad 3) | "they said in unison", "the cultists intoned in unison" | One group character with one ordinary voice. |
| **Call and response** | crimson-robed eunuch + eunuch chorus, "Praise her" / "Welcome" (Belgariad 2, Mallorean 2/4/5); Faldor's people answering "The Gods" (Belgariad 1) | a leader speaks; "the chorus replied", "the others responded ritualistically" | Leader and chorus are separate characters (good); chorus is one voice. |
| **Subset of a group** | "half of the kneeling eunuchs murmured" (Mallorean 2) | a part of a known group | One voice. |
| **Large crowd shout or chant** | assembled Alorns "Hail Belgarion!"; legions chanting "Ce-Ne-dra!" (Belgariad 4); Chorus of ancestors "No!" (Belgariad 5); "More!" (Mallorean 2) | "the great shout shook the Hall", "a chant that soon spread" | One voice, sometimes with "roaring" as a direction. |
| **Anonymous members, one at a time** | Dryad girls (Belgariad 2, 12 lines); Grolim escorts (Belgariad 5); Tolnedran legionnaires; Mob goon (Long Chills) | "one of the others observed", "Another one giggled", "one of the other Grolims added", "another man roared" | Every member speaks in one shared voice, so different Dryads sound like one girl. |
| **Same anonymous member continues** | Dryad girls | "one of the Dryads squealed" … "she demanded" … "the Dryad offered" | Fine today by accident; a naive random pool would break it. |
| **Swelling, staggered repetition** | fenling chorus "Mother, mother, mother" (Belgariad 4); Birds "Polgara. Polgara." (Belgariad 3); the herd "He is dead, he is dead!" (*Firebringer*); "Let us down" from the hanging bodies (Belgariad 3) | "a growing babble of small, squeaky voices, all repeating", "a chorus of chirping voices repeating" | One voice. |
| **Echo cascade / hive mind** | Switcher hive mind (Quest Academy 7) | "One voice started it, but it ended up getting parroted back deeper" | One voice. |
| **Spokesperson for a shared mind** | Andel and the Dals (Mallorean 3) | "that peculiar murmur as of many voices" before she speaks | Andel's ordinary voice; the murmur is narration only. |
| **Chorus behind one clear speaker** | Cyradis and the Seers of Kell (Mallorean 5) | "The vast chorus of the Seers of Kell swelled" | Cyradis alone. |
| **Group as one spokesperson (legend register)** | Arendish nobles, Ulgo elders (Belgariad prologues) | "the nobles … came to Brand, saying" | One voice. Arguably right: a legend's "they said" is often heard as one voice. |
| **Repeated remark from many people** | guests: "It never snows in Tol Honeth." (Mallorean 2) | "the endlessly repeated comment" | One voice. |
| **Two unseen voices** | first voice / second voice (Mallorean 4) | "a voice somewhere off to his left", "another voice replied" | Two ordinary characters. Right: these are two individuals, not a collective. |

Two misattributions the survey turned up show why the guards in §6 matter:

- **A plural line cast as one ordinary voice.** In Mallorean 2, "As your Majesty commands," is tagged "the Murgos replied in unison". It went to the single character "burly guard".
- **A single speaker absorbed by a group character.** In Belgariad 2, the "Arendish nobles" entry holds lines tagged "the voice promised" and "the voice in his mind said". Those lines belong to the singular "dry voice".

Name-based guesses also mislead. "dry voice", "first voice" and "Chorus" (the name) say nothing about plurality. "Voices" can be one entity, and "the twins" can be two named characters.

## 2. What the project does today

- **Analysis.** In the narrator read, `who: "id&id"` marks characters saying the same words together. `narrator_read` keeps a joint line only when the paragraph proves it: a model quote, or a narration match of `_JOINT` ("in unison", "together", "in chorus", "both", …). Otherwise the line goes back to a person as unresolved. The output is `segment.simultaneous_speakers = [{role_id, voice_id}, …]` with `role_id`/`voice_id` null (FORMAT.md §segments). It requires ≥2 distinct existing dialogue roles and distinct voices. Overlap and staggered entrances are explicitly out of scope there. A group with no named members becomes an ordinary character (often "Minor speaker added during markup") with one voice.
- **Review.** `review-ui` speaker reassignment is multi-select; choosing two or more builds `simultaneous_speakers`. `verify-attribution --instructions` can choose joint lines among existing roles.
- **Renderer.** `bundle.chunk_speakers` expands a simultaneous row into lanes. `workflow` renders each lane as an ordinary single-voice take with `performer` in its request. Every lane passes ordinary verification, and the mix is never put through ASR. `assembly.mix_speakers` aligns lanes at frame zero, zero-pads, and averages at a fixed `1/N` weight. `sync.json` lanes already carry relative `start_frame`/`end_frame`. Mixture cache keys bind every lane's asset, request, verification and hash (`chapter_cache`).
- **Continuity.** A lane hears its own voice's history (`continuity.request_speakers` → performer). It never joins history (`scope["history"] = "performer" not in request`), because lanes overlap and `history_dependencies` orders chunks by voice subsets.
- **Neighbouring features to reuse.** Derived voices (`derived_from`, `derive_voice_take`) clone a voice from a base with an instruction and pass reference verification. Voice effects (`renderer/voice_effects.py`) are closed FFmpeg presets applied after verification on a dry take, with the recipe in chapter identity. Accents are grouped by origin (`accent_assignment`). Character merge rebinds descendants. Sidecars inherit nearest-book-first.

Two side findings about the current mix, worth fixing whatever is chosen here:

1. **Loudness drop.** Averaging N uncorrelated voices at `1/N` gives RMS ≈ `1/√N` of a solo line: 2 voices ≈ −3 dB, 4 ≈ −6 dB, 8 ≈ −9 dB. Chapter normalization is chapter-wide, so unison lines sit quieter than the lines around them.
2. **No onset alignment.** Lanes start at frame zero, so a take with a longer lead-in is heard late. Even "perfectly in unison" twins sound sloppy.

## 3. How collective voices are used in fiction, and how each should sound

| Use | Ideal sound | Mode (proposed name) |
|---|---|---|
| Twins or a pair as one mind, in unison | 2 distinct but related voices, onsets aligned, near-identical timing | `unison` (tight) |
| Twins finishing each other's sentences | Alternate the two voices line by line within the exchange | `alternate` |
| Ritual chorus, congregation, cult chant | 3–6 voices, onsets within ~30 ms, slight timing spread, even level | `unison` (loose) |
| Call and response | Leader (an individual role) then the chorus role in unison | two roles; chorus `unison` |
| Crowd shout or roar | up to 8 voices, wider spread (50–150 ms), even level, maybe a "hall" room preset | `unison` (crowd) |
| A chant that spreads | Voices join one by one with rising level, then full unison | `staggered` (swell) |
| Babble of small voices repeating one phrase | 3–5 voices, entrances 150–400 ms apart, out of phase | `staggered` |
| Echo cascade ("parroted back deeper") | Leader full, then copies later, quieter and duller each time | `staggered` (echo) + effect |
| Hive mind or shared-mind spokesperson | One clear leading voice, others under it at −10 to −14 dB, 20–60 ms behind | `layered` |
| Anonymous members speaking one at a time ("one of them", "another") | A different pool voice per individual; the *same* voice while the same individual continues | `pool` (with slots) |
| Group addressed as one in a legend ("the elders said") | One voice; plural sound would distract | `single` (today's behaviour) |
| Crowd murmur behind one clear speaker, or under narration ("the hall buzzed") | A low, unintelligible bed of the group's voices under the passage, ducked under speech | murmur bed (§5.7) |
| A voice that later splits into individuals | Pool voices become named characters and keep their voice | promotion (§5.6) |
| A subset of a group ("half of them murmured") | Unison of fewer members, softer | per-line `members` subset |

## 4. Representation options

### Option A: extend `simultaneous_speakers` only

Add an optional `simultaneous_mode` (`unison|staggered|layered`) to rows that already list ≥2 existing roles. Unnamed groups need a character per member.

- **For:** smallest change; reuses the lane path end to end.
- **Against:** an unnamed chorus would need invented characters ("eunuch 1…4") and repeated lists on every line. It has no `pool`/`alternate`, no series identity for the group itself, and nothing for a person to recast in one place.

### Option B: a collective *character* expanded at analysis time

`cast.characters[i].collective = {members, mode}`. Analysis writes `simultaneous_speakers` for each of its lines.

- **For:** identity sits with the character.
- **Against:** it bakes rendering choices into every segment. A change of membership becomes an attribution revision across the book and every descendant. `pool` picks would be frozen into the script.

### Option C (recommended): a group character with an *ensemble voice*

The group is an ordinary character with one role and one `voice_id`, exactly as today. Once a person sets it up, that voice becomes an **ensemble voice**. It keeps its own designed recording as the group's *lead*, the first lane. It adds a list of further member voices. Segments keep `role_id`/`voice_id` pointing at the group, so attribution, review, merges, series carrying, chapter partitions and every "role → voice" consumer keep working. Only the renderer, casting and readiness learn to expand it.

- **For:** identity (who speaks) stays in attribution. Sound (members, mode, mix) stays in the cast, where a person already makes voice choices. Recasting a chorus changes no segment. Pool picks are deterministic at plan time. Series inheritance of voices already exists.
- **Against:** every voice consumer must learn that a voice can be an ensemble: casting, accents, voice aliases, `import-voices`, prune, audition scripts, pronunciation previews. Readiness must check the members, not the ensemble.

`simultaneous_speakers` stays for ad-hoc unison of individually named characters ("Lift!"). It also gains the optional `simultaneous_mode` from Option A for "Borton and Jezerey asked, overlapping".

## 5. Recommended design

### 5.1 Format

```jsonc
// cast.characters — the identity and analysis's proposal; analysis owns it
{"id": "char_40", "name": "eunuch chorus", "voice_id": "voice_40",
 "group": {"size": 24,                     // an exact number from the source, or null
           "scale": "crowd",               // pair | few | several | many | crowd, from the source's words
           "proposed_voices": 8,           // how many voices to cast, from size/scale (§5.2 item 7), at most 8
           "members": [],                  // named characters the source places in the group (e.g. the twins)
           "proposed_mode": "unison",      // the mode of its strongest-evidence lines
           "evidence": [{"block_id": "b001780", "quote": "the two dozen shaven-headed men"}]}}

// cast.voices — the sound; added only by a person in the cast screen
{"id": "voice_40", "description": "Male temple eunuchs, ritual monotone",   // its own recording = lead lane
 "ensemble": {"members": ["voice_40_1", "voice_40_2", "voice_40_3"],       // further lanes, after the lead
              "mode": "unison",            // default for this group's lines
              "mix": "loose",              // named recipe, see 5.4
              "basis": "user"}}
{"id": "voice_40_1", "description": "…", "member_of": "voice_40"}   // a pool voice: no character of its own

// segments — per-line override, only when the source says this line differs
{"role_id": "role_40", "voice_id": "voice_40", …,
 "ensemble": {"mode": "unison", "members": ["voice_40_1", "voice_40_2"],  // optional subset
              "lanes": 4,                  // optional: how many voices this line takes (≤ the ensemble's)
              "slot": null,                // pool: which anonymous individual (scene-local), see 5.3
              "basis": "explicit", "evidence": [{"block_id": "…", "quote": "half of the kneeling eunuchs murmured"}]}}
```

Contract rules (`contracts.py`):

- **Ensemble voices.** An ensemble voice is an ordinary designed voice (its own recording is the lead lane) plus ≥1 further distinct members (none only for a declined `single` record, §6.2). Every member is a non-ensemble voice other than the lead. The ensemble voice cannot be `derived_from`, but a member can be. It never appears inside another ensemble. Casting, accents and voice aliases treat the lead like any voice, so a group that was cast before keeps its casting.
- **Who may be a member.** The narration voice may be a member only with `basis: "user"`.
- **Pool voices.** `member_of` must name an ensemble voice that lists that pool voice. Pool voices belong to no character. Pool voices are written only by a person's cast-screen action (scope `ensembles`), never by analysis.
- **Per-line overrides.** Segment `ensemble` is allowed only on rows whose voice is an ensemble. `members` must be a subset of the ensemble's members. Analysis-made overrides need evidence; user ones do not.
- **Group membership.** `group.members` must be existing characters. Each needs evidence placing it in the group (a name match is not evidence).
- Modes: `single | unison | alternate | pool | staggered | layered`. `single` speaks every line in the lead voice, which is today's behaviour. It is the safe default when a person declines a collective sound.

### 5.2 Analysis: deciding from source evidence

The decision is about **the speaker tag and its subject**, never the character's name.

Scope: the narrator read (the default strategy), like voice changes. Consensus-path books get only the §6.1 guards as flags. A person can still set up a group by hand in the cast screen and `review-ui`.

1. **Group identity (ledger).** The prep read may mark an entry as a group (`kind: group`). That needs a quote whose grammatical subject is plural and refers to the speakers: "the twins said", "the cultists intoned", "the chorus replied", "several of them answered", "the Dryads", "two dozen … men". `size` comes only from a number in the source. Named members come only from a sentence that names them as the group ("Beltira and Belkira, the twins"). A batched check, like the existing merge and alias checks, confirms each group with exact quotes and splits on "n".
2. **Line mode cues.** These are read from the tag the way `speech_cues` reads sounds:
   - "in unison / as one / chorused / intoned in unison / all together" → `unison`;
   - "one of them … the other finished/completed" → `alternate`;
   - "one of the others / another / someone (of the group)" → `pool` with a new slot;
   - "the Dryad / she (continuing)" after a pool line → the same slot;
   - "repeating / babble / swelled / echoed back / overlapping" → `staggered`;
   - "half of them / some of them" → a `members` subset;
   - "a murmur as of many voices" before one named speaker → that speaker's group, `layered`.
   A cue applies only when its subject is the line's own speaker. Narration that merely mentions a crowd is not a cue. These are the same subject rules `speech_cues` already enforces.
3. **Proposed mode.** `group.proposed_mode` is the mode of the group's strongest-evidence lines. Per-line overrides are written only where a line's own tag differs from it. Both stay inert until a person adds an `ensemble` to the group's voice.
4. **Ad-hoc unison of named characters** stays `id&id` → `simultaneous_speakers`, with its existing evidence rule.
5. **Rerun review.** `analyze BOOK` adds a group check beside `cast_review`/`gender_review` (`group_review.review_groups`). It looks at:
   - every group entry without confirmed evidence;
   - every line hit by the guards in §6.1.
   It uses one batched `GROUP_CHECK` prompt with exact quotes and is recorded in `provenance.group_checks` by text hash. When no voice exists yet, it fixes at once; otherwise it reports and asks, as `gender_review` does.
6. **How many voices.** Analysis proposes a count; it never casts. `group.scale` comes from the source's words about the group, never the entry's name: "both / the twins / the pair" → `pair`; "a few / three of them" → `few`; "several / a handful" → `several`; "a dozen / the elders / the council" → `many`; "the crowd / the throng / two dozen / hundreds / the legions" → `crowd`. An exact `size` wins when the source gives one. `group.proposed_voices` is then `size` (clamped to 2–8), else pair 2, few 3, several 4, many 6, crowd 8. A line whose own tag changes the scale gets a per-line `lanes` hint with its quote: "half of the kneeling eunuchs murmured" → half; "several of them answered" → 3–4; "a chant that soon spread" → the full ensemble, with `swell`. Lines without a hint use every member.
7. **Prompt identity.** The new prompt text and ledger kinds are prompt/cache identity changes. Bump them deliberately and add a `direction_policy`-style version, so books analyzed earlier get the group check on their next rerun.

### 5.3 Pool slots: anonymous individuals who keep their voice

A `pool` line carries `slot`: a scene-local index of the anonymous individual. "One of the Dryads squealed" opens slot 1. "Another one giggled" opens slot 2. "The Dryad offered" continues the current slot. The renderer maps slots to pool voices deterministically:

- Within a scene, a slot keeps its voice. A new slot takes the pool voice with the highest rendezvous hash of `(ensemble voice, scene, slot, member)` that is not already used by another slot in the scene. When slots outnumber members, they wrap.
- Rendezvous hashing means adding a member moves only the slots that now prefer it. Removing one moves only its own slots. Lines elsewhere keep their cached takes.
- With no slot (an old analysis, or no cue), each line is its own slot, and two consecutive lines never share a voice.

`alternate` works the same way with strict rotation inside a run of consecutive group lines. A person can pin which member starts.

### 5.4 Renderer

Expansion happens in `bundle.chunk_speakers`, which already returns performer lanes:

| Mode | Lanes | Verification | History |
|---|---|---|---|
| `single`, `pool`, `alternate` | one: the resolved member | ordinary | an ordinary turn of that member's voice: hears and joins its history |
| `unison`, `staggered`, `layered` | one per member, or the line's `lanes` hint (a deterministic subset that always includes the lead), capped by `[render] max_ensemble_lanes` (default and ceiling 8) | each lane independently, unchanged; the mix is never put through ASR | each lane hears its own voice's history and does not join it (today's rule) |

The mix is generalized from `mix_speakers` to a versioned **mix recipe**, a closed named set like the voice-effect presets. The recipe name, parameters and `RECIPE_VERSION` are part of the mixture cache key:

- `tight` / `loose` / `crowd` (unison):
  1. align each lane's first speech onset, measured on the verified take;
  2. add deterministic offsets of 0–15 / 0–35 / 0–120 ms;
  3. use equal gains, then loudness-match the mix to the median of its lanes, with a peak limiter instead of `1/N` (§2, finding 1). This loudness matching applies to every mix, including today's `simultaneous_speakers` (decided).
- `swell` / `babble` / `echo` (staggered):
  - per-lane entry offsets: 250 / 150–400 / 300 ms steps;
  - gains: rising / even / −4 dB per step;
  - `echo` adds a darker filter per step.
  - Lanes keep full length, and the entry ends at the last lane's end. `sync.json` already carries per-lane `start_frame`.
- `lead` (layered): the lead voice at full level, the others at −12 dB and 20–60 ms behind.

Every lane still says the complete segment text. Partial echoes (repeating only the last words) would need new spoken-text spans. They are out of scope (§9, open question 6).

**Identity and caching**

- **Chunk IDs** stay source-position based. As with `revoiced` segments, a collective line's chunk identity adds the sorted resolved performer set. Take marks and manual selections then never move to a different voice.
- **Per-lane requests** keep today's `performer` identity, so recasting one member re-renders only that member's lanes and the mixes that contain them.
- **Changing mode** between a lane mode and a one-voice mode changes the request (a lane is not a solo turn), so those lines re-render.
- **Pool picks** are resolved in `make_plan` and shown in `plan`. A changed membership therefore names exactly the lines that move.

**Continuity in practice.** A chorus's pool voices may speak only in lanes, so their history is just their pinned reference context. Voices stay closest to their references, which is desirable for a chorus. A twin who also speaks alone keeps building normal history from the solo lines.

**Cost.** A full 8-lane crowd line costs 8 syntheses and 8 verifications, and a crowd group needs 8 designed voices. `plan` should report ensemble lane counts per chapter. Render pools already spread lanes across workers.

**Take review.** In the run shell `TakesScreen` and in `review-ui` playback, list the mix and each lane. Good, bad and truncated marks apply per lane, and `retire_judged_bad` rebuilds the mix.

### 5.5 A person's review and changes

**Cast TUI (`cast BOOK`)**

- Group characters show `GROUP ×N` with their members and mode. A new action (for example `G`) opens an ensemble screen built from `tui_widgets` components:
  - see the proposed number of voices with the quote it came from ("two dozen shaven-headed men" → 8), and choose a different number (2–8);
  - add or remove members: existing voices, or a new pool voice;
  - set the default mode and the mix recipe;
  - **audition** one real line of the group rendered with every member and mixed;
  - convert to `single`, which records the decision so it is never proposed again (the same keep-apart idea as `cast-review.json`).
- **New pool voices are designed separately** (decided), each through normal casting. Each is designed from the group's description with its own varied traits (age, pitch, texture, pace) and the group's accent, and the person picks among takes as for any voice. Voices are never cloned from a sibling. Clones would sound alike (the voice-convergence findings say Drama 3 cloning separates voices weakly), and `derived_from` means *the same person changed*. A casting-batch job designs a group's members together, so a crowd of 8 is one batch. The designer gets the varied traits of the members already designed, to keep them apart.
- **Accents.** `accent_assignment` treats a group's pool voices as one origin group by default.
- Every change is a voice choice and so a person's. The analysis proposal (`group`) stays a proposal until a person confirms it, and saves go through `analysis_revision` with a new scope `ensembles` (receipt-bound, descendant rebinding).

**`review-ui`**

- Speaker choices list group characters like any other.
- For a group line, the metadata panel shows the mode, the resolved members or pool voice, and the slot. The edit dialog can:
  - change this line's mode;
  - choose a member subset;
  - pin the pool or alternate member.
  These save as segment `ensemble` with `basis: "user"` through scope `attribution`.
- Multi-selecting characters still builds `simultaneous_speakers`. A new toggle on such a row sets `simultaneous_mode`.

**`verify-attribution` and `refresh-directions`** treat group roles as ordinary roles. Their jurors also see the group check's line cues, so "the twins said in unison" is not "corrected" to one twin.

### 5.6 Series inheritance

- The group character and its ensemble voice carry like any series character and voice. Its members are carried with it: pool voices as series voices, member characters by their own series IDs.
- A later book may add, remove or recast members. The nearest book wins, as with voice choices, and an earlier book's membership is never rewritten. An inherited ensemble stays fixed in the later book unless a person edits it there.
- **Promotion.** A pool voice that becomes a named individual later in the series (a Dryad who gets a name) is linked to the new character keeping its voice ID. The new character takes `voice_id = voice_40_2`, and the pool voice drops `member_of` in that book. Earlier books keep it as a member. This is a person's choice offered by the cast review, never automatic from a name.
- **Merges.** `merge-characters` refuses to merge a group into an individual or the reverse. Splitting is promotion, not merge.

### 5.7 Crowd murmur (a bed under a passage)

A **murmur bed** is the group's voices talking quietly underneath a passage: under one clear speaker (the Seers of Kell's chorus swelling behind Cyradis) or under narration ("a babble of small, squeaky voices", a hall full of guests). Its words are deliberately not intelligible. It is ambience, like a voice effect, not a line.

**Where it comes from (analysis).** A narration cue whose subject is a group making sustained, non-specific vocal sound proposes a span:
- cues: "murmur", "hubbub", "babble", "buzz of talk", "the crowd muttered", "the chorus swelled", "the chanting went on";
- the span starts at the cue and ends at an end cue ("fell silent", "the murmur died away"), a scene divider, or the chapter end, whichever comes first;
- a narration mention of a past or distant crowd ("the crowd had gone home") is not a cue;
- the subject rules are the ones `speech_cues` uses.

A group with no lines of its own (the Seers) gets a group cast entry with zero lines, so a person can cast it.

```jsonc
// analysis: chapter-local spans, partitioned into chapter scripts like segments
"murmurs": [{"group": "char_52",
             "start": {"block_id": "b004410", "offset": 0}, "end": {"block_id": "b004418", "offset": 212},
             "shape": "swell",             // steady | swell | fade | swell_fade
             "from_line": null,            // optional segment ID: the bed repeats that line (below)
             "basis": "explicit", "evidence": [{"block_id": "b004410", "quote": "The vast chorus of the Seers of Kell swelled"}]}]
```

**What the bed is made of.** The bed is made from verified takes, so nothing unchecked reaches the audio:
- **Murmur takes per member voice** (*walla* in film sound). Each member speaks a few short conversational sentences with a "talking quietly to a neighbour" direction. These are group assets like voice references (`purpose="murmur"`). They are verified as ordinary takes, rendered once per voice without chapter history, cached and carried through the series with the voice. Adding a member renders only that member's takes.
- **Text.** The sentences come from a small project-owned, language-level set of ordinary conversational sentences. It is generic and versioned, never book-specific. They are not book text, so they must stay unintelligible (below).
- **`from_line`.** The bed repeats a real line from the book that the text says many people said (the guests' "It never snows in Tol Honeth", the fenlings' "Mother, mother, mother"). Members speak that line, each take verified against it. Fragments that can be heard are then true to the book.

**How it is built (`renderer/murmur.py`, a closed versioned recipe like `voice_effects`):**
1. Tile each lane's murmur takes with seeded gaps and offsets, so that 4–8 talkers overlap at every moment (lanes = the group's voices, at most 8).
2. Low-pass at about 3 kHz, add light room reverb, and decorrelate lanes.
3. Shape the envelope (`swell`/`fade`).
4. Set the level well under speech (about −22 dB relative to the foreground by default), ducking 4–6 dB further under every foreground take.
5. Sum the bed under the chapter's composed audio without moving a single frame. Takes, verification, censorship, history and `sync.json` timings are untouched. `sync.json` gains a `murmurs` record of the span's frames.

**Unintelligibility check** (generic text only). The processed bed alone is transcribed with the primary recognizer. If it recognizes a run of 3 or more words of the murmur text, the recipe adds talkers or filtering and checks again. If it still can, the bed fails closed: the passage renders without it, and the failure is reported like a rejected take. `from_line` beds skip this check, because their words are the book's.

**Identity and caching.** The chapter identity includes the recipe version, the murmur take hashes and the span frames. Changing a bed re-composes that chapter only, with no new synthesis of lines.

**Review.**
- **Cast screen:** audition a 10-second bed for the group and set its default level.
- **`review-ui`:** a passage shows "murmur: Seers of Kell, swell". A person can remove the span, move its ends, change the shape or level, or add a span to any passage. Edits save through scope `attribution` with `basis: "user"`.
- Analysis proposals render only once the group's voices are cast. Until then the span is listed and skipped, never blocking.

## 6. Making sure it is used properly

### 6.1 Guards in analysis (`self_check`, `cast_audit`, rerun review)

| Guard | Detects | Real example |
|---|---|---|
| plural tag → ordinary role | a line whose tag subject is plural ("the Murgos replied in unison", "they all replied", "several of them") attributed to one non-group character | "As your Majesty commands" → burly guard (Mallorean 2) |
| singular tag → group role | a group's line whose tag subject is singular and not a group member ("the voice in his mind said", "she hissed") | "Be quiet" → Arendish nobles (Belgariad 2) |
| invented identity inside a pair | "one of them" / "the other" lines given to specific named twins with no evidence telling them apart | Beltira/Belkira half-lines (Belgariad 3) |
| name-only group | a group entry whose only plural evidence is its name ("Chorus", "Voices") | — |
| oversized subset | a `members` subset with no "half / some of them" evidence | — |

Each guard produces a flagged line for the rerun group check or attribution recheck, never a silent fix.

### 6.2 Readiness before render (`renderer/ensemble_readiness.py`, next to `accent_readiness`)

Rendering is blocked, with a message naming the group and what is missing, when:

- an ensemble member has no accepted voice, or a pool voice has no accent;
- a lane mode resolves to fewer than 2 members;
- a per-line `members` override names a voice outside the ensemble;
- `pool`/`alternate` cannot resolve a member.

An unconfirmed proposal (a `group` whose voice has no `ensemble`) does **not** block. Until a person confirms it in the cast screen, the group renders in its own voice (today's behaviour), and `plan`/`render` list "N groups proposed, not confirmed". That matters because pool voices are new voices, and only a person creates voices. Confirming is what makes lane modes and pool voices take effect. Declining records `ensemble: {members: [], mode: "single", basis: "user"}` (the one case with no further members) and is never proposed again. Analysis never creates pool voice records itself, only the proposal (size, mode, cues).

### 6.3 Verification

- **Every lane** is verified as an ordinary take, with the same gates, `acceptance_gate.refusal`, history-echo checks and chosen pronunciations.
- **Exhaustion selection** may choose per lane under its usual audited policy. A mix is built only from lanes that are each accepted.
- **ASR on the mix** is never used as proof (as today).
- **The mixer** refuses lanes whose hashes changed and silent mixtures (as today). Each recipe also checks that the mix keeps every lane's frames.
- **Murmur takes** pass ordinary verification. A bed built from generic text must also *fail* recognition (§5.7), so no invented words can be heard.

### 6.4 What stays a person's decision

- whether a group sounds collective or `single`;
- the number of voices (analysis only proposes it) and the choice of members;
- creating or designing pool voices;
- the mode and mix recipe;
- whether a passage gets a murmur bed, and its level;
- per-line overrides beyond what the source states;
- promotion of a pool voice to a named character.

Analysis may *propose* group identity and per-line cues from evidence, just as it writes `simultaneous_speakers` today. Agents never confirm a proposal, create performers, or set `basis: "user"`.

## 7. How the survey examples would come out

- **Twins (Belgariad 3).**
  - Group "the twins" with members Beltira and Belkira, default `unison` (tight).
  - The "one of them … the other finished" lines become `alternate` on the group role, so the invented split of the half-lines disappears.
  - Lines with a named twin stay with that twin.
- **Eunuch chorus.**
  - "two dozen shaven-headed men" proposes 8 separately designed voices with `unison` (loose); a person may choose fewer.
  - The crimson-robed leader stays an individual, which gives call and response.
  - "half of the kneeling eunuchs murmured" becomes a `lanes: 4` line.
- **Dryad girls.** A 4-voice female pool in `pool` mode with slots: "one of the others" and "another one" get different Dryads, and "the Dryad offered" keeps the same one.
- **Legions chanting "Ce-Ne-dra!"** `crowd` scale, 8 voices, `staggered` (swell). "another man roared" becomes a `pool` line. "several of them answered" gets `lanes: 3`.
- **Fenlings and birds.** `staggered` (babble) with small, high pool voices.
- **Switcher hive mind.** Its default stays `single`. The "parroted back" line is `staggered` (echo), plus a `pa`/`helmet`-style effect if a person picks one.
- **Andel and the Dals.** An optional `layered` group led by Andel's voice. Without a person's choice, Andel stays an individual.
- **Seers of Kell behind Cyradis.** A zero-line group cast entry, plus a `swell` murmur span from "The vast chorus … swelled" to the end of the scene. Cyradis stays clear on top.
- **Guests at the reception.** A `from_line` murmur: members repeat "It never snows in Tol Honeth" quietly under the passage.
- **Assembled Alorns after "Hail Belgarion!"** The shout is `unison` (crowd), and the hall's noise that follows ("The great shout shook the Hall") gets a `fade` murmur.
- **Arendish nobles, Ulgo elders.** `single`. The guard moves the misattributed "dry voice" lines back.
- **"Lift!", "Sorry!"** Unchanged `simultaneous_speakers`. "Guild do what?" gets `simultaneous_mode: staggered`.

## 8. Suggested phases

1. **Guards and mix fixes.** The §6.1 guards as flags only, plus onset alignment and loudness-matched mixing for existing `simultaneous_speakers` (re-mixes those lines; takes are reused).
2. **Format, renderer and cast TUI.** Ensemble voices, every mode, recipes, readiness, plan output, the cast ensemble screen and audition, and separately designed pool voices (batched casting).
3. **Crowd murmur.** Murmur takes, the bed recipe, the unintelligibility check, and manual spans in `review-ui`.
4. **Review tools.** `review-ui` per-line overrides, run-shell per-lane takes, and `simultaneous_mode`.
5. **Analysis proposals.** Ledger group entries, scale and voice-count proposals, line cues, `lanes` hints and slots, murmur spans, the rerun group check, and prompt version bumps.
6. **Series.** Carrying, nearest-book edits and promotion.

## 9. Decisions and open questions

**Decided** (your answers, 2026-11-19):
- **Pool voices are designed separately**, never cloned from a sibling (§5.5).
- **Every mix is loudness-matched**, including existing `simultaneous_speakers` lines (§5.4).
- **Up to 8 lanes.** Analysis proposes how many voices a group gets from the source's words and numbers, with per-line `lanes` hints; the cast screen shows the proposal, and a person chooses (§5.2 item 6, §5.5).
- **Crowd murmur is in scope** (§5.7).
- **Unconfirmed proposals never block render.** They render as today and are listed (§6.2).

**Still open** (each with a proposed answer):
1. **Tight unison.** Is a small pitch-preserving tempo match of lanes (±8 %) acceptable for "perfectly in unison" pairs? The verified dry takes would stay unchanged, but the mixer's "no time stretching" rule would be relaxed for this recipe only.
   *Proposed: not in the first version.* Onset alignment alone may be enough. Decide after listening to `tight` on the twins.
2. **Lane history.** Should unison lanes ever join their own voice's history? Today they do not, because chunk ordering would become nondeterministic.
   *Proposed: no.* Members are heard most in their solo lines.
3. **Doubling.** For a crowd, should a member also be rendered a second time with another seed, to sound like more than 8 people?
   *Proposed: no.* 8 distinct voices plus the `crowd` spread should be enough.
4. **Murmur text.** Should beds built from generic text use the project-owned conversational set (proposed), or only `from_line` book lines, so a bed never contains words that aren't in the book?
   *Proposed: the generic set, guarded by the unintelligibility check.* Most crowds in the survey have no line to repeat.
5. **Murmur level.** Is about −22 dB under speech, with 4–6 dB of ducking, the right default?
   *Proposed: yes, adjustable per group and per span.*
6. **Partial echoes** (only the last words repeated). Worth new spoken-text spans, or keep "every lane says the whole line"?
   *Proposed: whole line only.* It is rare in the survey (one hive-mind passage).
7. **Legend register** ("the elders said"). Should analysis propose a collective sound there?
   *Proposed: only when the tag itself says they speak together* ("in unison", "chorused").

## 10. Implementation status (branch `claude/collective-voices`)

Every phase in §8 is implemented:
- **Mixing:** lane recipes `tight/loose/crowd/swell/babble/echo/lead` (`renderer/lane_mix.py`), onset alignment, loudness matching under a peak ceiling, censorship over offset lanes. Version-1 mixes restore exactly and are re-mixed from cached takes on the next render.
- **Format and renderer:**
  - fields `group`, `ensemble`, `member_of`, segment `ensemble`, segment `murmur`, `simultaneous_mode` (`ensembles.py`, `FORMAT.md`);
  - `renderer/ensemble_plan.py` resolves every mode in `make_plan`, with performers in chunk identity;
  - the chapter service checks resolutions without the cast.
- **Casting and readiness:**
  - casting offers and auditions member voices, and pool voices follow the group's accent;
  - `plan` reports `collective_voices`, including murmurs and the groups they wait for, and `render` names uncast proposals.
- **Cast screen `C`:**
  - set-up choices: first the group's own proposed voices in one step, else voice count, mode, mix, and whether named cast members are among them; decline;
  - previews of a line in all voices and of the murmur;
  - published through revision scope `ensembles`, with new pool voices described separately;
  - a pool voice a named member keeps is shared with `d`.
- **review-ui:**
  - `v`: a group line's mode, voices or pinned member, and `simultaneous_mode`;
  - `m`: add, end, shape, level or remove a murmur;
  - takes: each voice's takes listed by name.
- **Crowd murmur** (§5.7, `renderer/murmur.py`):
  - verified murmur takes from a project-owned sentence set, or the repeated book line;
  - one filtered loop per group that must fail recognition (three heard words in a row → filtered harder → refused);
  - mixed under the span, ducked under speech, in the chapter identity;
  - local composition only.
- **Guards** (§6.1): `group_cues.py` flags `plural_tag`, `singular_tag` and `invented_member`. They appear as narrator-read issues and in `verify-attribution`'s risk set.
- **Analysis proposals** (§5.2), made from tags without a model call (`group_proposals.py`):
  - group, size, scale, voice count, mode;
  - line `lanes`, `mode` and `slot`;
  - murmur spans from narration cues.

  They run in fresh narrator reads, and on rerun through scope `group-proposals`. `group_check.py` asks the model about characters with a single plural tag; "y" needs an exact quote.
- **Series:** ensemble, member and group IDs are remapped; a later book carries an inherited ensemble's pool voices.

Differs from the plan above:
- Proposals come from speaker tags and narration cues in code (like `speech_cues`). The model is asked only about unclear characters (`GROUP_CHECK`), rather than through a ledger `kind: group`.
- Promotion of a pool voice to a named character uses voice sharing (`d`), not a format change. A later book carries pool voices only when the group itself returns.
- Murmur beds are mixed in local chapter composition; a remote chapter service renders without them and says so.
- The plain (curses) `review-ui` shows group and murmur details but changes them only in the full-screen view.
