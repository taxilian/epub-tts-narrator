---
name: fish-phoneme-tags
description: Write CMUdict-style ARPAbet for Fish Audio phoneme_start/phoneme_end English phoneme tags, transcribing any word (rare, names, brands, acronyms, call signs, loanwords, made-up) with stress digits. Use when preparing Fish TTS text (Drama 3, S2, S1) with likely mispronunciations, fixing a Fish mispronunciation, or asked for ARPAbet/CMU phonemes, even if Fish isn't named.
---

# Fish Audio English phoneme tags

`I am an <|phoneme_start|>EH1 N JH AH0 N IH1 R<|phoneme_end|>.`

A tag replaces one written word. The content must be CMU ARPAbet written exactly as CMUdict writes it, because the model was trained on CMUdict. Nothing validates the tag: a typo, a lowercase symbol or an IPA character silently degrades the audio.

## Tag rules

- **One written word per tag,** suffix included: `engineers` → `EH1 N JH AH0 N IH1 R Z`, `don't` → `D OW1 N T`. Two neighboring words get two tags.
- **Inside:** only uppercase symbols and single spaces. Punctuation and cue tags stay outside. Spell the markers exactly; keep them paired, never nested.
- **Digits:** every vowel ends in `0`, `1` or `2`; consonants never carry a digit. Normally exactly one `1`.
- **Tag only what the voice would get wrong:** names, brands, acronyms, jargon, homographs (read `R IY1 D` / `R EH1 D`) and invented words.
- **Numbers:** write them as words in plain text; local Fish runtimes don't expand digits.
- **Sentence length:** keep sentences that contain tags under ~250 bytes, or break them with commas. Drama 3's local chunker can otherwise split a tag in half.

## Symbols (only these 39)

| Vowel | As in | Vowel | As in |
|---|---|---|---|
| `AA` | father, cot | `EY` | say |
| `AE` | cat | `IH` | bit |
| `AH` | cut (`AH1`), schwa (`AH0`) | `IY` | bee |
| `AO` | caught, for, dog | `OW` | go |
| `AW` | cow | `OY` | boy |
| `AY` | my | `UH` | book |
| `EH` | bed | `UW` | food |
| `ER` | bird, butter | | |

Consonants: `B CH D DH`(this) `F G HH JH`(judge) `K L M N NG`(sing) `P R S SH T TH`(thin) `V W Y`(yes) `Z ZH`(measure).

Never use `AX`, `IX`, `AXR`, `DX`, `EL`, `EN`, `EM`, `NX`, `UX` or `WH`. Write `AH0`, `IH0`, `ER0`, `T`/`D`, `AH0 L`, `AH0 N`, `AH0 M` instead, and drop glottal stops.

## Priorities

Get these right, in order:

1. Every sound present and correct.
2. Which syllable carries the `1`.
3. The stressed vowel.

Don't fuss over AH0 vs IH0, `0` vs `2` after the main stress, or AA vs AO in caught/coffee words. CMUdict itself is inconsistent on all three.

For a known word, write the CMUdict entry, then check it against the style below. If you can run code, look it up instead: `pip install cmudict`, then `cmudict.dict()[word.lower()][0]`. Either way, apply the traps.

## CMUdict style

### Stress

- One-syllable words take `1` (`of AH1 V`).
- When the primary falls on the 3rd syllable or later, put a `2` on the first syllable or the one two before the primary. The syllable just before the primary stays reduced: `operation AA2 P ER0 EY1 SH AH0 N`, `Kubernetes K UW2 B ER0 N EH1 T IY0 Z`.
- Full vowels after the primary take `2`:
  - `-ize AY2 Z`, `-ify F AY2`, verb `-ate EY2 T`
  - `-ship IH2 P`, `-hood UH2 D`, `-ide`/`-ite` `AY2`
  - the second part of a compound (`football F UH1 T B AO2 L`)
- `-ary` and `-ory`:
  - `EH2 R IY0` / `AO2 R IY0` when an unstressed syllable separates them from the primary (`military M IH1 L AH0 T EH2 R IY0`)
  - otherwise `ER0 IY0` (`boundary`)
  - `-ery` is almost always `ER0 IY0`.
- Prefixes before a later stress:
  - in-, im-: `IH2`, even in `invade IH2 N V EY1 D`
  - un-: `AH0` right before the stress, `AH2` when the stress comes two or more syllables later (untraditional)
  - con-, com-: `K AH0`
  - de-, dis-, be-: `IH0`
  - ex-: `IH0 K S`, or `IH0 G Z` before a vowel
  - re-, pre-: `IY0` or `IH0`
  - pro-: `P R AH0`
  - trans-: `T R AE0`
- Inflections and -ness, -less, -ful, -ly, -ment, -ism, -able never move stress.

### Unstressed vowels

Unstressed vowels reduce, and a vowel next to the primary stress almost always does. A full vowel marked `0` is rare outside final -o, trans- and names. The default is `AH0`. The exceptions:

- **`IH0`:** -ing, -ic, -ish, -ive; -ed after t/d; -es after s/z/sh/ch/j sounds; word-initial e- and i-.
- **`ER0`:** any vowel + r.
- **`IY0`:** final -y, -ie, -ey, -i, and i or e before a vowel (`-ia IY0 AH0`, `-ious IY0 AH0 S`).
- **`OW0`:** final -o.

-edly and -edness keep the -ed as a syllable: `allegedly AH0 L EH1 JH AH0 D L IY0`.

### Endings

| Ending | Phonemes |
|---|---|
| -tion | `SH AH0 N` |
| -sion after a vowel | `ZH AH0 N` |
| -cial, -tial | `SH AH0 L` |
| -cious | `SH AH0 S` |
| -ture | `CH ER0` |
| -sure after a vowel | `ZH ER0` |
| -able, -ible | `AH0 B AH0 L` |
| -ism | `IH2 Z AH0 M` |
| -ity | `AH0 T IY0` |
| -ology | `AA1 L AH0 JH IY0` |

Plural -s and past -ed: `S`/`T` after a voiceless sound, `Z`/`D` after a voiced one, `IH0 Z` after s/z/sh/ch/j sounds, `IH0 D` after t/d.

### Consonant habits

- **No syllabic consonants, flaps or glottal stops:** `bottle B AA1 T AH0 L`, `button B AH1 T AH0 N`, `butter B AH1 T ER0`.
- **Doubled letters are one sound,** except across a prefix (`unknown AH0 N N OW1 N`). Silent letters are dropped.

### Vowels before R

- **`ER`:** her, hurry `HH ER1 IY0`. Never write `ER R`.
- **`AA R`:** car.
- **`AO R`:** for, more, four, story.
- **`EH R`:** care, very, marry.
- **`IH R`:** near, hero.
- **`UH R`:** tour, sure; `Y UH R` in cure, pure.
- **R after a diphthong:** fire `F AY1 ER0`, hour `AW1 ER0`.

### AO vs AA

- **`AO`:** au/aw, al + consonant, all, wa + l or t, war, and o before ng, ff, ss, st, th or g (long, off, lost, dog).
- **`AA`:** other short o (hot, stock), father, calm, wash/watch/want, and foreign a (drama, pasta).
- **Unsure:** `AA`.

### The u sounds

- **`Y UW`** at the start of a word and after P, B, M, F, V, K, G or HH (use, music, cute).
- **`UW`** in a stressed syllable after T, D, N, S, Z, L or TH (new `N UW1`, student).
- **Unstressed u after L or N keeps the Y:** value `V AE1 L Y UW0`, regular `R EH1 G Y AH0 L ER0`, failure `F EY1 L Y ER0`.

### Other consonants

- **ng / nk:**
  - n + k or g inside a root is `NG` (think `TH IH1 NG K`); across a prefix it stays `N` (include).
  - ng + vowel inside a root is `NG G` (finger, longer), but singer is just `NG`.
- **wh** is `W`.
- **x** is `K S`; `G Z` in ex + vowel; `Z` at the start of a word.
- **ch** is `CH`; `K` in Greek words (school); `SH` in French words (chef).
- **th** is `TH`; `DH` in function words, -ther and -the.
- **Palatalized sounds:** tu → `CH`, du → `JH`, ti/ci/si + vowel → `SH`, si/su after a vowel → `ZH`.
- **Letter pairs don't span morpheme boundaries:** lighthouse `T HH`.

### Compounds

The second part's `1` becomes `2`. Some second parts reduce instead: -man, -ton, -son `AH0 N`; -ford, -ward `ER0 D`; -berg `B ER0 G`.

## Traps

Override what CMUdict lists when:

- **It's the wrong sense.**
  - coax, the cable: `K OW1 AE2 K S`
  - sake, the drink: `S AA1 K IY0`
  - ICAO radio words: Kilo `K IY1 L OW0`, Lima `L IY1 M AH0`, Papa `P AH0 P AA1`, Quebec `K EH0 B EH1 K`
  - the letter A: `EY1`, not the article `AH0`
- **It's a foreign name read as English spelling:** Goethe `G ER1 T AH0`, not `G OW1 TH`.
- **The owner has stated a pronunciation:** nginx `EH2 N JH AH0 N EH1 K S`, GNU `G N UW1`.

## Invented and unfamiliar words

If the word, or a relative of it, already sounds a certain way to you, trust that. Otherwise write these steps out:

1. **Split** off inflections, suffixes, prefixes and compound parts, and copy their forms from above.
2. **Name 2–3 real words** that share the root's spelling and ending, and borrow their stressed vowel.
3. **Stress.** The rightmost telling suffix decides:
   - On the suffix: -ee, -eer, -ese, -ette, -esque, -ique, -oon.
   - On the syllable just before it: -ic, -ical, -tion, -sion, -ial, -ian, -ious, -ia, -ium, -ity, -ify, -ible, -logy, -graphy (bi-OL-o-gy).
   - Two syllables before it: -ize, verb -ate (CAL-cu-late).

   With no telling suffix:
   - Two syllables: stress the first (nouns, names, invented words).
   - Longer: stress the second-to-last syllable if it is heavy, meaning a long vowel or closed by a consonant (a-GEN-da). Otherwise stress the third-to-last (CAN-a-da).
   - Words ending in a vowel that look Romance or Japanese: second-to-last (ba-NA-na).
4. **Stressed vowel,** checked in this order:
   - Digraphs: ee/ea `IY`, ai/ay `EY`, oa `OW`, oo `UW`, ou/ow `AW`, oi/oy `OY`, au/aw `AO`, igh `AY`.
   - A following r.
   - The wa- and al- effects above.
   - Short before -ic, -ity and -ify, and in a stressed third-to-last syllable.
   - Long before a consonant + silent e: a `EY`, e `IY`, i `AY`, o `OW`, u `Y UW`.
   - Short before two consonants: `AE EH IH AA AH`.
   - Before one consonant + another vowel, follow the nearest familiar word.
5. **Reduce** every unstressed vowel to `AH0`, `IH0`, `ER0` or `IY0` unless a rule above keeps it full. Check that there is one vowel sound per syllable and exactly one `1`.

Example: glorpify = glorp + -ify → `G L AO1 R P AH0 F AY2`.

## Acronyms and letters

| Letter | Phonemes | Letter | Phonemes | Letter | Phonemes |
|---|---|---|---|---|---|
| A | `EY1` | J | `JH EY1` | S | `EH1 S` |
| B | `B IY1` | K | `K EY1` | T | `T IY1` |
| C | `S IY1` | L | `EH1 L` | U | `Y UW1` |
| D | `D IY1` | M | `EH1 M` | V | `V IY1` |
| E | `IY1` | N | `EH1 N` | W | `D AH1 B AH0 L Y UW0` |
| F | `EH1 F` | O | `OW1` | X | `EH1 K S` |
| G | `JH IY1` | P | `P IY1` | Y | `W AY1` |
| H | `EY1 CH` | Q | `K Y UW1` | Z | `Z IY1` |
| I | `AY1` | R | `AA1 R` | | |

Only the phonemes go in the tag: FBI is `EH2 F B IY2 AY1`.

- **Word or letters:** say it as a word when that's the convention (NASA `N AE1 S AH0`, NATO, scuba, radar). Otherwise spell it out; when unsure, spell.
- **Stress when spelled:** the last letter takes `1`, the others `2`. A plural adds `Z`.
- **Call signs:** clearest as separate words in plain text. If one tag is unavoidable, use letter names and number words with the same `2 … 1` pattern.

## Names and foreign words

Use, in priority order: the owner's stated pronunciation, then a published IPA or respelling, then how an American reads it at first sight.

**Unknown surnames.**
- Trust your first-sight American reading, which works by analogy (Davidow like David). Don't run the invented-word steps on names.
- Use English letter values, plus a few habits:
  - Italian c or cc before e/i: `CH`
  - German ei `AY`, ie `IY`, sch `SH`
  - Mc: `M AH0 K`, with the stress on the root
- Initial w stays `W`, th stays `TH`, and unstressed vowels reduce normally.
- Common endings: -son `S AH0 N`, -berg `B ER0 G`, -ski `S K IY0` (-owski `AO1 F S K IY0`), -ini `IY1 N IY0`, -ez `EH0 Z`.

**Stress when unsure:**
- Spanish, Italian, Polish, Japanese: second-to-last syllable.
- German, Czech, Dutch, Nordic: first syllable.
- French: last syllable.

**Foreign sounds,** for famous names and loanwords only:

| Source sound | ARPAbet |
|---|---|
| French nasal vowel | vowel + `N` |
| ü | `UW` |
| ö / eu | `ER` |
| any r | `R` |
| German ch, Arabic kh | `K` |
| Spanish j | `HH` |
| ñ / gn | `N Y` |
| pinyin x, q, zh | `SH`, `CH`, `ZH` |
| Polish sz, cz, w | `SH`, `CH`, `V` |
| glottal stop | drop |

**From IPA:**
- `ˈ` puts `1` on the next vowel, `ˌ` puts `2`.
- ə `AH0`; ɚ/ɝ `ER`; ɑ/ɒ `AA`; ɔ `AO`; ʊ `UH`; oʊ/əʊ `OW`; ju `Y UW`.
- Syllabic l̩/n̩ become `AH0 L`/`AH0 N`; ɾ becomes `T` or `D`.
- From British IPA, restore every R the spelling shows.

## Check each tag

- The markers are exact.
- Only the 39 symbols appear, with a digit on every vowel and none on consonants.
- There is one `1` (spelled acronyms excepted).
- Nothing else is inside the tag, and punctuation comes after it.
