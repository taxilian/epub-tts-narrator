set positional-arguments

python := ".venv-renderer/bin/python"

default:
    @just --list

analyze *args:
    @"{{ python }}" -m epub_tts_narrator analyze "$@"

verify-attribution *args:
    @"{{ python }}" -m epub_tts_narrator verify-attribution "$@"

refresh-directions *args:
    @"{{ python }}" -m epub_tts_narrator refresh-directions "$@"

review-ui *args:
    @"{{ python }}" -m epub_tts_narrator review-ui "$@"

plan *args:
    @"{{ python }}" -m epub_tts_narrator plan "$@"

cast *args:
    @"{{ python }}" -m epub_tts_narrator.renderer cast "$@"

render *args:
    @"{{ python }}" -m epub_tts_narrator.renderer render "$@"

assemble *args:
    @"{{ python }}" -m epub_tts_narrator.renderer assemble "$@"

runpod-cleanup *args:
    @"{{ python }}" -m epub_tts_narrator.renderer runpod cleanup "$@"

test *args:
    @"{{ python }}" tests/run_detached.py "$@"

pronounce *args:
    @"{{ python }}" -m epub_tts_narrator.renderer pronounce "$@"

pronounce-name *args:
    @"{{ python }}" -m epub_tts_narrator.renderer pronounce "$@"

# BOOK --from OTHER_BOOK: reuse OTHER_BOOK's chosen voices for the same characters.
import-voices *args:
    @"{{ python }}" -m epub_tts_narrator.renderer import-voices "$@"

# BOOK FILE: write BOOK's confirmed pronunciations to a YAML file.
export-pronunciations book file *args:
    @book="$1"; file="$2"; shift 2; "{{ python }}" -m epub_tts_narrator.renderer pronounce "$book" --export "$file" "$@"

# BOOK FILE: confirm a YAML file's pronunciations in BOOK, then tag their uses.
import-pronunciations book file *args:
    @book="$1"; file="$2"; shift 2; "{{ python }}" -m epub_tts_narrator.renderer pronounce "$book" --import "$file" "$@"
