set positional-arguments

python := ".venv-renderer/bin/python"

default:
    @just --list

analyze *args:
    @"{{ python }}" -m epub_tts_narrator analyze "$@"

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
    @"{{ python }}" -m unittest discover -s tests -q "$@"
