import sys

if sys.version_info < (3, 11):
    sys.exit("epub-tts-narrator requires Python 3.11 or newer.")

from .cli import main

raise SystemExit(main())
