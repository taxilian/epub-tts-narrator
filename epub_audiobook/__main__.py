import sys

if sys.version_info < (3, 11):
    sys.exit("epub-audiobook requires Python 3.11 or newer.")

from .cli import main

raise SystemExit(main())
