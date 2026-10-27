# Bundled CMU Pronouncing Dictionary

Upstream: https://github.com/cmusphinx/cmudict
Revision: `74790861f652b15e4ac49015a90074ad62a27690`

Exact source: https://raw.githubusercontent.com/cmusphinx/cmudict/74790861f652b15e4ac49015a90074ad62a27690/cmudict.dict

Uncompressed SHA-256: `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`
Uncompressed size: 3,618,488 bytes; 135,166 lines.

`cmudict.dict.gz` contains the complete, unmodified source bytes, compressed
with Python stdlib `gzip.GzipFile(filename="", mode="wb", compresslevel=9,
mtime=0)`. Download the revision-pinned source over HTTPS, verify its size and
SHA-256 above, then compress it. There is no runtime download.

`CMUDICT-LICENSE` is the unmodified upstream LICENSE from the same revision:
https://raw.githubusercontent.com/cmusphinx/cmudict/74790861f652b15e4ac49015a90074ad62a27690/LICENSE

LICENSE SHA-256: `bd4ce8e44170a5f9f481310ca85c51de3c4f851a65e679b40e603b143bd3542a`

The loader strips inline `#` comments and numeric alternate-reading suffixes
such as `(2)`, retaining every pronunciation in source order. Lookup is
case-insensitive only; book word identities and replacements remain exact-case.
Dictionary readings are suggestions, never approval or acoustic evidence.
