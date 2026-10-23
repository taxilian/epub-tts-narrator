# OVERVIEW

VENDORED third-party unofficial MLX port of Fish Audio Drama 3 TTS.
Upstream: https://github.com/Sogni-AI/drama3-mlx @ 3b76bf8, v0.1.1.
Provides the local Mac synthesis engine used by the renderer's default drama3_mlx engine.

Keep this file current: if the vendored snapshot or the local pyproject diff changes, update it in the same commit.

# HARD RULES

1. NEVER edit vendored runtime files: .py files, LICENSE, NOTICE, or README.
   Rule 1 is the primary guardrail. This tree is vendor code, not application code.
2. Make changes only through the PROVENANCE.md reproduction process.
   Then update UPSTREAM_SHA256SUMS and verify with:
   shasum -a 256 -c UPSTREAM_SHA256SUMS
3. Only pyproject.toml carries local modifications.
   Current local change: 3 lines, including version 0.1.1+epub.1,
   mlx 0.32.4, and mlx-audio >=0.6.1,<0.7.
   Keep the +epub.N local version suffix for any local version change.
4. Do not format, lint, or "fix" this tree with parent-project tooling.
5. Parent-project conventions do not apply here. Upstream style governs.

# VERIFY

Run from this directory:
shasum -a 256 -c UPSTREAM_SHA256SUMS

# REFERENCES

- PROVENANCE.md: authoritative provenance, diff, and reproduction process.
- UPSTREAM_SHA256SUMS: integrity manifest for unmodified upstream runtime files.
- Upstream: https://github.com/Sogni-AI/drama3-mlx
