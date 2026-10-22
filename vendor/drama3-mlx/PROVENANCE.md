# Vendored Sogni Drama 3 runtime

This is the Sogni Drama 3 runtime with dependency metadata adjusted for this
project's shared MLX-Audio environment. All Python runtime files, `README.md`,
`LICENSE`, and `NOTICE` are unchanged from upstream. The upstream README is
retained as source documentation; use this project's renderer setup instructions
to install the shared environment.

- Upstream: <https://github.com/Sogni-AI/drama3-mlx>
- Revision: `3b76bf84714d3766d7872f3409639a438e296cc8`
- Upstream package version: `0.1.1`
- Local package version: `0.1.1+epub.1`
- Archive: <https://codeload.github.com/Sogni-AI/drama3-mlx/tar.gz/3b76bf84714d3766d7872f3409639a438e296cc8>
- Downloaded archive SHA-256:
  `e01c17a89a0e64401db7c6fd15ba9ae63fd5be39577c4644824644b30a2769aa`

The local version distinguishes this package from the original distribution.
Normal pip and uv dependency resolution sees the adjusted requirements; no
dependency checks are bypassed.

## Metadata changes

Only these entries in upstream `pyproject.toml` differ:

```diff
-version = "0.1.1"
+version = "0.1.1+epub.1"
-    "mlx==0.32.3",
+    "mlx==0.32.4",
-    "mlx-audio==0.5.7",
+    "mlx-audio>=0.6.1,<0.7",
```

The `mlx-lm==0.31.3` pin and all other upstream dependencies remain unchanged.
The root renderer requirements select the tested MLX-Audio version within this
range. `MANIFEST.in` includes this provenance and `UPSTREAM_SHA256SUMS` in source
distributions. No model weights or generated assets are included.

## Reproduction and verification

Download the archive URL above and verify its SHA-256 before extracting it.
Retain only `drama3_mlx/*.py`, `pyproject.toml`, `README.md`, `LICENSE`, and
`NOTICE`; apply the three metadata replacements above. Add this provenance,
the file checksum list, and the two-line `MANIFEST.in` that includes them.

The checksum list covers every retained upstream file except the deliberately
modified `pyproject.toml`. Run from this directory to verify those files:

```sh
shasum -a 256 -c UPSTREAM_SHA256SUMS
```

Build a normal wheel or source distribution using Python's build frontend:

```sh
python -m build .
```

Install through the project's `renderer.requirements.txt` so that the renderer,
Sogni, and MLX-Audio use the same dependency set. The original upstream license
and attribution remain in `LICENSE` and `NOTICE`.
