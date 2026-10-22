# Partial staging — splitting a file's hunks across commits

Read this when one file contains hunks belonging to different logical
changes. The goal is to stage exactly the hunks that belong to the current
commit and leave the rest in the working tree for a later commit (or for
deletion, if they're cruft).

## Why not `git add -p`

`git add -p` is the canonical interactive tool, but it expects a human at a
terminal answering per-hunk prompts. Driving it by piping answers
(`printf 'y\nn\ny\n' | git add -p file`) requires predicting git's hunk
segmentation and ordering in advance, and a single misprediction silently
stages the wrong content. Don't rely on it. The patch-based approach below
achieves the same result deterministically, and every step is inspectable.

## Patch-based staging

The index accepts arbitrary patches, so build the patch you want and apply
it there:

1. Export the file's full diff:
   ```bash
   git diff -- path/to/file > /tmp/split.patch
   ```
2. Edit the patch to keep only the hunks that belong to this commit. Delete
   whole hunks (from one `@@` header to the next); keep the file header
   lines (`diff --git`, `index`, `---`, `+++`) intact. Don't edit inside a
   hunk unless you also fix its `@@` line counts — removing entire hunks
   needs no count adjustments, which is why whole-hunk granularity is the
   safe default.
3. Apply to the index only:
   ```bash
   git apply --cached /tmp/split.patch
   ```
4. Verify both sides:
   ```bash
   git diff --cached -- path/to/file   # exactly what this commit gets
   git diff -- path/to/file            # what remains for later
   ```
   The verification step is not optional — it's the whole reason this method
   is trustworthy.

## When changes land in the same hunk

Nearby changes get merged into one hunk because of the default 3 lines of
context. Before resorting to hunk surgery, regenerate the diff with less
context — this alone separates most "same hunk" cases:

```bash
git diff -U1 -- path/to/file > /tmp/split.patch
```

`git apply --cached` accepts these patches as-is. (`-U0` gives maximum
separation but requires `git apply --cached --unidiff-zero` and loses the
anchoring context that makes patch application robust — prefer `-U1`.)

If the two changes touch *literally adjacent or identical lines*, no context
setting separates them. Then either split the hunk manually in the patch —
duplicating the `@@` header and recomputing counts — or, often simpler and
less error-prone, temporarily edit the working file to its "first commit"
state, `git add` the whole file, then restore the remaining change. Judge
which is safer for the case at hand; verify with `git diff --cached` either
way.

## Recovering from staging mistakes

- Unstage a file: `git restore --staged path/to/file`
- Unstage everything *you staged this session*: prefer per-path unstaging
  over `git restore --staged .` — in a shared checkout the broad form could
  clobber index state that isn't yours (only ever use it when you've
  verified the entire index is your own work)
- If a partial application went sideways, unstage the file entirely and
  redo the patch — partial fixes to a wrong index state are harder to reason
  about than starting over.

## Partially staging a new (untracked) file

Untracked files don't appear in `git diff`, so the patch method can't see
them. `git add -N` (intent-to-add) registers the path without staging any
content; afterward the file's full contents show up in `git diff` as
ordinary hunks and the patch workflow above applies unchanged. This matters
when a scratch file evolved into a deliverable but still contains debug
sections that shouldn't ship.

## A note on `git stash`

Avoid stash-based splitting techniques (`git stash --keep-index` etc.) in a
shared working tree. Stashing sweeps up *all* working-tree changes,
including ones that aren't yours — the exact files the ownership boundary
says not to touch. The patch method never modifies the working tree at all,
which is why it's the right tool in this context.
