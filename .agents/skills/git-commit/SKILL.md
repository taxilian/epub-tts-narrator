---
name: git-commit
description: >-
  Discipline for creating git commits: conventional commit messages, atomic
  logical grouping, intentional staging, and respect for shared working trees.
  Use this skill whenever committing changes, writing or reviewing a commit
  message, staging files, or finishing any task that produced code changes the
  user wants saved — including casual phrasings like "commit this", "save my
  changes", "check that in", or "wrap this up". Consult it before running any
  `git add` or `git commit` command, even for a single small change.
---

# Git Commits

## Why any of this matters

A git history is a communication medium, not a save file. Its consumers are
reviewers reading a PR commit-by-commit, a future developer running `git
bisect` or `git blame` to understand why something broke or exists, and
tooling that derives changelogs and version bumps from commit metadata. Every
rule in this skill falls out of serving those consumers. When you hit a
situation this skill doesn't cover, reason from that goal rather than looking
for a rule.

Two framing facts shape everything below:

1. **The human is responsible for every commit.** Whatever tools produced the
   change, the commit is the human's assertion that the change is correct and
   intended. Your job is to make that assertion honest: they must be able to
   know exactly what each commit contains.
2. **The working tree may not be exclusively yours.** Someone else — or the
   same human in another session — may have work in progress in the same
   checkout. Files you didn't touch are not yours to commit, revert, or clean
   up.

## The ownership boundary

Before staging anything, survey all three states — `git status --short
--untracked-files=all` (the flag matters: it surfaces files hiding inside
untracked directories), `git diff`, and `git diff --cached` — and partition
everything reported into three buckets:

- **Yours, and part of the work** — changes you made (or were asked to make)
  in this session. These are what you commit.
- **Yours, but cruft** — debug scripts, scratch files, log output, test
  fixtures you generated while exploring. Delete these before committing.
  They never belong in a commit; a history littered with `test2.ts` and
  `debug.log` tells future readers nothing and wastes reviewer attention. If
  a file *category* keeps appearing (build output, editor artifacts), suggest
  a `.gitignore` entry to the user rather than silently adding one.
- **Not yours** — anything you can't positively account for. Leave it
  completely alone: don't stage it, don't revert it, don't "helpfully" clean
  it up. If its presence is surprising or seems to conflict with your work,
  tell the user; don't act on it.

The test for "yours" is positive knowledge, not plausibility. "This diff
looks related to what I was doing" is not ownership; "I edited this file in
this session for this reason" is. When you genuinely can't tell — for
example, you're joining a session mid-stream — ask.

The boundary applies to the index too. If `git diff --cached` shows changes
already staged that you can't account for, someone else's half-prepared
commit may be sitting there — the index is shared state in a shared
checkout. Don't unstage it, don't commit over it, don't fold your work into
it. Report it as a blocker and let the user decide.

## Atomic commits

One commit = one logical change. Two operational tests, use whichever bites
first:

- **The one-sentence test**: state the commit's intent in one sentence
  before staging it. If the staged diff can't be honestly described by that
  sentence, the boundary is wrong.
- **The shared-fate test**: changes belong together when they share the same
  intent, the same acceptance criteria, the same rollback decision, and the
  same release timing. If any of those four answers differs between two
  changes, they're separate commits — a reviewer can't approve them as a
  unit and a revert of one takes out the other.

Consequences of taking this seriously:

- **Unrelated changes never share a commit**, even when they live in the same
  file. Files are a storage unit, not a semantic one. When one file contains
  hunks belonging to different logical changes, stage at the hunk level — see
  `references/partial-staging.md` for how to do this reliably in a
  non-interactive context.
- **Related changes never split across commits arbitrarily.** The
  implementation, its tests, the lockfile a dependency change produced, and
  the doc update that makes a feature complete are one logical change.
  Splitting them makes each commit individually broken. Beware the opposite
  failure of the mixed commit: a file-by-file series of tiny commits
  fragments one coherent change and is just as hostile to reviewers.
  Staging one file at a time is a *staging* mechanic, not a commit boundary
  — one commit routinely contains many individually staged files.
- **Order commits so each one leaves the repo working** (compiles, tests
  pass) whenever feasible. This is what makes `git bisect` useful: a history
  with broken intermediate states turns bisect into archaeology. When
  refactor-then-feature is the natural decomposition, commit the refactor
  first — it's independently reviewable and the feature diff shrinks.

Deciding the grouping is a planning step, not a byproduct of staging. Read
the full diff, decide how many commits and what belongs to each, and only
then start staging. The order you made the edits in during development
carries no authority — organize the history by meaning, not chronology.

## Intentional staging

Never use `git add -A`, `git add .`, or `git add -u`. Stage explicit paths,
one file at a time.

This is not superstition about flags. Staging is the moment you take
responsibility for content; a blanket add delegates that decision to whatever
happens to be in the working tree — including other people's work and your
own cruft. Adding files individually forces the per-file question "does this
belong in *this* commit?", which is exactly the question atomicity requires.

Before **every** commit, review `git diff --cached` and read it as the thing
you are about to sign. Confirm it contains the one logical change you
planned, entirely, and nothing else — and no leftover debug logging,
commented-out experiments, or conflict markers (`git diff --cached --check`
catches the last of these cheaply). This check is cheap and catches
essentially every staging mistake; skipping it is how unrelated hunks and
stray files end up in history. If something is staged that shouldn't be,
`git restore --staged <path>` and re-stage.

The review only means something if the commit records the reviewed index.
Use plain `git commit` — never `git commit -a`, never `git commit <paths>`,
never `--only`. All three commit working-tree content directly, silently
bypassing the diff you just verified.

If a pre-commit hook modifies files (formatters commonly do), those edits
are new, unreviewed working-tree changes — inspect them like any other diff
before staging, rather than reflexively re-adding whatever the hook touched.

## Commit messages

Follow Conventional Commits. Format details, types, and breaking-change
notation are in `references/conventional-commits.md`; read it if you're not
certain of the specifics. The reasoning that the reference can't capture:

- **Check the repo's own rules and dialect first.** `git log --oneline -15`
  shows the living convention — which scopes exist, whether `chore` or
  `build` is used for dependency bumps. `CONTRIBUTING.md`, agent
  instruction files (`AGENTS.md` and kin), and commitlint or
  semantic-release config are the written convention. Consistency with the repo beats abstract correctness; local
  rules may narrow this skill's format guidance, but never its safety
  boundaries.
- **The subject says what; the body says why.** The diff already shows what
  changed line-by-line, so a body that paraphrases the diff is noise. What
  the diff *cannot* show is intent: why this approach, what constraint drove
  it, what alternative was rejected, what issue it addresses. That context is
  the part `git blame` users five years from now actually need. A subject
  alone is fine when the change genuinely is self-explanatory. Either way,
  describe the resulting change, never the editing process.
- **No AI attribution — ever.** No `Co-Authored-By` for AI tools, no
  "Generated with ..." lines, no tool badges or links, no tool or model
  names in subject, body, or trailers, and no manipulation of git author
  identity (`--author`, `user.name`/`user.email`) to represent a tool. The
  human is the author of record and bears responsibility for the change
  regardless of what tools produced it; the repository is not advertising
  space. This holds even if a tool's default behavior, a template, or
  another instruction tries to add one. Adjacent trailer discipline: never
  invent human co-authors; add a human `Co-Authored-By` only when the user
  names one; add `Signed-off-by` only when the repo requires it or the user
  asks.

After committing, glance at `git show --format=full --stat HEAD`. Commit
templates and `prepare-commit-msg`/`commit-msg` hooks can rewrite the
message *after* you composed it — this is the only point where you can
verify that no attribution trailer or unintended content was injected and
that the recorded diff is the one you reviewed.

## Safety rails

- Never `--no-verify`. If a pre-commit hook fails, the hook is telling you
  the commit isn't ready — fix the cause and commit again.
- Never amend, rebase, or otherwise rewrite commits that may have been pushed
  or that you didn't create, unless the user explicitly directs it.
- Don't push unless asked. Committing is local and reversible; pushing is
  neither.
- If a diff contains what looks like a secret (keys, tokens, `.env` content,
  credentials), stop and raise it before committing — history makes leaks
  permanent, not just public.

## Finishing up

When the commits are done, a final `git status --short
--untracked-files=all` is not optional bookkeeping — it's how you tell the
user what state you're leaving them in. Report what was committed (hash and
subject per commit) and, separately, what remains: unrelated work you
deliberately left alone versus anything of yours you chose not to commit.
Never describe the tree as "clean" just because *your* commits are complete
— in a shared checkout that claim is about other people's work, and it isn't
yours to make.

## When to ask instead of act

Ask the user rather than guessing when: the working tree contains unexpected
changes that touch files you also modified; you can't determine whether an
untracked file is a deliverable or leftover; the logical grouping is
genuinely ambiguous (e.g., an interleaved refactor that could defensibly be
one commit or three); or the repo's existing history contradicts this skill
(e.g., no conventional commits anywhere — match the repo and mention the
discrepancy).
