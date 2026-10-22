# Conventional Commits — format reference

Based on Conventional Commits v1.0.0. This file covers mechanics; the
reasoning for message content lives in SKILL.md.

## Structure

```
<type>[optional scope][!]: <description>

[optional body]

[optional footer(s)]
```

- **description**: imperative mood ("add", not "added" or "adds"),
  lowercase, no trailing period. Keep the full subject line ≤ 72 characters;
  ≤ 50 is better where it doesn't cost clarity. Imperative reads as "if
  applied, this commit will *<description>*".
- **body**: separated from the subject by a blank line. Wrap at 72
  characters. Explains why, not what (the diff shows what).
- **footers**: git-trailer format (`Token: value`), one per line, separated
  from the body by a blank line. Examples: `Fixes: #123`, `Refs: #456`,
  `Reviewed-by: <human>`. Never AI attribution of any kind.

## Types

The spec requires only `feat` and `fix`; the rest are the widely used
Angular-convention set. Match whatever set the repo's history already uses.

| Type       | Meaning                                              | SemVer  |
| ---------- | ---------------------------------------------------- | ------- |
| `feat`     | New user-facing capability                           | MINOR   |
| `fix`      | Bug fix                                              | PATCH   |
| `refactor` | Code restructuring; behavior unchanged               | —       |
| `perf`     | Performance improvement without behavior change      | —       |
| `docs`     | Documentation only                                   | —       |
| `test`     | Adding or correcting tests only                      | —       |
| `style`    | Formatting/whitespace; no semantic change            | —       |
| `build`    | Build system, tooling, or dependency changes         | —       |
| `ci`       | CI configuration                                     | —       |
| `chore`    | Maintenance that fits none of the above              | —       |
| `revert`   | Reverts a prior commit (reference it in the body)    | —       |

Type selection is about the change's *purpose*, not the files touched:
editing a `.ts` file to fix a typo in a doc comment is `docs`; adding a test
alongside a fix is part of the `fix` commit, not a separate `test` commit;
and `style` means source formatting with no semantic change — CSS or visual
UI work that changes what users see is `feat` or `fix`, not `style`.
If a single change seems to need two types, that's usually a sign it should
be two commits.

## Scope

Optional noun in parentheses naming the affected area: `feat(auth):`,
`fix(parser):`. Derive scopes from the repo's history, not invention — an
inconsistent scope vocabulary is worse than none. Omit the scope when the
change is cross-cutting or the repo doesn't use scopes.

## Breaking changes

Two equivalent notations; use whichever the repo's history uses (both is
also valid):

- `!` before the colon: `feat(api)!: remove legacy session endpoint`
- Footer: `BREAKING CHANGE: <description of the break and migration path>`

A breaking change maps to a SemVer MAJOR bump regardless of type. The footer
form is preferred when the break needs explanation, which it almost always
does.

## Writing the commit non-interactively

Use a heredoc so multi-line messages survive shell quoting:

```bash
git commit -m "$(cat <<'EOF'
fix(catalog): prevent duplicate SKU on concurrent variant creation

Two simultaneous requests could both pass the uniqueness check before
either wrote. Take a per-product advisory lock for the check-and-insert.

Fixes: #482
EOF
)"
```
