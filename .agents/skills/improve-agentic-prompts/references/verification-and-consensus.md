# Verification And Consensus

Use this page when an agent output must be usable by a downstream system, when correctness matters beyond schema validity, or when combining multiple models, judges, critics, or consensus methods.

## Contents

- Core principle
- Usable-result checks
- Postcondition checks
- Verification ladder
- Verifier models
- Critic-revise loop
- Self-consistency
- Multi-model consensus
- Debate
- Consensus for tool use
- Disagreement handling
- Anti-patterns

## Core Principle

Separate four questions:

- **Is it well-formed?** It parses and matches the schema.
- **Is it semantically valid?** It satisfies business rules and source constraints.
- **Is it usable?** The downstream consumer can render, execute, apply, or act on it successfully.
- **Is it correct enough?** It passes the required verification threshold for the risk level.

Do not use model agreement as a substitute for deterministic validation, authorization, or human approval.

## Usable-Result Checks

Define the consumer and its acceptance criteria before improving the prompt.

Examples:

- JSON must parse, validate, and round-trip through the consuming type.
- Generated SQL must parse, run in read-only mode, and return expected columns.
- Generated code must compile, pass tests, and avoid forbidden APIs.
- A browser action plan must reference visible UI state and pass preflight checks.
- A support reply must include required policy elements and no forbidden claims.
- A report must cite source IDs for each factual claim.
- A tool plan must contain only currently allowed tools and approved targets.

Prompt addition:

```text
Before finalizing, check that the result is directly usable by [consumer].
If it is not usable, either repair it within the allowed budget or return a blocked result with the first failed check.
```

Runtime addition:

```text
Run parser -> schema validator -> semantic validator -> consumer dry-run -> verifier/judge if needed.
Return the first failing check as a typed error.
```

## Postcondition Checks

For each important action or output, define postconditions.

Good postconditions are observable:

- The created record exists and has the approved fields.
- The email draft recipient, subject, and body match the approved plan.
- The file was modified and tests pass.
- The final answer cites only retrieved source IDs.
- The generated artifact opens successfully.
- No forbidden data class appears in the outbound payload.

Avoid vague postconditions such as "the answer is high quality" unless paired with a rubric.

## Verification Ladder

Use the cheapest reliable check first:

1. Parser, schema, and type checks.
2. Deterministic business-rule validation.
3. Dry-run, sandbox execution, compile, render, or test.
4. Source/evidence coverage checks.
5. Independent model critique or judge.
6. Multiple samples or multiple models.
7. Human review or approval.

Escalate only when the lower layer cannot answer the question or when risk requires redundancy.

## Verifier Models

Use a verifier model for judgments that are semantic, fuzzy, or expensive for code to check.

Good verifier tasks:

- Does the answer follow the rubric?
- Are all claims supported by the cited sources?
- Does the response satisfy the user's constraints?
- Is the proposed tool plan safe under this policy?
- Does this summary omit required fields?
- Which candidate is better under explicit criteria?

Verifier prompt pattern:

```text
You are an independent verifier.
Evaluate the candidate against the rubric only.
Use the provided sources and policy; do not add new assumptions.
Return JSON with: verdict, failed_checks, evidence, confidence, and recommended_action.
```

Verifier rules:

- Give the verifier the rubric, candidate, sources, and policy.
- Hide or randomize candidate order for pairwise judging when possible.
- Prefer structured verdicts over prose.
- Use a stronger or different model when bias/correlation matters.
- Treat verifier output as another signal, not as authority.

## Critic-Revise Loop

Use for open-ended outputs where a single pass often misses requirements.

Pattern:

1. Generator creates candidate.
2. Critic identifies concrete failures against the rubric.
3. Reviser fixes only those failures.
4. Validator/verifier checks again.
5. Stop after a small fixed budget.

Avoid unlimited self-repair. If the same failure persists, return blocked or escalate.

## Self-Consistency

Use multiple samples from the same model when the task has a stable answer but the path is uncertain.

Works best for:

- Math and logic.
- Extraction with ambiguous phrasing.
- Classification.
- Tool-plan selection.
- Short factual answers when evidence is available.

Process:

- Generate N independent candidates with temperature high enough for diversity.
- Normalize final answers or decisions.
- Aggregate by majority, weighted score, or verifier ranking.
- Escalate when no answer reaches the required threshold.

Limits:

- Samples from one model share blind spots.
- Majority can converge on a common misconception.
- It increases cost and latency.

## Multi-Model Consensus

Use multiple models when diversity is valuable:

- Different model families have different failure patterns.
- One model is better at generation and another at verification.
- A local or cheap model can screen, while a stronger model adjudicates.
- Regulatory, safety, or high-stakes workflows require redundancy.

Common patterns:

- **Generate-and-rank:** several models generate; a judge ranks against a rubric.
- **Generator-verifier:** one model proposes; another checks.
- **Specialist panel:** separate models check factuality, safety, style, and format.
- **Model jury:** several judges vote independently; aggregate votes.
- **Adjudicator:** a stronger model sees candidates, judge verdicts, and evidence, then selects or escalates.
- **Router:** choose the best model based on task class, risk, cost, and needed tool support.

Consensus policy:

```text
Accept only if at least [threshold] independent checks pass and no critical validator fails.
Escalate to human review when judges disagree on safety, authorization, factual support, or high-impact actions.
```

## Debate

Use debate for hard reasoning or factual disputes where candidates can expose each other's errors.

Structure:

- Agents answer independently.
- Each critiques the others using sources and rubric.
- Agents may revise once.
- A judge or deterministic rule selects a result.

Keep debate bounded. Do not let agents introduce new unverified facts during critique.

## Consensus For Tool Use

Consensus can reduce bad tool calls, but runtime policy still decides what may execute.

Useful checks:

- Ask an independent planner to choose the next tool from the same allowed set.
- Require agreement before high-cost read-only calls.
- Require verifier approval before prepare actions.
- Never execute state-changing calls based only on model consensus.
- Use runtime approvals and prepared action IDs for commits.

## Disagreement Handling

Disagreement is useful signal. Define it explicitly:

- If validators fail, repair or block.
- If judges disagree on low-risk quality, choose best-scored candidate or ask user.
- If judges disagree on factuality, retrieve more evidence or answer with uncertainty.
- If judges disagree on safety, authorization, or side effects, fail closed or request human review.
- If all candidates differ, ask for clarification or reduce task scope.

## Anti-Patterns

- Same model generates and judges without any independent signal.
- Judge sees the generator's hidden reasoning and becomes anchored.
- Majority vote over outputs that all lack evidence.
- Consensus used to bypass policy, approval, or authorization.
- Unbounded critic-revise loops.
- Vague judge rubrics such as "is this good?"
- Aggregating scores without checking critical failures.
- Treating confidence text as calibrated probability.
