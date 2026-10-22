# Loop Controls

Use this page when improving the agentic loop itself: step policy, tool availability, state machine phases, retries, context management, memory, approvals, or termination.

## Contents

- Mental model
- Phase-gated tools
- Tool choice
- Explicit done states
- Stop conditions
- Cycle detection
- No-progress detection
- Prompt patterns that help termination
- Parallel calls
- Context management
- Memory writes
- Approvals and idempotency
- Trace diagnosis

## Mental Model

Treat the loop as a state machine, not as an open-ended conversation.

At each step:

1. The model proposes a tool call or final answer.
2. The runtime validates the proposal against the current state.
3. The executor performs the smallest authorized action.
4. The result returns typed status and provenance.
5. The loop advances, asks, waits for approval, fails, or terminates.

The runtime should know what terminal states exist. Do not leave completion as a vague model preference.

## Phase-Gated Tools

Expose only the tools needed for the current phase.

Example phases:

- `understand_request`: no write tools.
- `retrieve_context`: search and lookup tools only.
- `draft_plan`: prepare tools only.
- `await_approval`: no autonomous side effects.
- `commit_action`: only the approved commit tool.
- `finalize`: no tools unless final verification is required.

If the framework supports `activeTools`, `toolChoice`, graph edges, interrupts, or per-step preparation hooks, use them to enforce phases.

## Tool Choice

Use deterministic tool choice when:

- The first step must retrieve or validate context.
- The model should call a specific submit/finalization tool.
- A phase should allow exactly one tool.
- A previous validation error indicates a specific repair action.

Use automatic tool choice when:

- Multiple read-only tools are safe alternatives.
- The model can choose based on context without increasing risk.

Disable tools when:

- The answer should be generated from already available context.
- The loop is in finalization.
- The user needs to approve or clarify.

## Explicit Done States

Make completion observable. Good patterns:

- **Natural final answer:** stop when the model returns a non-tool response.
- **Submit/final-answer tool:** require the model to call `submit_final_answer` or `done`; treat that tool call as terminal.
- **No-execute done tool:** in runtimes that stop when a tool has no executor, define a `done` tool with final answer fields and no executor.
- **Graph terminal edge:** route to `END` or a terminal node once success criteria are met.
- **State latch:** once `status=complete`, `approval_required`, `blocked`, or `failed`, prevent the loop from re-entering tool-use phases.

Use a submit/done tool when natural-language finalization is unreliable or when the final answer must be structured. Give it a clear description: "Call this exactly once when the task is complete; do not call other tools afterward."

Avoid "keep going until complete" without a terminal signal. It tells the model to continue but does not tell the runtime when to stop.

## Stop Conditions

Add hard stops outside the prompt:

- Max steps.
- Max wall-clock time.
- Max tokens or cost.
- Max retries per tool.
- Max repair attempts.
- Max repeated equivalent tool calls.
- Max no-progress steps.
- Max repeated observations from tools.
- Stop on approval requirement.
- Stop on unauthorized or fatal error.
- Stop on done/submit/final tool.
- Stop when success postconditions are satisfied.

Prompt the model to stop appropriately, but enforce stops in runtime code.

## Cycle Detection

Detect repeated work by canonicalizing:

- Tool name.
- Normalized arguments.
- Relevant state or phase.
- Error status.
- Tool result status.
- Key observations or returned IDs.

If the same call repeats without new information, stop, ask for clarification, or change strategy. Do not rely on the model to notice loops.

Implementation rule:

```text
fingerprint = hash(phase, tool_name, normalized_args, relevant_state, last_status)
if fingerprint has appeared before and no new evidence/postcondition changed:
  stop or route to clarification/failure
```

Use semantic canonicalization where needed: normalize whitespace, ordering, casing, timestamps that do not affect the decision, and equivalent IDs.

## No-Progress Detection

Track whether each step changed the known state.

Examples of progress:

- New validated fact.
- New source or record ID.
- Ambiguity reduced.
- Postcondition satisfied.
- Approval obtained.
- A planned side effect prepared or committed.
- A different repair attempted after a typed error.

Examples of no progress:

- Same tool, same arguments, same result.
- Re-reading the same source without new query terms.
- Re-validating an already valid result.
- Rewriting a final answer that already satisfies the contract.
- Asking a tool for confirmation of facts already in trusted state.
- Retrying an unauthorized, fatal, or policy-blocked action.

After one or two no-progress steps, do not keep looping. Return a final answer, ask a clarification question, or report the blocker.

## Prompt Patterns That Help Termination

Prompt text should reinforce runtime controls:

```text
Before calling any tool, check whether the success criteria are already satisfied.
Call a tool only if it can provide missing information or perform the next allowed state transition.
Do not call a tool merely to reconfirm information already available in trusted state.
If the next tool call would repeat the same tool with equivalent arguments and no new information, stop and return the best final result or ask for clarification.
When all required postconditions are satisfied, finalize immediately.
```

For tool-choice-heavy agents, add a short per-step decision rule:

```text
Next action must be one of: call an allowed tool for missing information, request approval, ask clarification, return final, or return blocked. Choose final when the answer is complete.
```

These prompt rules reduce looping but are not substitutes for runtime stop conditions.

## Tool Results For Loop Control

Shape tool results so the loop can tell whether anything changed:

```json
{
  "status": "ok | no_change | not_found | ambiguous | unauthorized | fatal_error",
  "summary": "short result",
  "new_information": true,
  "state_version": "optional monotonic version or etag",
  "postconditions_satisfied": [],
  "retry_recommended": false
}
```

Use `no_change`, `retry_recommended`, and typed fatal statuses to prevent blind retries.

## Parallel Calls

Allow parallel tool calls only when they are:

- Read-only.
- Independent.
- Idempotent.
- Bounded in number.
- Not competing for the same state.

Do not parallelize state-changing calls unless the runtime has explicit transaction and conflict handling.

## Context Management

- Summarize old steps into compact state, not vague prose.
- Keep current objective, constraints, approvals, and validated IDs visible.
- Drop irrelevant raw tool output.
- Preserve provenance for facts that affect later actions.
- Mark untrusted content as untrusted after summarization.
- Avoid mixing memory with current instructions.

## Memory Writes

Persist memory only through an explicit runtime path:

- Define what may be remembered.
- Exclude secrets, raw external content, transient IDs, and unverified facts.
- Require provenance and confidence.
- Consider user review for personal or durable memory.
- Treat retrieved memory as data, not policy.

## Approvals And Idempotency

For side effects:

- Stage a plan with a prepare tool.
- Show the exact target, action, payload, and risk.
- Store approval outside the model.
- Commit using a trusted prepared action ID.
- Generate idempotency keys in code.
- Log the approved payload and actual execution result.

## Trace Diagnosis

When reviewing a failed trace, identify the first bad transition:

- Wrong objective.
- Wrong tool set exposed.
- Wrong tool chosen.
- Bad arguments.
- Bad tool result shape.
- Missing validation.
- Missing approval.
- Bad retry policy.
- Missing terminal state or done latch.
- Missing no-progress detector.
- Bad context or memory.
- Bad final output.

Fix the earliest layer that would have prevented the failure.
