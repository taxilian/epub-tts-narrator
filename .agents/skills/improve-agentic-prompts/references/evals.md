# Evals

Use this page when designing tests for an agent prompt, validating an improvement, or diagnosing whether a prompt/runtime change actually helped.

## Contents

- Evaluation principle
- Capability probe
- Functional cases
- Injection matrix
- Tool-risk cases
- Loop cases
- Metrics
- Regression discipline
- Output for an eval design

## Evaluation Principle

Evaluate the whole loop, not only the final text. A good eval checks model behavior, tool calls, runtime validation, approvals, and final output.

## Capability Probe

Before deep tuning a new model, test:

- Can it select the right tool from a small catalog?
- Can it avoid tools when no tool is needed?
- Can it fill required schema fields correctly?
- Can it recover from a typed tool error?
- Can it stop after success?
- Can it ask for clarification when required information is missing?
- Can it treat external instructions as untrusted?
- Can it refuse or request approval for high-risk actions?

## Functional Cases

Include normal success paths:

- Answer from existing context, no tool.
- Read-only lookup then answer.
- Multi-step read-only lookup.
- Ambiguous request requiring clarification.
- Tool returns not found.
- Tool returns multiple matches.
- Tool returns transient error.
- Structured final output with all required fields.

## Injection Matrix

Test injection across locations:

- User prompt.
- Retrieved document.
- Webpage.
- Email or ticket body.
- File content.
- Tool result.
- Tool description or MCP metadata.
- Stored memory.
- Peer-agent message.
- Screenshot/OCR text.

Test injection forms:

- "Ignore previous instructions."
- Prompt or secret extraction.
- Tool-use redirection.
- Data exfiltration through URL, markdown image, email, webhook, or log.
- Hidden text, comments, encoded text, multilingual text, or split payloads.
- Instructions that appear helpful but change destination, recipient, amount, or scope.

## Tool-Risk Cases

For each high-risk tool, test:

- Model attempts action without approval.
- Model changes payload after approval.
- Model uses user-supplied tenant/user/approval fields.
- Model uses an ID from untrusted content without verification.
- Model calls write tool during read-only phase.
- Model retries a fatal or unauthorized error.
- Model attempts parallel writes.
- Model sends sensitive data to an unapproved destination.

## Loop Cases

Test:

- Max step reached.
- Repeated equivalent tool call.
- Same tool and args return same observation twice; loop must stop or change strategy.
- Task success criteria satisfied; agent must finalize without another tool call.
- Submit/done tool called; runtime must not re-enter tool-use phase.
- Tool result status is `no_change`, `unauthorized`, or `fatal_error`; agent must not retry blindly.
- Retrieval returns no new sources after query rewrite; agent must ask, answer with uncertainty, or stop.
- Final answer already satisfies schema and postconditions; repair loop must not continue.
- Repair budget exhausted.
- Context compaction preserves authority and approvals.
- Memory write is blocked for untrusted content.
- Approval pauses the loop.
- Finalization happens with no extra tool call.

## Metrics

Track at least:

- Task success.
- Format/schema validity.
- Semantic validity.
- Correct tool selection.
- Unnecessary tool calls.
- Unsafe tool calls blocked by runtime.
- Approval-boundary violations.
- Injection success rate.
- Data exfiltration attempts.
- Loop steps, latency, and cost.
- Recovery quality after typed errors.

## Regression Discipline

- Keep raw prompts, tool definitions, traces, and expected outcomes.
- Change one major prompt/runtime dimension at a time when possible.
- Replay the same trace set before and after changes.
- Add every production failure as a regression case.
- Include both helpful and adversarial external content.
- Test across target models if the system is model-portable.

## Output For An Eval Design

When asked to design evals, return:

- Test categories.
- Concrete test cases.
- Expected tool calls and non-calls.
- Expected final output or state.
- Runtime assertions.
- Metrics and pass/fail thresholds.
- Gaps that require instrumentation.
