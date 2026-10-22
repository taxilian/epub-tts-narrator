# Reasoning Traces

Use this page when an agent exposes, logs, streams, preserves, summarizes, or hides reasoning, thinking text, scratchpads, intermediate commentary, or debug traces.

## Contents

- Core distinctions
- Does reasoning stay in context?
- How to see reasoning
- What to show users
- Scratchpads and agent state
- Tool loops and replay
- Prompt guidance
- Security and privacy
- Evals and debugging
- Anti-patterns

## Core Distinctions

Separate these categories:

- **Hidden model reasoning:** internal tokens used by a reasoning model. Usually not visible or replayable as text.
- **Reasoning summary:** provider-generated summary of reasoning, when explicitly requested and supported.
- **Thinking block:** provider-specific content block that may be returned and may need to be passed back for API/tool-loop continuity.
- **Agent scratchpad:** framework-managed intermediate state, tool calls, observations, plans, and notes.
- **Visible commentary:** user-facing progress text, preambles, explanations, or decision summaries.
- **Trace log:** application telemetry: prompts, tool calls, tool results, approvals, errors, costs, timings, and optional summaries.

Do not assume these are interchangeable. Treat each as a different data class with different retention, visibility, and replay rules.

## Does Reasoning Stay In Context?

It depends on the provider and framework.

- Hidden reasoning generally does not become ordinary context text.
- Reasoning summaries usually appear in the API response only if requested. They remain in future context only if the application stores and re-injects them.
- Provider thinking blocks may need to be passed back unchanged in tool loops even if they are not counted like ordinary prior text.
- Agent scratchpads remain in context only if the framework includes them in subsequent model inputs.
- Visible commentary and final answers remain in context if the app stores them as conversation messages.

Design rule: decide explicitly what is replayed into the next step. Do not let debug traces accidentally become instructions.

## How To See Reasoning

Use the provider or framework surface rather than asking the model to reveal hidden chain-of-thought.

Common patterns:

- Read a non-streaming `reasoningText`, `reasoning`, `summary`, or equivalent result field when the SDK exposes one.
- Listen for streamed parts such as `reasoning`, `thinking`, or provider-specific content blocks.
- Enable provider options such as reasoning effort, thinking budget, or reasoning summary where supported.
- Inspect agent traces that include tool calls, observations, state transitions, and verifier outputs.
- Add your own structured trace events for plan, action, observation, postcondition, and finalization.

For user-facing debug UIs, prefer a "trace" view that shows tool calls, decisions, observations, and concise rationales. This is usually more useful and safer than raw hidden reasoning.

## What To Show Users

Show:

- Progress updates.
- Tool calls and tool results, redacted as needed.
- Decision summaries.
- Evidence and source IDs.
- Why an approval is required.
- Why the agent stopped, asked, blocked, or failed.

Usually do not show:

- Raw hidden chain-of-thought.
- Secrets, credentials, private data, or prompt internals.
- Unredacted external content that may contain injection.
- Internal policy text.
- Full scratchpads when they include sensitive observations.

Useful prompt line:

```text
Provide concise decision summaries, evidence, and next-step rationale. Do not reveal hidden reasoning or private scratchpad content.
```

## Scratchpads And Agent State

If the framework uses a scratchpad:

- Keep it structured, not free-form rambling.
- Separate trusted state from untrusted observations.
- Store current objective, phase, approved actions, validated IDs, and unresolved blockers.
- Avoid storing raw external content as trusted notes.
- Compact old scratchpad entries into factual state plus provenance.
- Do not make scratchpad text visible unless it has been redacted and intentionally formatted for users.

## Tool Loops And Replay

Reasoning/thinking artifacts can affect loop correctness.

- Preserve provider-required thinking blocks or reasoning items when the API requires them for continuation.
- Preserve phase markers, tool-call IDs, and tool-result IDs when replaying history manually.
- Do not drop intermediate assistant messages if the runtime needs them to distinguish commentary from final answer.
- Do not feed reasoning summaries back as authoritative policy.
- Do not let a summarized trace replace the actual validated state needed for tool execution.

When context gets long, compact traces into:

```text
objective, current phase, validated facts, tool results with provenance, approvals, blockers, postconditions, and final answer status
```

## Prompt Guidance

Use prompts to request observable reasoning artifacts, not hidden chain-of-thought:

```text
Keep private reasoning internal. When helpful, provide a concise decision summary explaining the evidence used, the tool chosen, and any uncertainty or blocker.
```

For agent loops:

```text
Before each tool call, provide at most one sentence of commentary describing the next action. After each tool result, update the structured state and decide whether to finalize, ask, approve, or continue.
```

For debugging mode:

```text
When debug_trace=true, include structured trace events: phase, decision, tool, arguments summary, observation summary, postcondition status, and stop reason. Do not include hidden reasoning.
```

## Security And Privacy

Reasoning traces can leak:

- System/developer prompts.
- Secrets or credentials from tool results.
- Private user data.
- Internal policy.
- Vulnerability details.
- Prompt-injection text copied from untrusted content.

Apply the same redaction and retention policy as logs. Treat reasoning summaries as potentially sensitive, even when they are not raw chain-of-thought.

## Evals And Debugging

When evaluating reasoning visibility, test:

- Reasoning summary requested and shown only in debug mode.
- Debug traces exclude secrets and prompt internals.
- Tool-call IDs and phase markers survive replay.
- Thinking/reasoning blocks are preserved when the provider requires them.
- Context compaction does not turn reasoning summaries into trusted facts.
- User-facing explanation is concise and sufficient without hidden reasoning.
- Failure traces include enough state to diagnose wrong tool choice or bad stopping.

## Anti-Patterns

- Asking the model to reveal its hidden chain-of-thought.
- Treating reasoning text as proof that the answer is correct.
- Feeding raw reasoning summaries back as instructions.
- Logging raw traces without redaction.
- Showing scratchpads to users without separating private state and untrusted content.
- Dropping provider-required thinking/tool blocks during tool-loop continuation.
- Letting debug commentary be mistaken for a final answer.
