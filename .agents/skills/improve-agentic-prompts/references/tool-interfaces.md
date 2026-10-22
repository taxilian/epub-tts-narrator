# Tool Interfaces

Use this page when improving tool/function/MCP definitions, tool schemas, tool descriptions, tool results, or tool-related prompt failures.

## Contents

- Core principle
- Where tool instructions belong
- Tool names
- Tool descriptions
- Schema design
- Split or consolidate
- Prepare/commit pattern
- Tool results
- Risk tiers
- Tool anti-patterns

## Core Principle

Tool definitions are part prompt, part API contract. Good tool design reduces ambiguity before the model chooses a tool and reduces damage after it chooses one.

## Where Tool Instructions Belong

Use the runtime's structured tool channel as the source of truth whenever it exists.

- Put **tool-specific details** in the tool definition: exact name, capability, when to use, when not to use, input schema, parameter semantics, result shape, and risk level.
- Put **cross-tool policy** in the main prompt: overall objective, trust boundaries, workflow phase, when to ask, when to stop, when approval is required, and how to treat untrusted tool outputs.
- Put **hard enforcement** in runtime code: active tool sets, authorization, validation, rate limits, egress policy, approvals, idempotency, and stop conditions.

If the agent uses a text-only protocol with no structured tool channel, the prompt must include a compact tool catalog, exact action syntax, examples, and parser constraints. Keep it much shorter than a full API reference.

If the prompt and tool definition disagree, fix the source of truth instead of adding more wording. Conflicting tool instructions cause tool-selection failures and make evals hard to interpret.

## Tool Names

- Use verb-object names that describe the actual capability: `search_customers`, `draft_email`, `commit_refund`.
- Avoid vague names: `process`, `execute`, `handle`, `do_action`, `run`.
- Split tools by security boundary, not by implementation convenience.
- Keep names stable. Model traces and evals depend on them.

## Tool Descriptions

Each description should answer:

- What the tool does.
- When to use it.
- When not to use it.
- Whether it reads, writes, sends, deletes, or changes permissions.
- What inputs must come from trusted state or validated tool results.
- What approval or workflow phase is required.

Example:

```text
Use `prepare_refund` to calculate and stage a refund for the authenticated user's current order.
This tool does not issue the refund. Use it only after the order has been found by `lookup_order`.
Do not use user-provided order IDs unless they have been validated by `lookup_order`.
```

## Schema Design

- Make invalid states unrepresentable with enums, discriminated unions, required fields, min/max constraints, and explicit units.
- Use separate fields for content and authority. Example: `email_body` is content; `recipient_user_id` is authority and should usually come from trusted state.
- Avoid free-form `action`, `command`, `query`, `url`, or `sql` fields unless tightly sandboxed.
- Remove fields the application already knows: tenant ID, current user ID, auth scope, approval status, secrets, idempotency token.
- Prefer IDs from prior validated tool results over user-supplied IDs.
- Include semantic constraints in field descriptions: allowed source, units, timezone, currency, max length, provenance requirement.
- Use enums for small closed sets instead of strings.
- Use arrays only when parallel or batch behavior is safe.

## Split Or Consolidate

Split tools when:

- One path reads and another writes.
- One path needs approval and another does not.
- One path can access sensitive data and another cannot.
- One path has different tenant/user authorization.
- One path has external egress and another does not.

Consolidate tools when:

- Several tools differ only in superficial names but share the same security boundary.
- The model repeatedly chooses among near-duplicates incorrectly.
- A single schema with an enum makes the distinction clearer and safer.

## Prepare/Commit Pattern

Use for side effects such as sending email, posting content, refunds, account changes, permission changes, deletes, or purchases.

1. `prepare_*`: read-only or reversible. Returns a compact plan, affected objects, risk level, and approval text.
2. Human or policy approval: outside the model.
3. `commit_*`: performs exactly the approved action by referencing a trusted `prepared_action_id`.

Do not let the model invent approval status or pass arbitrary action details into the commit tool.

## Tool Results

Return compact structured results optimized for the next decision.

Recommended shape:

```json
{
  "status": "ok | not_found | ambiguous | unauthorized | validation_error | rate_limited | transient_error | fatal_error",
  "summary": "short human-readable fact",
  "data": {},
  "provenance": {
    "source": "tool/system/external",
    "record_ids": [],
    "retrieved_at": "ISO-8601 timestamp"
  },
  "next_allowed_actions": []
}
```

Guidelines:

- Return enough detail to decide the next step, not entire raw documents.
- Preserve provenance for values the model may later use.
- Use typed errors instead of paragraphs.
- Tell the model whether retrying is useful.
- Mark untrusted text as untrusted text in the result.

## Risk Tiers

- **Read-only low risk:** search, lookup, summarize owned data.
- **Read-only sensitive:** user data, secrets, private records, regulated data.
- **Draft or prepare:** creates a plan but no external side effect.
- **State-changing:** writes, sends, deletes, purchases, posts, permission changes.
- **Open-ended execution:** shell, browser, arbitrary HTTP, SQL, code execution.

Higher tiers need narrower schemas, stronger validation, fewer active tools, explicit approval, and more evals.

## Tool Anti-Patterns

- A universal `execute(action: string)` tool.
- Arbitrary shell, SQL, browser, or HTTP tools with broad access.
- Tool descriptions that contain hidden policy or secrets.
- Model-supplied `user_id`, `tenant_id`, `is_approved`, `auth_token`, or `idempotency_key`.
- Read and write operations in one broad tool.
- Tool results that return huge raw external text without provenance or truncation.
- Retrying unknown tool names by guessing the closest name.
- Tool metadata from untrusted servers treated as trusted instructions.
