# Structured Outputs

Use this page when the user needs reliable JSON, function calls, tool arguments, typed final answers, parser repair, or schema design.

## Contents

- Reliability hierarchy
- Tool calls versus final output
- Schema guidelines
- Semantic validation
- Repair strategy
- Text-only fallback
- Structured output anti-patterns

## Reliability Hierarchy

Prefer stronger mechanisms before prompt-only formatting:

1. Native structured output or strict JSON Schema support.
2. Native tool/function calling with validated schemas.
3. Constrained decoding, grammar, regex, or guided decoding.
4. Parser plus bounded repair loop.
5. Text instructions and examples only.

Prompt text can improve compliance, but it is the weakest layer.

## Tool Calls Versus Final Output

Keep these separate:

- **Tool/function calling:** the model requests an external action or lookup by producing structured arguments for an executor.
- **Final structured output:** the model returns the answer to the application/user in a specific schema.

Do not use a final-answer schema when the model needs to call tools first. Do not use a tool call when the only requirement is final formatting.

## Schema Guidelines

- Set required fields explicitly.
- Disallow unexpected properties when the provider/runtime supports it.
- Use enums for finite choices.
- Use discriminated unions for different result types.
- Include units, currency, timezone, locale, and ID source in field descriptions.
- Prefer arrays of typed objects over loosely formatted strings.
- Avoid schema designs that require hidden reasoning.
- Keep schemas smaller for weaker or local models.
- Use a `status` or `result_type` field for non-success states.
- Include `evidence` or `source_ids` fields when factual grounding matters.

Example final schema shape:

```json
{
  "result_type": "success | needs_clarification | blocked | failed",
  "answer": "string",
  "evidence": [
    {
      "source_id": "string",
      "claim": "string"
    }
  ],
  "next_action": "none | ask_user | request_approval | retry_later"
}
```

## Semantic Validation

Schema validity is not semantic correctness. Validate business rules outside the model:

- Date ranges.
- User and tenant authorization.
- Record ownership.
- Currency and amount limits.
- Whether referenced IDs exist.
- Whether the value came from an allowed source.
- Whether a side effect was approved.
- Whether generated text is safe for the downstream sink.

When semantic validation fails, return a typed error and let the loop recover within a small retry budget.

## Repair Strategy

Use repair only for narrow, observable failures.

- Repair malformed syntax or missing required fields.
- Do not use repair to bypass authorization or approval failure.
- Do not silently reinterpret user intent during repair.
- Limit repair attempts, usually one or two.
- Feed the validator error back in a compact form.
- If repair fails, return a clear failure or ask the user.

Repair prompt pattern:

```text
The prior output did not match the required schema.
Validator error: [compact error]
Return only a corrected object that satisfies the schema.
Do not add new facts or change the intended answer.
```

## Text-Only Fallback

For models without native tool or structured output support:

- Use one unambiguous fenced block or tagged section.
- Keep the grammar simple.
- Avoid nested optional fields.
- Include a parser and validation step.
- Use low temperature.
- Prefer small schemas and short field names only when readability does not suffer.

Example:

```text
Return exactly one JSON object between <final_json> and </final_json>.
Do not include markdown outside the tags.
```

## Structured Output Anti-Patterns

- Asking for "valid JSON" without a schema.
- Treating schema conformance as authorization.
- Allowing arbitrary extra fields.
- Asking for prose and JSON interleaved.
- Letting the model invent IDs or approval flags.
- Retrying indefinitely until JSON parses.
- Using one giant schema for every state of the workflow.
