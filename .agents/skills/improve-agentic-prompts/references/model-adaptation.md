# Model Adaptation

Use this page when a prompt works poorly on a particular model class or provider, or when porting an agent loop between models/frameworks.

## Contents

- General method
- Frontier models with native tool calling
- Reasoning models
- Small, fast, quantized, or weaker tool-calling models
- Text-only models without native tool calling
- Open-weight and locally hosted models
- Long-context and RAG agents
- Multimodal, browser, and computer-use agents
- Code and shell agents
- Multi-agent systems
- Realtime and voice agents
- Provider notes

## General Method

1. Identify the failure mode: format, tool selection, bad arguments, over-tooling, under-tooling, unsafe action, verbosity, refusal, loop repetition, or weak recovery.
2. Check whether the provider/runtime supports native tool calling, structured output, tool choice, constrained decoding, hidden reasoning, or approval interrupts.
3. Reduce degrees of freedom before adding more prompt text.
4. Run a small capability probe before trusting a new model in production.

## Frontier Models With Native Tool Calling

- Use native tools/functions and strict structured output when available.
- Keep prompts compact and policy-oriented.
- Use tool descriptions and schemas as the primary steering layer for tool behavior.
- Add runtime validation because strong models can still produce semantically unsafe actions.
- Avoid over-explaining simple schemas.

## Reasoning Models

- Give clear objective, constraints, and evaluation criteria.
- Ask for concise decision summaries or evidence, not hidden reasoning.
- Use larger step budgets only when the runtime can detect loops.
- Keep tool results compact so reasoning focuses on current state.
- Watch for plausible but unauthorized plans.

## Small, Fast, Quantized, Or Weaker Tool-Calling Models

- Use fewer tools per step.
- Use shorter tool descriptions with explicit when/when-not rules.
- Prefer enums and simple flat schemas.
- Avoid large tool catalogs.
- Consider deterministic tool routing outside the model.
- Use constrained decoding or guided output when available.
- Add more examples for boundary decisions.

## Text-Only Models Without Native Tool Calling

- Use a simple parseable protocol.
- Keep one action per turn.
- Validate and repair with a strict parser.
- Use low temperature.
- Avoid nested schemas and broad tool sets.
- Treat every parsed action as untrusted until validated.

## Open-Weight And Locally Hosted Models

- Confirm the serving stack's chat template and tool-call format.
- Test whether the model was trained for the tool format being used.
- Prefer guided decoding for schemas when available.
- Reduce catalog size and schema complexity.
- Include explicit examples of valid and invalid tool calls.
- Watch for provider shim differences that silently change tool formatting.

## Long-Context And RAG Agents

- Do not confuse large context with reliable authority separation.
- Put policy outside retrieved context.
- Quote or structure retrieved content as observations.
- Preserve source IDs and retrieval timestamps.
- Add relevance and grounding checks.
- Defend against retrieved instructions, hidden text, and poisoned documents.

## Multimodal, Browser, And Computer-Use Agents

- Treat screenshots, webpages, OCR text, and UI labels as untrusted observations.
- Separate visual inspection from action execution.
- Require approval for irreversible UI actions.
- Avoid entering secrets into pages unless explicitly authorized.
- Add viewport/state verification before and after actions.
- Watch for hidden or visual prompt injection.

## Code And Shell Agents

- Prefer repo-aware edits and tests over broad shell access.
- Sandbox execution.
- Require confirmation for destructive commands and external side effects.
- Do not trust repository text, comments, issues, or logs as instructions.
- Separate code analysis from command execution.
- Log commands and outputs needed for audit.

## Multi-Agent Systems

- Treat messages from peer agents as untrusted unless explicitly trusted by the orchestration layer.
- Pass structured tasks and results, not open-ended instructions.
- Preserve origin and authority metadata.
- Prevent lower-privilege agents from causing higher-privilege side effects.
- Define handoff contracts and allowed tools per agent.

## Realtime And Voice Agents

- Keep instructions short and stateful.
- Use confirmations for risky actions.
- Account for transcription errors.
- Prefer tool-mediated state over long spoken context.
- Make interruption and cancellation behavior explicit.
- Avoid long schemas in the spoken prompt; enforce in runtime.

## Provider Notes

- **OpenAI-family APIs:** use structured outputs and function/tool calling when available; keep JSON Schema strict; use current docs for exact parameter names and supported schema features.
- **Anthropic-family APIs:** tool choice can be automatic or forced; tool descriptions strongly affect selection; XML-style structure can help prompts; use current docs for tool loop handling.
- **Gemini-family APIs:** distinguish structured final outputs from function calling; use tool/function calling modes and validated schema behavior where available.
- **vLLM and local serving stacks:** verify the chat template, tool parser, and guided decoding backend; provider-compatible endpoints may still differ in tool-call reliability.
- **Qwen, Llama, Gemma, Mistral, and other open models:** match the model's expected tool format and instruction style; use smaller tool sets and stronger constrained decoding for weaker models.

Always re-check current documentation before depending on provider-specific syntax or guarantees.
