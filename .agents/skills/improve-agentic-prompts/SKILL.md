---
name: improve-agentic-prompts
description: Improve, review, harden, or redesign prompts for LLM applications that run agentic loops, including tool/function/MCP calling agents, Vercel AI SDK ToolLoopAgent-style loops, LangGraph/ReAct agents, browser or computer-use agents, code/shell agents, multi-agent systems, structured-output workflows, reasoning/thinking traces, and agents with memory or RAG. Use when Codex needs to write or critique system/developer prompts, tool descriptions and schemas, output contracts, loop controls, reasoning visibility, injection defenses, approval boundaries, usability checks, verifier models, multi-model consensus, model/provider adaptations, or eval cases for reliable and safe agent behavior.
---

# Improve Agentic Prompts

Use this skill as the routing overview for improving prompts and runtime contracts for agentic loops. Keep this file in context first; load the referenced topic pages only when their topic is needed.

## Core Principle

Treat the model as a proposal generator. Prompts steer behavior and judgment; schemas, validation, permissions, approvals, stop conditions, and execution code enforce invariants. When a requirement cannot be enforced by prompting, name the runtime control separately.

## First Pass

1. Identify the request type: new prompt, prompt review, trace diagnosis, tool/schema design, structured output, loop control, security hardening, reasoning trace policy, verifier/consensus design, model migration, or eval design.
2. Gather only the context needed: runtime/framework, model(s), prompt text, tool definitions/schemas/results, loop controls, trusted and untrusted inputs, side effects, output consumer, usability criteria, failure traces, and existing validation/evals.
3. Read the reference file(s) that match the problem. Do not load every reference by default.
4. Return layered recommendations: prompt changes, tool/schema changes, runtime controls, verification/eval cases, and residual risks.

## Reference Map

Read only the detail pages needed for the task:

- `references/prompt-contracts.md`: main system/developer prompts, authority, trust boundaries, examples, completion states, and prompt anti-patterns.
- `references/tool-interfaces.md`: what tool information belongs in prompts versus structured tool definitions, tool names, descriptions, schemas, result shapes, typed errors, provenance, prepare/commit tools, and tool anti-patterns.
- `references/structured-outputs.md`: final output schemas, tool calls versus final answers, constrained output, parser repair, and semantic validation.
- `references/loop-controls.md`: state-machine design, active tools, tool choice, prepare-step behavior, stop conditions, cycle detection, context, memory, approvals, and idempotency.
- `references/verification-and-consensus.md`: usable-result checks, postconditions, verifier models, critics, self-consistency, debate, model juries, consensus routing, and escalation.
- `references/reasoning-traces.md`: hidden reasoning versus visible summaries, provider thinking blocks, scratchpads, trace logging, context retention, redaction, and UI exposure.
- `references/security-boundaries.md`: direct/indirect prompt injection, excessive agency, tool poisoning, memory poisoning, cross-agent risks, egress, secrets, MCP hardening, and approval boundaries.
- `references/model-adaptation.md`: adjustments for frontier, reasoning, small, text-only, open/local, RAG, multimodal, code/shell, multi-agent, realtime/voice, and provider-specific tool behavior.
- `references/evals.md`: functional tests, adversarial injection matrices, capability probes, metrics, regression practice, and trace-based iteration.

## Triage Guide

- Main prompt, authority, trust, task completion, or prompt anti-patterns: read `references/prompt-contracts.md`.
- Tool availability, tool-use instructions, tool schemas, or tool results: read `references/tool-interfaces.md`.
- Malformed JSON, missing fields, output schemas, or repair loops: read `references/structured-outputs.md`.
- Repeated loops, tool spam, stop conditions, progress checks, retries, or idempotency: read `references/loop-controls.md`.
- Usability checks, postconditions, verifier models, critics, juries, or multi-model consensus: read `references/verification-and-consensus.md`.
- Reasoning/thinking visibility, summaries, traces, logging, retention, or redaction: read `references/reasoning-traces.md`.
- Prompt injection, untrusted content, approvals, secrets, egress, memory poisoning, or cross-agent risk: read `references/security-boundaries.md`.
- Model/provider migration, small-model adaptation, local/open models, multimodal, code/shell, voice, or multi-agent behavior: read `references/model-adaptation.md`.
- Proof that an improvement works, regression coverage, adversarial cases, or trace metrics: read `references/evals.md`.

## Output Shape

- Provide concrete revised prompt sections when the user asks for prompt text.
- Separate prompt guidance from runtime-enforced controls.
- Include tool/schema, verification, reasoning trace, security, and eval changes only when relevant to the request.
- Fetch current provider/framework docs when API behavior, SDK syntax, tool-calling modes, structured output settings, or loop controls matter.

## Guardrails

- Do not present prompt wording as sufficient for security or reliability requirements that need runtime enforcement.
- Do not require visible chain-of-thought; prefer concise reasons, evidence, decision summaries, or structured trace events.
- Do not treat majority vote, model confidence, or judge agreement as authorization.
- Do not expose every tool on every step when a smaller active tool set is enough.
