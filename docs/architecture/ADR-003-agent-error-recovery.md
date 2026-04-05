# ADR-003: Pure LLM Baseline — Environment-Driven Error Recovery

**Status**: Accepted (Revised 2026-04-06)  
**Date**: 2026-04-05 (revised 2026-04-06)  
**Author**: Team Mana Mitra  
**Reviewers**: Internal  

## Context

LLM agents in multi-step environments exhibit three failure modes:

1. **Action loops**: Agent repeats the same action indefinitely
2. **Error cascades**: One wrong action leads to an error, agent retries the same thing
3. **Stagnation**: Agent makes valid but non-progressing actions (e.g., re-filling an already-filled form)

These are real challenges observed during testing with Qwen2.5-72B and GPT-4o.

### Design Philosophy: Let the Environment Teach

Rather than building guardrails into the inference script (which would hardcode agent strategy and defeat the purpose of RL training), we designed the **environment itself** to provide rich signals that any LLM can learn from:

- **Error messages** tell the agent exactly what went wrong ("Unknown fix target: 'aadhaar_address_outdated'. Valid: aadhaar_name, aadhaar_address, photo, address_proof")
- **Reward trend** shows whether the last action helped (reward increased) or not (decreased/unchanged)
- **Available actions** are curated per phase — invalid actions are excluded
- **Pending issues** list problems that need resolution
- **Completed steps** show what's already done

## Decision

### 1. Pure LLM Baseline (No Guardrails in inference.py)

The inference script is intentionally minimal — a pure LLM baseline suitable for RL:

```
Observation → Build Prompt → LLM → Parse Action → Execute → Repeat
```

No guards, no overrides, no hardcoded strategy. The LLM receives the full observation and decides every action autonomously.

### 2. Environment-Side Error Recovery Signals

Instead of agent-side recovery logic, the environment provides actionable feedback:

| Signal | Source | Example |
|--------|--------|---------|
| **Error messages** | `last_action_error` | "No appointment has been scheduled." |
| **Valid targets** | Error text | "Valid: aadhaar_name, aadhaar_address, photo" |
| **Reward trend** | Prompt injection | "Last reward: 0.38 (reward unchanged — try a different action)" |
| **Phase status** | `status_summary` | "Application preparation phase." |
| **Pending issues** | `pending_issues` | ["aadhaar_address_outdated"] |

### 3. Reward Trend Feedback

The inference script computes reward deltas between consecutive steps and annotates the prompt:

- **Increased**: `(reward increased +0.08)` — positive reinforcement
- **Decreased**: `(reward DECREASED -0.02 — try a different action)` — signals to change strategy
- **Unchanged**: `(reward unchanged — try a different action)` — signals stagnation

This breaks action loops without prescribing which action to take.

### 4. Repeat and Step Penalties (Environment-Side)

The environment applies automatic penalties to discourage exploitation:

- **Repeat penalty**: 2% per repeat of the same action+parameters beyond 3 occurrences
- **Step penalty**: 1% per step beyond 2× optimal path length
- **Score cap**: 0.95 maximum

These are environment mechanics, not agent logic — any agent interacting with the environment experiences them.

## Alternatives Considered

1. **Four-guard architecture (previous design)**: Guards validated actions, auto-filled forms, mapped errors to corrective actions, and detected loops. **Removed** — this hardcoded agent strategy into the inference script, making it unsuitable as an RL baseline. The environment should be the teacher, not the inference script.

2. **Hint-to-action mapping**: The environment provided hints and the agent converted them to actions via keyword matching. **Removed** — hints were replaced with non-prescriptive phase-level status summaries. The agent must reason about what to do, not follow instructions.

3. **Task-specific agent logic**: Rejected — violates the principle that the inference script should be task-agnostic.

## Consequences

- **Positive**: Inference script is a genuine RL baseline (339 lines, zero task-specific logic)
- **Positive**: Any LLM can be dropped in — no model-specific tuning
- **Positive**: Environment signals are rich enough for Qwen2.5-72B to score 0.77-0.86 with zero training
- **Positive**: Clear separation: environment teaches, agent learns
- **Negative**: Untrained LLMs may loop or stagnate on harder tasks. This is expected and desirable — it's exactly what RL training should fix.

## Validation

| Configuration | PAN (Easy) | Passport (Med) | DL (Hard) | VR (Expert) |
|---------------|-----------|----------------|-----------|-------------|
| Pure LLM (Qwen 72B) | 0.86 | 0.77 | 0.79 | 0.70 |
| Local rule-based agent | 0.91 | 0.92 | 0.90 | 0.90 |

The gap between the pure LLM and the rule-based agent (0.78 vs 0.91 avg) represents the **training opportunity** — the value that RL would add.
