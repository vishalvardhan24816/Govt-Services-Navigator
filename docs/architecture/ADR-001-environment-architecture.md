# ADR-001: Environment Architecture — OpenEnv-Compliant Government Services Simulator

**Status**: Accepted  
**Date**: 2026-04-04  
**Author**: Team Mana Mitra  
**Reviewers**: Internal  

## Context

> *There are 900+ environments on HuggingFace Spaces tagged with `openenv`. Not one simulates something a billion people actually do every year.*

India processes **1.5 billion+ government service transactions annually**. Citizens encounter document mismatches, missing prerequisites, multi-phase workflows with mandatory wait periods, and inspection failures — the kind of multi-step, consequence-heavy, error-prone processes that LLMs are supposed to help with but nobody has built a training environment for.

The closest existing environments are WebArena (web UI clicks) and MiniWoB++ (toy browser tasks). These test mouse-clicking, not **procedural reasoning with document dependencies, timing constraints, and cascading consequences**. An agent that can click a button can't help a farmer link his PAN with Aadhaar.

We built what was missing.

## Decision

Build a **stateful, multi-task RL environment** that simulates 4 Indian government services with increasing complexity:

| Task | Difficulty | Steps | Key Challenge |
|------|-----------|-------|---------------|
| PAN-Aadhaar Linking | Easy | 4-6 | Document mismatch detection |
| Fresh Passport | Medium | 10-13 | Multi-document gathering, form accuracy |
| Driving Licence | Hard | 11-15 | Two-phase workflow, timing constraints, test failures |
| Vehicle Registration | Expert | 10-18 | Multi-agency coordination, inspection, hypothecation |

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FastAPI Server                     │
│                    (port 7860)                        │
├──────────┬──────────┬──────────┬─────────────────────┤
│  /reset  │  /step   │ /state   │  /health /schema    │
├──────────┴──────────┴──────────┴─────────────────────┤
│                  GovtServicesEnv                      │
│         (env.py — orchestrator layer)                │
├──────────┬──────────┬──────────┬─────────────────────┤
│ PAN Task │ Passport │    DL    │    Vehicle Reg      │
│          │   Task   │   Task   │      Task           │
├──────────┴──────────┴──────────┴─────────────────────┤
│              Grader (7-dimension scoring)             │
├──────────────────────────────────────────────────────┤
│         Pydantic Models (typed, validated)            │
└──────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Stateful MDP**: Each task is a turn-level Markov Decision Process. State = (citizen profile, documents, completed steps, pending issues, simulated day). Actions are function calls with typed parameters.

2. **Citizen Profile Generation**: Each `reset(seed=N)` generates a unique citizen with randomized complications — name mismatches, expired documents, age-based restrictions. The seed ensures reproducibility while the randomization ensures generalizability.

3. **Ground Truth Precomputation**: On reset, we compute the optimal solution path, expected issues, correct form values, and correct fees. The grader compares the agent's trajectory against this hidden ground truth.

4. **Forgiving Environment (Principle of Least Surprise)**: Both `fix_document` and `gather_document` resolve the same issues. At a real government office, the clerk doesn't care which verb you use — they care that you brought the right document. Our environment mirrors this.

5. **Dense Reward Signal**: Per-step cumulative progress (0.0 → 1.0) provides continuous learning signal. Not just terminal reward.

## Alternatives Considered

1. **UI-based environment (Selenium/Playwright)**: Rejected — adds browser complexity without testing the core skill (procedural reasoning). Also fails the vcpu=2 constraint.

2. **Single-task environment**: Rejected — doesn't demonstrate curriculum learning. Ben from HuggingFace explicitly said "long running tasks with multiple trajectories" are rewarded.

3. **Hardcoded scenarios (no randomization)**: Rejected — agents would memorize solutions. Seed-based citizen generation ensures every episode is unique.

## Consequences

- **Positive**: Environment is usable by ANY LLM agent (tested with GPT-4o and Qwen2.5-72B). No task-specific agent code needed.
- **Positive**: 102 unit tests + 48/48 seed tests ≥ 0.80 demonstrate robustness.
- **Negative**: Single-agent, single-thread — doesn't model queue contention or inter-department delays. Acceptable for v1.
- **Negative**: Deterministic action outcomes — `fix_document` always succeeds. Real-world has stochastic failures. Planned for v2.

## References

- [OpenEnv Specification](https://github.com/huggingface/openenv)
- [Mana Mitra — AP Government](https://manamitra.ap.gov.in/) — 2 crore+ services, 40 lakh+ users
- [AgentPRM: Process Reward Models for LLM Agents](https://arxiv.org/abs/2502.10325)
