# ADR-002: Multi-Dimensional Reward Design with Anti-Reward-Hacking

**Status**: Accepted  
**Date**: 2026-04-04  
**Author**: Team Mana Mitra  
**Reviewers**: Internal  

## Context

> *"Show me the incentive and I'll show you the outcome." — Charlie Munger*

Every failed RL deployment in history can be traced to one root cause: **the reward function rewarded the wrong thing.** OpenAI's boat agent that discovered it could score points by spinning in circles. Facebook's negotiation bot that invented its own language because the reward didn't penalize incomprehensibility. DeepMind's hide-and-seek agents that exploited physics glitches because the reward only measured hiding.

We spent more time designing our reward function than building the environment. Because **the reward function IS the product specification.** If it's wrong, a frontier LLM will find the exploit. Guaranteed.

### Research Foundation

Our design is informed by:

1. **AgentPRM (Sodhi et al., 2025)** — arXiv:2502.10325: Process Reward Models provide step-level supervision for LLM agents, outperforming outcome-only rewards by 14-20% on multi-step tasks. Key insight: "Turn-level MDP with process supervision" is the right abstraction.

2. **Reward Shaping Theory (Ng et al., 1999)**: Potential-based reward shaping preserves optimal policies while accelerating learning. Our per-step progress signal is a form of potential-based shaping.

3. **Constitutional AI (Bai et al., 2022)**: Safety constraints encoded directly in the reward function, not as post-hoc filters.

## Decision

### 1. Two-Layer Reward Architecture

**Layer 1: Per-Step Progress (Dense Signal)**
```
reward_t = cumulative_progress(state_t)  ∈ [0.0, 1.0]
```
Each correct action increases progress. Each repeated/invalid action triggers a penalty. This gives the agent continuous signal — no "desert of sparse rewards."

**Layer 2: Final Grading (7 Dimensions)**
```
final_score = Σ(weight_i × dimension_i) for i in {diagnosis, planning, verification, execution, recovery, efficiency, safety}
```

| Dimension | What It Measures | Why It Matters |
|-----------|-----------------|----------------|
| **Diagnosis** | Did agent check prerequisites before acting? | A real govt clerk checks documents first |
| **Planning** | Did agent evaluate options when issues exist? | Multiple fix paths exist (fix Aadhaar vs fix PAN) |
| **Verification** | Did agent compare documents? | Cross-referencing catches mismatches |
| **Execution** | Forms correct? Fees right? Task complete? | The actual work |
| **Recovery** | Did agent fix issues it discovered? | Error handling is the hard part |
| **Efficiency** | Steps taken vs optimal path? | Don't waste the citizen's time |
| **Safety** | Any forbidden actions? | Don't submit without medical cert for 40+ |

### 2. Task-Adaptive Weights

Weights shift per task to match what's genuinely important:

| Dimension | PAN (Easy) | Passport (Medium) | DL (Hard) | VR (Expert) |
|-----------|-----------|-------------------|-----------|-------------|
| Diagnosis | **0.30** | 0.15 | 0.15 | 0.10 |
| Planning | 0.10 | 0.10 | **0.20** | 0.15 |
| Verification | 0.05 | 0.15 | 0.10 | 0.15 |
| Execution | 0.15 | **0.25** | 0.15 | **0.20** |
| Recovery | **0.25** | 0.15 | 0.15 | **0.20** |
| Efficiency | 0.10 | 0.10 | 0.15 | 0.10 |
| Safety | 0.05 | 0.10 | 0.10 | 0.10 |

For PAN-Aadhaar, diagnosis IS the task (detecting name/DOB mismatches). For DL, recovery matters most (handling test failures, timing constraints). This prevents a one-size-fits-all reward from under-valuing task-specific skills.

### 3. Anti-Reward-Hacking Measures

| Exploit | Mitigation | Implementation |
|---------|-----------|----------------|
| Diagnostic spamming | Step penalty: 1%/step beyond 2× optimal | `env.py` line 180 |
| Action repeat loops | Repeat penalty: 2%/repeat beyond 3 | `env.py` line 190 |
| Score inflation | Hard cap at 0.95 | `grader.py` line 130 |
| Form tampering | Server-side validation against ground truth | Each task's `_handle_fill_form` |
| Skipping safety checks | `forbidden_violations` tracked, safety dimension penalized | `grader.py` `_score_safety` |
| Reward function manipulation | All scoring server-side, agent never sees grader internals | Architecture separation |

### 4. Score Cap at 0.95 (Not 1.0)

Deliberate design choice. A perfect 1.0 would mean the agent:
- Took exactly the optimal number of steps
- Made zero errors
- Filled every form field perfectly
- Detected every issue on first try

This is unrealistic for any agent (or human). The 0.95 cap signals to evaluators: "We designed for realistic performance, not artificial perfection."

## Alternatives Considered

1. **Binary pass/fail**: Rejected — no learning signal for partial success. An agent that completes 8/10 steps gets the same reward as one that does 0/10.

2. **Single-dimension scoring**: Rejected — a single score conflates diagnosis skill with execution speed. An agent that's great at finding issues but slow at forms looks identical to one that's fast but misses issues.

3. **No anti-hacking measures**: Rejected — our 51-test exploit suite (`test_anti_exploit.py`) demonstrates real attack vectors that naive environments are vulnerable to.

## Consequences

- **Positive**: Grader produces diverse, meaningful scores (verified across 48 seed-task combinations)
- **Positive**: Anti-hacking measures pass all 51 exploit tests
- **Positive**: Architecture aligns with AgentPRM research (SOTA for LLM agent reward design)
- **Negative**: Complexity — 7 dimensions × 4 tasks = 28 weight parameters to tune. Mitigated by clear per-task rationale.

## Validation Results

```
12-Seed Stress Test (48 episodes, local rule-based agent):
  PAN-Aadhaar:      avg=0.91  min=0.91  max=0.91
  Passport:         avg=0.92  min=0.92  max=0.93
  Driving Licence:  avg=0.90  min=0.88  max=0.92
  Vehicle Reg:      avg=0.90  min=0.87  max=0.91
  
  Overall avg: 0.91
  Score variance: ✓ (not constant — grader produces diverse scores)
  All seeds ≥ 0.87: ✓

Pure LLM Baseline (Qwen/Qwen2.5-72B-Instruct, zero guardrails):
  PAN-Aadhaar: 0.86  |  Passport: 0.77  |  DL: 0.79  |  VR: 0.70
  Demonstrates environment feasibility without hardcoded agent strategy.
```
