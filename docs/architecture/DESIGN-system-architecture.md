# System Architecture: Government Services Navigator

**Version**: 1.0  
**Date**: April 2026  
**Author**: Team Mana Mitra  

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INFERENCE LAYER                              │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  inference.py (Agent)                                        │    │
│  │                                                              │    │
│  │  LLM (GPT-4o / Qwen / Nemotron)                           │    │
│  │       │                                                      │    │
│  │       ▼                                                      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │    │
│  │  │ GUARD 1  │→ │ GUARD 2  │→ │ GUARD 3  │→ │ GUARD 4  │    │    │
│  │  │ Validate │  │ Auto-fill│  │  Error   │  │  Loop    │    │    │
│  │  │ Actions  │  │  Forms   │  │ Recovery │  │Detection │    │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │    │
│  │  Guards 1,2,4 relaxed on Hard/Expert tasks after step 3    │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                              │ HTTP                                   │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────────┐
│                         SERVER LAYER                                  │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  FastAPI (app.py) — port 7860                                │    │
│  │  /reset  /step  /state  /health  /schema  /metadata  /tasks  │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  GovtServicesEnv (env.py) — Orchestrator                     │    │
│  │  • Episode lifecycle (reset → step → done)                   │    │
│  │  • Step penalty calculation                                  │    │
│  │  • Action repeat penalty                                     │    │
│  │  • Trajectory recording                                      │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│  ┌──────────┬──────────┬──────────┬─────────────────────────────┐    │
│  │   PAN    │ Passport │    DL    │     Vehicle Registration    │    │
│  │  (Easy)  │ (Medium) │  (Hard)  │        (Expert)             │    │
│  │          │          │          │                              │    │
│  │ • Name   │ • Docs   │ • 2-phase│ • Insurance                 │    │
│  │   match  │ • Forms  │ • Tests  │ • Inspection                │    │
│  │ • DOB    │ • Photo  │ • Timing │ • Hypothecation             │    │
│  │   match  │ • Addr   │ • Medical│ • Road tax                  │    │
│  └──────────┴──────────┴──────────┴─────────────────────────────┘    │
│                              │                                        │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Grader (grader.py) — 7-Dimension Scoring                   │    │
│  │                                                              │    │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌────┐ ┌──┐ │    │
│  │  │Diag  │ │Plan  │ │Verify│ │Exec  │ │Recov │ │Eff │ │Sa│ │    │
│  │  │0.05- │ │0.10- │ │0.05- │ │0.15- │ │0.15- │ │0.10│ │fe│ │    │
│  │  │0.30  │ │0.20  │ │0.15  │ │0.25  │ │0.25  │ │0.15│ │ty│ │    │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └────┘ └──┘ │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Pydantic Models (models.py)                                 │    │
│  │  Action, Observation, EnvironmentState, Reward, Trajectory   │    │
│  │  CitizenProfile, DocumentInfo, GroundTruth                   │    │
│  │  All extend OpenEnv base classes                             │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: One Episode

```
1. Agent → POST /reset {task: "driving_licence", seed: 42}
   └→ Server generates CitizenProfile (age=45, medical needed, address mismatch)
   └→ Server computes GroundTruth (optimal=11 steps, expected_issues=[medical_cert_missing, address_mismatch])
   └→ Returns Observation (citizen_summary, available_actions, status_summary)

2. Agent → POST /step {action_type: "check_prerequisites", parameters: {}}
   └→ Task state machine processes action
   └→ Discovers: medical_cert_missing, address_mismatch
   └→ Adds to pending_issues
   └→ Returns Observation (progress=0.18, hints="Resolve pending issues...")

3. [Steps 2-10: Agent follows hints, fixes issues, fills forms, pays fees, takes tests]

4. Agent → POST /step {action_type: "check_status", parameters: {}}
   └→ Task marks done=True
   └→ Grader evaluates full trajectory against GroundTruth
   └→ Returns final reward=0.87, info={final_grade: {breakdown: [...]}}
```

---

## Episode State Machine (Generic)

```
                    ┌─────────────────┐
                    │   reset()       │
                    │ Generate citizen│
                    │ Compute ground  │
                    │ truth           │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ check_          │ ← Diagnostic phase
                    │ prerequisites   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
              ┌─────│ Issues found?   │─────┐
              │ Yes └─────────────────┘ No  │
              │                             │
     ┌────────▼────────┐           ┌────────▼────────┐
     │ evaluate_options │           │ fill_form        │
     │ fix_document     │           │ pay_fee          │
     │ gather_document  │           │ book_appointment │
     └────────┬────────┘           │ submit           │
              │                    └────────┬─────────┘
              │ (issues resolved)           │
              └────────────┬────────────────┘
                           │
                  ┌────────▼────────┐
                  │ check_status    │ ← Terminal
                  │ done=True       │
                  └─────────────────┘
```

---

## Testing Strategy

```
┌─────────────────────────────────────────────────────┐
│                  Test Pyramid                        │
│                                                      │
│                    ┌──────┐                          │
│                   /  GPT   \     ← 2 models tested   │
│                  /  4o/Qwen  \      (real LLM)       │
│                 /____________\                        │
│                /  12-Seed     \   ← 48 episodes      │
│               /  Stress Test   \    (hint agent)     │
│              /__________________\                     │
│             /  51 Anti-Exploit   \  ← Security       │
│            /  Tests               \                   │
│           /________________________\                  │
│          /  41 Unit Tests (env)     \  ← Core logic  │
│         /  7 Smoke Tests             \               │
│        /  3 VR-specific Tests         \              │
│       /________________________________\             │
│                                                      │
│  Total: 102 tests, all passing                       │
└─────────────────────────────────────────────────────┘
```

---

## File Structure

```
govt-services-navigator/
├── inference.py              # Baseline agent (OpenAI client)
├── openenv.yaml              # Environment metadata
├── Dockerfile                # Container deployment
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── pyproject.toml            # Project config
│
├── server/
│   ├── app.py                # FastAPI endpoints
│   ├── env.py                # Environment orchestrator
│   ├── grader.py             # 7-dimension grader
│   ├── models.py             # Pydantic models (OpenEnv base)
│   └── tasks/
│       ├── task_pan_aadhaar.py       # Easy (4-6 steps)
│       ├── task_passport.py          # Medium (10-13 steps)
│       ├── task_driving_licence.py   # Hard (11-15 steps)
│       └── task_vehicle_registration.py  # Expert (10-18 steps)
│
├── tests/
│   ├── test_env.py           # 41 environment tests
│   ├── test_smoke.py         # 7 smoke tests
│   ├── test_vehicle_reg.py   # 3 VR-specific tests
│   └── test_anti_exploit.py  # 51 anti-reward-hacking tests
│
├── docs/
│   ├── VISION.md
│   ├── architecture/
│   │   ├── ADR-001-environment-architecture.md
│   │   ├── ADR-002-reward-design.md
│   │   ├── ADR-003-agent-error-recovery.md
│   │   └── DESIGN-system-architecture.md
│   ├── product/
│   │   └── PRD-mana-mitra-phase3.md
│   └── research/
│       ├── TECH-PROPOSAL-process-reward-models.md
│       └── RESEARCH-rl-best-practices.md
│
└── scripts/
    ├── quick_test.py         # 12-seed stress test
    ├── debug_test.py         # Single-task debugger
    └── stress_test.py        # Extended stress test
```
