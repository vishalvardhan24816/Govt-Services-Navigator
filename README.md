# Government Services Navigator — OpenEnv RL Environment

An RL environment that simulates Indian government service navigation. Trains AI agents to help citizens complete multi-step bureaucratic processes including PAN-Aadhaar linking, passport application, driving licence acquisition, and vehicle registration.

**Inspired by [Mana Mitra](https://manamitra.ap.gov.in/)** — a real government initiative in Andhra Pradesh that has served 2 crore+ services to 40 lakh+ users, proving the demand for AI-assisted government navigation.

## Why This Environment?

Every year, millions of Indian citizens struggle with government services — wrong documents, name mismatches, missed deadlines, unclear processes. This environment captures these real-world challenges:

- **Multi-step workflows** with dependencies and prerequisites
- **Realistic failure modes** sourced from official government portals
- **Multiple valid solution paths** (e.g., fix Aadhaar name vs fix PAN name)
- **Cascading consequences** (wrong action = weeks of delay for the citizen)
- **Process supervision** — rewards diagnostic thinking, not just final outcomes

## Tasks

| Task | Difficulty | Description | Max Steps |
|------|-----------|-------------|-----------|
| `pan_aadhaar_link` | Easy | Link PAN with Aadhaar on Income Tax portal. Identity documents may contain discrepancies. | 15 |
| `passport_fresh` | Medium | Apply for fresh passport via PSK. Document verification, application, in-person verification. | 25 |
| `driving_licence` | Hard | Two-phase driving licence process with age restrictions and document validity constraints. | 30 |
| `vehicle_registration` | Expert | Register a new vehicle at RTO. Compliance checks, vehicle inspection, permanent registration. | 35 |

### Difficulty Progression

- **Easy**: Linear flow, 2 documents, common single mismatch
- **Medium**: Multiple documents, form validation, multi-stage process with document cross-verification
- **Hard**: Two-phase process with timing constraints, test failures, age restrictions, LL expiry window
- **Expert**: 8+ documents, multi-agency coordination (RTO, insurance, bank), inspection failures, hypothecation clearance

## Action Space

### Diagnostic Actions (show reasoning)
| Action | Description | Parameters |
|--------|-------------|------------|
| `check_prerequisites` | Verify documents and eligibility | `{}` |
| `compare_documents` | Compare fields across documents | `{}` |
| `evaluate_options` | List resolution options for issues | `{}` |
| `check_eligibility` | Check citizen eligibility | `{}` |
| `check_status` | Check current progress | `{}` |

### Execution Actions
| Action | Description | Parameters |
|--------|-------------|------------|
| `gather_document` | Collect a document | `{"target": "<doc_type>"}` |
| `fill_form` | Fill application form | `{field: value, ...}` |
| `pay_fee` | Pay required fee | `{"amount": <number>}` |
| `book_appointment` | Book appointment | `{}` |
| `submit_application` | Submit at government office | `{}` |
| `fix_document` | Fix document issue | `{"target": "<field>"}` |
| `take_test` | Take written/driving test | `{"test_type": "written"\|"practical"}` |
| `wait` | Wait for time to pass | `{"days": <number>}` |

## Observation Space

Each observation includes:
- **task_description**: What the agent needs to accomplish
- **citizen_summary**: Full citizen profile with document details
- **current_phase**: Current stage of the process
- **services_status**: Status of each service (not_started/in_progress/blocked/completed)
- **completed_steps**: What has been done so far
- **pending_issues**: Unresolved problems blocking progress
- **last_action_result**: Detailed result of the last action taken
- **available_actions**: Currently valid actions
- **status_summary**: Current phase of the process (non-prescriptive — agent reasons from state)
- **progress_pct**: Cumulative progress (0.0 to 1.0)

## Reward Design

### Per-Step Reward
Cumulative progress signal (0.0 → 1.0) that increases with each correct action. Provides continuous learning signal, not just end-of-episode.

### Final Grading (7 Dimensions)
| Dimension | Description | Weight varies by task |
|-----------|-------------|----------------------|
| Diagnosis | Checked prerequisites before acting? | 0.05–0.30 |
| Planning | Evaluated options, detected issues? | 0.10–0.20 |
| Verification | Verified documents correctly? | 0.05–0.15 |
| Execution | Forms correct, fees right, docs submitted? | 0.15–0.25 |
| Recovery | Handled failures and resolved issues? | 0.15–0.25 |
| Efficiency | Steps taken vs optimal path? | 0.10–0.15 |
| Safety | Avoided harmful/forbidden actions? | 0.05–0.10 |

Weights adapt per task — e.g., "diagnosis" is 0.30 for PAN-Aadhaar (where mismatch detection IS the task) but 0.15 for passport (where form accuracy matters more).

Final score capped at 0.95. Safety violations reduce the safety dimension toward 0.0 (each violation costs 0.3).

## Setup & Usage

### Local Development
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t govt-services-navigator .
docker run -p 7860:7860 govt-services-navigator
```

### API Endpoints
```
GET  /health  → Health check
POST /reset   → Start episode: {"task": "pan_aadhaar_link", "seed": 42}
POST /step    → Take action: {"action_type": "check_prerequisites", "parameters": {}}
GET  /state   → Current state
GET  /tasks   → List available tasks
```

### Run Inference
```bash
export HF_TOKEN=your_token
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

### Run Tests
```bash
pytest tests/ -v
```

## Baseline Scores

### Pure LLM Baseline (Qwen/Qwen2.5-72B-Instruct, zero guardrails, seed=42)
| Task | Score | Notes |
|------|-------|-------|
| `pan_aadhaar_link` | **0.86** | Full run, correct action sequence |
| `passport_fresh` | **0.77** | Task completed, learned from errors |
| `driving_licence` | **0.79** | Near-complete, two-phase reasoning |
| `vehicle_registration` | **0.70** | Partial run (API credits) |

The inference script is a pure LLM baseline — no hardcoded strategy, no guardrails. The LLM reasons entirely from observations, rewards, and error messages.

### 12-Seed Stress Test (Local Rule-Based Agent)
| Task | Avg | Min | Max |
|------|-----|-----|-----|
| `pan_aadhaar_link` | 0.91 | 0.91 | 0.91 |
| `passport_fresh` | 0.92 | 0.92 | 0.93 |
| `driving_licence` | 0.90 | 0.88 | 0.92 |
| `vehicle_registration` | 0.90 | 0.87 | 0.91 |

All 48/48 task-seed combinations score ≥ 0.87. Overall avg: **0.91**.

**Anti-reward-hacking**: Step penalty (1%/step beyond 2× optimal) + action repeat penalty (2%/repeat beyond 3) prevent agents from inflating scores via diagnostic spamming. Score capped at 0.95.

*Scores vary by citizen profile (randomized complications). Run multiple seeds for representative distribution.*

## Documentation

> **Start here: [The Vision — Making Government Work for Every Citizen](docs/VISION.md)**

### Strategy & Product
- **[VISION](docs/VISION.md)** — Why this matters, the $40B problem, Meta/WhatsApp connection, roadmap to national scale
- **[PRD: Mana Mitra Phase 3](docs/product/PRD-mana-mitra-phase3.md)** — Full product spec: Mana Mitra's evolution from digitization → chatbots → AI agents. The stories of Rajesh, Priya, and Suresh — real citizens, real problems.

### Research & Technical
- **[Tech Proposal: Process Reward Models](docs/research/TECH-PROPOSAL-process-reward-models.md)** — AgentPRM-aligned architecture, comparison vs WebArena/SWE-bench/ALFWorld
- **[Research: RL Best Practices](docs/research/RESEARCH-rl-best-practices.md)** — Audit against 5 major RL problems with evidence of mitigation
- **[System Architecture](docs/architecture/DESIGN-system-architecture.md)** — Full architecture diagrams, data flow, testing pyramid

### Architecture Decision Records
- **[ADR-001: Environment Architecture](docs/architecture/ADR-001-environment-architecture.md)** — Stateful MDP, citizen generation, forgiving API
- **[ADR-002: Reward Design](docs/architecture/ADR-002-reward-design.md)** — 7-dimension grading, anti-reward-hacking (51 exploit tests)
- **[ADR-003: Pure LLM Baseline](docs/architecture/ADR-003-agent-error-recovery.md)** — Environment-driven error recovery, reward trend feedback, no hardcoded strategy

## Sources

All rules verified against official Indian government portals (April 2026):
- [incometax.gov.in](https://incometax.gov.in) — PAN-Aadhaar linking
- [passportindia.gov.in](https://passportindia.gov.in) — Passport application
- [parivahan.gov.in](https://parivahan.gov.in) — Driving licence & vehicle registration
- [uidai.gov.in](https://uidai.gov.in) — Aadhaar updates

## Limitations

- Covers standard adult citizens with common document types
- Edge cases (NRI, minors under 16, name changes post-marriage) planned for v2
- Simulated time is simplified — real-world processing times vary
- Fee amounts and rules as of April 2026; may change

## License

MIT
