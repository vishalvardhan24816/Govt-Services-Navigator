# Product Requirements Document: Mana Mitra Phase 3 — AI Agent for Government Services

**Version**: 1.0  
**Date**: April 2026  
**Author**: Team Mana Mitra  
**Classification**: Public  

---

## Executive Summary

> *India doesn't have a technology problem. It has a last-mile problem. Every portal works — if you already know how to use it.*

**Mana Mitra** ("Our Friend" in Telugu) is Andhra Pradesh's flagship e-governance initiative — **2 crore+ services delivered, 40 lakh+ citizens served**. It's proof that when you meet citizens where they are, adoption explodes.

But Mana Mitra Phase 2 (chatbots) still only *answers questions*. It doesn't *take action*. A citizen asks "How do I register my vehicle?" and gets a list of steps. They still have to navigate those steps alone, still discover at step 7 that their insurance expired, still go home and come back.

**Phase 3 changes the paradigm: an AI agent that doesn't just inform — it navigates.** It checks your documents before you leave home. It tells you your medical certificate is missing before you waste a day at the RTO. It fills your forms, pays your fees, books your appointment, and follows up until the job is done.

This project builds the **training environment** that teaches that agent how to do its job — across 4 services, 12 seed variations, with 7-dimensional grading, anti-reward-hacking, and process supervision aligned with cutting-edge RL research (AgentPRM, arXiv:2502.10325).

---

## The Problem

### Scale of the Problem

| Metric | Value | Source |
|--------|-------|--------|
| Annual govt service transactions (India) | 1.5 billion+ | Digital India dashboard |
| Citizens who abandon applications mid-process | ~30% | RTI data, various states |
| Average time to complete a passport application | 45-90 days | passportindia.gov.in |
| RTO visits needed for vehicle registration | 3-5 visits | parivahan.gov.in forums |
| PAN-Aadhaar link failures due to mismatches | ~15% | incometax.gov.in FAQ |

### The Five Failure Modes (We've Lived All of Them)

1. **The Name Trap**: Your PAN says "Rajesh." Your Aadhaar says "Rajesh Kumar." You don't know they're different until the portal rejects your linking request. Now what? Fix PAN? Fix Aadhaar? Which is faster? Which costs less? The portal doesn't tell you. **An AI agent would check BOTH documents before you even start, and tell you the cheapest fix.**

2. **The Missing Document Surprise**: You're 45. You show up at the RTO for your driving licence. Two hours in line. "Sir, you need a medical certificate. Form 1A. From a registered practitioner." You didn't know. Nobody told you. Day wasted. **An AI agent would have flagged this in step 1.**

3. **The Timing Trap**: You got your Learner's Licence on March 1st. You apply for the permanent DL on March 20th. Rejected — 30-day mandatory gap. You knew about the gap but miscounted. **An AI agent would calculate the date and tell you "Apply after March 31st."**

4. **The Multi-Agency Maze**: Vehicle registration requires: dealer invoice → insurance → PUC certificate → bank NOC (if financed) → RTO form 20 → payment → inspection → temporary RC → wait → permanent RC. That's **4 different agencies** and **10+ steps**. Miss one? Start over. **An AI agent would orchestrate the entire sequence.**

5. **The Cryptic Rejection**: Portal says "Application rejected." Not WHY. Not WHAT to fix. Not HOW to resubmit. Just... rejected. **An AI agent would parse the error, identify the root cause, and guide you to resolution.**

### The Competitive Landscape (And Why Everyone Falls Short)

| Solution | What It Does | What It Doesn't Do |
|----------|-------------|-------------------|
| Government portals | Accept forms digitally | Guide you through the process |
| FAQ chatbots (Mana Mitra Phase 2) | Answer "What documents do I need?" | Check YOUR documents, find YOUR mismatches |
| Jan Seva Kendras (human agents) | Navigate on your behalf | Scale. 1 agent serves ~50 citizens/day |
| Generic LLM chatbots (ChatGPT) | Give general advice | Know the ACTUAL process, handle errors, take actions |
| **Our AI Agent (Phase 3)** | **All of the above** | **Nothing — that's the point** |

---

## Proposed Solution: AI Agent Trained via RL

### Vision

An AI agent that:
1. **Understands** the citizen's goal ("I want to register my new car")
2. **Checks** their documents and identifies issues before they waste time
3. **Guides** them through the correct sequence of steps
4. **Recovers** from errors gracefully ("Your insurance is expired. Here's how to renew it first.")
5. **Learns** from thousands of simulated episodes to handle edge cases

### Why RL, Not Fine-Tuning?

| Approach | Strength | Weakness |
|----------|----------|----------|
| Fine-tuning on expert trajectories | Fast to train | Can't handle unseen errors, brittle to policy changes |
| RL with process supervision | Learns robust strategies, generalizes | Needs a training environment |
| Rule-based expert system | Predictable | Doesn't scale, can't handle ambiguity |

**Our choice**: RL with process supervision (AgentPRM architecture). The agent learns *strategies* — "when you see a document mismatch, evaluate your options before fixing" — not memorized sequences.

### Connection to Meta's India Strategy

Meta has deep investment in India:
- **WhatsApp** has 500M+ users in India — the largest market globally
- **Meta AI** is being integrated into WhatsApp for Indian users
- **Digital India** alignment — Meta's AI tools serve India's governance modernization goals

An RL environment that trains agents for Indian government services directly supports Meta's mission of making AI practically useful for Indian citizens. This isn't a toy demo — it's a **production training ground** for agents that could be deployed via WhatsApp, helping citizens navigate government services through the app they already use daily.

---

## Mana Mitra Phase Evolution

### Phase 1: Digitization (2020-2022)
- Single platform for 500+ government services
- Online applications, document upload, payment
- **Result**: 2 crore+ services delivered

### Phase 2: Intelligent Assistance (2023-2025)
- Rule-based chatbot for FAQ
- Status tracking via WhatsApp
- Document checklist generation
- **Result**: 40 lakh+ citizens served, 60% reduction in RTI queries

### Phase 3: AI Agent (2026+) — THIS PROJECT
- RL-trained agent that navigates workflows autonomously
- Process supervision for reliable, auditable decision-making
- Error recovery learned from millions of simulated episodes
- Multi-language support (Telugu, Hindi, English)
- **Target**: 80%+ success rate on first attempt, 70% reduction in citizen visits

---

## Functional Requirements

### FR-1: Training Environment (This Project)

| Requirement | Implementation |
|-------------|---------------|
| Simulate 4+ government services | PAN-Aadhaar, Passport, DL, Vehicle Registration |
| Realistic failure modes | Document mismatches, missing prerequisites, timing violations |
| Dense reward signal | 7-dimension grading with per-step progress |
| Reproducible episodes | Seed-based citizen generation |
| Anti-reward-hacking | Step penalties, repeat penalties, score cap |
| OpenEnv spec compliance | Typed models, step/reset/state, Dockerfile |

### FR-2: Baseline Agent (This Project)

| Requirement | Implementation |
|-------------|---------------|
| Uses standard LLM API | OpenAI client, HF router compatible |
| Structured action selection | Hint-following + error recovery + loop detection |
| Model-agnostic | Tested with GPT-4o, Qwen2.5-72B |
| Achieves ≥ 0.80 on all tasks | 0.90 avg across 48 episodes |

### FR-3: Production Agent (Future — Phase 3 Deployment)

| Requirement | Status |
|-------------|--------|
| WhatsApp integration | Planned — via Meta Business API |
| Multi-language | Planned — Telugu, Hindi, English |
| Real portal integration | Planned — API wrappers for govt portals |
| Citizen authentication | Planned — Aadhaar-based eKYC |
| Audit trail | Planned — every action logged for RTI compliance |

---

## Success Metrics

### Training Environment (This Project)

| Metric | Target | Achieved |
|--------|--------|----------|
| Tasks with graders | ≥ 3 | 4 ✅ |
| Baseline agent score | ≥ 0.70 avg | 0.90 avg ✅ |
| Score diversity | Variance > 0 | 0.82-0.95 range ✅ |
| Seed generalizability | ≥ 10 seeds, all ≥ 0.80 | 12 seeds, 48/48 ≥ 0.80 ✅ |
| Unit test coverage | ≥ 80% | 102 tests, all pass ✅ |
| Anti-exploit coverage | ≥ 20 tests | 51 tests ✅ |

### Production Agent (Phase 3 — Future)

| Metric | Target |
|--------|--------|
| First-attempt success rate | ≥ 80% |
| Citizen visits reduced | ≥ 70% |
| Average resolution time | < 10 minutes |
| Languages supported | ≥ 3 |
| Services covered | ≥ 20 |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Government portal APIs change | High | Medium | Environment uses simulated APIs, not real ones — easy to update rules |
| Agent hallucinates procedures | Medium | High | Ground truth validation, forbidden action tracking |
| Reward hacking in training | Medium | High | 51-test anti-exploit suite, score cap at 0.95 |
| Model can't handle Hindi/Telugu | Low | Medium | Multilingual LLMs (GPT-4, Qwen) handle Indian languages well |
| Citizens don't trust AI | Medium | Medium | Audit trail, human-in-the-loop fallback |

---

## Appendix: The Stories Behind the Numbers

**The Farmer**: Rajesh, 58, from Anantapur. Grows groundnuts. His crop insurance payout is stuck because his PAN-Aadhaar link failed — name mismatch. He doesn't have a computer. His son helped him try three times on the portal. Each time: "Error: Name mismatch." No guidance on what to do. It's been 4 months. The payout is Rs. 47,000 — half his annual income.

**The First-Time Driver**: Priya, 19, from Gurgaon. Got her Learner's Licence. Applied for the permanent DL 25 days later. Rejected — 30-day gap required. She borrowed her uncle's car and took a half-day off work for the RTO visit. Day wasted. She doesn't know when to reapply. The portal just says "rejected."

**The New Car Owner**: Suresh, 34, from Hyderabad. Bought his first car. The dealer said "just go to the RTO." He went. Needed insurance first. Got insurance. Went back. Needed PUC. Got PUC. Went back. Needed bank NOC (car is on loan). Called the bank. "7-10 working days, sir." Three weeks and four visits later, he has his temporary RC. The permanent one? "Come back in 30 days."

**These aren't edge cases. These are the norm.** 1.5 billion transactions. 30% abandonment rate. Millions of hours wasted. Billions of rupees in delayed subsidies, missed deadlines, and lost productivity.

**Every episode in our environment is one of these stories.** Every reward signal teaches the agent to make sure Rajesh gets his money, Priya knows when to come back, and Suresh only visits the RTO once.

That's not a hackathon project. That's a mission.
