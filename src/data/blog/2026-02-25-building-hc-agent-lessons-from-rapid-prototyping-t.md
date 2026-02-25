---
author: Robin
pubDatetime: 2026-02-25T07:30:00-03:00
title: "Building hc-agent: Lessons from Rapid Prototyping to Production"
description: "A technical post-mortem on building an AI-first cloud automation framework: from a rapid prototype built with Codex on a plane to a production-ready system, and the hard lessons learned along the way."
tags:
  - ai
  - llm
  - architecture
  - refactoring
  - huawei-cloud
  - lessons-learned
  - codex
  - agent
featured: true
draft: false
---

# Building hc-agent: Lessons from Building an AI-First Cloud Automation Framework

**Author**: Robin  
**Date**: February 25, 2026  
**Project**: hc-agent (Huawei Cloud Agent)

---

## TL;DR

What started as a rapid prototype built with Codex on a plane became a 15K+ line codebase with serious architectural debt. This is the story of how we evolved from "move fast and break things" to "move deliberately and build right," and the hard lessons learned along the way.

**Key Takeaways**:
1. 📋 **Define tasks before writing code** — not after
2. 🎯 **Code ≠ Productivity** — resist the illusion
3. 🤖 **Ask the LLM how, not just what** — keep learning
4. 🏗️ **Own the architecture decisions** — LLMs can't do this for you
5. 🔄 **Embrace refactoring** — structural change is not failure
6. 🧬 **Agent code must co-evolve with LLMs** — it's never "done"

---

## The Beginning: A Plane Ride and a Bold Idea

It started on a flight from São Paulo to Santiago. I had an idea: what if we could make Huawei Cloud resources as easy to manage as talking to an assistant?

Armed with Codex and 8 hours of flight time, I built the first working prototype of **hc-agent** — a natural language interface to Huawei Cloud.

By the time we landed, it could:
- Create VPCs, subnets, and security groups
- Launch ECS instances
- Set up basic networking

The demo worked. The architecture? Not so much.

---

## The Problem: When "Moving Fast" Becomes "Moving Blindly"

**What we did right:**
- ✅ Shipped a working prototype in 8 hours
- ✅ Validated the core idea with real users
- ✅ Learned what customers actually needed

**What went wrong:**
- ❌ No task definitions upfront
- ❌ Letting Codex write code without reviewing architecture
- ❌ Treating "lines of code" as a success metric
- ❌ Skipping design docs to "move faster"

By the time we reached production, we had:
- **15,000+ lines of code**
- **5 architectural layers** (with circular dependencies)
- **12% success rate** on E2E tests
- **3.7 errors per task** on average

---

## The Turning Point: Real Tests Don't Lie

The wake-up call came when we ran **real E2E tests** against actual Huawei Cloud APIs.

**First test (validation only)**:
```
✅ 4/4 tests passed
💵 Cost: $0.00
```

We celebrated. Then we looked closer.

**Second test (real API calls)**:
```
❌ 0/4 resources created
❌ Payload bugs: missing password, wrong AZ format, invalid VPC ID
💵 Cost: $0.00 (because the cloud rejected all requests)
```

The "success" was an illusion. We'd been testing code paths, not functionality.

**Lesson learned**: If a test costs $0, it's probably not testing the right thing.

---

## The Lessons: Hard-Won Truths

### 1. Define Tasks Before Writing Code

**The mistake:**
```
❌ "Build a CCE cluster creation feature"
   → Codex generates 500 lines
   → We discover the task was wrong
   → Refactor everything
```

**The fix:**
```
✅ "Build a CCE cluster creation feature"
   → Write task definition (inputs, outputs, edge cases, dependencies)
   → Review with team
   → THEN write code
   → 80% less refactoring needed
```

**Impact**: Task definition time went from 0 minutes to 15 minutes. Refactoring time went from 4 hours to 30 minutes.

---

### 2. Code ≠ Productivity

**Dangerous metric:**
- Lines of code written per day
- Number of features "completed"
- Speed of initial implementation

**Better metric:**
- Success rate on real tests
- Errors per task
- Time to fix bugs
- Code that survives refactoring

We went from celebrating "3,000 lines in one day" to celebrating "deleted 2,000 lines and tests still pass."

---

### 3. Ask the LLM "How?" — Not Just "What"

**Before:**
```
Prompt: "Create a function to query RDS instances"
→ Gets code
→ Code works (maybe)
→ Learn nothing
```

**After:**
```
Prompt: "Explain the best way to query RDS instances with error handling"
→ Gets explanation + code
→ Understand the approach
→ Can debug/extend it later
→ Knowledge compounds
```

**Result**: We stopped being "code copiers" and became "informed builders."

---

### 4. Own the Architecture Decisions

**What LLMs can do:**
- Write individual functions
- Suggest patterns
- Implement known algorithms

**What LLMs cannot do:**
- Decide if you need a state machine or event-driven architecture
- Choose between monolith vs microservices for your use case
- Refactor 15K LOC with circular dependencies

**The hard truth**: Architecture debt compounds faster than code debt.

We spent 3 days refactoring because we let Codex "just add another layer" instead of stepping back and designing.

---

### 5. Embrace Refactoring as Part of the Process

**Old mindset:**
- "We already wrote the code, refactoring is wasted time"
- "If we refactor, we admit the first version was wrong"

**New mindset:**
- "Refactoring is a sign we learned something"
- "Good code is rewritten code"

**Metrics before refactoring:**
- Success rate: 12%
- Errors per task: 3.7
- Code complexity: 5 nested layers

**Metrics after refactoring:**
- Success rate: 94%
- Errors per task: 0.3
- Code complexity: Clean state machine (7 steps)

**Time investment**: 2 weeks  
**Time saved over next 6 months**: Estimated 8+ weeks

---

### 6. Agent Code Must Co-Evolve with LLMs

**The insight:**
AI-first applications are never "done" because:
- LLMs improve every quarter
- Prompts that work today break tomorrow
- New capabilities unlock new patterns

**Example:**
- v1 (Jan 2026): Used GPT-4 for intent parsing
- v2 (Feb 2026): Switched to DeepSeek-V3.1 (10x cheaper, same quality)
- v3 (planned): Add multi-turn dialogue (impossible in v1 architecture)

**Strategy:**
- Design for prompt evolution
- Version prompts like code
- Test prompt changes like code changes
- Budget for LLM experiments

---

## The Numbers: Before and After

| Metric | Before Refactoring | After Refactoring |
|--------|-------------------|------------------|
| **Success Rate** | 12% | 94% |
| **Errors per Task** | 3.7 | 0.3 |
| **Code Size** | 15,000 LOC | 12,000 LOC |
| **Circular Dependencies** | 5 layers | 0 |
| **Test Coverage** | 60% | 95% |
| **Time to Add New Service** | 2-3 days | 30 minutes* |

\* With Service Profile Architecture (YAML-based service definitions)

---

## The Architecture Evolution

### v1.0: The Prototype (8 hours, on a plane)
```
User Input → LLM → SDK Call → Done
```
- ✅ Fast to build
- ❌ No error handling
- ❌ No validation
- ❌ No memory

### v2.0: The Production Attempt (3 months)
```
User Input → Intent Parser → LLM Planner → SDK Wrapper → Error Recovery → LLM Retry → Done
```
- ✅ Feature-complete
- ❌ 5 layers of abstraction
- ❌ Circular dependencies
- ❌ Hardcoded per service

### v3.0: The Refactored System (2 weeks redesign)
```
User Input → 7-Step State Machine → Service Profile (YAML) → Done
```
- ✅ Clean separation of concerns
- ✅ Service-specific logic in YAML configs
- ✅ LLM used strategically (not everywhere)
- ✅ New service = 30 min (was 3 days)

---

## The Biggest Surprise: The Fix Was Simple

After weeks of struggling with error recovery, we discovered the bug was trivial:

**Before (broken):**
```python
def generate_password(self, field_name):
    if "password" in field_name.lower():
        return None  # ❌ Returns None!
```

**After (fixed):**
```python
def generate_password(self, field_name):
    if "password" in field_name.lower():
        return secrets.token_urlsafe(16)  # ✅ Returns password
```

**Impact**: This one-line fix took success rate from 12% → 94%.

**Lesson**: Don't over-engineer before you've done real testing.

---

## What We'd Do Differently

### ✅ Do Again:
1. Rapid prototyping with Codex to validate ideas
2. Real API testing (even if it costs money)
3. Refactoring when architecture debt gets too high
4. Documenting lessons learned in real-time

### ❌ Avoid Next Time:
1. Skipping task definitions to "save time"
2. Treating LOC as a success metric
3. Letting LLMs make architecture decisions
4. Writing code before understanding the problem

### 🔄 Change:
1. Design-first approach (mandatory design docs)
2. Real tests from day 1 (not just validation)
3. Architecture reviews every 2 weeks
4. Metrics that matter (success rate, error rate, refactor frequency)

---

## The Team: Humans + AI

This project was built with:
- **Codex (OpenAI)**: Rapid prototyping, code generation
- **DeepSeek-V3.1**: Intent parsing, error analysis
- **Kimi (Moonshot AI)**: Testing, documentation
- **Human (me)**: Architecture, task definition, debugging, learning

**Key insight**: The best results came when we treated AI as a **collaborator**, not a **replacement**.

- LLMs are great at "how to implement X"
- Humans are great at "should we even build X?"

---

## Open Source & Community

**hc-agent** is designed to be open-sourced. We're finalizing:
- License (likely MIT)
- Documentation cleanup
- Community contribution guidelines

**Repo**: [github.com/huaweicloud/hc-agent](https://github.com/huaweicloud/hc-agent) (coming soon)

**Why open source?**
- Cloud automation should be accessible
- AI-first patterns need more public examples
- Community can extend to other clouds (AWS, Azure, GCP)

---

## Conclusion: The Journey Continues

Building hc-agent taught me that:
- Speed matters, but direction matters more
- Code is cheap, architecture is expensive
- LLMs are powerful tools, but humans must architect
- Real tests reveal truth that validation tests hide

**The next challenge**: Integrate hc-agent with OpenClaw (an AI orchestration framework) to enable true multi-cloud, AI-driven operations.

Stay tuned for Part 2: "Building the AI-Native Cloud."

---

## Appendix: Tech Stack

**Core Technologies:**
- Python 3.9+
- Huawei Cloud SDK
- OpenAI API (GPT-4, Codex)
- DeepSeek-V3.1 (via Huawei Cloud MaaS)
- Kimi (Moonshot AI)

**Infrastructure:**
- Huawei Cloud ECS (la-south-2, Santiago Chile)
- CCE (Kubernetes)
- RDS (MySQL, PostgreSQL)
- OBS (Object Storage)

**Development Tools:**
- Git (version control)
- pytest (testing framework)
- OpenClaw (AI orchestration)
- Codex CLI (rapid prototyping)

---

**Questions? Feedback?**
- 📧 Email: [Contact via GitHub]
- 🐦 Twitter: [@robin_tech]
- 💬 Discuss: [GitHub Discussions]

---

*This blog post is part of a series on building AI-first cloud automation frameworks. All metrics and code examples are from real production systems.*

*Published: February 25, 2026*  
*Last updated: February 25, 2026*
