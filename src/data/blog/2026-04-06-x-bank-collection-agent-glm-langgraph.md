---
author: Robin
pubDatetime: 2026-04-06T09:13:00-03:00
title: "Building a Multilingual AI Collections Agent for X Bank with GLM and LangGraph"
description: "How to build a multilingual AI collections agent with deterministic workflow orchestration, GLM-assisted classification and generation, retrieval-backed script selection, and session-aware negotiation logic."
tags:
  - ai-agent
  - collections
  - banking
  - glm
  - langgraph
  - multilingual
  - negotiation
  - retrieval-augmented-generation
  - fastapi
featured: true
draft: false
---

# Building a Multilingual AI Collections Agent for X Bank with GLM and LangGraph

## Introduction

X Bank wanted to build an AI collections agent for a simple reason: a large share of low-balance delinquent cases were still consuming expensive collector time, yet the conversations themselves were often repetitive.

The operational pattern was familiar:

- customers asked for more time
- customers negotiated for lower monthly payments
- many conversations repeated the same objections
- prior interactions existed, but were rarely operationalized into strategy
- the team needed something explainable, testable, and easy to demo

This repository is the result: a demo-grade but implementation-backed collections agent that combines **deterministic workflow orchestration**, **GLM-assisted classification and generation**, **retrieval-backed script selection**, and **session-aware negotiation logic**.

The project is now much more than a static “chatbot.” It is closer to a small decision engine for collections:

- it remembers customer context
- it distinguishes between personas and negotiation states
- it routes to a next-best action instead of repeating a single payment menu
- it supports English and Brazilian Portuguese
- it can be exercised through API tests, smoke tests, and Playwright end-to-end tests

This article explains how the system was designed, why key technology choices were made, and how the implementation evolved toward higher collection efficiency.

One important note up front: **this repository currently has no git commit history**. That means this write-up does not reconstruct literal commit-by-commit evolution. Instead, it derives the engineering story from the **current codebase**, the **current test suite**, and the **current architecture and runtime shape**.

## The Product Requirement X Bank Actually Needed

The ask was not “build a generic chatbot for debt collection.” The real requirement was narrower and more practical:

1. automate repetitive collections interactions for smaller balances
2. preserve strategy discipline instead of allowing a free-form LLM to improvise
3. surface negotiation paths that increase the chance of a same-day payment
4. keep the system explainable enough for business, risk, and engineering review
5. support multilingual collections, starting with English and Brazilian Portuguese

That combination matters.

If the project had been framed as a pure prompt-engineering problem, the result would have been brittle. A single prompt can generate plausible collector language, but it does not naturally provide:

- stable state transitions
- deterministic payment-ladder logic
- controllable escalation
- structured audit output
- consistent multilingual behavior
- reliable regression testing

The architecture therefore had to separate:

- **what the system decides**
- **how it explains or phrases that decision**

That split shows up throughout the codebase.

## Why GLM Was Chosen

The LLM layer in this repository is implemented through [`src/llm/glm_client.py`](/Users/latamcloudsolution/Documents/Claude/Projects/I_bank_AI_project/src/llm/glm_client.py). The choice of GLM makes sense for this kind of system for several reasons.

### 1. OpenAI-compatible integration surface

The project uses a simple chat-completions style client with structured request building, environment-based enablement, and controllable prompt payloads. That gives the team a familiar integration model without forcing the entire application to be designed around the model vendor.

In practice, that means the rest of the system depends on a small set of capabilities:

- generate a collection reply
- classify intent and sentiment
- interpret ambiguous payment amounts in context
- interpret turn-level dialogue behavior
- generate the opening message when live generation is enabled

This is a good boundary. It avoids model lock-in at the workflow layer.

### 2. GLM is used as an assistant, not the system of record

The most important design choice is not “which model,” but **what authority the model has**.

In this project, GLM is deliberately constrained. It does not own the whole negotiation. Instead:

- deterministic code owns state transitions
- deterministic code owns offer construction
- deterministic code owns payment ladder logic
- deterministic code owns guardrails
- GLM assists with language understanding and response phrasing

That is visible in the code:

- `generate_collection_reply(...)` receives a large structured context block rather than raw conversation alone
- amount interpretation has a dedicated structured path via `GLMAmountInterpretationResult`
- turn interpretation also returns structured fields instead of arbitrary reasoning text

This is exactly the right pattern for a collections system. A model is good at **interpreting messy human language** and **turning approved decisions into natural dialogue**. It is much less reliable at being the only source of truth for:

- whether a number is a monthly budget or a same-day payment
- whether a customer already confirmed a choice
- whether the agent already probed a higher payment this turn
- whether the reply should reopen installment negotiation

### 3. GLM fits the “deterministic-first AI” philosophy

The older architecture document in [`docs/architecture.md`](/Users/latamcloudsolution/Documents/Claude/Projects/I_bank_AI_project/docs/architecture.md) described the project as “deterministic-first AI.” The current implementation still follows that principle, even though the system has become much richer since that doc was written.

The design philosophy is simple:

- use the model where human language is ambiguous
- use code where business logic must be predictable

That split produces a system that is easier to test and easier to trust.

## Why LangGraph Was Chosen

The orchestration layer lives in [`src/agent/graph.py`](/Users/latamcloudsolution/Documents/Claude/Projects/I_bank_AI_project/src/agent/graph.py). This file captures why LangGraph is a good fit for collections work.

### 1. Collections is a stateful workflow problem

A collections turn is not just “input text -> output text.”

It is a workflow where each turn depends on:

- session language
- prior turns
- customer history
- payment signals
- selected strategy
- retrieval mode
- policy outcomes
- whether negotiation is still open or already narrowed

That is exactly the kind of problem LangGraph is useful for: a stateful graph with explicit node boundaries and predictable routing.

### 2. The graph makes the workflow visible

The current ordered pipeline is:

```text
detect_language
-> build_session_memory
-> classify_intent_and_sentiment
-> determine_collection_strategy
-> determine_next_best_action
-> select_collection_skill
-> determine_retrieval_mode
-> retrieve_scripts_and_cases
-> compute_offer_options
-> generate_response
-> run_policy_guardrails
-> finalize
```

That sequence is easy to reason about.

If something goes wrong, engineers can ask:

- did the model misclassify the intent?
- did strategy routing choose the wrong branch?
- did the next-best-action engine override the obvious payment path?
- did retrieval miss the right scripts?
- did guardrails rewrite the answer?

Without a graph, those questions tend to collapse into “the prompt seems wrong.” With a graph, problems become diagnosable.

### 3. LangGraph improves fallback resilience

The implementation also includes a sequential fallback if LangGraph runtime components are unavailable. That matters in demo-grade systems, where dependency stability is often uneven.

In other words, the project keeps the **graph model** even when it cannot rely on the full runtime environment. That is a pragmatic engineering choice: preserve the mental model, preserve the node boundaries, preserve the behavior.

### 4. Collections needs conditional routing, not just linear chains

The graph supports conditional routing after strategy selection. For example, some strategies can skip retrieval or offer construction when those steps are irrelevant.

That is a better fit than a single prompt chain because the system is not just generating text. It is:

- deciding whether retrieval is useful
- deciding whether manual review should short-circuit offer generation
- deciding whether a fast path should be used

LangGraph helps keep those branches explicit rather than hidden inside model instructions.

## The Real Center of the System: Rich Agent State

The most revealing file in the project is probably [`src/agent/state.py`](/Users/latamcloudsolution/Documents/Claude/Projects/I_bank_AI_project/src/agent/state.py). It shows that the system has grown far beyond a minimal collections chatbot.

The `CollectionAgentState` now includes categories like:

- account and customer facts
- conversation history
- intent and sentiment
- persona and contact history
- retrieval mode and evidence
- offer tracks and offer candidates
- payment ladder targets
- amount-resolution metadata
- next-best-action outputs
- business-risk flags
- offline scoring bundle
- session memory
- language-lock information

That tells an important story: the system evolved from **message response generation** toward **stateful negotiation management**.

Some fields are especially meaningful:

- `customer_persona`
- `contact_attempt_index`
- `prior_ptp_count`
- `broken_promise_count`
- `next_best_action`
- `payment_ladder_targets`
- `resolved_payment_amount`
- `resolved_payment_timeframe`
- `session_language`
- `response_pattern_id`
- `dedupe_mode`

This is the difference between a toy assistant and an operationally meaningful one. The agent is not just producing language. It is carrying a structured view of the collection situation from node to node.

## Runtime Architecture

The runtime bootstrapping is intentionally simple and lives in [`src/agent/runtime.py`](/Users/latamcloudsolution/Documents/Claude/Projects/I_bank_AI_project/src/agent/runtime.py).

At startup, the system builds:

- a `KnowledgeBase`
- a `DeterministicPolicyEngine`
- an optional `GLMClient`
- a `CollectionAgentGraph`
- an active runtime profile

This is a clean dependency graph:

```text
FastAPI app
  -> runtime bundle
     -> knowledge base
     -> policy engine
     -> GLM client (optional/live)
     -> agent graph
     -> active profile
```

That structure keeps the app layer thin and the orchestration layer isolated.

## API Layer and Session Model

The FastAPI entrypoint is [`src/app/main.py`](/Users/latamcloudsolution/Documents/Claude/Projects/I_bank_AI_project/src/app/main.py). It handles:

- session creation
- session lookup
- chat execution
- dual-agent runs
- customer loading
- simulator persona listing
- profile introspection

This file also contains one of the most practical pieces of engineering in the repository: **in-memory session management backed by log-aware history summarization**.

For a local demo, in-memory sessions are enough. But the application still behaves like a stateful multi-turn system because:

- sessions are stored explicitly
- history is persisted into the session structure
- customer-level history summaries are built across sessions and log artifacts

That history summarization powers business-relevant fields such as:

- prior session count
- likely broken-promise count
- recent agent signatures
- last contact strategy
- last contact outcome

This is a good example of demo engineering done well. The storage is lightweight, but the behavior models the shape of a more serious production system.

## Retrieval: Why This Is Not Just Prompting

The retrieval subsystem in [`src/rag/knowledge_base.py`](/Users/latamcloudsolution/Documents/Claude/Projects/I_bank_AI_project/src/rag/knowledge_base.py) uses a three-tier deterministic strategy:

1. exact metadata lookup
2. BM25 over core scripts
3. BM25 over reference records

This matters because collections quality depends heavily on **using the right tactic for the right situation**, not merely phrasing a generic response well.

The retrieval layer supports filtering on dimensions such as:

- language
- stage
- intent
- strategy
- pressure stage
- consequence level
- target outcome
- customer persona
- contact-attempt bucket
- offer track

That gives the system something stronger than “nearest text chunk.” It gives the workflow a way to retrieve scripts and cases that actually align with the current negotiation state.

In deep retrieval mode, the system can explicitly bias toward a bundle of:

- tactic cards
- policy snippets
- phrase banks
- conversion cases

That is a sensible collections retrieval design because good negotiation rarely comes from one source type alone.

## Policy Design: Minimal Hard Guardrails, Not a Soft Handbrake

The policy engine in [`src/policy/rules.py`](/Users/latamcloudsolution/Documents/Claude/Projects/I_bank_AI_project/src/policy/rules.py) is intentionally narrow.

It blocks:

- threatening or abusive language
- structurally invalid offers
- impossible debt values

It also emits business-risk flags for cases like:

- contact pressure
- aggressive discounting
- overly long installment structures

This is important for understanding the project’s current philosophy. The system is **not** optimized for the most conservative possible collections posture. It is optimized for **recovery effectiveness under hard redline controls**.

That means:

- the model is allowed to be firm
- the workflow is allowed to push for same-day action
- the system is allowed to negotiate dynamically
- but it should not fabricate or threaten illegal consequences

This is a practical middle ground for a product that wants to improve collections efficiency without turning into unconstrained free-form generation.

## How Collections Efficiency Improved in the Implementation

The most interesting part of this repository is how the strategy evolved away from static offer menus and toward a more intelligent collections flow.

Even without commit history, the current system clearly encodes several major design lessons.

### Stage 1: Installment-first logic is not enough

The basic starting point for many collections bots is:

- offer 12 installments
- maybe offer a second structured option
- ask the customer to choose

That approach is easy to implement, but weak in practice.

It fails when customers say:

- “I can’t afford that monthly amount”
- “I can only pay something small today”
- “Maybe next week”
- “How about 500?”

At that point, a static installment menu becomes repetitive and low-conversion.

### Stage 2: Persona-aware strategy routing

The current system introduces persona-level behavior such as:

- cash constrained
- skeptical
- disputing
- hostile but bounded
- stalling
- hard bargainer
- promise breaker
- evasive negotiator

This makes a major difference. A customer asking for explanation should not be handled the same way as a customer who keeps promising and then backing away.

Persona is not the final answer, but it is a powerful routing feature.

### Stage 3: History-aware negotiation

The system does not only inspect the latest message. It also considers:

- how many times the customer has been contacted
- whether prior promises were likely broken
- which strategies were used before
- recent agent signature history

This is one of the highest-ROI changes in the project.

Collections quality improves when the system stops pretending every turn is the first turn.

### Stage 4: Next-best-action instead of next-best-reply

One of the clearest signs of architectural maturation is the use of `next_best_action`.

That shifts the mindset from:

- “What should the bot say?”

to:

- “What should the system try to achieve on this turn?”

Examples include:

- collect a small payment now
- counter to a floor amount
- convert monthly affordability into a same-day first payment
- unlock a longer arrangement only after a down payment
- hold the policy boundary and re-anchor

That is a much stronger abstraction than raw dialogue generation.

### Stage 5: Payment-first capture instead of installment obsession

One of the most important efficiency improvements in the current design is the move from:

- “insist on the best approved installment”

to:

- “capture viable money now, then structure the rest”

This is visible throughout the state model and prompt structure:

- `payment_offer_status`
- `accepted_payment_amount`
- `counter_required_amount`
- `resolved_payment_amount`
- `resolved_payment_timeframe`

That is how the agent starts behaving like a collector instead of a billing FAQ.

### Stage 6: Dynamic payment ladders

The project no longer revolves around a fixed set of trivial amounts. Instead, it computes ladder targets such as:

- floor
- mid
- high

This allows the agent to do something much closer to real negotiation:

1. hear what the customer can do
2. probe once for a better amount
3. if the customer holds, lock the viable amount instead of losing the deal

This is one of the strongest ideas in the codebase.

### Stage 7: Context-aware amount resolution

Short customer replies are a classic failure mode for conversational systems. Consider:

- `500`
- `ok, 500`
- `200, no more`

If the system reads those replies in isolation, it may misclassify them or drop them entirely.

The current design addresses this through:

- rule-based amount extraction
- negotiation context
- GLM amount interpretation assistance
- deterministic final resolution

This is a good example of hybrid AI done right: use the model to help interpret ambiguity, but keep the workflow in charge.

### Stage 8: Response compaction and anti-robotic behavior

Another major efficiency improvement is not purely strategic. It is compositional.

The system learned to reduce repetition:

- duplicate offer menus
- duplicate consequence paragraphs
- duplicate CTA phrasing
- overlong probe/lock wording

That matters because repetitive systems lose credibility quickly, especially in collections.

## How GLM and Deterministic Logic Work Together

The project gets stronger when viewed as a layered decision stack:

```text
customer input
-> deterministic extraction and state
-> GLM-assisted interpretation where needed
-> deterministic strategy routing
-> deterministic offer computation
-> GLM-assisted or template-based phrasing
-> deterministic guardrails
```

This division of labor is deliberate.

### What deterministic logic owns

- graph execution order
- session memory
- history summarization
- persona routing
- next-best-action routing
- payment ladder thresholds
- offer generation
- language lock
- guardrails

### What GLM owns or assists with

- intent/sentiment classification
- ambiguous amount interpretation
- turn interpretation
- natural reply generation
- opening message generation when enabled

This is likely the single most important architectural decision in the project.

If the model were given total control, the system would be easier to prototype but harder to trust. If the model were excluded entirely, the system would be rigid and worse at handling natural conversation. The current design sits in a productive middle ground.

## The Negotiation Engine in Practice

The most interesting workflow in the repository is the payment negotiation loop.

At a high level, it now behaves like this:

```text
Start with an approved structured path
-> customer says it is too high
-> ask what is actually possible
-> extract amount or budget
-> determine whether it is today or monthly
-> map to payment ladder
-> probe upward once if useful
-> if the customer holds, lock the viable amount
-> keep the remainder eligible for structured follow-up, without promising approval
```

That is significantly closer to a real collector interaction than the usual “choose option A or B.”

The architecture also shows another lesson: some failures in conversational AI are not model failures at all. They are **state-management failures**.

For example, short replies like `yes` after a payment lock proposal expose the need for:

- pending commitment state
- confirmation-aware routing
- transactional subflows

Even where that architecture is not yet fully finalized, the code and tests already point toward it clearly.

## Multilingual Design: English and Brazilian Portuguese

One of the major capabilities added in the current system is session-locked language.

That means:

- language is chosen when the demo session starts
- the session language becomes authoritative
- the collector does not switch language mid-session because of one message
- the dual-agent simulator also respects the same language

This is much better than per-message language guessing for production-like flows.

### Portuguese support is not just translation

The repository now supports:

- Portuguese opening messages
- Portuguese collector templates
- Portuguese offer titles and descriptions
- Portuguese simulator behavior
- Brazilian currency formatting in PT sessions

This matters because multilingual quality is rarely just about translating words. In a collections workflow, language and presentation affect:

- credibility
- comprehension
- tone
- payment behavior

The addition of Brazilian currency formatting is a good example. `BRL 650.00` is technically understandable, but `R$ 650,00` is more natural and more trustworthy in a Brazilian Portuguese session.

### Spanish was not implemented, but the design now allows it

The language layer already separates:

- internal state and decisioning
- display language
- localized offer presentation

That means adding Spanish later should be a language-pack exercise, not an architectural rewrite.

## The Demo Layer Matters More Than It Looks

The simulator in [`src/demo/customer_simulator.py`](/Users/latamcloudsolution/Documents/Claude/Projects/I_bank_AI_project/src/demo/customer_simulator.py) is not just a demo convenience. It is one of the strongest engineering assets in the repository.

Why?

Because it lets the team repeatedly test:

- persona-specific behavior
- payment negotiation logic
- multilingual runtime
- dual-agent flows

The simulator includes personas such as:

- cash constrained
- skeptical
- disputing
- hostile but bounded
- stalling
- hard bargainer
- promise breaker
- evasive negotiator

That gives the project a practical way to stress the system without relying entirely on manual UI clicking.

## Runtime Profiles and Optimization

The profile layer in [`src/agent/profile_store.py`](/Users/latamcloudsolution/Documents/Claude/Projects/I_bank_AI_project/src/agent/profile_store.py) is another sign that the architecture is moving toward a more mature treatment engine.

Profiles hold behavior-level knobs such as:

- overdue thresholds for urgency and final warning
- retrieval depth triggers
- standard and enhanced installment counts
- payment floor and ladder ratios
- generation variants
- autoresearch reward weights

This is smart for two reasons.

First, it decouples strategy tuning from code changes.

Second, it enables systematic experimentation. Instead of debating every collections behavior in code review, the team can eventually optimize profile values against simulated or historical outcomes.

## Testing Strategy: Why This System Is More Trustworthy Than a Pure Prompt Demo

One of the strongest parts of the repository is its test coverage.

### 1. API-level tests

[`tests/test_api_demo.py`](/Users/latamcloudsolution/Documents/Claude/Projects/I_bank_AI_project/tests/test_api_demo.py) verifies session behavior, chat behavior, multilingual locking, dual-agent behavior, and several negotiation regressions.

This is important because it exercises the system the same way the UI does.

### 2. Graph/smoke tests

[`tests/test_agent_smoke.py`](/Users/latamcloudsolution/Documents/Claude/Projects/I_bank_AI_project/tests/test_agent_smoke.py) covers core strategy behavior, including:

- classification behavior
- payment-amount interpretation
- offer generation expectations
- payment probing and locking
- compact-response behavior
- Portuguese-specific behavior

These tests function as a specification for the negotiation engine.

### 3. Playwright end-to-end tests

The UI is also covered through Playwright. That matters because multilingual and session-locking bugs often only show up at the surface layer.

Playwright validates:

- demo session startup
- Portuguese session locking
- dual-agent behavior
- realistic browser interaction

For conversational systems, that is the right combination:

- unit-like behavior tests
- API tests
- end-to-end browser tests

## What This Project Gets Right

Several engineering decisions stand out as especially good.

### 1. The system is workflow-first, not prompt-first

That gives it structure.

### 2. The agent is stateful in a business-meaningful way

It remembers more than just the raw chat transcript.

### 3. GLM is used where it adds leverage, not where it adds risk

This is the right use of LLMs for collections.

### 4. The project is highly testable

That is still unusual for LLM-heavy applications.

### 5. The multilingual design is now session-based, not accidental

This avoids many edge cases.

## Current Limitations

The system is strong for a demo-grade implementation, but it is not pretending to be a production collections platform yet.

Current limitations include:

- no durable production session store
- no real messaging-channel integrations
- no production observability stack
- no online scoring service
- no true historical repayment-outcome training loop
- no full transactional commitment subflow for every confirmation edge case
- no actual git commit history for timeline reconstruction

These limitations should be treated as engineering backlog, not architectural failures. The current boundaries are good enough that a production path is visible.

## If X Bank Wanted to Take This Further

The next logical steps would be:

1. **durable session and history storage**
   - move beyond in-memory sessions

2. **real collections event model**
   - payment posted
   - promise kept
   - promise broken
   - plan approved
   - plan denied

3. **confirmation-aware commitment state**
   - treat `yes`, `no`, and amount changes as transactional states, not generic text turns

4. **offline-to-online scoring**
   - use the existing score bundle shape as the bridge

5. **channel orchestration**
   - SMS
   - WhatsApp
   - callback scheduling

6. **Spanish language pack**
   - built on the now session-locked language architecture

## Final Thoughts

The most interesting thing about this project is not that it uses an LLM. Many systems do.

What makes it technically interesting is that it combines:

- a graph-based workflow
- a rich collections state model
- retrieval-aware guidance
- payment-capture-first negotiation logic
- deterministic control over risky behaviors
- multilingual runtime support
- serious regression testing

That combination is what turns a collections chatbot into an early collections engine.

For X Bank, that is the real opportunity. The value does not come from making the agent sound clever. It comes from making the system:

- disciplined
- measurable
- adaptable
- harder to break
- and more effective at converting messy customer dialogue into structured payment action

GLM gives the agent linguistic flexibility. LangGraph gives it operational structure. The surrounding deterministic logic gives it business discipline.

That is the architecture pattern worth reusing.
