---
author: Robin
pubDatetime: 2026-04-18T14:40:00-03:00
title: "Why I Created Huawei Cloud Adoption Skills: Helping AI Agents Use Huawei Cloud Faster, More Accurately, and More Reliably"
description: "Why I created the open source Huawei Cloud Adoption Skills project, and how scenario-first skill packages can help AI agents support real Huawei Cloud delivery work with more speed, accuracy, and reuse."
tags:
  - huawei-cloud
  - ai-agents
  - cloud-adoption
  - open-source
  - enterprise-ai
  - cloud-delivery
featured: false
draft: false
---

# Why I Created *Huawei Cloud Adoption Skills*: Helping AI Agents Use Huawei Cloud Faster, More Accurately, and More Reliably

AI is changing how engineers build, migrate, operate, and optimize cloud systems. But in real delivery work, one problem becomes obvious very quickly: a general-purpose AI model is not the same thing as a cloud delivery expert.

A large model may know how to write code, summarize architecture, or explain APIs. But that does not automatically mean it can guide a real Huawei Cloud migration, design a practical landing zone, adapt a data platform, or turn a use case into a repeatable implementation pattern. In most cases, AI only becomes truly useful when domain knowledge, delivery method, cloud product understanding, and validation steps are packaged together.

That is why I created the open source project **Huawei Cloud Adoption Skills**.

The goal of this project is simple: make it possible for AI Agents to use Huawei Cloud **faster, more accurately, and more efficiently**, while continuously accumulating Huawei Cloud best practices into reusable assets.

## The Problem: AI Is Powerful, But Cloud Delivery Is Context-Heavy

In theory, AI can help with architecture design, migration analysis, operations, code generation, troubleshooting, and optimization. In practice, cloud work is highly contextual.

A useful Huawei Cloud answer usually requires more than generic intelligence. It requires:

- understanding the business scenario
- knowing which Huawei Cloud services and patterns apply
- knowing how those services are usually used in real delivery
- understanding integration constraints, security boundaries, and operational tradeoffs
- producing outputs that are not just "smart," but actually reusable and verifiable

Without that context, AI often produces answers that are plausible but not operationally useful. It may sound correct while missing the real delivery path. It may recommend technology patterns without grounding them in Huawei Cloud usage. It may generate output that is hard to validate or impossible to reuse across projects.

That gap is exactly what this project is designed to solve.

## The Core Idea: An AI Skill Is Not Just a Prompt

One of the core ideas behind this repository is that an **AI skill is not just prompt engineering**.

In this project, an AI skill is defined as a **reusable capability unit built around a cloud scenario**, where AI is used to support one or more of the following:

- understanding
- design
- migration
- operations
- development
- analysis
- optimization

Each skill must contain five elements:

- **Scenario**: what business problem or cloud scenario it addresses
- **Knowledge**: what cloud products, architectures, and domain logic are required
- **Tools**: what models, APIs, scripts, platforms, or workflows are used
- **Method**: how AI is embedded into the delivery process
- **Output and validation**: what is produced, and how the result is checked

This definition matters because it shifts the focus from "what the model can do" to "what business value the skill can deliver."

That is also why the repository is not organized around abstract AI categories such as prompt, RAG, agent, or fine-tuning. Those techniques are important, but they are not the right top-level structure for delivery. Real enterprise work needs to start from the cloud scenario first.

## Why This Repository Uses a Scenario-First Structure

The structure of the project is intentional.

Instead of starting with model capabilities, the repository starts with Huawei Cloud adoption domains and delivery use cases. At the top level, it follows a `1+3` structure:

- **Cloud Foundation**
- **Application Modernization**
- **Big Data**
- **AI**

Under those domains, the repository organizes content into Level 2 use cases, such as:

- landing zone and organization management
- automation and IaC
- cross-cloud and hybrid network
- application migration
- database migration
- data governance
- AI-ready data and knowledge base
- AI coding
- agent platform
- responsible AI and governance

This matters because AI only creates value when it is attached to real delivery scenarios. A skill that cannot support a real use case is difficult to operationalize, difficult to measure, and difficult to reuse.

The structure is designed so that an engineer, architect, delivery team, or AI Agent can move from:

**domain -> use case -> skill package -> execution assets**

That is a much more useful path than:

**model feature -> generic technique -> disconnected demo**

## What the Project Is Trying to Build

At a high level, *Huawei Cloud Adoption Skills* is building a reusable knowledge and execution layer for AI-enabled Huawei Cloud delivery.

The repository is meant to grow into a place where Huawei Cloud best practices are continuously captured and structured into reusable forms, including:

- solution materials
- demos
- scripts
- best practices
- test reports
- FAQ and troubleshooting guides
- repeatable templates

This is important because enterprise cloud delivery has a repeatability problem. Teams often solve the same patterns multiple times across different projects:

- the same landing zone decisions
- the same migration questions
- the same data pipeline adaptation issues
- the same operational troubleshooting flows
- the same AI integration patterns

Without a structured repository of reusable skills, those lessons remain fragmented across slides, notebooks, scripts, personal experience, and project history. AI Agents then have nothing consistent to build on.

This project turns that fragmented knowledge into an operating asset.

## The Role of General Skills

Another key part of the repository is the concept of **General Skills**.

These are the common capabilities that every Huawei Cloud adoption scenario depends on, regardless of whether the work is in cloud foundation, modernization, data, or AI.

The current framework defines eight general areas:

- scenario understanding and requirement abstraction
- AI interaction and prompt design
- cloud knowledge retrieval and knowledge injection
- API / SDK / CLI / IaC automation
- security and governance
- observability, evaluation, and optimization
- integration and workflow orchestration
- assetization and replication

These general skills are the baseline that makes AI useful in real delivery.

For example, even a strong domain skill becomes weak if the AI output is not validated, if the knowledge source is stale, if the automation path is unclear, or if the result cannot be packaged into reusable material. The general skill layer is what makes the overall system operational rather than experimental.

## Helping AI Agents Become Delivery-Ready

One of the project's most practical goals is to help AI Agents act less like generic assistants and more like delivery-ready collaborators.

That means the agent should not only answer questions. It should be able to:

- understand a Huawei Cloud scenario
- identify the right use case and migration path
- apply known Huawei Cloud patterns
- call the right tools or scripts
- generate reusable outputs
- validate the result
- preserve the best solution as an asset for future reuse

This is the difference between an AI demo and an AI-enabled engineering workflow.

For example, a good Huawei Cloud skill package should help an agent do things like:

- map a Databricks workload to OBS plus MRS
- adapt SQL Server or PostgreSQL code to GaussDB
- integrate AI coding workflows with OpenShift and Huawei Cloud MaaS
- design a Kafka plus GaussDB transactional architecture on Huawei Cloud
- package migration logic, checks, scripts, and references into repeatable delivery assets

These are not generic prompts. They are repeatable delivery units.

## Why Open Source

I made this project open source because Huawei Cloud adoption knowledge becomes more valuable when it is structured, reviewed, expanded, and reused across teams and scenarios.

Open source helps in several ways:

- it makes the framework inspectable
- it makes skills reusable across projects
- it encourages cleaner documentation and clearer structure
- it allows best practices to accumulate instead of being trapped in individual projects
- it creates a practical bridge between AI workflows and cloud engineering work

Open source also creates discipline. If a skill is meant to be reused, it must be documented well, scoped correctly, and packaged cleanly. That pressure improves quality.

## What "Good" Looks Like in This Project

This repository is not trying to collect random cloud notes. It is trying to build a usable skill system.

A good skill in this project should have several properties:

- it is attached to a real Huawei Cloud use case
- it is specific enough to guide actual delivery
- it includes not only explanation but also method and artifacts
- it can be validated
- it can be reused by a person or by an AI Agent
- it contributes to business outcomes, not just technical discussion

That is also why the framework includes maturity levels:

- **Understand**
- **Execute**
- **Replicate**

The point is not to say a team has "heard of" a topic. The point is to capture whether the topic can actually be applied and then reproduced.

## The Long-Term Vision

The long-term vision for *Huawei Cloud Adoption Skills* is not a static document library. It is a living skill repository for AI-native cloud delivery.

Over time, I want this project to become a place where Huawei Cloud delivery knowledge is continuously accumulated in a form that both humans and AI Agents can use effectively.

That means:

- more domain-specific skill packages
- more use-case-aligned assets
- more references and runnable examples
- more validation patterns
- more migration and operations guidance
- more AI-ready packaging of Huawei Cloud expertise

In other words, the project is trying to turn Huawei Cloud best practices into a structured execution layer for AI-assisted engineering.

## Final Thoughts

I created *Huawei Cloud Adoption Skills* because I believe AI becomes truly useful in cloud work only when it is grounded in scenario, knowledge, tools, method, and validation.

Huawei Cloud has a growing ecosystem of products, architectures, and best practices. But for AI Agents to use that ecosystem well, they need more than raw model intelligence. They need reusable skills.

This project is my way of building that layer.

It is meant to help AI Agents use Huawei Cloud faster, more accurately, and more efficiently. It is also meant to make sure Huawei Cloud best practices are not repeatedly rediscovered, but continuously summarized, accumulated, and turned into reusable assets.

That is the real purpose of this open source project.
