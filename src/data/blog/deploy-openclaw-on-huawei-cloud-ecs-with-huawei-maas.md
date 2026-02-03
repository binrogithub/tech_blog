---
author: Robin
pubDatetime: 2026-02-02T22:25:00-03:00
title: "Deploy OpenClaw on Huawei Cloud ECS and Use Huawei ModelArts MaaS (OpenAI-Compatible)"
description: "A production-like runbook to deploy OpenClaw on Huawei Cloud ECS, wire it to Huawei ModelArts Studio (MaaS) via OpenAI-compatible APIs, and validate automation with cron-driven health checks (Telegram example)."
tags:
  - openclaw
  - huawei-cloud
  - ecs
  - modelarts
  - maas
  - openai-compatible
  - devops
  - runbook
featured: false
draft: false
---

> This post is written as a practical runbook and follows the same structure as my previous ops-focused posts (TL;DR → architecture → prerequisites → steps → validation → pitfalls).
>
> Research note: Gemini CLI was unavailable in my environment, so I used public docs via web fetch and combined them with my own operational history (cron-based Telegram health checks, etc.).

## TL;DR

- Run **OpenClaw Gateway** on **Huawei Cloud ECS** (Linux VM) and keep it running 24/7.
- Connect OpenClaw to **Huawei ModelArts Studio (MaaS)** using **OpenAI-Compatible** settings:
  - Get the **API URL** + **Model ID** from MaaS “View Call Description”.
  - Use **Base URL = API URL with `/chat/completions` removed**.
- Validate:
  - `openclaw gateway status`
  - send a test message in Telegram
  - schedule a cron job (cron runs inside the Gateway)

---

## 1) Reference Architecture

**Single-node, production-like** (recommended baseline):

- **Huawei Cloud ECS** (Ubuntu 22.04 / Rocky Linux 9)
- **OpenClaw Gateway** (daemon process)
- **Workspace** on disk (durable memory + artifacts)
- **Messaging channel** (Telegram recommended for ops)
- **Huawei ModelArts Studio (MaaS)** as LLM provider (OpenAI-compatible)

Data flow:

Telegram/WhatsApp → OpenClaw Gateway → LLM (Huawei MaaS) + Tools (shell/files/browser/cron) → Reply

---

## 2) Why ECS + MaaS is a good pairing

- ECS gives you a clean, isolated server for agent automation (safer than running on your laptop).
- MaaS gives you managed inference endpoints with a familiar **OpenAI-compatible** API surface.
- OpenClaw’s **cron scheduler** runs inside the Gateway, so once the Gateway is stable, you can reliably automate recurring tasks.

---

## 3) Prerequisites

### 3.1 ECS sizing

Baseline for an always-on agent:

- 2–4 vCPU
- 4–8 GB RAM
- 40–100 GB disk (depends on how much you store in workspace)

### 3.2 Network / Security Group

Minimum:

- SSH 22 (lock down to your IP)
- Outbound access to:
  - Huawei MaaS endpoint
  - Telegram API

Optional (only if you expose a dashboard/UI):

- 80/443 behind a reverse proxy

### 3.3 System packages

Ubuntu example:

```bash
sudo apt-get update
sudo apt-get install -y curl git jq
```

---

## 4) Step-by-step: Install & start OpenClaw on ECS

> Install method can vary (package manager vs binary vs container). The key is: **OpenClaw CLI is available**, and **Gateway can run continuously**.

### 4.1 Create a dedicated workspace

```bash
sudo mkdir -p /mnt/data/clawd
sudo chown -R $USER:$USER /mnt/data/clawd
cd /mnt/data/clawd
```

Recommended structure:

```text
/mnt/data/clawd/
  MEMORY.md
  memory/
  scripts/
  insights/
```

### 4.2 Verify OpenClaw CLI

```bash
openclaw help
```

### 4.3 Start Gateway and confirm health

```bash
openclaw gateway status
openclaw gateway start
openclaw gateway status
```

If you change config later:

```bash
openclaw gateway restart
```

---

## 5) Step-by-step: Configure Huawei ModelArts MaaS (OpenAI-Compatible)

Huawei’s own best-practice guide for using MaaS with a client (Cline) documents the key idea:

- Choose **API Provider: OpenAI Compatible**
- Use **Base URL = API URL with `/chat/completions` removed**
- Set **Model ID** to the MaaS model name you see in the console

Source: Huawei Cloud ModelArts Studio (MaaS) best practice (DeepSeek + Cline).

### 5.1 Create MaaS API key

In ModelArts Studio (MaaS) console:

- Go to **API Key Management**
- Create a key
- Copy it (it’s typically shown only once)

### 5.2 Deploy / select a MaaS real-time inference service

In MaaS console:

- Go to **Real-Time Inference**
- Use **My Services**
- Deploy a model service

Then for the running service:

- **More → View Call Description**
- Copy:
  - API URL (often ends with `/chat/completions`)
  - Model name / Model ID

### 5.3 Set OpenAI-compatible variables (example)

Use placeholders because endpoint shapes vary by tenant/region:

```bash
export HUAWEI_MAAS_API_KEY="<YOUR_MAAS_KEY>"
export HUAWEI_MAAS_API_URL="https://<...>/v1/chat/completions"   # from MaaS call description
export HUAWEI_MAAS_BASE_URL="${HUAWEI_MAAS_API_URL%/chat/completions}"  # remove suffix
export HUAWEI_MAAS_MODEL_ID="<MODEL_ID_FROM_MAAS>"
```

### 5.4 Sanity check from ECS (curl)

```bash
curl -sS "$HUAWEI_MAAS_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $HUAWEI_MAAS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$HUAWEI_MAAS_MODEL_ID"'",
    "messages": [{"role":"user","content":"Hello from Huawei ECS"}],
    "temperature": 0.2,
    "max_tokens": 128
  }' | jq
```

If this works, your ECS outbound network + MaaS credentials are good.

---

## 6) Step-by-step: Wire OpenClaw to MaaS

OpenClaw is model-agnostic. The practical target is:

- provider type: **OpenAI-compatible**
- base URL: `https://<...>/v1`
- api key: MaaS API key
- model: MaaS Model ID

Where you set these depends on how your OpenClaw deployment is configured (config file vs environment variables). A safe operational pattern is:

- Keep secrets **outside git**
- Inject them as environment variables into the Gateway service

Example (conceptual):

```env
OPENAI_COMPAT_BASE_URL=https://<HUAWEI_MAAS_OPENAI_ENDPOINT>/v1
OPENAI_COMPAT_API_KEY=<YOUR_KEY>
OPENAI_COMPAT_MODEL=<MODEL_ID>
```

> Tip: start by validating MaaS with curl first; only then integrate into OpenClaw. It reduces debugging surface area.

---

## 7) Validation: Messaging + cron automation (production sanity)

### 7.1 Confirm Gateway is up

```bash
openclaw gateway status
```

### 7.2 Confirm you can send/receive a Telegram message

Send a message to your bot and confirm:

- the agent replies
- model calls succeed (no auth or base URL errors)

### 7.3 Add a cron job (cron runs inside Gateway)

OpenClaw cron jobs are persisted on the Gateway host and survive restarts.

You can create a one-shot reminder:

```bash
openclaw cron add \
  --name "ECS smoke test" \
  --at "5m" \
  --session main \
  --system-event "Reminder: OpenClaw on ECS is running. Check MaaS connectivity." \
  --wake now \
  --delete-after-run
```

List jobs:

```bash
openclaw cron list
```

Reference: OpenClaw cron docs.

---

## 8) Operational pattern: Health checks + failure thresholds

In real deployments, the important question is not “does it work once?” but “does it keep working?”

A lightweight pattern:

- A script that checks Telegram reachability
- A cron schedule to run it periodically
- A state file that tracks consecutive failures
- Escalate after N consecutive failures

This is the same pattern used in my own workspace for Telegram health monitoring.

---

## 9) Security hardening checklist

- Run OpenClaw under a dedicated non-root user
- Restrict SSH (IP allowlist), disable password login
- Keep Gateway ports private unless you explicitly need a UI
- Store MaaS API keys in env vars / secret manager
- Consider outbound egress allowlists (only MaaS + Telegram)

---

## 10) Common pitfalls (and how to fix them)

### Pitfall A: Cron doesn’t run

- Cron runs **inside the Gateway**.
- If the Gateway isn’t running continuously, cron won’t trigger.

Check:

```bash
openclaw gateway status
```

### Pitfall B: MaaS “OpenAI compatible” fails

Common causes:

- Base URL is wrong (you forgot to remove `/chat/completions`)
- Model ID mismatch
- API key not in effect

Fix: re-test with curl (Section 5.4) before blaming OpenClaw.

### Pitfall C: Costs creep up

Cron + agents can silently increase token usage.

Mitigations:

- keep cron prompts short
- use isolated cron sessions for noisy jobs
- add quotas / alerting on token spend

---

## References

- OpenClaw Cron Jobs (Gateway scheduler): https://docs.openclaw.ai/automation/cron-jobs
- Huawei ModelArts Studio (MaaS) best practice (DeepSeek + Cline, OpenAI Compatible guidance):
  https://support.huaweicloud.com/intl/en-us/bestpractice-modelarts/modelarts_10_25188.html
- OpenClaw deployment tutorial (DigitalOcean, good security checklist inspiration):
  https://www.digitalocean.com/community/tutorials/how-to-run-openclaw
