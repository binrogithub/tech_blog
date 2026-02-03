---
author: Robin
pubDatetime: 2026-02-02T22:10:00-03:00
title: "Runbook: Huawei Cloud ECS + Huawei MaaS (OpenAI-Compatible) for Production-Like AI Apps"
description: "A practical, production-like runbook to deploy an app on Huawei Cloud ECS and consume Huawei MaaS through an OpenAI-compatible API (chat + embeddings). Includes curl/Python tests, deployment patterns, observability, and common pitfalls."
tags:
  - huawei-cloud
  - ecs
  - maas
  - openai-compatible
  - llm
  - devops
  - runbook
featured: false
draft: true
---

## TL;DR

- Deploy your app on **Huawei Cloud ECS** (one VM) using Docker Compose.
- Treat **Huawei MaaS** as an **OpenAI-compatible** provider:
  - `BASE_URL = https://<HUAWEI_MAAS_OPENAI_ENDPOINT>/v1`
  - `API_KEY = <YOUR_KEY>`
  - `MODEL = <MODEL_ID>`
- Validate with:
  - `curl /v1/chat/completions`
  - (optional) `curl /v1/embeddings`
- Add production basics: reverse proxy, TLS, outbound allowlist, logs/metrics, cost guards.

---

## 1) Reference Architecture

A minimal production-like layout:

- **ECS VM** (Ubuntu 22.04 / Rocky 9)
- **Reverse proxy** (Nginx/Caddy) for TLS + routing
- **Application** (containerized)
- Optional: **vector DB** (Qdrant) if you do RAG yourself

Data flow:

User → HTTPS → Reverse Proxy → App → (Huawei MaaS OpenAI-compatible API)

---

## 2) Prerequisites

### 2.1 ECS sizing (baseline)

- 2–4 vCPU, 8–16 GB RAM for typical API + light RAG
- 100 GB disk (logs + cache + vectors can grow)

### 2.2 Network / Security Group

Inbound:
- 22 (SSH) — restrict to your IP
- 80/443 (recommended)

Outbound:
- allow access to Huawei MaaS endpoint domain/IP

### 2.3 System packages

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin curl jq
sudo usermod -aG docker $USER
newgrp docker
```

---

## 3) Step-by-step: Deploy an App on ECS (Docker Compose)

### 3.1 Directory layout

Example:

```text
/mnt/data/app/
  docker-compose.yml
  .env
  logs/
```

### 3.2 Example compose (API service)

```yaml
services:
  app:
    image: your-org/your-app:latest
    restart: unless-stopped
    env_file: .env
    ports:
      - "127.0.0.1:8080:8080"  # bind locally; reverse proxy exposes 443
```

Start:

```bash
cd /mnt/data/app
docker compose up -d
docker compose logs -f --tail=200 app
```

---

## 4) Configure Huawei MaaS as an OpenAI-Compatible Provider

Huawei MaaS typically provides an OpenAI-compatible interface. Exact endpoint and model IDs may vary by tenant/region.

### 4.1 Required variables

Set these in `.env`:

```env
# OpenAI-compatible base URL (placeholder)
HUAWEI_MAAS_BASE_URL=https://<HUAWEI_MAAS_OPENAI_ENDPOINT>/v1

# Bearer token / API key
HUAWEI_MAAS_API_KEY=<YOUR_KEY>

# Model IDs (use the exact IDs shown in your MaaS console)
HUAWEI_MAAS_CHAT_MODEL=<MODEL_ID>
HUAWEI_MAAS_EMBED_MODEL=<EMBED_MODEL_ID>

# Optional: timeouts / retries
HTTP_TIMEOUT_SECONDS=60
```

### 4.2 Quick sanity test (chat)

```bash
BASE_URL="https://<HUAWEI_MAAS_OPENAI_ENDPOINT>/v1"
API_KEY="<YOUR_KEY>"
MODEL="<MODEL_ID>"

curl -sS "$BASE_URL/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\
    \"model\": \"$MODEL\",\
    \"messages\": [{\"role\":\"user\",\"content\":\"Hello from ECS\"}],\
    \"temperature\": 0.2,\
    \"max_tokens\": 128\
  }" | jq
```

### 4.3 Optional sanity test (embeddings)

Only if your MaaS endpoint supports `/v1/embeddings`:

```bash
EMBED_MODEL="<EMBED_MODEL_ID>"

curl -sS "$BASE_URL/embeddings" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\
    \"model\": \"$EMBED_MODEL\",\
    \"input\": \"This is a test embedding\"\
  }" | jq
```

---

## 5) Python client example (requests)

```python
import os
import requests

BASE_URL = os.environ["HUAWEI_MAAS_BASE_URL"].rstrip("/")
API_KEY = os.environ["HUAWEI_MAAS_API_KEY"]
MODEL = os.environ["HUAWEI_MAAS_CHAT_MODEL"]

r = requests.post(
    f"{BASE_URL}/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Explain what 1+3 means."}],
        "temperature": 0.2,
        "max_tokens": 256,
    },
    timeout=60,
)

r.raise_for_status()
print(r.json())
```

---

## 6) RAG notes (if you build RAG outside MaaS)

If you run your own RAG stack on ECS:
- store embeddings in a vector DB (e.g., Qdrant)
- **chunking** and **top_k** usually impact latency more than model choice

Practical defaults:
- chunk size: 300–800 tokens
- top_k: 3–6
- keep prompts short; cap per-chunk characters

---

## 7) Observability & Operations

Minimum checks:

```bash
# container health
docker compose ps

# app logs
docker compose logs --tail=200 app
```

Production suggestions:
- reverse proxy access logs
- app structured logs (JSON)
- request IDs + latency histograms
- alert on: 5xx rate, timeouts, MaaS 429/5xx, queue backlog

---

## 8) Security checklist

- Put the app behind HTTPS (Nginx/Caddy)
- Do not expose internal ports publicly (bind to 127.0.0.1)
- Store API keys as secrets (env vars at runtime; avoid committing)
- Add outbound egress controls if possible
- Add rate limits at proxy

---

## 9) Common pitfalls (field notes)

- **Wrong Base URL**: must typically end with `/v1` for OpenAI-compatible routing.
- **Wrong model id**: use the exact MaaS model identifier.
- **429 rate limits**: add retries with exponential backoff; cache where possible.
- **Timeouts**: set client + proxy timeouts; large outputs need higher timeouts.
- **Embeddings not configured**: RAG indexing will fail or be empty without an embedding model.

---

## References

- Huawei Cloud documentation (ModelArts / ECS) — use the official portal for your region.
- OpenAI API (compatibility reference): https://platform.openai.com/docs/api-reference
