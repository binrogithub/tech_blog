---
author: Robin
pubDatetime: 2026-02-02T20:05:00-03:00
title: "Runbook: Deploy Dify + RAG on Huawei Cloud ECS (Docker Compose)"
description: "A production-like, single-node deployment runbook for Dify on Huawei Cloud ECS with RAG: web/api/worker + Postgres + Redis + Qdrant, plus optional SearXNG, plugin daemon, and sandbox. Includes troubleshooting and MaaS (OpenAI-compatible) model binding." 
tags:
  - huawei-cloud
  - dify
  - rag
  - docker
  - devops
  - runbook
featured: false
draft: false
---

> Source: adapted from `/mnt/data/clawd/memory/tech-notes/2026-02-02_Deploy_Dify_RAG_on_Huawei_Cloud_ECS.md`.

## 1) Target Architecture

### Containers (recommended minimum)

- `web` (UI) → **port 3000**
- `api` (REST / console API) → **port 5001**
- `worker` (indexing + async tasks)
- `postgres` (metadata)
- `redis` (queue/cache)
- `qdrant` (vector DB) → **port 6333**

### Common add-ons

- `plugin_daemon` (Dify plugins; avoids console/plugin errors) → port varies (often 5002/5003)
- `sandbox` (code execution; fixes “Failed to execute code… Name or service not known”) → internal **8194**
- `searxng` (web search tool) → map to host port **8081**
- `reranker` (optional) → e.g., **6006**

---

## 2) Prerequisites on Huawei Cloud ECS

### Compute & OS

- Recommended: **4 vCPU / 16 GB RAM** (8 GB minimum for small workloads)
- Rocky Linux 9.x / Ubuntu 22.04 both OK
- Disk: **100 GB+** (datasets + embeddings grow fast)

### Network / Security Group

Open inbound (at least):
- TCP **3000** (Dify web)
- TCP **5001** (Dify API / console API)

Optional (prefer keep internal-only unless needed):
- TCP **6333** (Qdrant)
- TCP **8081** (SearXNG)
- TCP **6006** (reranker)
- TCP **5003** (plugin debug/install)

### System packages

Install Docker + Compose + git.

---

## 3) Deployment Steps (Docker Compose)

### Step 1: Prepare directories

Example:

- `/mnt/data/dify/docker` (compose, configs, volumes)
- `./volumes/...` mounted into containers

### Step 2: Use **one** Compose file only

Avoid keeping both `docker-compose.yml` and `docker-compose.yaml` in the same folder.

This prevents confusing errors like:
- `Found multiple config files ... docker-compose.yml/yaml`

### Step 3: Key environment variables (must-have)

#### 3.1 Public URL correctness (fixes blank pages / wrong redirects / embed issues)

Set these to your public IP / domain:

- `APP_URL=http://<PUBLIC_IP>:3000`
- `API_URL=http://<PUBLIC_IP>:5001`
- `CONSOLE_API_URL=http://<PUBLIC_IP>:5001`

> If you put a reverse proxy (recommended), use your **https** domain for all of the above.

#### 3.2 Database / Redis

```env
DB_HOST=postgres
DB_PORT=5432
DB_USERNAME=...
DB_PASSWORD=...
DB_DATABASE=dify

REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
CELERY_BROKER_URL=redis://redis:6379/1
```

#### 3.3 Vector store (critical for indexing)

If missing on worker, you’ll see:
- `Vector store must be specified`
- Qdrant collections won’t appear

```env
VECTOR_STORE=qdrant
QDRANT_URL=http://qdrant:6333
```

#### 3.4 Storage (fixes worker crash “root is not specified”)

If using OpenDAL local FS:

```env
STORAGE_TYPE=opendal
OPENDAL_SCHEME=fs
OPENDAL_FS_ROOT=/app/api/storage
```

And mount:

```text
./volumes/app/storage:/app/api/storage
```

#### 3.5 Worker modes

- API container: `MODE=api`
- Worker container: `MODE=worker`

### Step 4: Plugin daemon (recommended)

If missing/misconfigured, you may get:
- “Failed to request plugin daemon…”

Add `plugin_daemon` service, and set in **api** and **worker**:

```env
PLUGIN_DAEMON_URL=http://plugin_daemon:5002
PLUGIN_DAEMON_KEY=...
PLUGIN_DIFY_INNER_API_URL=http://api:5001
INNER_API_KEY_FOR_PLUGIN=...
```

### Step 5: Sandbox (required for code execution)

Symptom:
- `Failed to execute code... (Error: [Errno -2] Name or service not known)`

Fix:

1) Add sandbox service (e.g. `langgenius/dify-sandbox:0.2.12`)

2) Set in **api** and **worker**:

```env
CODE_EXECUTION_ENDPOINT=http://sandbox:8194
CODE_EXECUTION_API_KEY=dify-sandbox
CODE_EXECUTION_SSL_VERIFY=false
```

3) Mount:

```text
./volumes/sandbox/dependencies:/dependencies
./volumes/sandbox/conf:/conf
```

### Step 6: Start stack

```bash
docker compose up -d
```

Run DB migrations:

```bash
docker compose run --rm -e MODE=job api upgrade-db
```

---

## 4) RAG Validation Checklist

### API & UI

- `http://<PUBLIC_IP>:3000`
- `http://<PUBLIC_IP>:5001/console/api/health` (or the equivalent health endpoint)

### Worker health

```bash
docker compose logs --tail=200 worker
```

### Qdrant collections appear

```bash
curl http://127.0.0.1:6333/collections
```

If empty while documents show “indexing”, worker is usually not writing vectors (vector store/env/embedding failures).

### Knowledge retrieval test

If recall test returns:
- `Collection ... doesn't exist`

That means indexing never created the Qdrant collection.
Fix worker vector env and re-index.

---

## 5) Troubleshooting (Most Common Issues & Fixes)

### A) “Found multiple config files … yml/yaml”

Cause: both compose files exist.
Fix: keep one.

### B) docker compose logs docker-worker-1 → “no such service”

Cause: you used container name.
Fix: `docker compose logs worker`.

### C) SearXNG restarting: Invalid settings.yml / Expected object, got null

Cause: settings schema mismatch or broken YAML.
Fix: replace with a known-good `settings.yml` and mount it.

### D) Worker crash: opendal.exceptions.ConfigInvalid ... root is not specified

Cause: missing `OPENDAL_FS_ROOT` or wrong mount.
Fix: set root and mount path consistently.

### E) UI shows Internal Server Error / plugin API failures

Cause: missing plugin daemon or wrong URL/keys.
Fix: add plugin daemon + correct env vars; restart.

### F) Knowledge base recall 404: Collection ... doesn't exist

Cause: worker didn’t write vectors.
Fix:
- ensure `VECTOR_STORE=qdrant` and `QDRANT_URL=http://qdrant:6333`
- check worker logs for embedding/provider errors
- reindex

### G) Image pull fails: TLS handshake timeout

Cause: network/DNS/egress issues.
Fix: configure registry mirror, ensure outbound network stable.

### H) Jinja2 prompt errors (workflow templating)

- `expected token ':' got '}'`: you used `{{ }}` inside `{% %}`.
- `'str object' has no attribute 'metadata'`: you iterated the wrong level (object vs array).

---

## 6) Bind Huawei Cloud MaaS Models via OpenAI-Compatible Interface (DeepSeek / Qwen)

Goal: use Huawei MaaS as an OpenAI-compatible endpoint inside Dify.

### What you need
- Base URL (OpenAI-compatible): `https://<...>/v1`
- API key
- Model name (Huawei’s exact model id)

### Typical Dify config
In Dify Console → Model Provider:
- Provider type: OpenAI Compatible
- Base URL: `https://<HUAWEI_MAAS_OPENAI_ENDPOINT>/v1`
- API Key: `<YOUR_KEY>`
- Model: `deepseek-chat` / `qwen-...` (use Huawei’s exact model id)

Sanity check from ECS:

```bash
curl -sS <BASE_URL>/chat/completions \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<MODEL_NAME>",
    "messages": [{"role":"user","content":"Hello"}],
    "temperature": 0.2,
    "max_tokens": 128
  }'
```

> Embeddings matter: configure an embedding model and make sure worker indexing uses it.

---

## 7) RAG Performance Tuning (Latency + Quality)

### A) LLM time optimization
- smaller model when acceptable
- disable “thinking mode”
- lower temperature (0.2–0.3)
- cap `max_tokens`

### B) Retrieval & indexing performance
- stable chunking strategy (e.g., 300–800 tokens)
- tune top_k (e.g., 3–6)
- rerank selectively
- ensure worker resources + concurrency are sufficient
- watch disk IOPS (Qdrant)

### C) Prompt size control
- cap per-segment chars
- keep system prompt concise

---

## 8) Operational Best Practices

- Backups: Postgres volumes + Qdrant storage
- Observability: tail logs for `api`, `worker`, `qdrant`, `sandbox`, `plugin_daemon`
- Change management: restart only impacted services after env changes

