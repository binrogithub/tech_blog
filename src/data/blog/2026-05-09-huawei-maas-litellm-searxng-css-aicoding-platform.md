---
author: Robin
pubDatetime: 2026-05-09T18:00:00-03:00
title: "Building an Enterprise-Grade AI Coding Platform on Huawei Cloud MaaS"
description: "A reference architecture and field report for fronting Huawei Cloud MaaS with LiteLLM, exposing SearXNG web search and CSS/OpenSearch code search as remote MCP tools, and integrating the whole stack into Claude Code via claude-code-router on a single ECS."
tags:
  - litellm
  - huawei-cloud
  - maas
  - mcp
  - searxng
  - css
  - opensearch
  - claude-code
  - ai-coding
  - ai-agent
featured: false
draft: false
---

# Building an Enterprise-Grade AI Coding Platform on Huawei Cloud MaaS

*A reference architecture and field report for fronting Huawei Cloud MaaS
with LiteLLM, web search via SearXNG, and code search via CSS/OpenSearch —
all integrated into Claude Code on a single ECS.*

---

## TL;DR

We packaged two reusable skills that together form an enterprise-grade
AI coding platform on Huawei Cloud:

1. **LiteLLM gateway** in front of Huawei Cloud MaaS, with a **SearXNG
   web-search MCP** sharing the same ECS — so spend, rate limits, audit,
   and the search tool live in one place.
2. **CSS/OpenSearch code-search MCP** that lets the AI agent grep an
   indexed Git repository as a native tool.

The integration target is Claude Code (`claude-glm`) routed via
`claude-code-router`. The user's regular `claude` install is left
untouched through `CLAUDE_CONFIG_DIR` isolation. Both skills are open
source:

- [`LiteLLM-SearXNG-AICoding-Gateway-Single-ECS`](https://github.com/binrogithub/1-3-Cloud-Adoption-Skills/tree/main/AI/AI-Development/LiteLLM-SearXNG-AICoding-Gateway-Single-ECS)
- [`CSS-Code-Search-MCP`](https://github.com/binrogithub/1-3-Cloud-Adoption-Skills/tree/main/AI/AI-Coding/CSS-Code-Search-MCP)

This post explains the architecture, the design decisions that survived
contact with reality, and the specific traps you will hit if you wire it
up yourself.

---

## Why this shape

Most teams trying to ship an AI coding agent on a regional cloud start
with a thin wrapper that calls the cloud's MaaS endpoint directly from
each laptop. That works for a demo and breaks the moment a second
engineer joins. The pain points are predictable:

- **Spend is invisible.** A MaaS API key handed to five laptops produces
  one rolled-up bill. Nobody knows whose bug-hunt loop just spent 40
  bucks on reasoning tokens.
- **Rate-limits and budgets are missing.** Region-level MaaS usually
  rate-limits per-account, not per-user. A loose script from one engineer
  starves everyone else.
- **Audit and rotation are ugly.** Rotating a MaaS key forces every
  laptop to be touched at once. Compromise of one laptop forces a
  fleet-wide rotation.
- **Tools are bolted onto each agent.** Web search, code search, and
  retrieval are configured per-laptop through environment variables,
  Cursor settings, and ad-hoc scripts. Nothing is reproducible.

Putting **LiteLLM** in front of MaaS solves the first three. Exposing
**search and retrieval as MCP servers** solves the fourth. We deliberately
deployed both on a single ECS for a small team — for two to ten engineers
this is the right unit of compute. The same architecture scales out to
multiple ECS or CCE without rethinking the contracts.

---

## Architecture at a glance

```
Laptop                                       Huawei Cloud (single ECS)
─────────────────────────────────────────    ──────────────────────────────────────
Claude Code  (untouched, → Anthropic)
                                             ┌─ LiteLLM Proxy            :4000
claude-glm                                   │    explicit model_list, FinOps,
  │ Anthropic-format request                 │    Redis cache, PostgreSQL keys
  ▼                                          │
claude-code-router (ccr) :3456 ──────────────┼─►  ────────────────►  Huawei MaaS
  │ rewrites to OpenAI-format                │     glm-5.1, etc.
  │                                          │
  ├── MCP "searxng"  (HTTP, bearer-auth) ────┼─►  SearXNG MCP            :8788
  │                                          │     ↓ HTTP
  │                                          │     SearXNG container     :8080 (loopback)
  │                                          │       └ public web search engines
  │                                          │
  └── MCP "css-search" (HTTP, bearer-auth) ──┼─►  CSS Code Search MCP    :8789
                                             │     ↓ HTTP                via nginx :8788/css/mcp
                                             │     CSS / OpenSearch      :9200 (private subnet)
                                             │       └ chunked Git repo, metadata, ACL fields
                                             │
                                             └─ nginx :8788 reverse-proxies
                                                /mcp        → SearXNG MCP (after migration)
                                                /css/mcp    → CSS Code Search MCP

Security group: allow tcp/22, tcp/4000, tcp/8788 from each operator's /32 only.
```

The salient property: **every external surface is one of three ports on
one ECS, locked to per-laptop `/32` allow-lists**. SearXNG and CSS
themselves are bound to loopback or the private subnet. Nothing
internet-facing is unauthenticated.

---

## Component 1 — LiteLLM in front of MaaS

LiteLLM speaks the OpenAI Chat Completions API and translates to dozens
of provider formats. For Huawei Cloud MaaS we use its OpenAI-compatible
endpoint and define explicit model mappings:

```yaml
model_list:
  - model_name: "huawei/glm-5.1"
    litellm_params:
      model: "openai/glm-5.1"
      api_base: os.environ/HUAWEI_MAAS_API_BASE
      api_key:  os.environ/HUAWEI_MAAS_API_KEY
      timeout:  120
      input_cost_per_token:  1.078e-06
      output_cost_per_token: 3.774e-06

  - model_name: "huawei-glm-5.1"   # alias without slash, friendlier for ccr
    litellm_params: { ...same... }
```

A few decisions earn their keep in production:

**Explicit prices, not zeros.** LiteLLM's `/model/info` reports
`input_cost_per_token` and `output_cost_per_token` to its budget engine.
If they are zero, every key has effectively unlimited spend — budgets do
not bite. We hardcode the validated GLM-5.1 unit prices.

**Local Redis and PostgreSQL.** Redis backs the response cache and
router state. PostgreSQL holds the master key, virtual keys, teams, and
spend logs. Both bind to `127.0.0.1` only and are managed by `systemd`.
There is no need to run a managed cluster for two-to-ten engineers.

**Master key is admin-only; clients use virtual keys.**

```bash
curl -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
     -H 'Content-Type: application/json' \
     -X POST http://<ECS>:4000/key/generate \
     -d '{"key_alias":"team-platform","models":["huawei-glm-5.1"],
          "max_budget":50,"budget_duration":"30d",
          "tpm_limit":40000,"rpm_limit":120}'
```

Each key is rotatable, revocable, and budgeted independently. Compromise
of one laptop only invalidates one key.

**Streaming works.** Claude Code sends streamed requests; LiteLLM
streams them through to MaaS and back. We verified the SSE path
end-to-end with `data: {...chunk...}` events flowing.

---

## Component 2 — SearXNG behind a FastMCP HTTP server

SearXNG is an open-source meta-search proxy. It does not have native
auth; we run it bound to `127.0.0.1:8080` and put a bearer-authenticated
**FastMCP HTTP server** in front. The MCP server exposes two tools to
the agent:

```python
@mcp.tool
async def web_search(query: str, num_results: int = 8,
                     language: str = "auto") -> list[dict]: ...

@mcp.tool
async def fetch_url(url: str, max_chars: int = 6000) -> str: ...
```

Why this split:

- **No new attack surface on the public web.** Only the MCP port
  (`8788`) is reachable, and only from operator `/32`s.
- **Output is normalized.** Snippets are capped to 500 chars; result
  count is capped to 20. SearXNG's verbose fields (engines, scores,
  parsed_url) are dropped because they balloon the agent's context
  budget without improving answers.
- **Auth is centralized.** A static bearer token in the `systemd` unit
  controls who can call the MCP. Rotation is one `sed` and a service
  restart.

SearXNG itself needs one non-default setting: `search.formats` must
include `json`. Without it, SearXNG cheerfully returns HTML for
`?format=json` and the MCP fails with an opaque parse error. This took
an embarrassing five minutes to diagnose the first time.

The HTTP MCP handshake follows the streamable transport spec:

```
POST /mcp   Authorization: Bearer <token>
            Accept: application/json,text/event-stream
            body  = { "jsonrpc":"2.0","id":1,"method":"initialize", ... }

200 OK      mcp-session-id: <session>
            data: { "result": { ... } }

POST /mcp   + mcp-session-id  body = { "method":"notifications/initialized" }
202 Accepted

POST /mcp   + mcp-session-id  body = { "method":"tools/list" }
data: { "result": { "tools":[ {...web_search...}, {...fetch_url...} ] } }
```

The `Accept: application/json,text/event-stream` header is mandatory at
this FastMCP version. Plain `application/json` is rejected with `406`.

---

## Component 3 — Code search via CSS/OpenSearch

The companion skill, `CSS-Code-Search-MCP`, indexes a Git repository
into Huawei Cloud CSS (Huawei's managed OpenSearch) and exposes three
tools:

| Tool | What it does |
|------|--------------|
| `search_code` | BM25 search across chunked code/doc files, with `category` and `skill` filter facets. |
| `list_skills` | Aggregations: every category and skill, with doc count. |
| `get_file` | Returns the full contents of one file by repo-relative path. |

Indexing strategy that worked in practice:

- **Chunk size 8000 characters with 500-character overlap.** Smaller
  chunks blow up the index; larger chunks lose retrieval precision when
  multiple concepts are in the same chunk.
- **Document fields:** `repo`, `category`, `skill`, `path`, `extension`,
  `chunk_id`, `content`. The first three drive UI filters and aggregations.
- **One index per repo.** This makes "re-index from main" a single index
  rebuild rather than a delete-by-query.

The CSS cluster sits on a private subnet (`192.168.0.x:9200`). The MCP
server on the ECS reaches it over the VPC; no public path exists.
External clients hit `nginx :8788/css/mcp` which reverse-proxies to the
local FastMCP on `:8789`.

Re-indexing after a repo update is one command on the ECS:

```bash
ssh root@<ECS> "
  cd /tmp && rm -rf <repo>
  git clone --depth 1 <repo-url>
  python3 /opt/index_repo_to_css.py /tmp/<repo> http://<css_private_ip>:9200
"
```

Because the agent calls `search_code` and `get_file` as native MCP
tools, the agent never has to clone the repo locally — useful for
laptops without enough disk or for repos with sensitive content that
should not leave the cloud boundary.

---

## Integration — Claude Code through `claude-code-router`

Claude Code speaks the **Anthropic Messages API**. Huawei MaaS speaks
**OpenAI Chat Completions**. We need a translator. The smallest
dependable one is `claude-code-router` (`ccr`):

```
claude-glm   — Anthropic format
  ↓ ANTHROPIC_BASE_URL=http://127.0.0.1:3456
ccr :3456     — rewrites to OpenAI format, applies "enhancetool" transformer
  ↓ http://<ECS>:4000/v1/chat/completions   Authorization: Bearer <virtual_key>
LiteLLM       — routes "huawei-glm-5.1" to MaaS, accounts spend
  ↓ https://api-...modelarts-maas.com/openai/v1
Huawei MaaS   — runs glm-5.1
```

Two design choices avoid common foot-guns:

**Point ccr at LiteLLM, not at MaaS.** The router config provider URL
is `http://<ECS>:4000/v1/chat/completions`. If you point ccr directly
at MaaS to "save a hop", you lose every spend, audit, and rate-limit
guarantee LiteLLM gives you, *and* you have to give every laptop the
real MaaS key. Don't.

**Use the no-slash model alias for ccr.** LiteLLM exposes both
`huawei/glm-5.1` and `huawei-glm-5.1`. ccr's router config strings are
parsed by comma (`provider,model`); slashes work but make scripts
harder to grep. We use `huawei-glm-5.1` everywhere on the laptop side.

### `CLAUDE_CONFIG_DIR` isolation

Claude Code stores user-scope MCP servers and settings in
`~/.claude.json` (or `$CLAUDE_CONFIG_DIR/.claude.json`). If we register
the SearXNG and CSS MCPs at default scope, they leak into the user's
plain `claude` invocations as well. That is almost never desired —
SearXNG is configured for the GLM coding agent, not for the user's
regular Anthropic-backed Claude flows.

The wrapper sets:

```bash
export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude-glm-config}"
```

`claude-glm` reads/writes `~/.claude-glm-config/.claude.json` (with the
two MCPs registered). Plain `claude` continues to use `~/.claude.json`
and never sees them. We verified isolation is preserved by listing MCPs
with and without the env var set.

### One wrapper script, two non-obvious gotchas

Two settings inside the wrapper bit us in production:

1. `DISABLE_COMPACT=true` was the inherited default. It disables Claude
   Code's auto-compaction. Long sessions then grow until they overflow
   GLM-5.1's hard input ceiling of **196,608 tokens** — LiteLLM
   surfaces a `400` with the upstream `Inference failed: prompt length
   N must less than the maximum input length 196608`. We changed the
   default to `false`.

2. `CLAUDE_CODE_MAX_CONTEXT_TOKENS=190000` left only ~6.6k headroom
   under the cap. Tool definitions, system prompt, and the next user
   turn easily push past it. We dropped it to `180000` (~8% headroom).

For an existing stuck session, `/compact` (in-place summary) or
`/clear` (fresh session) unblocks immediately. New sessions self-manage.

---

## What survived contact with reality

A field-tested list of things that wasted hours the first time and now
have one-liner fixes in the skills:

### Infrastructure

- **VPC quota exhausted.** `VPC.0114 Quota exceeded for resources:
  ['router']`. Don't delete shared infra; reuse the existing default
  VPC and subnet.
- **CSS COMMON disk sold out.** `CSS.0065`. Auto-fall back to `HIGH`
  disk type; the flavor stays the same.
- **Ubuntu image logs in as `root`, not `ubuntu`.** Several Huawei
  Ubuntu 22.04 images bake `PermitRootLogin yes` and don't activate
  the `ubuntu` account. Test both, record the working user.
- **EIP recycled to a new ECS.** Old SSH host key fingerprints fail with
  "REMOTE HOST IDENTIFICATION HAS CHANGED". `ssh-keygen -f
  ~/.ssh/known_hosts -R <ip>` and reconnect with `accept-new`.

### LiteLLM

- **`Unable to find Prisma binaries`.** `pip install prisma` does not
  fetch the platform-specific query engine. Run `prisma generate
  --schema <path>` then `prisma db push --schema <path> --accept-data-loss
  --skip-generate` in that order.
- **`Not connected to the query engine`.** The generated Prisma client
  bakes the engine path absolute (e.g. `/root/.cache/prisma-python/...`).
  When LiteLLM runs as the `litellm` user, that `0700` path is
  unreadable. Either `chmod -R o+rX /root/.cache/prisma-python` or
  regenerate with `HOME=/opt/litellm`.
- **`subprocess.run(["prisma"])` fails under systemd.** The default
  `PATH` is `/usr/sbin:/usr/bin:/sbin:/bin`. The venv's `prisma` CLI
  is invisible. Set `PATH=/opt/litellm-venv/bin:/usr/local/sbin:...`
  in `litellm.env`.

### SearXNG / FastMCP

- **`format=json` returns HTML.** `search.formats` must include
  `json`; the default is HTML-only.
- **`from fastmcp.server.auth.providers.bearer import StaticTokenVerifier`**
  no longer exists in FastMCP 2.14+. Use
  `from fastmcp.server.auth import StaticTokenVerifier`.
- **`406 Not Acceptable`.** The client sent `Accept: application/json`.
  The streamable HTTP transport requires `application/json,text/event-stream`.

### Claude Code / ccr

- **ccr returns 401 from upstream.** `$LITELLM_VIRTUAL_KEY` was not in
  ccr's environment when the daemon started. Always source the env
  file before `ccr start`; the wrapper guards against this on auto-
  start, but manual restarts can lose it.
- **Model id in chunks reads `glm-5.1`, not `huawei-glm-5.1`.** ccr is
  bypassing LiteLLM. The provider URL in `~/.claude-code-router/config.json`
  is wrong; verify it ends in `:4000/v1/chat/completions`.

### Security group hygiene

- **Outbound IP drift.** Mobile networks, VPNs, ISP rebalancing. The
  symptom in `claude-glm` is `Retrying ... attempt N/10 ·
  API_TIMEOUT_MS=...`. First diagnostic is `curl ifconfig.me`. Add a
  new `/32` rule, remove the stale one. **Never** widen to
  `0.0.0.0/0`.
- **Two rules per laptop, not one.** SSH (`tcp/22`) and LiteLLM
  (`tcp/4000`) are commonly added together; the MCP port (`tcp/8788`)
  is forgotten. List rules and confirm all three for each `/32`.

Every one of these is now documented in the skills' `troubleshooting.md`
with the exact command that fixes it.

---

## Onboarding and day-2 operations

For the first laptop, the operator runs the install scripts on the
ECS, mints a virtual key, registers both MCPs locally, and validates.
About 30 minutes start-to-finish on a clean account.

Adding a teammate's laptop later is intentionally cheap:

```bash
# Operator: mint a per-laptop virtual key, share it + ECS IP + MCP
# bearer token via 1Password / encrypted message.
# Operator: add the teammate's /32 to the SG on tcp/22, tcp/4000, tcp/8788.

# Teammate, on their laptop:
ECS_PUBLIC_IP='<...>' \
LITELLM_VIRTUAL_KEY='<...>' \
MCP_TOKEN='<...>' \
bash scripts/install_claude_glm_client.sh
```

The installer refuses to run if any of the three values are missing,
detects whether the laptop's `/32` is in the SG (by probing
`/health/liveliness`), and writes only into the user's home.

Token rotation has a clear separation of blast radius:

- **Virtual key compromise.** One key, one laptop. Operator revokes via
  `POST /key/delete`, mints a new one, hands it to the affected
  teammate.
- **MCP bearer compromise.** Shared across clients today. Operator
  edits `Environment=MCP_TOKEN=...` in the systemd unit, restarts the
  service, re-shares. Each client runs `mcp remove` then `mcp add` with
  the new bearer.
- **MaaS API key compromise.** One value on the ECS. Operator updates
  `/etc/litellm/litellm.env`, restarts `litellm.service`. No client
  changes.

---

## What this gives you

For a team of two-to-ten engineers using AI coding agents on Huawei
Cloud:

- **One bill, attributed.** LiteLLM logs every request with its virtual
  key. Per-laptop spend is queryable.
- **Real budgets.** Per-key `max_budget`, `budget_duration`,
  `tpm_limit`, `rpm_limit`. The proxy is the only egress path, so
  budgets actually bite.
- **Web search and code search as native tools.** No per-laptop scripts,
  no per-IDE settings. The agent calls `mcp__searxng__web_search` and
  `mcp__css-search__search_code` like it would call any built-in tool.
- **A clean separation between the user's regular `claude` and the
  GLM-routed `claude-glm`.** Two CLIs, two configs, no cross-talk.
- **A reproducible deploy.** Two skills, four scripts, a `validate_e2e.sh`
  that runs seven checks in order and fails loud.

The two skills together — about 20 files, ~1.5k lines of documentation,
~1k lines of scripts and configs — codify a setup we wish we had built
on day one. They do not pretend to be a replacement for a managed
gateway product; they are the thing you stand up when you need
production-shaped behavior in a week, not a quarter.

---

## Where to read next

- [`LiteLLM-SearXNG-AICoding-Gateway-Single-ECS`](https://github.com/binrogithub/1-3-Cloud-Adoption-Skills/tree/main/AI/AI-Development/LiteLLM-SearXNG-AICoding-Gateway-Single-ECS)
  — gateway skill: deployment walkthrough, AI-coding-agent
  integration, FastMCP transport detail, troubleshooting, laptop client
  onboarding, and four runnable scripts (`install_litellm.sh`,
  `install_searxng_and_mcp.sh`, `wire_claude_glm.sh`,
  `install_claude_glm_client.sh`, `validate_e2e.sh`).
- [`CSS-Code-Search-MCP`](https://github.com/binrogithub/1-3-Cloud-Adoption-Skills/tree/main/AI/AI-Coding/CSS-Code-Search-MCP)
  — code-search skill: provisioning CSS + ECS, indexing strategy,
  FastMCP server, nginx co-hosting pattern, and a one-shot
  `deploy_css_mcp.sh`.

If you adopt either skill on Huawei Cloud and hit a snag the
troubleshooting docs do not cover, file an issue with the symptom and
the fix; we will fold it back into the skill so the next person does
not lose the same hour.
