---
author: Robin
pubDatetime: 2026-03-11T16:34:00-03:00
title: "Deploying GLM-5 W4A8 on Huawei Cloud Ascend 910B x8"
description: "A field report on deploying GLM-5 W4A8 on ModelArts with 8 Ascend 910B cards. We solved Python stack issues and got the API server ready, but hit a decode blocker at the Ascend runtime layer. This is a practical summary of what worked, what broke, and where the real problem turned out to be."
tags:
  - glm-5
  - ascend
  - huawei-cloud
  - modelarts
  - vllm
  - deployment
  - ai
  - llm
  - inference
featured: true
draft: false
---

# Deploying GLM-5 W4A8 on Huawei Cloud Ascend 910B x8

## Introduction

Deploying a modern large model is rarely a straight line. On paper, the target in this project was simple enough: run **GLM-5 W4A8** on a **Huawei Cloud ModelArts Notebook with 8 x Ascend 910B** cards. In reality, the work turned into a layered engineering exercise involving model selection, image selection, environment persistence, dependency alignment, large-scale model transfer, and finally low-level Ascend runtime compatibility.

This article is a field report from that process. It is not a polished one-command tutorial. It is a practical summary of what worked, what broke, what we fixed, and what still blocked the final decode path.

The short version is this:

- We successfully brought up the **GLM-5 W4A8** service on 8 Ascend 910B cards.
- We solved the Python and vLLM stack issues.
- We got the API server ready and verified `/v1/models`.
- But the **first real generation request** still failed because the Notebook base runtime lacked a required Ascend operator.

That distinction matters. The remaining problem is no longer "how to deploy GLM-5 with vLLM." It is now "how to provide the right Ascend runtime beneath an otherwise working GLM-5 stack."

## Why GLM-5

There were easier choices.

We had already explored smaller and more forgiving models on Ascend, including single-card options such as Qwen-family models. Those are useful when the goal is to validate an environment quickly or get a lightweight local assistant running. But the target here was different. The goal was to stand up a more serious model on a proper 8-card 910B setup and see how far we could push a realistic coding-capable MoE deployment path on ModelArts.

That is why **GLM-5** became the focus.

It is a high-value target for an 8-card Ascend machine because it is large enough to justify the cluster resources, new enough to stress the modern inference stack, and meaningful enough to expose whether the platform can support a current-generation MoE-style workload end to end.

Once the objective became "make a real 8-card Ascend deployment work," GLM-5 was the right kind of model to test against.

## Choosing the Right GLM-5 Variant

The final model target was:

- **GLM-5 W4A8**

This was the practical choice for Ascend 910B x8.

Trying to force a raw full-precision route would have made memory pressure and runtime instability worse, while drifting to random community quantized variants would have introduced compatibility risk at the wrong layer. The point of this deployment was not just to boot "some GLM-like model," but to follow a version of GLM-5 that had a realistic chance of working with the Ascend vLLM path.

The working model directory in the Notebook was:

```text
/home/ma-user/work/models/GLM-5-w4a8
```

That path choice later turned out to matter almost as much as the model itself.

## The First Hard Lesson: Pick the Right Base Image Early

One of the earliest blockers had nothing to do with GLM-5 directly.

The deployment failed on older environments because the base userspace was too old. In practice, the relevant symptom was that a low `glibc` baseline prevented the newer vLLM stack from behaving normally. Once you hit that class of error, everything downstream becomes noisy: wheels stop matching, source builds become more fragile, and even when individual packages install, the environment is already tilted against you.

The fix was to move to a newer Ascend PyTorch image:

```text
swr.cn-southwest-2.myhuaweicloud.com/atelier/pytorch_ascend:pytorch_2.7.1-cann_8.3.rc1-py_3.11-hce_2.0.2509-aarch64-snt9b-20251205091605-e41006e
```

Why this image mattered:

- It provided a usable modern Python baseline.
- It was aligned with `aarch64`.
- It was a realistic starting point for Ascend + PyTorch work on ModelArts.
- It removed the "old glibc" class of deployment failure.

That did not mean the image was perfect. In fact, a later low-level runtime issue still remained. But this image was the first base that let the deployment move forward in a meaningful way.

## A 500 GB-Class Model Changes the Download Strategy

GLM-5 W4A8 is not a model you casually pull inside an interactive Notebook and forget about.

The model is large enough that direct Notebook download becomes an operational problem:

- it is slow
- it is easy to interrupt
- it wastes expensive accelerator time
- recovery after partial failure is annoying

The better workflow was:

1. Download in an **ECS instance in the same region**
2. Transfer the model into the ModelArts Notebook
3. Use **parallel transfer** for ECS-to-Notebook copy

That approach saves time for a simple reason: region-local transfer is much better than forcing the Notebook to do everything itself. It also makes retries less painful and keeps the Notebook focused on what it should really do: environment setup and inference.

This became one of the most practical lessons from the whole deployment.

For large models, download strategy is infrastructure design, not just housekeeping.

## ModelArts Persistence Rules Matter More Than People Think

Another failure mode had nothing to do with AI at all. It was a storage layout mistake.

When ModelArts Notebook paths are used incorrectly, important tools and environments can disappear after restart. That happened with `vllm`: parts of the environment were installed outside the correct persistent location, which made the setup fragile and easy to lose.

The fix was simple but non-negotiable:

- Use the ModelArts persistent work directory:

```text
/home/ma-user/work/
```

From then on, long-lived assets were treated as first-class persistent state:

- model files
- Python virtual environments
- startup scripts
- logs

That changed the operational stability of the deployment immediately.

Just as important, the environment had to be checked repeatedly before launch:

- confirm `vllm` still exists
- confirm the correct venv is active
- confirm scripts still point to the right paths
- confirm the model directory is complete

In ModelArts, persistence discipline is part of deployment correctness.

## Preparing the vLLM Environment

The initial strategy was conservative:

- keep a persistent venv under `/home/ma-user/work/venvs/`
- keep logs under `/home/ma-user/work/logs/`
- keep helper scripts under `/home/ma-user/work/bin/`

The first serious stack that came up was the older stable direction:

- `vllm-ascend 0.11.0`
- `vllm 0.11.0`

That choice was understandable. Older stacks are easier to install, easier to reason about, and often good enough for many models. The CLI came up. Basic checks passed. It looked promising.

But GLM-5 is not a forgiving target for an older inference stack.

What looked like a working environment at the command-line level was not actually ready for this specific model family.

## Problem Review: `glm_moe_dsa` and the Transformers Mismatch

The next blocker was more model-specific.

The venv contained `transformers 4.57.1`, but that version did not properly support the GLM-5 architecture identifier:

```text
model_type = glm_moe_dsa
```

At this point, the debugging strategy was deliberately narrow. Instead of changing everything at once, only `transformers` in the existing venv was upgraded to GitHub main. Then three focused checks were used to separate recognition problems from deeper runtime problems:

1. `GlmMoeDsaConfig`
2. `CONFIG_MAPPING`
3. `tiny-random/glm-moe-dsa`

This was the turning point in the investigation.

Once those checks passed, the problem was no longer "Transformers does not recognize GLM-5." That entire class of failure was eliminated. The remaining issue had to be elsewhere.

The new conclusion was much sharper:

- `transformers` recognition had been fixed
- if GLM-5 still failed, the issue was now the `vllm` / `vllm-ascend` stack combination

That is exactly what happened.

## Why the Old vLLM Stack Was Not Enough

After upgrading only `transformers`, a new error appeared. That error was useful because it proved the old stable stack was the wrong fit for GLM-5.

In other words:

- old `transformers` was too old for `glm_moe_dsa`
- new `transformers` fixed that
- but new `transformers` no longer fit cleanly with `vllm 0.11.0 / vllm-ascend 0.11.0`

This meant the environment had entered a mixed-stack state:

- correct model recognition
- incorrect cross-package compatibility

That was the signal to stop patching around the edges and move to the correct stack shape.

## Rebuilding the Stack the Right Way

At this point, the environment was rebuilt in a new persistent venv using a main-branch-oriented GLM-5-compatible stack.

Conceptually, the changes were:

- rebuild in a clean venv
- move `vllm` to the newer main-branch line
- move `vllm-ascend` to the matching main-branch line
- keep `transformers` on GitHub main

This was the right move for two reasons:

1. It matched the reality of GLM-5 better than the old stable stack.
2. It kept the deployment aligned with the actual behavior of the software, rather than trying to force an older compatibility story that was clearly no longer true.

The result was significant:

- the earlier tokenizer and architecture errors disappeared
- `glm_moe_dsa` was no longer the blocker
- GLM-5 could move from "recognized" to "actually booting"

## What Finally Worked

This is the part that matters most when evaluating real deployment progress.

With the rebuilt stack, the system successfully achieved all of the following:

- 8-card HCCL initialization
- expert parallel placement
- shard-by-shard model loading across all `96` checkpoint shards
- successful weight loading on the full machine
- API server startup
- a working `/v1/models` response on port `8077`

That is a meaningful success milestone.

It proves that:

- the model path was correct
- the startup scripts were correct
- the persistent environment strategy was correct
- the main-branch Python stack was good enough to boot the model
- the 8-card topology itself was not the reason for failure

This was no longer a fake success like "the CLI works." The deployment had genuinely crossed into server-ready territory.

## The Final Blocker: Decode Fails on the Ascend Runtime

The final failure only appeared when the first real generation request was sent.

The service had already become ready:

- `:8077` was listening
- `/v1/models` worked

But the first `/v1/chat/completions` request returned `500`, and the actual root cause was lower in the stack:

```text
RuntimeError: aclnnLightningIndexer or aclnnLightningIndexerGetWorkspaceSize not in libopapi.so, or libopapi.so not found.
```

This error matters because it changes the diagnosis completely.

At that stage, the problem was **not**:

- the model
- the scripts
- `transformers`
- service boot
- shard loading
- HCCL initialization

Instead, the problem was an Ascend runtime capability gap.

The Notebook base image was still tied to:

- `CANN 8.3.RC1`

while the newer `vllm-ascend` path was clearly expecting a more capable underlying Ascend runtime. The failure occurred inside the decode path when `torch_npu.npu_lightning_indexer` was invoked and the needed low-level symbol was not available from `libopapi.so`.

That is why this deployment should be described carefully:

- **GLM-5 booted**
- **weights loaded**
- **the API came up**
- **but first-token decode failed because the base Ascend runtime was still behind the required operator capability**

This is not a minor detail. It is the dividing line between a Python packaging problem and a real platform runtime problem.

## Lessons Learned

Several practical lessons came out of this project.

### 1. Base image selection is not a boring detail

If the base image is too old, deployment becomes a fight against the system before the model is even relevant.

### 2. For very large models, download strategy is part of engineering design

Using regional ECS for download and parallel transfer into ModelArts saves both time and failure recovery effort.

### 3. Persistent paths are part of deployment correctness

On ModelArts Notebook, `/home/ma-user/work/` is not optional housekeeping. It is where the deployment state must live if you want the setup to survive.

### 4. "The CLI works" is not the same as "the model serves"

Many teams stop too early once `vllm --help` works or once imports succeed. Real validation starts much later:

- model recognition
- shard loading
- API readiness
- first real decode

### 5. For GLM-5, old stable inference stacks are not enough

The combination of GLM-5 and `glm_moe_dsa` pushed the deployment toward newer `transformers`, newer `vllm`, and newer `vllm-ascend`.

### 6. On Ascend, Python-layer compatibility is only half the story

Even when the Python stack is finally aligned, low-level Ascend runtime capabilities can still decide whether inference actually works.

## Recommended Next Step

If I had to summarize the next move in one sentence, it would be this:

**Do not restart the Python debugging cycle. Upgrade the underlying Ascend runtime baseline.**

The evidence now strongly suggests that the remaining blocker is not in the application layer. The deployment has already done enough to prove that.

The right follow-up path is:

- keep the validated GLM-5 main-branch Python stack
- keep the persistent Notebook layout under `/home/ma-user/work/`
- move to a newer Ascend runtime baseline that provides the missing decode operator capability

In other words, the next iteration should focus on the platform runtime, not on rewriting startup scripts or re-litigating model selection.

## Final Takeaway

This GLM-5 deployment was not a failure.

It was a successful narrowing of the problem.

We now know all of the following with confidence:

- **GLM-5 W4A8 is the right class of model for this 8-card Ascend target**
- **the ModelArts persistent deployment layout is understood**
- **the correct vLLM stack direction is the newer main-branch path**
- **the service can boot and become API-ready**
- **the remaining blocker is the Ascend runtime beneath the Notebook, not the Python stack above it**

That is exactly the kind of result you want from a serious deployment attempt: not false optimism, but a precise boundary around the real next problem.
