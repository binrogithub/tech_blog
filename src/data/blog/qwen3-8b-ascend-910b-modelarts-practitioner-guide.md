---
author: Robin
pubDatetime: 2026-02-02T15:35:00-03:00
title: "Deploying Qwen3-8B on Huawei Cloud ModelArts with Ascend 910B (Practitioner’s Runbook)"
description: "A practical runbook to deploy Qwen3-8B on Huawei Cloud ModelArts Notebook with Ascend 910B NPUs: custom image via SWR, model download, inference, and an optional FastAPI endpoint—plus the common pitfalls." 
tags:
  - huawei-cloud
  - modelarts
  - ascend
  - qwen
  - llm
  - runbook
featured: false
draft: true
---

## Why this runbook exists

Getting an LLM like **Qwen3-8B** running on **Ascend 910B** is very doable, but the workflow differs from the CUDA-first ecosystem most teams are used to. This post consolidates a reproducible path on **Huawei Cloud ModelArts Notebook**.

What you’ll end up with:
- a custom container image (CANN + PyTorch + torch_npu + transformers)
- model weights downloaded (ModelScope / HuggingFace / OBS)
- interactive inference in Notebook
- optional API serving (FastAPI)

## Important caveats (read first)

1) **Version availability may vary by region/mirror.** PyTorch 2.6.0 was released on **2026-01-29**; depending on your region and mirrors, `torch_npu==2.6.0.post3` may not be immediately available. Treat the exact versions below as a target, not a guarantee.

2) **Long-context memory can OOM on 32GB 910B.** Qwen3-8B with large context windows (e.g., 32K) may exceed 32GB due to KV cache + overhead. Prefer 64GB when possible, or reduce context / generation length.

3) **Model download paths can differ.** ModelScope’s `snapshot_download()` uses nested cache layouts. Always verify the real model path by locating `config.json`.

## Prerequisites

Huawei Cloud prerequisites:
- ModelArts enabled in your region
- SWR available (for custom images)
- access to **Ascend 910B** resource pool
- IAM permissions for ModelArts + SWR (+ OBS if you stage weights)

Local prerequisites (for building images):
- Docker 20.10+
- Either an ARM64 machine **or** Docker buildx/QEMU for cross-build

## Stack overview

| Component | Version (target) |
|---|---|
| CANN | 8.3.RC2 |
| PyTorch | 2.6.0 |
| torch_npu | 2.6.0.post3 |
| transformers | >= 4.51.0 |
| Python | 3.11 |

> As of early 2026, ModelArts official examples often lag behind (e.g., older PyTorch/CANN). In practice, a custom image is usually the most reliable approach.

---

## Step 1 — Build a custom image (SWR)

ModelArts supports custom images via **SWR**, and commonly expects a `ma-user` user inside the image. The Dockerfile below uses a CANN base image and then installs PyTorch + torch_npu.

### Dockerfile (baseline)

```dockerfile
FROM quay.io/ascend/cann:8.3.rc2-910b-ubuntu22.04-py3.11

LABEL description="PyTorch 2.6.0 + torch_npu 2.6.0.post3 for Ascend 910B"

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl vim htop \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# PyTorch 2.6.0 (CPU wheel; NPU acceleration comes from torch_npu)
RUN pip install --upgrade pip \
    && pip install torch==2.6.0 torchvision==0.21.0 \
       --index-url https://download.pytorch.org/whl/cpu

# Ascend NPU support
RUN pip install torch_npu==2.6.0.post3

# LLM dependencies
RUN pip install \
    "transformers>=4.51.0" \
    "accelerate>=0.26.0" \
    safetensors sentencepiece protobuf tiktoken einops \
    modelscope huggingface_hub

# Optional serving dependencies
RUN pip install fastapi uvicorn pydantic

# ModelArts compatibility user
RUN default_user=$(getent passwd 1000 | cut -d: -f1) || true && \
    default_group=$(getent group 100 | cut -d: -f1) || true && \
    if [ -n "$default_user" ] && [ "$default_user" != "ma-user" ]; then userdel -r "$default_user" 2>/dev/null || true; fi && \
    if [ -n "$default_group" ] && [ "$default_group" != "ma-group" ]; then groupdel -f "$default_group" 2>/dev/null || true; fi && \
    groupadd -g 100 ma-group 2>/dev/null || true && \
    useradd -d /home/ma-user -m -u 1000 -g 100 -s /bin/bash ma-user && \
    chmod 750 /home/ma-user

RUN mkdir -p /home/ma-user/work/models /home/ma-user/work/scripts \
    && chown -R ma-user:ma-group /home/ma-user

RUN echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true' \
    >> /home/ma-user/.bashrc

USER ma-user
WORKDIR /home/ma-user/work
CMD ["/bin/bash"]
```

### Build & push

- **ARM64 native build** is simplest.
- On x86, use buildx:

```bash
docker buildx create --name ascend-builder --use
docker buildx inspect --bootstrap

docker buildx build --platform linux/arm64 \
  -t ascend-pytorch26-qwen3:v1.0 \
  --load .
```

Push to SWR (use the login command provided by the console; do not hardcode credentials):

```bash
docker tag ascend-pytorch26-qwen3:v1.0 swr.${REGION}.myhuaweicloud.com/${SWR_ORG}/ascend-pytorch26-qwen3:v1.0
docker push swr.${REGION}.myhuaweicloud.com/${SWR_ORG}/ascend-pytorch26-qwen3:v1.0
```

Register the image in ModelArts → Image Management.

---

## Step 2 — Create a ModelArts Notebook

Create Notebook:
- Resource: Ascend
- Spec: 1× Ascend 910B (prefer 64GB for long context)
- Image: your custom image

---

## Step 3 — Download Qwen3-8B weights

### Option A: ModelScope

```bash
mkdir -p /home/ma-user/work/models
cd /home/ma-user/work/models

python3 << 'EOF'
from modelscope import snapshot_download

cache_dir = '/home/ma-user/work/models'
model_dir = snapshot_download('Qwen/Qwen3-8B', cache_dir=cache_dir, revision='master')
print('Model downloaded to:', model_dir)
EOF

# Verify the real path
find /home/ma-user/work/models -name "config.json" -type f 2>/dev/null | head
```

### Option B: HuggingFace

```bash
mkdir -p /home/ma-user/work/models
cd /home/ma-user/work/models
huggingface-cli download Qwen/Qwen3-8B \
  --local-dir ./Qwen3-8B \
  --local-dir-use-symlinks False
```

### Option C: OBS (pre-staged)

```bash
obsutil cp -r obs://your-bucket/models/Qwen3-8B /home/ma-user/work/models/Qwen/Qwen3-8B
```

---

## Step 4 — Inference (Notebook)

Create `inference.py` and run it:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python /home/ma-user/work/scripts/inference.py --model_path <YOUR_MODEL_PATH> --prompt "Hello" --thinking
```

If you enable thinking mode, Qwen3 may output `<think>...</think>` blocks. If you need to separate “reasoning” from “final answer”, parse the tags.

---

## Step 5 — Optional: expose an API (FastAPI)

You can wrap the model with FastAPI for demos, but do not expose it publicly without authentication.

---

## Troubleshooting quick hits

- `torch_npu` install fails: confirm which pip index/mirror you’re using; versions may differ by region.
- OOM: reduce `max_new_tokens`, reduce context, or move to 910B 64GB.
- CANN mismatch: check `npu-smi info` and toolkit version.

---

## References

- Huawei Cloud ModelArts docs: https://support.huaweicloud.com/intl/en-us/modelarts/
- Ascend PyTorch: https://github.com/Ascend/pytorch
- Qwen docs: https://qwen.readthedocs.io/
