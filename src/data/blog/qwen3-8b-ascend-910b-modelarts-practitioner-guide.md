---
author: Robin
pubDatetime: 2026-02-02T17:45:00-03:00
title: "Qwen3-8B Deployment on Huawei Cloud ModelArts Notebook (Ascend 910B + PyTorch 2.6.0)"
description: "Comprehensive operational runbook: deploy Qwen3-8B on Huawei Cloud ModelArts Notebook with Ascend 910B and PyTorch 2.6.0 (Ascend build), including custom image, model download, inference, and optional API serving."
tags:
  - huawei-cloud
  - modelarts
  - ascend
  - qwen
  - llm
  - runbook
featured: false
draft: false
---

# Qwen3-8B Deployment on Huawei Cloud ModelArts Notebook (Ascend 910B + PyTorch 2.6.0)

## Comprehensive Operational Runbook

**Document Version:** 1.0  
**Date:** January 29, 2026  
**Target Audience:** MLOps Engineers, AI Infrastructure Teams

---

## 1. Overview & Prerequisites

### 1.1 Objective

Deploy **Qwen3-8B** large language model on **Huawei Cloud ModelArts Notebook** using **Ascend 910B NPU** with **PyTorch 2.6.0** (Ascend build), including:
- Custom container image creation and registration
- Model weights download from ModelScope/HuggingFace
- Interactive inference in Notebook environment
- Optional API service deployment (FastAPI/Gradio)

### 1.2 User-Configurable Variables

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `REGION` | Huawei Cloud region | `cn-north-4`, `ap-southeast-1` |
| `SWR_ENDPOINT` | SWR registry endpoint | `swr.{REGION}.myhuaweicloud.com` |
| `SWR_ORG` | SWR organization name | `my-ai-org` |
| `MODEL_PATH` | Local model storage path | `/home/ma-user/work/models/Qwen/Qwen3-8B` |
| `HF_TOKEN` | HuggingFace access token (optional) | `hf_xxx...` |
| `MODELSCOPE_TOKEN` | ModelScope access token (optional) | `ms_xxx...` |

### 1.3 Access & Authorization Prerequisites

1. **Huawei Cloud Account** with:
   - ModelArts service activated
   - SWR (SoftWare Repository for Container) access
   - Ascend 910B resource pool allocation (dedicated or public)
   - IAM permissions for ModelArts, SWR, and OBS

2. **Local Development Environment** (for custom image build):
   - Docker 18.09.7+ installed
   - ARM64 cross-compilation capability (if building on x86)
   - Internet access for pulling base images and packages

3. **Model Access**:
   - ModelScope account (recommended for China regions)
   - HuggingFace account (optional, for international access)

---

## 2. Evidence Map (Task A)

| # | Claim | Source URL | Credibility | Notes |
|---|-------|------------|-------------|-------|
| 1 | ModelArts Notebook supports custom images via SWR; requires `ma-user` (UID 1000) and `ma-group` (GID 100) | https://support.huaweicloud.com/intl/en-us/usermanual-standard-modelarts/docker-modelarts_0011-0.html | High | Official Huawei Cloud docs |
| 2 | PyTorch 2.6.0 + torch_npu 2.6.0.post3 requires CANN 8.2+ | https://docs.sglang.io/platforms/ascend_npu.html | High | SGLang official documentation |
| 3 | Official CANN 8.3.RC2 Docker image for 910B: `quay.io/ascend/cann:8.3.rc2-910b-ubuntu22.04-py3.11` | https://docs.vllm.ai/projects/ascend/en/latest/installation.html | High | vLLM-Ascend official docs |
| 4 | Qwen3-8B requires `transformers>=4.51.0`; available at Qwen/Qwen3-8B | https://huggingface.co/Qwen/Qwen3-8B | High | Official Qwen model card |
| 5 | Qwen3-8B: 8B parameters, 32K native context, Apache 2.0 license | https://qwenlm.github.io/blog/qwen3/ | High | Official Qwen blog |
| 6 | ModelScope download: `from modelscope import snapshot_download` | https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html | High | Official Qwen docs |
| 7 | SWR login command generation via console; temporary (6h) and long-term commands available | https://support.huaweicloud.com/intl/en-us/usermanual-swr/swr_01_1000.html | High | Official SWR docs |
| 8 | torch_npu auto-imports in version 2.5.1+; CANN env via `source /usr/local/Ascend/ascend-toolkit/set_env.sh` | https://github.com/Ascend/pytorch | High | Official Ascend PyTorch repo |
| 9 | Ascend 910B supports 32GB and 64GB HBM variants | https://www.aidoczh.com/llamafactory/en/advanced/npu.html | Medium | LLaMA-Factory documentation |
| 10 | vLLM-Ascend v0.12+ uses CANN 8.3.RC2, PyTorch 2.8.0, torch-npu 2.8.0 | https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html | High | vLLM-Ascend release notes |

### 2.1 Compatibility Gap Statement

> **⚠️ COMPATIBILITY GAP IDENTIFIED**
> 
> Official ModelArts documentation (as of January 2026) provides examples using **PyTorch 2.1.0 + CANN 7.0.0**. The target combination of **PyTorch 2.6.0 + torch_npu 2.6.0.post3 + CANN 8.3.RC2** is verified by community projects (vLLM-Ascend, SGLang) but **not explicitly documented** in ModelArts official guides.
> 
> **Recommendation:** Use the custom image approach (Path 2) for production deployments with PyTorch 2.6.0.

---

## 3. Reasonableness & Compatibility Check

### 3.1 Model Size Analysis

| Metric | Value | Calculation |
|--------|-------|-------------|
| Parameters | 8 billion | Dense model |
| Weight Size (BF16) | ~16 GB | 8B × 2 bytes |
| Weight Size (FP32) | ~32 GB | 8B × 4 bytes |
| KV Cache (32K context, BF16) | ~4-6 GB | Varies with batch size |
| **Total Inference Memory** | **~20-24 GB** | Weights + KV Cache + overhead |

### 3.2 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| NPU | 1× Ascend 910B (32GB HBM) | 1× Ascend 910B (64GB HBM) |
| System Memory | 32 GB | 64 GB |
| Disk Space | 50 GB | 100 GB |
| Network | Public internet or NAT | Public internet for model download |

### 3.3 Version Compatibility Matrix

| Component | Version | Verified Source |
|-----------|---------|-----------------|
| PyTorch | 2.6.0 | SGLang docs |
| torch_npu | 2.6.0.post3 | SGLang docs |
| CANN | 8.2+ (recommend 8.3.RC2) | vLLM-Ascend docs |
| Python | 3.11 | CANN image default |
| transformers | ≥4.51.0 | Qwen model card |
| Ubuntu | 22.04 | CANN image default |

### 3.4 Download Time Estimation

| Source | Model Size | Network Speed | Estimated Time |
|--------|------------|---------------|----------------|
| ModelScope (China) | ~16 GB | 100 Mbps | ~25 minutes |
| HuggingFace (International) | ~16 GB | 50 Mbps | ~50 minutes |
| OBS (Pre-staged) | ~16 GB | 1 Gbps intranet | ~2 minutes |

---

## 4. Environment Setup Paths

### Path 1: Check ModelArts Preset Images (Recommended First)

> **⚠️ Unverified — Confirm in Huawei Cloud console**
> 
> ModelArts may provide preset Ascend PyTorch 2.6.0 images. Check the console before building a custom image.

**Verification Steps:**

1. Log in to ModelArts Console → Development Workspace → Notebook
2. Click "Create Notebook"
3. In "Image" section, expand "Ascend" or "NPU" category
4. Look for images containing:
   - `pytorch2.6` or `pytorch-2.6`
   - `cann8.2` or `cann8.3`
5. If found, note the image name and skip to Section 6 (Model Download)

**If no suitable preset image exists, proceed to Path 2.**

---

### Path 2: Custom Docker Image via SWR

#### Step 4.1: Create Dockerfile

Create a working directory and Dockerfile:

```bash
mkdir -p ~/ascend-qwen3-image/context
cd ~/ascend-qwen3-image/context
```

Create `Dockerfile`:

```dockerfile
# ==============================================================================
# Dockerfile: Ascend 910B + PyTorch 2.6.0 + Qwen3-8B Inference Environment
# Base: Official CANN 8.3.RC2 image for Ascend 910B
# ==============================================================================

FROM quay.io/ascend/cann:8.3.rc2-910b-ubuntu22.04-py3.11

# Maintainer info
LABEL maintainer="your-team@example.com"
LABEL description="PyTorch 2.6.0 + torch_npu 2.6.0.post3 for Qwen3-8B on Ascend 910B"

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Configure pip mirrors (Huawei Cloud mirrors for faster download)
RUN pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple \
    && pip config set global.trusted-host repo.huaweicloud.com

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    vim \
    htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.6.0 (CPU wheel, NPU support via torch_npu)
RUN pip install --upgrade pip \
    && pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

# Install torch_npu 2.6.0.post3 (Ascend NPU support)
RUN pip install torch_npu==2.6.0.post3

# Install ML/AI dependencies
RUN pip install \
    "transformers>=4.51.0" \
    "accelerate>=0.26.0" \
    "safetensors>=0.4.0" \
    "sentencepiece>=0.1.99" \
    "protobuf>=3.20.0" \
    "tiktoken>=0.5.0" \
    "einops>=0.7.0" \
    modelscope \
    huggingface_hub

# Install optional service dependencies
RUN pip install \
    fastapi \
    uvicorn \
    gradio \
    pydantic

# ==============================================================================
# ModelArts Notebook Compatibility: Create ma-user (UID 1000, GID 100)
# This is REQUIRED for ModelArts Notebook to work correctly
# ==============================================================================
RUN default_user=$(getent passwd 1000 | awk -F ':' '{print $1}') || true && \
    default_group=$(getent group 100 | awk -F ':' '{print $1}') || true && \
    if [ ! -z "${default_user}" ] && [ "${default_user}" != "ma-user" ]; then \
        userdel -r ${default_user} || true; \
    fi && \
    if [ ! -z "${default_group}" ] && [ "${default_group}" != "ma-group" ]; then \
        groupdel -f ${default_group} || true; \
    fi && \
    groupadd -g 100 ma-group || true && \
    useradd -d /home/ma-user -m -u 1000 -g 100 -s /bin/bash ma-user && \
    chmod 750 /home/ma-user

# Create working directories with proper permissions
RUN mkdir -p /home/ma-user/work /home/ma-user/work/models /home/ma-user/work/scripts \
    && chown -R ma-user:ma-group /home/ma-user

# Set CANN environment (will be sourced at runtime)
ENV ASCEND_HOME=/usr/local/Ascend
ENV LD_LIBRARY_PATH=${ASCEND_HOME}/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${ASCEND_HOME}/ascend-toolkit/latest/python/site-packages:${PYTHONPATH}

# Copy CANN environment setup to bashrc
RUN echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true' >> /home/ma-user/.bashrc

# Switch to ma-user for runtime
USER ma-user
WORKDIR /home/ma-user/work

# Health check script
RUN echo '#!/bin/bash\npython3 -c "import torch; print(f\"PyTorch: {torch.__version__}\"); import torch_npu; print(f\"NPU available: {torch.npu.is_available()}\")"' > /home/ma-user/healthcheck.sh \
    && chmod +x /home/ma-user/healthcheck.sh

# Default command
CMD ["/bin/bash"]
```

#### Step 4.2: Build the Docker Image

**Option A: Build on ARM64 machine (native)**

```bash
cd ~/ascend-qwen3-image/context

# Build the image
docker build -t ascend-pytorch26-qwen3:v1.0 -f Dockerfile .

# Verify the build
docker images | grep ascend-pytorch26-qwen3
```

**Option B: Build on x86 machine with cross-compilation**

```bash
# Enable buildx for multi-platform builds
docker buildx create --name multiarch --use
docker buildx inspect --bootstrap

# Build for ARM64
docker buildx build --platform linux/arm64 \
    -t ascend-pytorch26-qwen3:v1.0 \
    -f Dockerfile \
    --load .
```

#### Step 4.3: Push Image to SWR

```bash
# ============================================================
# Configure these variables for your environment
# ============================================================
export REGION="cn-north-4"  # Change to your region
export SWR_ENDPOINT="swr.${REGION}.myhuaweicloud.com"
export SWR_ORG="your-organization"  # Your SWR organization name
export IMAGE_NAME="ascend-pytorch26-qwen3"
export IMAGE_TAG="v1.0"

# ============================================================
# Step 1: Login to SWR
# Get login command from: SWR Console → Dashboard → Generate Login Command
# ============================================================
# Option A: Temporary login (valid 6 hours)
# Copy the command from SWR console, it looks like:
# docker login -u {region}@{AK} -p {login_key} ${SWR_ENDPOINT}

# Option B: Using environment variables (DO NOT hardcode in scripts!)
docker login -u "${REGION}@${HUAWEI_AK}" -p "${HUAWEI_LOGIN_KEY}" ${SWR_ENDPOINT}

# ============================================================
# Step 2: Tag the image for SWR
# ============================================================
docker tag ${IMAGE_NAME}:${IMAGE_TAG} \
    ${SWR_ENDPOINT}/${SWR_ORG}/${IMAGE_NAME}:${IMAGE_TAG}

# ============================================================
# Step 3: Push to SWR
# ============================================================
docker push ${SWR_ENDPOINT}/${SWR_ORG}/${IMAGE_NAME}:${IMAGE_TAG}

# ============================================================
# Step 4: Verify upload
# ============================================================
echo "Image pushed successfully!"
echo "SWR URL: ${SWR_ENDPOINT}/${SWR_ORG}/${IMAGE_NAME}:${IMAGE_TAG}"
```

> **🔐 Security Note:**
> - Never hardcode AK/SK or login keys in scripts
> - Use environment variables or Huawei Cloud IAM for credential management
> - Temporary login commands expire after 6 hours

#### Step 4.4: Register Image in ModelArts

1. **Log in to ModelArts Console**
2. Navigate to: **Image Management** (left sidebar)
3. Click **Register**
4. Configure:
   - **SWR Source:** Paste full SWR URL: `swr.{REGION}.myhuaweicloud.com/{ORG}/ascend-pytorch26-qwen3:v1.0`
   - **Architecture:** `ARM` (aarch64)
   - **Type:** `Ascend`
   - **Specifications:** Select appropriate NPU type (e.g., `Ascend 910B`)
5. Click **Register**

> **⚠️ Unverified — Confirm field names in Huawei Cloud console**
> The exact field names and options may vary. Refer to the console interface for accurate labels.

---

## 5. Create Notebook Instance

1. **Navigate to:** ModelArts Console → Development Workspace → Notebook
2. **Click:** Create Notebook
3. **Configure:**

| Setting | Value |
|---------|-------|
| Name | `qwen3-8b-inference` |
| Image | Select registered custom image OR preset Ascend image |
| Resource Type | Ascend |
| Specifications | `Ascend: 1*Ascend 910B 32GB` or `64GB` variant |
| Storage | Attach OBS bucket if needed (for model storage) |
| Auto Stop | Configure based on usage (e.g., 4 hours) |

4. **Click:** Create
5. **Wait** for instance to start (may take 3-5 minutes for image pull)
6. **Open** JupyterLab when status shows "Running"

---

## 6. Download Qwen3-8B Weights (Task E)

### Method 1: ModelScope SDK (Recommended for China)

Open a terminal in JupyterLab and run:

```bash
# ============================================================
# Qwen3-8B Download via ModelScope
# ============================================================

# Create model directory
mkdir -p /home/ma-user/work/models
cd /home/ma-user/work/models

# Download using Python script
python3 << 'EOF'
from modelscope import snapshot_download
import os

# Configure download path
cache_dir = '/home/ma-user/work/models'

# Download Qwen3-8B
print("Starting Qwen3-8B download from ModelScope...")
model_dir = snapshot_download(
    'Qwen/Qwen3-8B',
    cache_dir=cache_dir,
    revision='master'  # or specific commit hash
)
print(f"Model downloaded to: {model_dir}")
EOF
```

**Alternative: ModelScope CLI**

```bash
# Install modelscope CLI if not present
pip install modelscope

# Download using CLI
modelscope download --model Qwen/Qwen3-8B --cache_dir /home/ma-user/work/models

# The model will be at: /home/ma-user/work/models/Qwen/Qwen3-8B
```

### Method 2: HuggingFace with git-lfs

```bash
# ============================================================
# Qwen3-8B Download via HuggingFace
# ============================================================

# Ensure git-lfs is installed
git lfs install

# Create and enter model directory
mkdir -p /home/ma-user/work/models
cd /home/ma-user/work/models

# Clone the model repository
# Note: This will download ~16GB of model weights
git clone https://huggingface.co/Qwen/Qwen3-8B

# Verify download
ls -lah Qwen3-8B/
du -sh Qwen3-8B/
```

**Alternative: huggingface-cli**

```bash
# Login (optional, for gated models)
# huggingface-cli login --token $HF_TOKEN

# Download using CLI
huggingface-cli download Qwen/Qwen3-8B \
    --local-dir /home/ma-user/work/models/Qwen/Qwen3-8B \
    --local-dir-use-symlinks False
```

### Method 3: Pre-staged from OBS (Offline/Restricted Network)

If model weights are pre-staged in OBS:

```bash
# ============================================================
# Download from OBS (if pre-staged)
# ============================================================

# Configure OBS path (example)
OBS_BUCKET="obs://your-bucket/models/Qwen3-8B"
LOCAL_PATH="/home/ma-user/work/models/Qwen/Qwen3-8B"

# Method A: Using obsutil (if available)
obsutil cp -r ${OBS_BUCKET} ${LOCAL_PATH}

# Method B: Using ModelArts SDK
python3 << EOF
from modelarts.session import Session
session = Session()
session.obs.download_dir(
    src_obs_dir="${OBS_BUCKET}",
    dst_local_dir="${LOCAL_PATH}"
)
EOF

# Method C: Mount OBS as filesystem (if configured in Notebook)
# The OBS bucket may already be mounted at /home/ma-user/obs-mount/
cp -r /home/ma-user/obs-mount/models/Qwen3-8B ${LOCAL_PATH}
```

> **⚠️ Unverified — Confirm OBS integration method in your environment**
> OBS access methods may vary based on your ModelArts configuration.

---

## 7. Run Inference (Task F)

### Method 1: Transformers Script (Recommended Baseline)

Create `/home/ma-user/work/scripts/qwen3_inference.py`:

```python
#!/usr/bin/env python3
"""
Qwen3-8B Inference Script for Ascend 910B NPU
==============================================

Usage:
    python qwen3_inference.py --model_path /home/ma-user/work/models/Qwen/Qwen3-8B
"""

import os
import sys
import argparse
import time
from typing import Optional, List, Dict, Any

# Initialize CANN Environment
def init_ascend_env():
    """Initialize Ascend CANN environment."""
    cann_env_script = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    if os.path.exists(cann_env_script):
        import subprocess
        result = subprocess.run(
            f"source {cann_env_script} && env",
            shell=True, capture_output=True, text=True, executable='/bin/bash'
        )
        for line in result.stdout.split('\n'):
            if '=' in line:
                key, _, value = line.partition('=')
                os.environ[key] = value
        print(f"[INFO] CANN environment initialized")

init_ascend_env()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def validate_environment() -> Dict[str, Any]:
    """Validate the runtime environment."""
    env_info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "npu_available": False,
        "npu_count": 0,
        "cann_version": "Unknown"
    }
    
    try:
        env_info["npu_available"] = torch.npu.is_available()
        if env_info["npu_available"]:
            env_info["npu_count"] = torch.npu.device_count()
    except Exception as e:
        print(f"[WARN] NPU check failed: {e}")
    
    cann_version_file = "/usr/local/Ascend/ascend-toolkit/latest/version.info"
    if os.path.exists(cann_version_file):
        with open(cann_version_file, 'r') as f:
            env_info["cann_version"] = f.read().strip()
    
    return env_info


def load_model(model_path: str, device: str = "npu:0"):
    """Load Qwen3-8B model and tokenizer."""
    print(f"[INFO] Loading model from: {model_path}")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    
    print(f"[INFO] Model loaded in {time.time() - start_time:.2f}s")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, enable_thinking: bool = False,
                     max_new_tokens: int = 2048, temperature: float = 0.7):
    """Generate response from Qwen3-8B."""
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    
    model_inputs = tokenizer([text], return_tensors="pt")
    device = next(model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.8, top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    input_length = model_inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][input_length:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    gen_time = time.time() - start_time
    print(f"[INFO] Generated {len(output_ids)} tokens in {gen_time:.2f}s")
    
    return response


def interactive_chat(model, tokenizer):
    """Run interactive chat session."""
    print("\n" + "=" * 60)
    print("QWEN3-8B INTERACTIVE CHAT")
    print("Commands: /think, /nothink, /quit")
    print("=" * 60 + "\n")
    
    enable_thinking = False
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == "/quit":
                break
            elif user_input.lower() == "/think":
                enable_thinking = True
                print("[System] Thinking mode enabled")
                continue
            elif user_input.lower() == "/nothink":
                enable_thinking = False
                print("[System] Thinking mode disabled")
                continue
            
            response = generate_response(
                model, tokenizer, user_input,
                enable_thinking=enable_thinking,
                temperature=0.6 if enable_thinking else 0.7
            )
            print(f"\nQwen3: {response}\n")
            
        except KeyboardInterrupt:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, 
                       default="/home/ma-user/work/models/Qwen/Qwen3-8B")
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=2048)
    
    args = parser.parse_args()
    
    # Validate environment
    env_info = validate_environment()
    print(f"PyTorch: {env_info['torch_version']}")
    print(f"NPU Available: {env_info['npu_available']}")
    print(f"CANN: {env_info['cann_version']}")
    
    if not env_info["npu_available"]:
        print("[ERROR] NPU not available!")
        sys.exit(1)
    
    model, tokenizer = load_model(args.model_path, args.device)
    
    if args.prompt:
        response = generate_response(
            model, tokenizer, args.prompt,
            enable_thinking=args.thinking, max_new_tokens=args.max_tokens
        )
        print(f"\nResponse:\n{response}")
    else:
        interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
```

**Run:**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python /home/ma-user/work/scripts/qwen3_inference.py \
    --model_path /home/ma-user/work/models/Qwen/Qwen3-8B
```

### Method 2: FastAPI Service

Create `/home/ma-user/work/scripts/qwen3_api_server.py`:

```python
#!/usr/bin/env python3
"""Qwen3-8B FastAPI Server for Ascend 910B"""

import os
import time
from typing import List, Optional
from contextlib import asynccontextmanager

# Initialize CANN
cann_script = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
if os.path.exists(cann_script):
    import subprocess
    result = subprocess.run(f"source {cann_script} && env", 
                          shell=True, capture_output=True, text=True, executable='/bin/bash')
    for line in result.stdout.split('\n'):
        if '=' in line:
            k, _, v = line.partition('=')
            os.environ[k] = v

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "/home/ma-user/work/models/Qwen/Qwen3-8B")
model, tokenizer = None, None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "qwen3-8b"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    enable_thinking: Optional[bool] = False

class ChatResponse(BaseModel):
    id: str
    choices: List[dict]
    usage: dict

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="npu:0", trust_remote_code=True
    )
    model.eval()
    print("Model loaded!")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy", "npu_available": torch.npu.is_available()}

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=request.enable_thinking
    )
    
    inputs = tokenizer([text], return_tensors="pt")
    inputs = {k: v.to("npu:0") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=request.max_tokens,
            temperature=request.temperature, top_p=0.8, do_sample=True
        )
    
    response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}}],
        "usage": {"prompt_tokens": inputs["input_ids"].shape[1], "completion_tokens": len(outputs[0])}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run:**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export MODEL_PATH="/home/ma-user/work/models/Qwen/Qwen3-8B"
uvicorn qwen3_api_server:app --host 0.0.0.0 --port 8000
```

---

## 8. Validation Checklist

| Check | Command | Expected |
|-------|---------|----------|
| NPU Driver | `npu-smi info` | Shows device info |
| CANN Version | `cat /usr/local/Ascend/ascend-toolkit/latest/version.info` | `8.3.RC2` |
| PyTorch | `python -c "import torch; print(torch.__version__)"` | `2.6.0` |
| NPU Available | `python -c "import torch; print(torch.npu.is_available())"` | `True` |
| transformers | `pip show transformers` | `>=4.51.0` |

---

## 9. Troubleshooting (Task G)

### 9.1 torch_npu Import Failures

```bash
pip uninstall torch_npu torch -y
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.6.0.post3
```

### 9.2 CANN/Driver Mismatch

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
cat /usr/local/Ascend/ascend-toolkit/latest/version.info
```

### 9.3 OOM Errors

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# Reduce max_new_tokens in generation config
```

### 9.4 Slow Generation

- Verify model is on `npu:0` not `cpu`
- Use `torch.bfloat16` for best NPU performance
- Always use `with torch.no_grad():`

---

## 10. Appendix

### Reference Links

| Resource | URL |
|----------|-----|
| ModelArts Docs | https://support.huaweicloud.com/intl/en-us/modelarts/ |
| Ascend PyTorch | https://github.com/Ascend/pytorch |
| Qwen3 Docs | https://qwen.readthedocs.io/ |
| vLLM-Ascend | https://docs.vllm.ai/projects/ascend/ |

### Disclaimer

> This runbook is based on publicly available documentation as of January 2026. Console UI may differ. Test in non-production first. Review Qwen3 license (Apache 2.0) before commercial use.

---

**End of Runbook**
