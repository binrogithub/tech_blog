---
author: Robin
pubDatetime: 2026-06-22T09:15:00-03:00
title: "Deploying Qwen3.6-27B-W8A8 on Huawei Ascend 910B with vLLM Ascend"
description: "A field-tested deployment guide for Qwen3.6-27B-W8A8 on Huawei Ascend 910B/A2 with vLLM Ascend 0.18.0rc1, covering the working Python 3.11 stack, Triton Ascend wheels, custom kernels, eager serving, and failure analysis."
tags:
  - huawei-cloud
  - ascend
  - modelarts
  - vllm
  - qwen
  - deployment
  - ai-infrastructure
  - inference
  - quantization
featured: true
draft: false
---

# Deploying Qwen3.6-27B-W8A8 on Ascend 910B3/A2 with vLLM Ascend

This is a deployment write-up from a real ModelArts Ascend 910B3 session. The original goal looked simple: serve `Qwen3.6-27B-w8a8` with vLLM Ascend. The actual lesson was stricter: Qwen3.6-27B is a Qwen3-Next-style hybrid model, and the deployment only becomes repeatable when the Python ABI, Triton Ascend wheel, torch-npu wheel, vLLM Ascend native extension, and serving mode all line up.

The final working path was **not** the Python 3.12 venv path. Python 3.12 was useful for debugging, and it even built some pieces later, but it was not the clean reproducible route for this model. The reproducible route is:

- Python 3.11 environment.
- `vllm-ascend 0.18.0rc1`.
- `vllm 0.18.0`.
- `torch 2.9.0`.
- `torch-npu 2.9.0.post1+gitee7ba04`.
- `triton-ascend 3.2.0.dev20260322`.
- Full `vllm_ascend_C` custom kernels built and importable.
- `--enforce-eager` for Qwen3.6 on this stack.
- Text-only serving by disabling image/video prompt limits unless you intentionally need multimodal profiling.

The validated target machine had one visible Ascend `910B3` card, 64GB HBM, and CANN 8.5.2. The model ran as a one-card constrained deployment at 4K, 8K, and 16K context in eager mode. Official docs recommend a larger A2/A3 node and TP=2 for the full long-context W8A8 recipe; the one-card deployment is a pragmatic constrained setup, not the official performance configuration.

## What Worked

- Model: `/home/ma-user/work/Qwen3.6-27B-w8a8`.
- Persistent Python env: `/home/ma-user/work/venvs/vllm-ascend-py311`.
- Source tree for successful native build: `/home/ma-user/work/vllm-ascend-src-py311`.
- Compiler: GCC/G++ 12.4.0 under `/home/ma-user/work/conda-gcc`.
- Working mode: eager.
- Validated contexts: 4K, 8K, 16K.
- Validated APIs: `/v1/completions` and `/v1/chat/completions`.

## What Did Not Work

- Treating Python 3.12 as the primary route.
- Installing `triton-ascend 3.2.1` and expecting Qwen3-Next kernels to work.
- Running without `vllm_ascend_C`.
- Graph/compile mode (`PIECEWISE` or `FULL_DECODE_ONLY`) on this exact stack.
- CPU KV offload with `OffloadingConnector`.

## External References

- vLLM Ascend Qwen3.5/Qwen3.6 tutorial: https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html
- vLLM Ascend supported model matrix: https://docs.vllm.ai/projects/ascend/en/v0.18.0/user_guide/support_matrix/supported_models.html
- Qwen3.6-27B model announcement: https://qwen.ai/blog?id=qwen3.6-27b
- BF16 model on Hugging Face: https://huggingface.co/Qwen/Qwen3.6-27B
- BF16 model on ModelScope: https://modelscope.cn/models/Qwen/Qwen3.6-27B
- W8A8 model on ModelScope: https://modelscope.cn/models/Eco-Tech/Qwen3.6-27B-w8a8

## Version Matrix

This table is the heart of the deployment. Most failed attempts came from being close but not exact.

| Component | Working value | Why it matters |
|---|---:|---|
| Python | `3.11` | The critical Triton Ascend wheel was available as cp310/cp311, not cp312. |
| CANN on tested host | `8.5.2` | The host had 8.5.2. Official image lineage referenced 8.5.1/8.5.x; the tested source route worked on the host after native kernels were built. |
| vLLM Ascend release | `0.18.0rc1` | Qwen3.6-27B is first supported in this release line in the vLLM Ascend docs. |
| vLLM | `0.18.0` / `0.18.0+empty` | Installed from the pinned vLLM commit with `VLLM_TARGET_DEVICE=empty`. |
| vLLM commit | `bcf2be96120005e9aea171927f85055a6a5c0cf6` | Modelers/source route pin. |
| vLLM Ascend commit | `99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f` | Modelers/source route pin. |
| Transformers commit | `fc9137225880a9d03f130634c20f9dbe36a7b8bf` | Modelers/source route pin; runtime showed Transformers `4.57.x`. |
| torch | `2.9.0+cpu` | Normal for this Ascend Python stack; NPU is provided by torch-npu. |
| torch-npu | `2.9.0.post1+gitee7ba04` | The cp311 wheel used in the working env. Plain `2.9.0` was not enough to close the final gap. |
| triton-ascend | `3.2.0.dev20260322` | Required because Qwen3-Next/GDN path needs `triton.language.extract_slice`; `3.2.1` was wrong for this deployment. |
| vLLM Ascend native extension | `vllm_ascend_C.cpython-311-aarch64-linux-gnu.so` | Must exist. It provides custom ops such as `npu_gemma_rms_norm`. |
| Compiler | GCC/G++ `12.4.0` | System GCC 7.3 cannot compile PyTorch/Triton C++17 pieces. |

The important correction: **Python 3.12 was not the successful deployment route.** In the logs, Python 3.12 initially failed because the available `triton-ascend 3.2.1` did not provide the needed Qwen3-Next behavior. The correct `triton-ascend 3.2.0.dev20260322` wheel was available for cp311, and the official image family is Python 3.11 oriented.

The working import check looked like this:

```text
python 3.11.x
torch 2.9.0+cpu
torch_npu 2.9.0.post1+gitee7ba04
triton extract_slice: True
vllm 0.18.0
vllm_ascend_C: vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
npu_gemma_rms_norm: True
```

If `triton extract_slice` is false, this deployment is not healthy. If `vllm_ascend_C` is missing, this deployment is not healthy. If `npu_gemma_rms_norm` is missing, Qwen3.6 will fail during model forward/profile.

## Model Download

There are two Qwen3.6-27B model families that are easy to confuse.

| Model | Purpose |
|---|---|
| `Qwen/Qwen3.6-27B` | BF16 base model. Larger memory footprint. |
| `Eco-Tech/Qwen3.6-27B-w8a8` | Ascend W8A8 quantized model used in this deployment. |

For the W8A8 deployment, download the W8A8 model:

```bash
mkdir -p /home/ma-user/work
python -m pip install -U modelscope

modelscope download \
  --model Eco-Tech/Qwen3.6-27B-w8a8 \
  --local_dir /home/ma-user/work/Qwen3.6-27B-w8a8
```

Use persistent storage. On ModelArts, `/root` and parts of the base conda tree may be ephemeral; `/home/ma-user/work` is the safe storage mount.

Progress check:

```bash
watch -n 5 'du -sh /home/ma-user/work/Qwen3.6-27B-w8a8; find /home/ma-user/work/Qwen3.6-27B-w8a8 -type f | wc -l'
```

Expected size from the tested download:

```text
34G    /home/ma-user/work/Qwen3.6-27B-w8a8
```

If you started a Hugging Face download first and then switch to ModelScope, stop the downloader and delete the partial download/cache. Mixed partial weights create confusing model-load failures.

## Preferred Route: Official vLLM Ascend Image

The official route should be preferred whenever possible. The source/venv route is useful when Docker is unavailable or when you need to patch vLLM Ascend, but most production teams should start with the image.

The vLLM Ascend documentation says Qwen3.6-27B is first supported in `vllm-ascend:v0.18.0rc1`. The docs also show that Qwen3.6-27B-w8a8 uses `--quantization ascend` and a TP=2 serve recipe for long context.

Example shape:

```bash
export IMAGE=quay.io/ascend/vllm-ascend:v0.18.0rc1
export NAME=vllm-ascend-qwen36

docker run --rm -it \
  --name "${NAME}" \
  --net=host \
  --shm-size=100g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home/ma-user/work:/home/ma-user/work \
  "${IMAGE}" bash
```

Adjust the image tag to the exact A2/A3 tag available in your registry. Adjust `/dev/davinci*` devices to the cards assigned to your job.

The official long-context W8A8 command from the docs is conceptually:

```bash
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve Eco-Tech/Qwen3.6-27B-w8a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --seed 1024 \
  --quantization ascend \
  --served-model-name qwen3.6 \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true}' \
  --async-scheduling
```

On the tested one-card 910B3 environment, this full official long-context TP=2 target could not be reproduced directly because only one card was visible. The stable one-card route used reduced context and eager mode, described below.

## Source Route That Actually Worked: Python 3.11

Use this when you cannot use the official image or need to reproduce the notebook environment. Do not start from Python 3.12 for this model.

### Persistent Layout

```bash
mkdir -p /home/ma-user/work/venvs
mkdir -p /home/ma-user/work/src
mkdir -p /home/ma-user/work/qwen36_run_record
```

Recommended final paths:

```text
/home/ma-user/work/venvs/vllm-ascend-py311
/home/ma-user/work/vllm-ascend-src-py311
/home/ma-user/work/conda-gcc
/home/ma-user/work/Qwen3.6-27B-w8a8
```

### Create Python 3.11 Environment

Use conda or micromamba, but place the env under `/home/ma-user/work`:

```bash
source /home/ma-user/anaconda3/bin/activate
conda create -p /home/ma-user/work/venvs/vllm-ascend-py311 python=3.11 -y
conda activate /home/ma-user/work/venvs/vllm-ascend-py311
python -m pip install -U pip setuptools wheel
```

### Install Modern GCC

System GCC 7.3 is too old. The working deployment used GCC/G++ 12.4:

```bash
export PATH=/home/ma-user/work/conda-gcc/bin:${PATH}
export CC=/home/ma-user/work/conda-gcc/bin/aarch64-conda-linux-gnu-gcc
export CXX=/home/ma-user/work/conda-gcc/bin/aarch64-conda-linux-gnu-g++
```

Keep these exports both for building and for serving, because Triton runtime JIT can compile C++ headers at serve time.

### Install the Critical Wheels

The key wheel pair was:

```text
torch_npu-2.9.0.post1+gitee7ba04-cp311-cp311-manylinux_2_28_aarch64.whl
triton_ascend-3.2.0.dev20260322-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
```

In the tested environment, the ModelArts machine could not reliably fetch the OBS bucket directly. The working approach was to fetch these wheels on a machine that could reach the bucket, validate them as wheels, then copy them to `/home/ma-user/work/`.

Install them into the py311 environment:

```bash
conda activate /home/ma-user/work/venvs/vllm-ascend-py311
python -m pip install torch==2.9.0
python -m pip install /home/ma-user/work/torch_npu-2.9.0.post1+gitee7ba04-cp311-cp311-manylinux_2_28_aarch64.whl
python -m pip install /home/ma-user/work/triton_ascend-3.2.0.dev20260322-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
```

Then install the runtime dependencies required by vLLM/vLLM Ascend. The exact lock can vary, but the tested environment needed at least:

```bash
python -m pip install \
  numpy \
  pybind11 \
  transformers \
  xgrammar \
  compressed-tensors \
  numba \
  quart \
  einops \
  fastapi \
  msgpack \
  scipy \
  regex \
  arctic-inference \
  torch-c-dlpack-ext
```

### Install vLLM

```bash
cd /home/ma-user/work/src
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout bcf2be96120005e9aea171927f85055a6a5c0cf6
VLLM_TARGET_DEVICE=empty python -m pip install -v -e .
```

### Install Transformers Pin

```bash
cd /home/ma-user/work/src
git clone https://github.com/huggingface/transformers.git
cd transformers
git reset --hard fc9137225880a9d03f130634c20f9dbe36a7b8bf
python -m pip install -e .
```

### Build vLLM Ascend with Custom Kernels

This is the non-negotiable step. Qwen3.6 needs custom Ascend ops. A `COMPILE_CUSTOM_KERNELS=0` install is not enough.

```bash
cd /home/ma-user/work
git clone https://github.com/vllm-project/vllm-ascend.git vllm-ascend-src-py311
cd /home/ma-user/work/vllm-ascend-src-py311
git checkout 99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f
git submodule update --init --recursive

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/home/ma-user/work/conda-gcc/bin:${PATH}
export CC=/home/ma-user/work/conda-gcc/bin/aarch64-conda-linux-gnu-gcc
export CXX=/home/ma-user/work/conda-gcc/bin/aarch64-conda-linux-gnu-g++
export C_COMPILER=${CC}
export CXX_COMPILER=${CXX}
export COMPILE_CUSTOM_KERNELS=1

python -m pip install -v --no-build-isolation -e .
```

Successful build markers include:

```text
[100%] Linking CXX shared module vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
Successfully built vllm_ascend
Successfully installed vllm_ascend-0.18.0rc1
```

### Verify the Stack

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
conda activate /home/ma-user/work/venvs/vllm-ascend-py311

python - <<'PY'
import torch
import torch_npu
import triton.language as tl
import vllm
import vllm_ascend.vllm_ascend_C as C

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("triton extract_slice:", hasattr(tl, "extract_slice"))
print("vllm", vllm.__version__)
print("vllm_ascend_C:", C.__file__.split("/")[-1])
print("npu_gemma_rms_norm:", hasattr(torch.ops._C_ascend, "npu_gemma_rms_norm"))
PY
```

Do not serve until this check passes.

## Serving Configuration

The constrained one-card deployment used text-only eager mode. It intentionally avoided graph/compile mode and disabled image/video prompt profiling.

### Common Environment

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /home/ma-user/anaconda3/bin/activate /home/ma-user/work/venvs/vllm-ascend-py311

export PATH=/home/ma-user/work/conda-gcc/bin:${PATH}
export CC=/home/ma-user/work/conda-gcc/bin/aarch64-conda-linux-gnu-gcc
export CXX=/home/ma-user/work/conda-gcc/bin/aarch64-conda-linux-gnu-g++

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
```

### 4K Smoke Test

```bash
vllm serve /home/ma-user/work/Qwen3.6-27B-w8a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --quantization ascend \
  --served-model-name qwen3.6 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.80 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt image=0,video=0 \
  --enforce-eager \
  --additional-config '{"enable_cpu_binding":true}'
```

Validated markers:

```text
Loading weights took ~151s
GPU KV cache size: 50,688 tokens
Maximum concurrency for 4,096 tokens per request: 22.50x
Application startup complete
```

### 8K Stable Test

```bash
vllm serve /home/ma-user/work/Qwen3.6-27B-w8a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --quantization ascend \
  --served-model-name qwen3.6 \
  --max-model-len 8192 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.80 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt image=0,video=0 \
  --enforce-eager \
  --additional-config '{"enable_cpu_binding":true}'
```

Validated markers:

```text
GPU KV cache size: 62,976 tokens
Maximum concurrency for 8,192 tokens per request: 18.44x
Application startup complete
```

### 16K Stable Test

```bash
vllm serve /home/ma-user/work/Qwen3.6-27B-w8a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --quantization ascend \
  --served-model-name qwen3.6 \
  --max-model-len 16384 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.82 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt image=0,video=0 \
  --enforce-eager \
  --additional-config '{"enable_cpu_binding":true}'
```

Validated markers:

```text
GPU KV cache size: 67,584 tokens
Maximum concurrency for 16,384 tokens per request: 12.64x
Application startup complete
```

Validated request:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6",
    "messages": [{"role": "user", "content": "Say hello in three languages."}],
    "max_tokens": 60,
    "temperature": 0
  }'
```

### Restart Hygiene

Before restarting, kill old server processes and wait for HBM to fall back to the baseline. In the tested environment the idle HBM baseline was about 3382MB.

```bash
pgrep -af "bin/vllm serve"
pkill -f "bin/vllm serve" || true
sleep 5
npu-smi info
```

If you relaunch too quickly, vLLM can fail with:

```text
ValueError: Free memory on device ... is less than desired GPU memory utilization
```

That is often stale memory/process cleanup, not a model sizing error.

## Failure Analysis

### Python 3.12 Was the Wrong Primary Route

Python 3.12 looked attractive because the base environment already had it. The mistake was assuming package names were enough. The model needs a specific Triton Ascend capability:

```text
triton.language.extract_slice
```

The working wheel was:

```text
triton_ascend-3.2.0.dev20260322-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
```

The accessible Python 3.12 path initially installed `triton-ascend 3.2.1`, but that was not the right functional match for Qwen3-Next. Later py312 native pieces could build, but the reliable deployment route in the run record remained py311 because the required wheel set matched py311.

### Missing `vllm_ascend_C`

This was the most important failure:

```text
ModuleNotFoundError: No module named 'vllm_ascend.vllm_ascend_C'
```

and later:

```text
torch.ops._C_ascend.npu_gemma_rms_norm
```

`npu_gemma_rms_norm` is not optional for this model path. It lives behind the vLLM Ascend native extension/custom kernels. Installing with `COMPILE_CUSTOM_KERNELS=0` can import Python packages, but Qwen3.6 fails during the first real forward/profile path.

The fix is to build vLLM Ascend with `COMPILE_CUSTOM_KERNELS=1` and verify:

```text
vllm_ascend_C: vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
npu_gemma_rms_norm: True
```

### GCC 7.3 Broke Triton Runtime JIT

Symptom:

```text
Failed to compile ... precompiled.h.gch
You're trying to build PyTorch with a too old version of GCC. We need GCC 9 or later.
cmd=['/usr/bin/g++', ...]
```

Fix:

```bash
export PATH=/home/ma-user/work/conda-gcc/bin:${PATH}
export CC=/home/ma-user/work/conda-gcc/bin/aarch64-conda-linux-gnu-gcc
export CXX=/home/ma-user/work/conda-gcc/bin/aarch64-conda-linux-gnu-g++
```

Do this at both build time and serve time.

### Multimodal Profiling Crashed Without Text-Only Limits

Qwen3.6 is multimodal. During startup, vLLM can profile the visual path using dummy image inputs. On the tested environment, this reached a device-side failure around `aclnnIndex`.

For text-only service, disable multimodal prompt profiling:

```bash
--limit-mm-per-prompt image=0,video=0
```

This is not a cosmetic flag. It changed the startup path.

### Graph Mode Was Not Stable on This Stack

The official long-context recipe recommends `FULL_DECODE_ONLY`. In this tested stack, graph mode could capture but failed on decode or hit Qwen3-Next torch.compile issues:

```text
qwen3_next.py:1358 -> torch._dynamo call_size
AttributeError: 'NoneType' has no attribute 'size'
```

Observed result:

| Mode | Result |
|---|---|
| 4K eager | Passed |
| 8K eager | Passed |
| 16K eager | Passed |
| FULL_DECODE_ONLY graph | Startup/capture could pass, first real request failed |
| PIECEWISE graph/default compile | Failed earlier |

Practical rule for this exact build:

```text
Use --enforce-eager.
```

### CPU KV Offload Was Architecturally Incompatible

The tested KV offload attempt used `OffloadingConnector` and failed for a model-architecture reason, not just memory tuning:

```text
ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type.
```

Qwen3.6 has a hybrid attention/cache design. `OffloadingConnector` disabled the hybrid KV cache manager, then vLLM tried to unify incompatible cache specs. This is not fixed by adding a little memory or changing `gpu_memory_utilization`.

Practical rule:

```text
Do not use CPU KV offload for this Qwen3.6 stack unless the connector explicitly supports the required hybrid cache path.
```

## Fast Redeploy Checklist

Use this order. Do not skip verification steps.

### 1. Confirm Hardware and Persistent Storage

```bash
npu-smi info
df -h /home/ma-user/work
```

Expected one-card constrained test environment:

```text
910B3, 64GB HBM
/home/ma-user/work on persistent /dev/sdb
```

### 2. Confirm Model

```bash
du -sh /home/ma-user/work/Qwen3.6-27B-w8a8
```

Expected:

```text
~34G
```

### 3. Activate Correct Python Environment

```bash
source /home/ma-user/anaconda3/bin/activate /home/ma-user/work/venvs/vllm-ascend-py311
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 4. Export Compiler Variables

```bash
export PATH=/home/ma-user/work/conda-gcc/bin:${PATH}
export CC=/home/ma-user/work/conda-gcc/bin/aarch64-conda-linux-gnu-gcc
export CXX=/home/ma-user/work/conda-gcc/bin/aarch64-conda-linux-gnu-g++
```

### 5. Run the Critical Import Check

```bash
python - <<'PY'
import torch, torch_npu
import triton.language as tl
import vllm
import vllm_ascend.vllm_ascend_C as C
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("triton extract_slice:", hasattr(tl, "extract_slice"))
print("vllm", vllm.__version__)
print("vllm_ascend_C:", C.__file__.split("/")[-1])
print("npu_gemma_rms_norm:", hasattr(torch.ops._C_ascend, "npu_gemma_rms_norm"))
PY
```

Do not continue if this fails.

### 6. Start 4K First

Use 4K as the fastest real startup test. It still loads the full model, so weight load takes time, but it limits KV/cache pressure.

### 7. Scale to 8K, Then 16K

The tested stable sequence was:

```text
4K eager -> 8K eager -> 16K eager
```

Do not jump directly to graph, KV offload, or high concurrency.

### 8. Validate End-to-End

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6","prompt":"The capital of France is","max_tokens":24,"temperature":0}'
```

and:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"Say hello in three languages."}],"max_tokens":60,"temperature":0}'
```

### 9. Record Runtime Markers

Save these from the log:

```text
Loading weights took ...
GPU KV cache size ...
Maximum concurrency ...
Application startup complete
```

Those markers are more useful than a generic "server started" note.

## Conclusion

The deployment became stable only after the stack was treated as a tightly coupled system, not as a set of independent Python packages.

The decisive fixes were:

- Move to Python 3.11.
- Use `triton-ascend 3.2.0.dev20260322`, not `3.2.1`.
- Use the matching `torch-npu 2.9.0.post1+gitee7ba04` wheel.
- Build `vllm_ascend_C` with custom kernels enabled.
- Use GCC/G++ 12.4 at build time and serve time.
- Disable multimodal prompt profiling for text-only serving.
- Use eager mode for this Qwen3.6 stack.

The wrong mental model was "Python 3.12 mostly works, so just patch missing dependencies." That got the service to model loading, but not to a reliable endpoint. The right mental model is "Qwen3.6 depends on Qwen3-Next/GDN-specific Triton and Ascend custom kernels; match the ABI and native extension first, then tune serving."

If deploying fresh and Docker is allowed, start from the official vLLM Ascend image for `0.18.0rc1`. If Docker is not available, reproduce the Python 3.11 source route exactly and verify the critical import checks before launching vLLM.

