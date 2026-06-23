---
author: Robin
pubDatetime: 2026-05-31T09:15:00-03:00
modDatetime: 2026-05-31T10:35:00-03:00
title: "Deploying Qwen3.6-27B-W8A8 on Huawei Ascend 910B with vLLM Ascend"
description: "A field-tested deployment guide for Qwen3.6-27B-W8A8 on Huawei Ascend 910B/A2 with vLLM Ascend v0.19.1rc1 and vLLM v0.19.1, covering CANN 8.5.2, Python 3.11, torch-npu 2.9, Triton Ascend, native kernels, ATB, graph mode, and the failure fixes needed to make deployment repeatable."
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

This is a field deployment note from a real Huawei ModelArts Ascend 910B3 session. The target was `Qwen3.6-27B-w8a8`, an Ascend W8A8 quantized Qwen3.6 model, served through vLLM Ascend.

The first working environment used vLLM Ascend `0.18.0rc1` in eager mode. That was useful as a safety baseline, but it was not the final answer: graph mode still failed. The final validated route upgraded the stack to **vLLM Ascend `v0.19.1rc1` with vLLM `v0.19.1`**, kept the host CANN at **8.5.2**, loaded the ATB runtime, and fixed both build-time and runtime compiler selection.

The validated one-card smoke test used:

- Ascend `910B3`, 64 GB HBM, visible as one NPU.
- CANN `8.5.2`, already present in the container.
- Python `3.11` under persistent storage.
- vLLM `0.19.1+empty`, built from `v0.19.1`.
- vLLM Ascend source checked out at `v0.19.1rc1`.
- torch `2.9.0`.
- torch-npu `2.9.0.post1+gitee7ba04`.
- triton-ascend `3.2.0.dev20260322`.
- `Qwen3.6-27B-w8a8` from ModelScope.
- `--quantization ascend`.
- `FULL_DECODE_ONLY` graph mode.

The final graph-mode service successfully returned both `/v1/models` and `/v1/chat/completions` responses.

## Final State

The final environment paths were:

| Item | Path |
|---|---|
| Model | `/home/ma-user/work/Qwen3.6-27B-w8a8` |
| Python env | `/home/ma-user/work/venvs/vllm-ascend-0191-py311` |
| vLLM source | `/home/ma-user/work/src/vllm-0191` |
| vLLM Ascend source | `/home/ma-user/work/src/vllm-ascend` |
| Logs | `/home/ma-user/work/logs` |
| Start scripts | `/home/ma-user/work/scripts` |

The final package check was:

```text
vllm: 0.19.1+empty
vllm-ascend: 0.19.1rc2.dev0+gda421afad.d20260622
torch: 2.9.0
torch-npu: 2.9.0.post1+gitee7ba04
triton-ascend: 3.2.0.dev20260322
numpy: 1.26.4
setuptools: 80.9.0
```

The source refs were:

```text
vLLM source:        v0.19.1
vLLM Ascend source: v0.19.1rc1-dirty
```

The `vllm-ascend` Python metadata reported `0.19.1rc2.dev0+gda421afad.d20260622`, but the checked-out source was the `v0.19.1rc1` tag at commit `da421afad7192dac64e39ae1d32305d57344f3cf`. This is a packaging/version-string detail from the source tree, not a reason to discard the environment. Always record both the Python package metadata and the git ref.

The `-dirty` suffix came from an already dirty `csrc/third_party/catlass` submodule state. It was not reset during the deployment.

## What Changed From the 0.18 Route

The old conclusion was:

- vLLM Ascend `0.18.0rc1`.
- vLLM `0.18.0`.
- eager mode only.
- graph mode failed.

The updated conclusion is:

- vLLM Ascend source at `v0.19.1rc1`.
- vLLM source at `v0.19.1`.
- graph mode works after loading ATB and setting a modern runtime compiler.
- CANN did not need to be upgraded from `8.5.2`.

The most important correction is that CANN `8.5.2` was not the blocker. The blockers were:

1. Build-time C++ compiler selection for the vLLM Ascend extension.
2. Runtime C++ compiler selection for Triton Ascend JIT.
3. Missing ATB library path in graph mode.

## External References

- vLLM Ascend Qwen3.5/Qwen3.6 tutorial: https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html
- vLLM Ascend release notes: https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html
- vLLM Ascend repository: https://github.com/vllm-project/vllm-ascend
- vLLM repository: https://github.com/vllm-project/vllm
- Qwen3.6-27B BF16 on Hugging Face: https://huggingface.co/Qwen/Qwen3.6-27B
- Qwen3.6-27B BF16 on ModelScope: https://modelscope.cn/models/Qwen/Qwen3.6-27B
- Qwen3.6-27B-W8A8 on ModelScope: https://modelscope.cn/models/Eco-Tech/Qwen3.6-27B-w8a8
- Modelers Qwen3.6-27B guide: https://modelers.cn/models/vLLM_Ascend/Qwen3.6-27B

## Version Matrix

This is the matrix that was actually validated.

| Component | Validated value | Notes |
|---|---:|---|
| Python | `3.11` | Keep the deployment on Python 3.11. Do not use Python 3.12 as the main route. |
| CANN | `8.5.2` | Already installed in the container. It did not need to change for this upgrade. |
| torch | `2.9.0` | CPU wheel name is normal in this stack; NPU support comes from torch-npu. |
| torch-npu | `2.9.0.post1+gitee7ba04` | Validated wheel. |
| triton-ascend | `3.2.0.dev20260322` | Validated with Qwen3.6/GDN path. |
| vLLM | `0.19.1+empty` | Built from `v0.19.1` with `VLLM_TARGET_DEVICE=empty`. |
| vLLM Ascend | source tag `v0.19.1rc1` | Package metadata may show a dev version from the same commit. |
| numpy | `1.26.4` | Pinned back from NumPy 2.x to avoid compatibility churn. |
| opencv-python-headless | `4.11.0.86` | vLLM Ascend pins this lower than the newest vLLM dependency request. |
| setuptools | `80.9.0` | Setuptools 82 removed behavior needed by torchair imports using `pkg_resources`. |
| GCC/G++ for Python extension | conda GCC/G++ `15` | Needed because PyTorch extension headers require GCC 9+. |
| GCC/G++ for CANN custom ops | system GCC `7.3` | CANN custom ops built successfully when not forced to use conda GCC globally. |
| GCC/G++ for Triton runtime JIT | conda G++ `15` via `CC`/`CXX` | Needed at serve time. |

There were non-blocking `pip check` warnings:

- `vllm-ascend` wanted `arctic-inference==0.1.1`; that package failed to build with system GCC 7.3 and was not needed for this Qwen serving path.
- `torch-npu` and `triton-ascend` version strings included local build suffixes.
- vLLM and vLLM Ascend disagreed on the preferred OpenCV range. The deployment favored the vLLM Ascend pin.

## Model Download

The W8A8 model used here is:

```text
Eco-Tech/Qwen3.6-27B-w8a8
```

Download it to persistent ModelArts storage:

```bash
mkdir -p /home/ma-user/work
python -m pip install -U modelscope

modelscope download \
  --model Eco-Tech/Qwen3.6-27B-w8a8 \
  --local_dir /home/ma-user/work/Qwen3.6-27B-w8a8
```

Progress check:

```bash
watch -n 5 'du -sh /home/ma-user/work/Qwen3.6-27B-w8a8; find /home/ma-user/work/Qwen3.6-27B-w8a8 -type f | wc -l'
```

Expected size from the tested download:

```text
34G    /home/ma-user/work/Qwen3.6-27B-w8a8
```

If you started a Hugging Face download first and switch to ModelScope later, stop the old process and remove the partial files. Mixing partial HF and ModelScope artifacts makes model-load failures harder to diagnose.

## Build the Python 3.11 Environment

Create a persistent Python 3.11 environment:

```bash
source /home/ma-user/anaconda3/bin/activate
conda create -p /home/ma-user/work/venvs/vllm-ascend-0191-py311 python=3.11 -y
conda activate /home/ma-user/work/venvs/vllm-ascend-0191-py311
python -m pip install -U pip wheel
python -m pip install setuptools==80.9.0
```

Install the base NPU stack:

```bash
python -m pip install torch==2.9.0
python -m pip install torch-npu==2.9.0.post1
python -m pip install triton-ascend==3.2.0.dev20260322
```

In a restricted environment, those wheels may need to be copied in manually. The validated versions were:

```text
torch_npu-2.9.0.post1+gitee7ba04
triton_ascend-3.2.0.dev20260322
```

Install conda compilers into the same environment:

```bash
conda install -p /home/ma-user/work/venvs/vllm-ascend-0191-py311 \
  gcc_linux-aarch64 \
  gxx_linux-aarch64 \
  -y
```

## Build vLLM v0.19.1

Clone and install vLLM:

```bash
mkdir -p /home/ma-user/work/src
cd /home/ma-user/work/src

git clone https://github.com/vllm-project/vllm.git vllm-0191
cd vllm-0191
git checkout v0.19.1

conda activate /home/ma-user/work/venvs/vllm-ascend-0191-py311
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation --no-deps -e .
```

Verify:

```bash
/home/ma-user/work/venvs/vllm-ascend-0191-py311/bin/python - <<'PY'
import vllm
print(vllm.__version__)
PY
```

Expected:

```text
0.19.1+empty
```

## Build vLLM Ascend v0.19.1rc1

Clone and checkout the release candidate:

```bash
cd /home/ma-user/work/src
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout v0.19.1rc1
```

Install vLLM runtime requirements, then correct the pins that matter for this stack:

```bash
conda activate /home/ma-user/work/venvs/vllm-ascend-0191-py311
python -m pip install -r /home/ma-user/work/src/vllm-0191/requirements/common.txt
python -m pip install numpy==1.26.4 opencv-python-headless==4.11.0.86 setuptools==80.9.0
```

The native build needs a split-compiler approach:

- Let CANN custom ops use the system compiler expected by the CANN toolchain.
- Let the Python extension CMake path use the conda GCC/G++ compiler.

Do not globally export `CC` and `CXX` during the build. Instead, pass `C_COMPILER` and `CXX_COMPILER` for the Python extension build:

```bash
cd /home/ma-user/work/src/vllm-ascend
conda activate /home/ma-user/work/venvs/vllm-ascend-0191-py311

export PATH=/home/ma-user/work/venvs/vllm-ascend-0191-py311/bin:$PATH
export C_COMPILER=/home/ma-user/work/venvs/vllm-ascend-0191-py311/bin/aarch64-conda-linux-gnu-gcc
export CXX_COMPILER=/home/ma-user/work/venvs/vllm-ascend-0191-py311/bin/aarch64-conda-linux-gnu-g++

python -m pip install --no-build-isolation --no-deps -v .
```

This avoids the failure:

```text
#error "You're trying to build PyTorch with a too old version of GCC. We need GCC 9 or later."
```

while still letting the CANN custom-op build complete with the system toolchain.

Verify the native extension:

```bash
/home/ma-user/work/venvs/vllm-ascend-0191-py311/bin/python - <<'PY'
import importlib.metadata as m
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C as C

print("vllm:", m.version("vllm"))
print("vllm-ascend:", m.version("vllm-ascend"))
print("torch:", m.version("torch"))
print("torch-npu:", m.version("torch-npu"))
print("npu available:", torch.npu.is_available())
print("vllm_ascend_C:", C)
PY
```

## Required Runtime Environment

Runtime is where the second major compiler problem appears. Triton Ascend reads `CC` for its C++ compiler. If `CC` resolves to `/usr/bin/g++`, the first serve can fail with the same GCC-too-old error.

Set `CC` and `CXX` to conda G++ before serving:

```bash
export ENV_DIR=/home/ma-user/work/venvs/vllm-ascend-0191-py311
export CC="$ENV_DIR/bin/aarch64-conda-linux-gnu-g++"
export CXX="$ENV_DIR/bin/aarch64-conda-linux-gnu-g++"
```

Graph mode also needs ATB. Without ATB, graph mode failed with:

```text
OSError: libatb.so: cannot open shared object file: No such file or directory
```

The fix was:

```bash
source /usr/local/Ascend/nnal/atb/set_env.sh
```

The full graph runtime environment was:

```bash
ENV_DIR=/home/ma-user/work/venvs/vllm-ascend-0191-py311
ASCEND_SRC=/home/ma-user/work/src/vllm-ascend

export PATH="$ENV_DIR/bin:$PATH"
source "$ASCEND_SRC/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash"
source /usr/local/Ascend/nnal/atb/set_env.sh

export CC="$ENV_DIR/bin/aarch64-conda-linux-gnu-g++"
export CXX="$ENV_DIR/bin/aarch64-conda-linux-gnu-g++"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
```

## Graph Mode Serve Command

This is the validated one-card smoke command. It is intentionally conservative: 4K context, one sequence, and small graph capture sizes.

```bash
ENV_DIR=/home/ma-user/work/venvs/vllm-ascend-0191-py311
MODEL_DIR=/home/ma-user/work/Qwen3.6-27B-w8a8

"$ENV_DIR/bin/vllm" serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 8010 \
  --served-model-name qwen3.6-w8a8-graph \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --quantization ascend \
  --trust-remote-code \
  --gpu-memory-utilization 0.82 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4],"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

The startup log showed:

```text
FULL_DECODE_ONLY compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
Using OOT custom backend for compilation.
enable_npugraph_ex is enabled, which will bring graph compilation optimization.
Compiling a graph for compile range (1, 2048) takes 47.28 s
torch.compile and initial profiling/warmup run together took 108.25 s in total
Graph capturing finished in 22 secs, took 0.22 GiB
Starting vLLM server on http://0.0.0.0:8010
```

The model-load and memory logs were:

```text
Loading safetensors checkpoint shards: 100% Completed | 9/9
Loading weights took 155.68 seconds
Loading model weights took 33.6045 GB
Available KV cache memory: 13.50 GiB
GPU KV cache size: 53,760 tokens
Maximum concurrency for 4,096 tokens per request: 23.67x
```

Even though the log says "GPU", this is the shared vLLM wording for the accelerator memory path; the device was Ascend NPU.

## API Verification

Check the model list:

```bash
curl http://127.0.0.1:8010/v1/models
```

Validated response shape:

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3.6-w8a8-graph",
      "object": "model",
      "owned_by": "vllm",
      "root": "/home/ma-user/work/Qwen3.6-27B-w8a8",
      "max_model_len": 4096
    }
  ]
}
```

Check chat completions:

```bash
curl -sS --max-time 180 http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-w8a8-graph",
    "messages": [{"role": "user", "content": "Say OK only."}],
    "max_tokens": 8,
    "temperature": 0
  }'
```

Validated response status was `200 OK`, with usage:

```json
{
  "prompt_tokens": 14,
  "completion_tokens": 8,
  "total_tokens": 22
}
```

The short prompt returned odd text because the request capped generation at eight tokens, but the important deployment signal was that graph-mode inference completed through the OpenAI-compatible API.

## Reusable Scripts

The final machine had these scripts:

```text
/home/ma-user/work/scripts/start-vllm-qwen36-w8a8-0191-graph.sh
/home/ma-user/work/scripts/start-vllm-qwen36-w8a8-0191-eager.sh
/home/ma-user/work/scripts/stop-vllm-qwen36.sh
```

Start graph mode:

```bash
/home/ma-user/work/scripts/start-vllm-qwen36-w8a8-0191-graph.sh
```

Check logs:

```bash
tail -f /home/ma-user/work/logs/serve-v0191-graph.log
```

Stop:

```bash
/home/ma-user/work/scripts/stop-vllm-qwen36.sh
```

The graph start script should include the ATB and custom-op env setup:

```bash
source /home/ma-user/work/src/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash
source /usr/local/Ascend/nnal/atb/set_env.sh
export CC=/home/ma-user/work/venvs/vllm-ascend-0191-py311/bin/aarch64-conda-linux-gnu-g++
export CXX=/home/ma-user/work/venvs/vllm-ascend-0191-py311/bin/aarch64-conda-linux-gnu-g++
```

## Eager Fallback

Keep an eager fallback script. Eager mode proved the model, tokenizer, quantization, and native kernels before graph mode was fixed.

```bash
"$ENV_DIR/bin/vllm" serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 8010 \
  --served-model-name qwen3.6-w8a8 \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --quantization ascend \
  --trust-remote-code \
  --gpu-memory-utilization 0.82 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --no-enable-prefix-caching \
  --enforce-eager
```

The eager smoke test loaded the same 9 shards, used about 33.6 GB for weights, exposed `/v1/models`, and returned a valid `/v1/chat/completions` response.

## Failure Log and Fixes

### Failure: Python 3.12 as the Main Route

Python 3.12 was the wrong primary route for this deployment. The working stack was Python 3.11. The key issue was not just Python syntax or package installation; the Ascend, Triton, torch-npu, and vLLM Ascend pieces had to match.

Use Python 3.11 unless the upstream wheel matrix explicitly supports the exact versions you need on Python 3.12.

### Failure: vLLM Ascend 0.18 Graph Mode

The `0.18.0rc1` stack could serve in eager mode, but graph mode failed. This made it a useful baseline but not the final deployment.

The corrected route was:

```text
vLLM Ascend source tag: v0.19.1rc1
vLLM source tag:        v0.19.1
```

### Failure: GCC Too Old During Build

System GCC 7.3 is too old for PyTorch C++ extension compilation:

```text
#error "You're trying to build PyTorch with a too old version of GCC. We need GCC 9 or later."
```

Installing conda GCC/G++ fixed the Python extension build, but globally exporting `CC` and `CXX` for the whole build can break the CANN custom-op path. Use the split-compiler approach:

```bash
export C_COMPILER=/home/ma-user/work/venvs/vllm-ascend-0191-py311/bin/aarch64-conda-linux-gnu-gcc
export CXX_COMPILER=/home/ma-user/work/venvs/vllm-ascend-0191-py311/bin/aarch64-conda-linux-gnu-g++
python -m pip install --no-build-isolation --no-deps -v .
```

### Failure: GCC Too Old During Serve

After the package built successfully, serve could still fail because Triton Ascend JIT used `/usr/bin/g++`.

Set runtime compiler variables:

```bash
export CC=/home/ma-user/work/venvs/vllm-ascend-0191-py311/bin/aarch64-conda-linux-gnu-g++
export CXX=/home/ma-user/work/venvs/vllm-ascend-0191-py311/bin/aarch64-conda-linux-gnu-g++
```

### Failure: Missing `libatb.so`

Graph mode failed with:

```text
OSError: libatb.so: cannot open shared object file: No such file or directory
```

The library was present under:

```text
/usr/local/Ascend/nnal/atb/8.5.2/atb/cxx_abi_1/lib/libatb.so
```

The fix was to source:

```bash
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### Failure: Setuptools 82

Setuptools 82 broke imports that still relied on `pkg_resources`.

Pin:

```bash
python -m pip install setuptools==80.9.0
```

### Failure: NumPy and OpenCV Drift

The deployment was stabilized with:

```bash
python -m pip install numpy==1.26.4 opencv-python-headless==4.11.0.86
```

This avoided unnecessary ABI and dependency-range churn while preserving the vLLM Ascend path.

## Fresh Redeploy Checklist

Use this checklist for a new machine:

1. Put everything under `/home/ma-user/work`.
2. Download `Eco-Tech/Qwen3.6-27B-w8a8` from ModelScope.
3. Create Python 3.11 env at `/home/ma-user/work/venvs/vllm-ascend-0191-py311`.
4. Install torch `2.9.0`, torch-npu `2.9.0.post1+gitee7ba04`, triton-ascend `3.2.0.dev20260322`.
5. Clone vLLM and checkout `v0.19.1`.
6. Install vLLM with `VLLM_TARGET_DEVICE=empty`.
7. Clone vLLM Ascend and checkout `v0.19.1rc1`.
8. Install conda GCC/G++ into the env.
9. Build vLLM Ascend with `C_COMPILER` and `CXX_COMPILER`, not global `CC/CXX`.
10. Pin `numpy==1.26.4`, `opencv-python-headless==4.11.0.86`, `setuptools==80.9.0`.
11. Verify `vllm_ascend.vllm_ascend_C` imports.
12. For graph mode, source both custom-op env and ATB env.
13. For runtime, export `CC` and `CXX` to conda G++.
14. Start with 4K context and small graph capture sizes.
15. Verify `/v1/models` and `/v1/chat/completions`.

## Operational Notes

Startup is slow on the first graph run. On the validated one-card smoke test:

- Weight loading took about 156 seconds.
- Torch compile and initial profiling/warmup took about 108 seconds.
- Graph capture took about 22 seconds.
- Total time to API readiness was several minutes.

HBM after startup was roughly:

```text
Weights:              33.6 GiB
Peak activation:       2.74 GiB
NPU graph memory:      0.22 GiB
Current KV cache:     13.5 GiB
```

For larger context or higher concurrency, scale carefully. The one-card 910B3 test is a smoke deployment, not the official long-context production shape. For 262K context, follow the official multi-card A2/A3 guidance and increase TP/DP according to the available hardware.

## Final Takeaway

The successful deployment was not unlocked by changing CANN. CANN stayed at `8.5.2`.

The real unlock was aligning the software stack around vLLM Ascend `v0.19.1rc1`, building the native extension with the right compiler split, using conda G++ for Triton runtime JIT, and sourcing ATB before graph serving.

For a fast redeploy, start from:

```text
Python 3.11
CANN 8.5.2
torch 2.9.0
torch-npu 2.9.0.post1+gitee7ba04
triton-ascend 3.2.0.dev20260322
vLLM v0.19.1
vLLM Ascend v0.19.1rc1
ModelScope Eco-Tech/Qwen3.6-27B-w8a8
FULL_DECODE_ONLY graph mode
ATB env sourced
runtime CC/CXX set to conda G++
```

That combination produced a working OpenAI-compatible endpoint on Ascend 910B3.
