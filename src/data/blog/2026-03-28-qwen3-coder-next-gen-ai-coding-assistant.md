---
author: Robin
pubDatetime: 2026-03-28T11:55:00-03:00
title: "How We Successfully Started Qwen3-Coder-Next on Huawei Ascend 910B with vLLM-Ascend 0.17"
description: "Complete deployment guide for Qwen3-Coder-Next on Huawei Ascend 910B with vLLM-Ascend 0.17. Documents the working configuration after multiple failed attempts, covering CANN 8.5.1, W8A8 quantization, worker startup modes, and KV cache management."
tags:
  - huawei-cloud
  - ascend
  - vllm
  - qwen3-coder
  - deployment
  - quantization
  - ai-infrastructure
  - model-serving
featured: true
draft: false
---

# How We Successfully Started Qwen3-Coder-Next on Huawei Ascend 910B with vLLM-Ascend 0.17

## Overview

This post documents the deployment path that finally worked for **Qwen3-Coder-Next** on a Huawei Ascend 910B environment after several failed attempts across older CANN and `vllm-ascend` combinations.

The important point is that this was not a "one-parameter fix." The successful result came from aligning four things at the same time:

1. A **supported runtime stack** based on `CANN 8.5.1`
2. A **quantized W8A8 model** instead of the original BF16 weights
3. The correct **worker startup mode** for the current stack
4. A hard **KV cache cap** to keep memory usage bounded

If you are trying to deploy Qwen3-Coder-Next on Ascend and keep hitting startup failures, KV cache OOMs, or inconsistent worker crashes, this is the configuration that actually worked.

## Final Working Environment

The final successful deployment ran on an Ascend notebook host with:

- `8` visible NPUs
- `CANN 8.5.1`
- driver `23.0.6`
- Python environment at:

```bash
/home/ma-user/work/venvs/unified-py311
```

Installed versions in the successful run:

- `vllm==0.17.0`
- `vllm-ascend==0.17.0rc1`
- `torch==2.9.0`
- `torch_npu==2.9.0`
- `transformers==4.57.6`
- `triton-ascend==3.2.0`
- `modelscope==1.35.0`
- `xgrammar==0.1.29`

Additional dependencies that also had to exist in the environment:

- `kaldi-native-fbank`
- `opentelemetry-api`
- `opentelemetry-exporter-otlp`
- `opentelemetry-sdk`
- `opentelemetry-semantic-conventions-ai`

This version alignment mattered. Earlier work on older stacks could sometimes load weights, but the deployment remained unstable or failed later during worker startup, Triton integration, or KV cache initialization.

## Why BF16 Was Not the Final Path

The original BF16 model was not the final serving format.

In earlier experiments, BF16 repeatedly failed for a practical reason:

- model weights could load
- but the service still died during **KV cache initialization**
- lowering only `max_model_len` was not a sufficient long-term fix

The critical lesson was:

- **weight compression** and **KV cache control** are different problems
- even if weights fit, the service can still fail when KV cache is allocated

That is why the final working route used:

- **W8A8 quantization** for the model
- a **hard cap** on GPU/NPU KV blocks

## How the W8A8 Model Was Produced

The final serving model was not downloaded directly as a public W8A8 artifact. It was produced from the original model using **ModelSlim**.

The model that was ultimately served was:

```bash
/home/ma-user/work/models/Qwen3-Coder-Next-w8a8-from-30011
```

This model came from an earlier successful quantization run on another notebook.

The important detail is that the successful quantization path was the **ModelSlim CLI**, not the older speculative `Qwen3-MOE` example script.

The working quantization pattern was:

```bash
python -m msmodelslim.cli quant \
  --model_path /home/ma-user/work/Qwen3-coder-next \
  --save_path /home/ma-user/work/Qwen3-coder-next-w8a8 \
  --device npu \
  --model_type Qwen3-Next-80B-A3B-Instruct \
  --quant_type w8a8 \
  --trust_remote_code True
```

Successful quantization was confirmed by log signals such as:

- `model.layers.0` through `model.layers.47`
- `FlexSmoothQuantProcessor`
- `AscendV1Saver`
- `QUANTIZATION: END`
- `SUCCESS`

The quantized output contained:

- `config.json`
- `quant_model_description.json`
- `quant_model_weights.safetensors.index.json`
- `quant_model_weights-00001-of-00019.safetensors` through `00019-of-00019`

## Moving Large Models Between Notebooks

Downloading a very large model repeatedly from ModelScope is slow and fragile. In practice, it was often faster to move the model directory from one notebook to another.

The reliable rule was:

- do **not** use the public notebook ingress host and external SSH port for notebook-to-notebook transfer
- do use **private IP + the notebook's internal SSH daemon**

For large model directories, a streaming transfer worked well:

```bash
tar -C /source/root -cf - model_dir | ssh target 'tar -C /dest/root -xf -'
```

Two operational lessons mattered:

1. Never transfer into a directory that is still being downloaded.
2. Always use a unique destination directory name first, then switch later if needed.

That avoided silent merges between:

- an incomplete local download
- and a transferred complete copy

## The Real Deployment Breakthrough

The final success did not come from one single version bump. It came from a set of choices that worked together.

### 1. Use `spawn`, not `fork`

Older `vllm-ascend` rescue paths sometimes relied on:

```bash
VLLM_WORKER_MULTIPROC_METHOD=fork
```

That was **not** the working answer on the final `0.17` stack.

On the successful host and package set:

- `fork` produced:
  - `Invalid thread pool!`
- `spawn` was the correct mode

This was the working choice:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### 2. Keep CPU thread counts pinned to 1

The stable route also used:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

This was not about peak throughput. It was about keeping worker startup and runtime behavior stable on this stack.

### 3. Treat KV cache as a separate memory budget

The deployment remained stable only because KV cache growth was constrained explicitly:

```bash
--num-gpu-blocks-override 512
```

That setting prevented the service from consuming the entire remaining memory budget during KV cache initialization.

This is the main reason the deployment was later able to move from a smaller context window to a working:

```bash
--max-model-len 4096
```

### 4. The quantization flag was mandatory

Because the model was produced by ModelSlim, serving had to use:

```bash
--quantization ascend
```

Without this, the deployment would not be using the quantized artifact correctly.

## Final Working Serve Command

This is the final serve shape that worked:

```bash
vllm serve /home/ma-user/work/models/Qwen3-Coder-Next-w8a8-from-30011 \
  --host 127.0.0.1 \
  --port 8008 \
  --served-model-name qwen3-coder-next \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.45 \
  --num-gpu-blocks-override 512 \
  --quantization ascend \
  --trust-remote-code
```

Supporting environment:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_USE_MODELSCOPE=False
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
HCCL_OP_EXPANSION_MODE=AIV
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

## What Was Successfully Verified

The final service was not treated as successful just because the process stayed alive.

Success was confirmed with:

- `GET /v1/models` returning `200`
- `POST /v1/chat/completions` returning `200`
- real text output from the model

In other words, this was not just a "port is open" result. The model completed actual inference requests successfully.

## The Triton Warning That Did Not Block Deployment

One warning still appeared during startup:

```text
No module named 'triton.language.target_info'
```

This looked alarming, but in the final deployment it was **non-blocking**.

That conclusion was based on actual behavior:

- the service started
- `/v1/models` worked
- `chat/completions` worked

So the practical rule is:

- do not ignore it blindly in every environment
- but do not treat it as an automatic deployment failure on this exact `0.17 + CANN 8.5.1` stack if real inference succeeds

## Performance Measurements

With the successful deployment running on `8` NPUs, the measured short-prompt performance was approximately:

- **Time to first token (TTFT)**: `~0.70s`
- **Overall generation rate**: `~11.36 tok/s`
- **Post-first-token decode rate**: `~13 tok/s`

These numbers were measured on short prompts and short completions.

The more important result for real coding-agent style usage was that the service also handled larger prompts after moving to `4096` context:

- about `1035` prompt tokens + `256` completion tokens: success
- about `2045` prompt tokens + `256` completion tokens: success

That was a major improvement over the earlier `1024` context configuration, which could reject longer requests immediately.

## Model Output Quality: Deployment Success Is Not The Same As Production Quality

The deployment was successful. That does **not** mean every generation was production-grade.

When tested on multiple coding tasks, the model behaved like this:

- deployment and inference pipeline: stable
- code editing / bug fixing: relatively strong
- clean-sheet code generation: mixed quality

Examples from testing:

- one clean code-generation case returned a function with indentation issues and an incorrect example output
- a code-repair scenario worked correctly and produced syntactically valid Python

So the fair engineering conclusion is:

- the service is usable
- the model can help with coding tasks
- but generation quality still needs task-specific evaluation before production use

This deployment solved the infrastructure problem. It did not eliminate the need for application-level evaluation.

## Mistakes Worth Avoiding

Several bad assumptions cost time during earlier attempts.

Do not repeat these:

1. **Do not trust the notebook image label without checking the actual visible CANN/toolkit files.**
2. **Do not assume `fork` remains the right answer across stack upgrades.**
3. **Do not assume W8A8 alone fixes memory problems without KV cache limits.**
4. **Do not transfer a complete model into a directory that is already being downloaded.**
5. **Do not treat `triton.language.target_info` as an automatic blocker on this exact final stack.**
6. **Do not copy an `x86_64` helper binary, such as `opencode`, onto an `aarch64` notebook and expect it to run.**

## Recommended Validation Checklist

If you want to reproduce this deployment cleanly, validate in this order:

1. Check actual toolkit and driver versions.
2. Verify `8` visible NPUs from the target environment.
3. Verify the quantized model directory is complete.
4. Confirm the serving environment contains the expected Python package versions.
5. Start the service with `spawn`, not `fork`.
6. Verify `/v1/models`.
7. Verify one real `chat/completions` request.
8. Only then benchmark latency, throughput, or long-context behavior.

## Conclusion

The final successful path for Qwen3-Coder-Next on Ascend was:

- **not BF16**
- **not the old `0.11` patched route**
- **not `fork`**

It was:

- `CANN 8.5.1`
- `vllm 0.17.0`
- `vllm-ascend 0.17.0rc1`
- `W8A8` quantization produced by ModelSlim
- `spawn`
- `8` NPUs
- explicit KV cache control with `--num-gpu-blocks-override 512`
- and final serving at `max_model_len=4096`

That combination finally moved the problem from "cannot start the model" to "the service is up, inference works, and output quality can now be evaluated like a normal model-serving problem."

---

## Appendix: Final Working Startup Script

The script below captures the final working deployment shape and intentionally avoids the stale self-killing `pkill` pattern that caused earlier startup confusion.

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/models/Qwen3-Coder-Next-w8a8-from-30011}"
PORT="${PORT:-8008}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-512}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-coder-next}"
LOG_DIR="${LOG_DIR:-/home/ma-user/work/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/qwen3_coder_next_w8a8_v017_spawn_4096.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/qwen3_coder_next_w8a8_v017_spawn_4096.pid}"
UNIFIED_ENV="${UNIFIED_ENV:-/home/ma-user/work/venvs/unified-py311}"

mkdir -p "${LOG_DIR}"

[ -d "${MODEL_PATH}" ] || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 1; }
[ -f "${MODEL_PATH}/config.json" ] || { echo "[ERROR] config.json not found under ${MODEL_PATH}"; exit 1; }
[ -d "${UNIFIED_ENV}" ] || { echo "[ERROR] Unified env not found: ${UNIFIED_ENV}"; exit 1; }

set +u
source "${UNIFIED_ENV}/bin/activate"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-False}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [ -f "${PID_FILE}" ]; then
  OLD_PID="$(cat "${PID_FILE}" || true)"
  if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" >/dev/null 2>&1; then
    kill "${OLD_PID}" || true
    sleep 5
    kill -9 "${OLD_PID}" >/dev/null 2>&1 || true
  fi
  rm -f "${PID_FILE}"
fi

nohup vllm serve "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --data-parallel-size "${DP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}" \
  --quantization ascend \
  --trust-remote-code \
  > "${LOG_FILE}" 2>&1 &

NEW_PID=$!
echo "${NEW_PID}" > "${PID_FILE}"

echo "[INFO] Started. PID=${NEW_PID}"
echo "[INFO] Log: ${LOG_FILE}"
echo "[INFO] PID file: ${PID_FILE}"
sleep 3
tail -n 40 "${LOG_FILE}" || true
```
