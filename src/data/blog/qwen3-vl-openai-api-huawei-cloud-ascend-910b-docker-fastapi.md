---
author: Robin
pubDatetime: 2026-02-03T00:00:00-03:00
title: "Deploying a Qwen3-VL OpenAI-Compatible API on Huawei Cloud Ascend (910B) with Docker + FastAPI"
description: "A comprehensive guide to deploying Qwen3-VL-8B vision-language model on Ascend 910B NPUs with an OpenAI-compatible FastAPI server, including container setup, load testing, and production deployment patterns."
tags:
  - qwen3-vl
  - huawei-cloud
  - ascend-910b
  - docker
  - fastapi
  - openai-api
  - vision-language-model
  - npu
featured: true
draft: false
---

## 1) Background and Goal

On a host equipped with 8 Ascend 910B NPUs, we needed to serve Qwen3-VL-8B (vision-language) from local weights and expose an OpenAI-compatible API so the client can call it via HTTP.

Key constraints:

- Ascend runtime and PyTorch-NPU toolchain are available inside a Huawei Cloud SWR container image, not on the host.
- Model weights are stored on the host at `/mnt/model_weights`.
- Customer-shared test data is available on the host at `/mnt/test_data`.
- The server must run in the container; the client/test runner runs on the host.

Deliverables:

- An OpenAI-compatible API server (FastAPI) supporting:
  - `GET /v1/models`
  - `POST /v1/chat/completions`
- A client script to validate single images and run dataset benchmarks.

---

## 2) Environment Overview

Host:

- 8x Ascend 910B
- Test data: `/mnt/test_data`
- Model weights: `/mnt/model_weights`
- Code workspace: `/mnt/GUI`

Container image (Ascend development environment):

- `swr.cn-north-9.myhuaweicloud.com/vllm-npu/912train_image_llamafactory:20260115`

Important note:

- Do NOT assume `:latest` exists. Use `docker images` to find the correct tag.

---

## 3) What We Built

### 3.1 OpenAI-compatible server (FastAPI)

Endpoints:

- `GET /health` (liveness)
- `GET /v1/models` (OpenAI-style model list)
- `POST /v1/chat/completions` (OpenAI-style chat completions)

Input format:

- Supports text parts: `{ "type": "text", "text": "..." }`
- Supports image parts: `{ "type": "image_url", "image_url": { "url": "data:image/...;base64,..." } }`

Notes:

- Streaming (`stream=true`) is intentionally not implemented and returns HTTP 501.
- The server uses the first image it sees in the request.

### 3.2 Benchmark client (host-side)

- Reads `/mnt/test_data/*.jsonl` and calls `/v1/chat/completions` concurrently.
- Added single-image mode: `benchmark_client.py -f /path/to/image.webp`.

---

## 4) Problems We Hit (and How We Fixed Them)

### 4.1 No `latest` tag on the SWR image

Attempting `...:latest` failed with a manifest error. Fix: run `docker images` and use the existing tag `20260115`.

### 4.2 Host cannot run server due to missing Ascend PyTorch stack

Running `server.py` on host failed with `ModuleNotFoundError: No module named 'torch'`. Fix: run the server inside the Ascend container image.

### 4.3 8 NPUs already occupied by another container

Symptoms:

- `/health` and `/v1/models` worked but inference requests hung.
- Inside the container, `torch.npu.is_available()` was false / NPU count was 0.

Root cause: another long-running container (`qwen3_vl_8b`) was holding the NPUs.

Fix:

- Stop the old container to release NPUs:
  - `docker stop qwen3_vl_8b`
- Restart the new server container process.

### 4.4 Transformers dependency conflicts inside the container

The image contains `llamafactory`, which requires `transformers <= 4.57.1`. We pinned transformers in `requirements.txt`:

- `transformers>=4.49.0,<=4.57.1,!=4.52.0,!=4.57.0`

---

## 5) One-Command Deployment

We provide `run_server_container.sh` to:

- Start a new container with NPU device mounts
- Mount model + test data + code
- Install deps in-container
- Start `server.py`
- Wait until `/health` is ready

On host:

```bash
cd /mnt/GUI
chmod +x run_server_container.sh
./run_server_container.sh
```

Verify from host:

```bash
curl http://127.0.0.1:9000/health
curl http://127.0.0.1:9000/v1/models
```

---

## 6) Functional Tests

### 6.1 Text-only

```bash
python - <<'PY'
import requests, time
payload = {
  "model": "/opt/models/qwen3-vl-8b",
  "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
  "max_tokens": 16,
  "temperature": 0.0
}
start=time.time()
resp = requests.post("http://127.0.0.1:9000/v1/chat/completions", json=payload, timeout=300)
print('status', resp.status_code, 'elapsed', round(time.time()-start,2))
print(resp.text[:800])
PY
```

### 6.2 Image (data URL)

```bash
python - <<'PY'
import base64, requests, time
img_path = "/mnt/test_data/pictures/test1.webp"
with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()
payload = {
  "model": "/opt/models/qwen3-vl-8b",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe the image."},
      {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{img_b64}"}}
    ]
  }],
  "max_tokens": 128,
  "temperature": 0.0
}
start=time.time()
resp = requests.post("http://127.0.0.1:9000/v1/chat/completions", json=payload, timeout=300)
print('status', resp.status_code, 'elapsed', round(time.time()-start,2))
print(resp.text[:1200])
PY
```

---

## 7) Benchmark Test (Single Image)

We added `-f` to `benchmark_client.py` for quick verification:

```bash
python /mnt/GUI/benchmark_client.py -f /mnt/test_data/pictures/test1.webp
```

Results are written to:

- `/mnt/GUI/test_results.jsonl`

---

## 8) Scaling to 8 NPUs (Practical Guidance)

To truly utilize all 8 NPUs, the server side must be parallelized. Two typical approaches:

1) Run multiple server instances (one per NPU) on different ports, setting:

- `ASCEND_RT_VISIBLE_DEVICES=<npu_id>`
- `SERVER_PORT=<unique_port>`

Then call with the benchmark client using `--endpoints` (round-robin).

2) Use a dedicated inference engine that manages multi-card execution internally (e.g., Ascend vLLM).

This blog focuses on correctness and an OpenAI-compatible API surface first.

---

## 9) Full Source Code

### 9.1 `requirements.txt`

```text
transformers>=4.49.0,<=4.57.1,!=4.52.0,!=4.57.0
pillow>=10.3.0
numpy>=1.26.0
concurrent-log-handler>=0.9.24
requests>=2.31.0
fastapi>=0.115.2
uvicorn>=0.24.0
httpx>=0.27.0
tqdm>=4.66.0
```

### 9.2 `run_server_container.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Container settings
IMAGE=${IMAGE:-swr.cn-north-9.myhuaweicloud.com/vllm-npu/912train_image_llamafactory}
CONTAINER_NAME=${CONTAINER_NAME:-qwen3_vl_server}
HOST_PORT=${HOST_PORT:-9000}
SERVER_PORT=${SERVER_PORT:-9000}

# Paths
HOST_MODEL_PATH=${HOST_MODEL_PATH:-/mnt/model_weights}
HOST_TEST_DATA=${HOST_TEST_DATA:-/mnt/test_data}
HOST_CODE_DIR=${HOST_CODE_DIR:-/mnt/GUI}

CONTAINER_MODEL_PATH=${CONTAINER_MODEL_PATH:-/opt/models/qwen3-vl-8b}
CONTAINER_TEST_DATA=${CONTAINER_TEST_DATA:-/opt/test_data}
CONTAINER_CODE_DIR=${CONTAINER_CODE_DIR:-/workspace/app}

echo "[INFO] Starting container ${CONTAINER_NAME} from ${IMAGE}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -itd \
  --name "${CONTAINER_NAME}" \
  --net=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/log/npu/:/usr/slog \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  -v "${HOST_MODEL_PATH}:${CONTAINER_MODEL_PATH}" \
  -v "${HOST_TEST_DATA}:${CONTAINER_TEST_DATA}" \
  -v "${HOST_CODE_DIR}:${CONTAINER_CODE_DIR}" \
  "${IMAGE}" \
  /bin/bash

echo "[INFO] Installing dependencies and starting server inside container"

docker exec -u root "${CONTAINER_NAME}" /bin/bash -lc "
set -e
source /etc/profile || true
cd ${CONTAINER_CODE_DIR}
pip install -r requirements.txt
export MODEL_PATH=${CONTAINER_MODEL_PATH}
export MODEL_ID=${CONTAINER_MODEL_PATH}
export SERVER_HOST=0.0.0.0
export SERVER_PORT=${SERVER_PORT}
nohup python server.py > /root/qwen3_vl_server.log 2>&1 &
"

echo "[INFO] Waiting for server to be ready..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:${HOST_PORT}/health >/dev/null 2>&1; then
    echo "[INFO] Server is ready!"
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "[ERROR] Server failed to start within 30 seconds"
    echo "[INFO] Container logs:"
    docker exec "${CONTAINER_NAME}" tail -50 /root/qwen3_vl_server.log 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

echo "[INFO] Server started. Health check: http://127.0.0.1:${HOST_PORT}/health"
echo "[INFO] Logs inside container: /root/qwen3_vl_server.log"
echo "[INFO] Test with: curl http://127.0.0.1:${HOST_PORT}/v1/models"
```

### 9.3 `server.py`

```python
import base64
import io
import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/model_weights")
MODEL_ID = os.environ.get("MODEL_ID", MODEL_PATH)
DEVICE = os.environ.get("DEVICE", "")
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", "2621440"))  # ~1.6K x 1.6K


def _detect_device() -> str:
    if DEVICE:
        return DEVICE
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_dtype(device: str) -> torch.dtype:
    if device in {"cuda", "npu"}:
        return torch.bfloat16
    return torch.float32


@lru_cache()
def _load_model():
    device = _detect_device()
    dtype = _resolve_dtype(device)
    try:
        import torch_npu  # type: ignore  # noqa: F401
    except Exception:
        torch_npu = None  # noqa: F841

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, processor, device


def _decode_data_url(data_url: str) -> Image.Image:
    if not data_url.startswith("data:image"):
        raise ValueError("Unsupported image_url; must be data:image/* base64.")
    try:
        header, b64 = data_url.split(",", 1)
    except ValueError as exc:
        raise ValueError("Invalid data URL format.") from exc
    img_bytes = base64.b64decode(b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        image.thumbnail((int(MAX_IMAGE_PIXELS**0.5),) * 2)
    return image


def _load_image(image_url: str) -> Image.Image:
    if image_url.startswith("data:image"):
        return _decode_data_url(image_url)
    if image_url.startswith("http://") or image_url.startswith("https://"):
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            image.thumbnail((int(MAX_IMAGE_PIXELS**0.5),) * 2)
        return image
    raise ValueError("Unsupported image_url scheme; use data URL or http(s).")


class ImageURL(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


class Message(BaseModel):
    role: str
    content: List[ContentPart] | str
    
    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v):
        """Normalize string content to ContentPart list for OpenAI compatibility."""
        if isinstance(v, str):
            return [ContentPart(type="text", text=v)]
        return v


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.5)
    ignore_eos: bool = False
    stream: bool = False
    stop_token_ids: Optional[List[int]] = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"


app = FastAPI(title="Qwen3-VL OpenAI-Compatible Server")


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {"object": "list", "data": [ModelCard(id=MODEL_ID).model_dump()]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
    # Handle streaming (return error - not implemented yet)
    if req.stream:
        raise HTTPException(status_code=501, detail="Streaming not yet implemented. Set stream=false")
    
    # Extract a single image (if provided) and a concatenated text prompt
    image: Optional[Image.Image] = None
    texts: List[str] = []
    
    for message in req.messages:
        content_list = message.content
        for part in content_list:
            if part.type == "text" and part.text:
                texts.append(part.text)
            if part.type == "image_url" and part.image_url:
                if image is None:
                    try:
                        image = _load_image(part.image_url.url)
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

    prompt = "\n".join(texts).strip()
    if not prompt:
        prompt = "Describe the image."

    try:
        model, processor, device = _load_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {str(e)}")
    
    messages = [{"role": "user", "content": []}]
    if image is not None:
        messages[0]["content"].append({"type": "image", "image": image})
    messages[0]["content"].append({"type": "text", "text": prompt})

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tokenization failed: {str(e)}")
    
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    do_sample = req.temperature > 0
    generation_kwargs = {
        "max_new_tokens": int(req.max_tokens),
        "do_sample": do_sample,
        "temperature": float(req.temperature) if do_sample else None,
        "top_p": float(req.top_p) if do_sample else None,
    }
    # Remove None values to avoid passing them to generate()
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    
    if req.top_k is not None and req.top_k > 0:
        generation_kwargs["top_k"] = int(req.top_k)
    if req.repetition_penalty and req.repetition_penalty != 1.0:
        generation_kwargs["repetition_penalty"] = float(req.repetition_penalty)
    if req.stop_token_ids:
        generation_kwargs["eos_token_id"] = req.stop_token_ids
    if req.ignore_eos:
        generation_kwargs["ignore_eos"] = True

    try:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_kwargs)
        
        # NPU memory cleanup after generation
        if device == "npu" and hasattr(torch, "npu"):
            torch.npu.empty_cache()
            
    except Exception as e:
        # Clear NPU cache on error to prevent OOM in subsequent requests
        if device == "npu" and hasattr(torch, "npu"):
            torch.npu.empty_cache()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    trimmed_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    
    try:
        output_text = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decoding failed: {str(e)}")

    prompt_tokens = int(inputs["input_ids"].shape[1])
    completion_tokens = int(trimmed_ids.shape[1])
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", "9000"))
    uvicorn.run(app, host=host, port=port)
```

### 9.4 `benchmark_client.py`

```python
#!/usr/bin/env python3
"""
OpenAI-compatible API Benchmark Client

Reads JSONL test data and sends concurrent requests to maximize throughput
across multiple NPUs with batching, retry logic, and robust error handling.
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from tqdm.asyncio import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/mnt/GUI/benchmark_client.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class RequestConfig:
    """Configuration for API requests."""
    url: str = "http://127.0.0.1:9000/v1/chat/completions"
    model: str = "/mnt/model_weights"
    max_tokens: int = 256
    temperature: float = 0.0
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    num_npus: int = 8
    batch_size: int = 8
    max_concurrent: int = 0  # 0 = auto (num_npus * concurrency_per_npu)
    concurrency_per_npu: int = 8
    rate_limit: float = 0.0  # requests per second, 0 = no limit
    output_path: str = "/mnt/GUI/test_results.jsonl"
    input_pattern: str = "/mnt/test_data/*.jsonl"
    base_dir: str = "/mnt/test_data"
    endpoints: Optional[List[str]] = None


@dataclass
class RequestResult:
    """Result of a single request."""
    request_id: str
    original_id: Optional[str]
    success: bool
    response: Optional[Dict[str, Any]]
    error: Optional[str]
    latency: float
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    retry_count: int = 0


class RateLimiter:
    """Token bucket rate limiter for controlling request rate."""
    
    def __init__(self, requests_per_second: float):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0
        self.last_request_time = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request."""
        if self.requests_per_second <= 0:
            return
        
        async with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                await asyncio.sleep(wait_time)
            self.last_request_time = time.time()


class AdaptiveConcurrency:
    """Adaptive concurrency controller based on success rate."""
    
    def __init__(self, initial: int, min_concurrent: int = 8, max_concurrent: int = 128):
        self.current = initial
        self.min_concurrent = min_concurrent
        self.max_concurrent = max_concurrent
        self.success_count = 0
        self.failure_count = 0
        self.lock = asyncio.Lock()
    
    async def report_success(self):
        async with self.lock:
            self.success_count += 1
            if self.success_count >= 10:
                self.current = min(self.current + 1, self.max_concurrent)
                self.success_count = 0
                logger.debug(f"Increased concurrency to {self.current}")
    
    async def report_failure(self):
        async with self.lock:
            self.failure_count += 1
            if self.failure_count >= 5:
                self.current = max(self.current // 2, self.min_concurrent)
                self.failure_count = 0
                logger.warning(f"Decreased concurrency to {self.current}")
    
    def get_limit(self) -> int:
        return self.current


def load_jsonl_files(pattern: str) -> List[Dict[str, Any]]:
    """Load all JSONL files matching the pattern."""
    items = []
    paths = list(Path("/").glob(pattern.lstrip("/")))
    
    if not paths:
        logger.error(f"No files found matching pattern: {pattern}")
        return items
    
    for path in paths:
        logger.info(f"Loading {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Extract ID if present, otherwise generate one
                        if isinstance(data, list) and len(data) > 0:
                            # Check if first item is system message
                            if isinstance(data[0], dict) and data[0].get("role") == "system":
                                item_id = str(uuid.uuid4())[:8]
                            else:
                                item_id = data[0].get("id", str(uuid.uuid4())[:8]) if isinstance(data[0], dict) else str(uuid.uuid4())[:8]
                        elif isinstance(data, dict):
                            item_id = data.get("id", str(uuid.uuid4())[:8])
                        else:
                            item_id = str(uuid.uuid4())[:8]
                        
                        items.append({
                            "id": item_id,
                            "data": data,
                            "source_file": str(path),
                            "line_number": line_num,
                        })
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in {path}:{line_num}: {e}")
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
    
    logger.info(f"Loaded {len(items)} items from {len(paths)} files")
    return items


def _file_to_data_url(path: str) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    suffix = p.suffix.lower().lstrip(".") or "png"
    with p.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{suffix};base64,{b64}"


def _normalize_image_url(url: str, base_dir: str) -> str:
    if url.startswith("data:image") or url.startswith("http://") or url.startswith("https://"):
        return url
    # Try resolve relative path under base_dir
    rel_path = Path(base_dir) / url
    data_url = _file_to_data_url(str(rel_path)) or _file_to_data_url(url)
    if data_url:
        return data_url
    return url


def _normalize_content(content: Any, base_dir: str) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        normalized = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url" and part.get("image_url"):
                img_url = part["image_url"].get("url", "")
                part = {
                    "type": "image_url",
                    "image_url": {"url": _normalize_image_url(img_url, base_dir)},
                }
            normalized.append(part)
        return normalized
    return [{"type": "text", "text": str(content)}]


def convert_to_openai_format(item: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    """Convert test data to OpenAI chat completion format."""
    data = item["data"]
    
    # If already in messages format (list of role/content dicts)
    if isinstance(data, list):
        # Filter to only include valid message roles
        messages = []
        for msg in data:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                content = _normalize_content(msg["content"], base_dir)
                messages.append({
                    "role": msg["role"],
                    "content": content
                })
        return {"messages": messages}
    
    # If dict format with expected fields
    if isinstance(data, dict):
        messages = []
        
        # Handle different input formats
        if "messages" in data:
            messages = []
            for msg in data["messages"]:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({
                        "role": msg["role"],
                        "content": _normalize_content(msg["content"], base_dir),
                    })
        elif "prompt" in data:
            messages = [{"role": "user", "content": _normalize_content(data["prompt"], base_dir)}]
        elif "text" in data:
            messages = [{"role": "user", "content": _normalize_content(data["text"], base_dir)}]
        elif "image_url" in data:
            img_url = _normalize_image_url(data["image_url"], base_dir)
            content = [{"type": "image_url", "image_url": {"url": img_url}}]
            if "text" in data and data["text"]:
                content.insert(0, {"type": "text", "text": data["text"]})
            messages = [{"role": "user", "content": content}]
        elif "image_path" in data:
            img_url = _normalize_image_url(data["image_path"], base_dir)
            content = [{"type": "image_url", "image_url": {"url": img_url}}]
            if "text" in data and data["text"]:
                content.insert(0, {"type": "text", "text": data["text"]})
            messages = [{"role": "user", "content": content}]
        else:
            # Wrap entire data as text
            messages = [{"role": "user", "content": _normalize_content(str(data), base_dir)}]
        
        return {"messages": messages}
    
    # Default: wrap as text
    return {"messages": [{"role": "user", "content": str(data)}]}


def prepare_request_payload(
    item: Dict[str, Any],
    config: RequestConfig,
    base_dir: str,
) -> Dict[str, Any]:
    """Prepare the request payload for the API."""
    openai_data = convert_to_openai_format(item, base_dir)
    
    payload = {
        "model": config.model,
        "messages": openai_data["messages"],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "stream": False,
    }
    
    return payload


async def make_request_with_retry(
    client: httpx.AsyncClient,
    item: Dict[str, Any],
    config: RequestConfig,
    base_dir: str,
    endpoints: Optional[List[str]],
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
) -> RequestResult:
    """Make a request with retry logic."""
    request_id = item["id"]
    original_id = item.get("original_id") or request_id
    payload = prepare_request_payload(item, config, base_dir)
    endpoint = config.url
    if endpoints:
        endpoint = endpoints[hash(request_id) % len(endpoints)]
    
    retry_count = 0
    last_error = None
    start_time = time.time()
    
    await rate_limiter.acquire()
    
    async with semaphore:
        for attempt in range(config.max_retries + 1):
            try:
                response = await client.post(
                    endpoint,
                    json=payload,
                    timeout=config.timeout,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                
                result_data = response.json()
                latency = time.time() - start_time
                
                # Extract token usage if available
                usage = result_data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
                
                return RequestResult(
                    request_id=request_id,
                    original_id=original_id,
                    success=True,
                    response=result_data,
                    error=None,
                    latency=latency,
                    tokens_used=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    retry_count=retry_count,
                )
                
            except httpx.TimeoutException as e:
                last_error = f"Timeout after {config.timeout}s"
                logger.warning(f"Request {request_id} timeout (attempt {attempt + 1})")
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                if e.response.status_code in (429, 503, 502, 504):  # Rate limit or server overload
                    retry_after = float(e.response.headers.get("Retry-After", config.retry_delay))
                    logger.warning(f"Request {request_id} got {e.response.status_code}, retrying after {retry_after}s")
                    await asyncio.sleep(retry_after * (config.retry_backoff ** attempt))
                else:
                    logger.error(f"Request {request_id} HTTP error: {last_error}")
                    break  # Don't retry client errors
            except httpx.NetworkError as e:
                last_error = f"Network error: {str(e)}"
                logger.warning(f"Request {request_id} network error (attempt {attempt + 1})")
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Request {request_id} unexpected error: {e}")
                break
            
            retry_count += 1
            if attempt < config.max_retries:
                delay = config.retry_delay * (config.retry_backoff ** attempt)
                logger.info(f"Retrying request {request_id} in {delay:.2f}s (attempt {attempt + 2})")
                await asyncio.sleep(delay)
    
    latency = time.time() - start_time
    return RequestResult(
        request_id=request_id,
        original_id=original_id,
        success=False,
        response=None,
        error=last_error,
        latency=latency,
        retry_count=retry_count,
    )


def extract_response_content(result: RequestResult) -> Optional[str]:
    """Extract the assistant's response content from the result."""
    if not result.success or not result.response:
        return None
    
    try:
        choices = result.response.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            return content
    except Exception as e:
        logger.warning(f"Failed to extract response content: {e}")
    
    return None


def write_result_to_file(
    result: RequestResult,
    output_path: str,
    lock: asyncio.Lock
):
    """Write a single result to the output file."""
    content = extract_response_content(result)
    
    output_record = {
        "id": result.original_id or result.request_id,
        "request_id": result.request_id,
        "success": result.success,
        "response": content,
        "error": result.error,
        "latency": round(result.latency, 3),
        "tokens_used": result.tokens_used,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "retry_count": result.retry_count,
    }
    
    # Only include full response if needed for debugging
    if not result.success and result.response:
        output_record["raw_response"] = result.response
    
    return output_record


async def process_batch(
    client: httpx.AsyncClient,
    batch: List[Dict[str, Any]],
    config: RequestConfig,
    base_dir: str,
    endpoints: Optional[List[str]],
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    pbar: tqdm,
) -> List[RequestResult]:
    """Process a batch of requests concurrently."""
    tasks = [
        make_request_with_retry(client, item, config, base_dir, endpoints, semaphore, rate_limiter)
        for item in batch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task raised exception: {result}")
            processed_results.append(RequestResult(
                request_id="unknown",
                original_id=None,
                success=False,
                response=None,
                error=f"Task exception: {str(result)}",
                latency=0,
            ))
        else:
            processed_results.append(result)
        pbar.update(1)
    
    return processed_results


async def run_benchmark(
    items: List[Dict[str, Any]],
    req_config: RequestConfig,
    bench_config: BenchmarkConfig,
) -> Tuple[List[RequestResult], Dict[str, Any]]:
    """Run the benchmark with optimized concurrency."""
    
    # Calculate optimal batch and concurrency settings
    num_npus = bench_config.num_npus
    batch_size = bench_config.batch_size or max(1, num_npus)
    max_concurrent = (
        bench_config.max_concurrent
        if bench_config.max_concurrent > 0
        else max(1, num_npus * bench_config.concurrency_per_npu)
    )
    
    logger.info(f"Starting benchmark with {len(items)} items")
    logger.info(f"NPUs: {num_npus}, Batch size: {batch_size}, Max concurrent: {max_concurrent}")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    rate_limiter = RateLimiter(bench_config.rate_limit)
    
    results = []
    start_time = time.time()
    
    # Configure HTTP client with connection pooling
    limits = httpx.Limits(
        max_keepalive_connections=max_concurrent,
        max_connections=max_concurrent * 2,
    )
    timeout = httpx.Timeout(
        connect=30.0,
        read=req_config.timeout,
        write=30.0,
        pool=30.0,
    )
    
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        with tqdm(total=len(items), desc="Processing") as pbar:
            # Process in batches
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_results = await process_batch(
                    client,
                    batch,
                    req_config,
                    bench_config.base_dir,
                    bench_config.endpoints,
                    semaphore,
                    rate_limiter,
                    pbar,
                )
                results.extend(batch_results)
                
                # Write results incrementally to avoid memory issues
                if i % (batch_size * 10) == 0:
                    await asyncio.sleep(0.01)  # Allow event loop to breathe
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    latencies = [r.latency for r in successful]
    
    stats = {
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) * 100 if results else 0,
        "total_time_seconds": round(total_time, 2),
        "throughput_rps": round(len(results) / total_time, 2) if total_time > 0 else 0,
        "avg_latency": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "min_latency": round(min(latencies), 3) if latencies else 0,
        "max_latency": round(max(latencies), 3) if latencies else 0,
        "total_tokens": sum(r.tokens_used or 0 for r in successful),
        "tokens_per_second": round(sum(r.tokens_used or 0 for r in successful) / total_time, 2) if total_time > 0 else 0,
    }
    
    return results, stats


def write_results_to_jsonl(
    results: List[RequestResult],
    output_path: str,
):
    """Write all results to JSONL file."""
    logger.info(f"Writing results to {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            record = {
                "id": result.original_id or result.request_id,
                "request_id": result.request_id,
                "success": result.success,
                "response": extract_response_content(result),
                "error": result.error,
                "latency": round(result.latency, 3),
                "tokens_used": result.tokens_used,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "retry_count": result.retry_count,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"Results written to {output_path}")


def print_statistics(stats: Dict[str, Any]):
    """Print benchmark statistics."""
    print("\n" + "=" * 60)
    print("BENCHMARK STATISTICS")
    print("=" * 60)
    print(f"Total Requests:      {stats['total_requests']}")
    print(f"Successful:          {stats['successful']}")
    print(f"Failed:              {stats['failed']}")
    print(f"Success Rate:        {stats['success_rate']:.1f}%")
    print(f"Total Time:          {stats['total_time_seconds']:.2f}s")
    print(f"Throughput:          {stats['throughput_rps']:.2f} req/s")
    print(f"Avg Latency:         {stats['avg_latency']:.3f}s")
    print(f"Min Latency:         {stats['min_latency']:.3f}s")
    print(f"Max Latency:         {stats['max_latency']:.3f}s")
    print(f"Total Tokens:        {stats['total_tokens']:,}")
    print(f"Tokens/Second:       {stats['tokens_per_second']:.2f}")
    print("=" * 60)


def _normalize_endpoint(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    if not url.startswith("http"):
        url = f"http://{url}"
    if url.endswith("/v1/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    if url.endswith("/"):
        return url + "v1/chat/completions"
    return url + "/v1/chat/completions"


async def main():
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible API Benchmark Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:9000/v1/chat/completions",
        help="API endpoint URL",
    )
    parser.add_argument(
        "--endpoints",
        default="",
        help="Comma-separated endpoints to round-robin (overrides --url)",
    )
    parser.add_argument(
        "--model",
        default="/mnt/model_weights",
        help="Model name/path",
    )
    parser.add_argument(
        "--input",
        default="/mnt/test_data/*.jsonl",
        help="Input file pattern",
    )
    parser.add_argument(
        "-f",
        "--file",
        default="",
        help="Single image file to test (overrides --input)",
    )
    parser.add_argument(
        "--output",
        default="/mnt/GUI/test_results.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--npus",
        type=int,
        default=8,
        help="Number of NPUs to utilize",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size for concurrent requests (0 = auto)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=0,
        help="Maximum concurrent requests (0 = auto: npus * concurrency_per_npu)",
    )
    parser.add_argument(
        "--concurrency-per-npu",
        type=int,
        default=8,
        help="Concurrency per NPU when max-concurrent=0",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0,
        help="Rate limit (requests per second), 0 = unlimited",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Initial retry delay in seconds",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of requests (0 = all)",
    )
    parser.add_argument(
        "--base-dir",
        default="/mnt/test_data",
        help="Base directory for resolving relative image paths",
    )
    
    args = parser.parse_args()
    
    # Load test data
    if args.file:
        if not os.path.isfile(args.file):
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        items = [{
            "id": str(uuid.uuid4())[:8],
            "data": {
                "image_path": args.file,
                "text": "Describe the image.",
            },
            "source_file": args.file,
            "line_number": 1,
        }]
        logger.info(f"Loaded single file: {args.file}")
    else:
        items = load_jsonl_files(args.input)
        if not items:
            logger.error("No test data loaded. Exiting.")
            sys.exit(1)
    
    if args.limit > 0:
        items = items[:args.limit]
        logger.info(f"Limited to {args.limit} items")
    
    # Configure
    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    endpoints = [_normalize_endpoint(e) for e in endpoints]
    url = _normalize_endpoint(args.url)

    req_config = RequestConfig(
        url=url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
    
    bench_config = BenchmarkConfig(
        num_npus=args.npus,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        concurrency_per_npu=args.concurrency_per_npu,
        rate_limit=args.rate_limit,
        output_path=args.output,
        input_pattern=args.input,
        base_dir=args.base_dir,
        endpoints=endpoints or None,
    )
    
    # Run benchmark
    try:
        results, stats = await run_benchmark(items, req_config, bench_config)
        
        # Write results
        write_results_to_jsonl(results, args.output)
        
        # Print statistics
        print_statistics(stats)
        
        # Exit with error code if any requests failed
        if stats["failed"] > 0:
            logger.warning(f"{stats['failed']} requests failed")
            if stats['success_rate'] < 90:
                sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Benchmark failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 10) Example Benchmark Output

`/mnt/GUI/test_results.jsonl` is JSON Lines. Example (first line):

```json
{"id": "c7091897", "request_id": "c7091897", "success": true, "response": "The image shows a close-up of a hand holding a small, white, rectangular object that appears to be a piece of paper or a card. The hand is positioned in the foreground, with the fingers gently gripping the object. The background is blurred, suggesting a shallow depth of field, which draws attention to the hand and the object it holds. The lighting is soft, and the overall tone of the image is neutral, with no strong colors or contrasts. The focus is on the interaction between the hand and the object, conveying a sense of care or attention. There are no other discernible elements or context provided in the image.", "error": null, "latency": 10.374, "tokens_used": 139, "prompt_tokens": 12, "completion_tokens": 127, "retry_count": 0}
```
