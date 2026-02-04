---
author: Robin
pubDatetime: 2026-02-03T22:00:00-03:00
title: "Building a Multimodal Agent on Huawei Cloud Ascend: Qwen3-VL + LangChain"
description: "End-to-end guide to deploying Qwen3-VL multimodal inference server on Huawei Cloud Ascend 910B with OpenAI-compatible API, and building practical multimodal agents using LangChain with caching strategies."
tags:
  - huawei-cloud
  - ascend-910b
  - qwen3-vl
  - langchain
  - multimodal
  - agent
  - openai-api
  - docker
featured: true
draft: false
---

# Building a Multimodal Agent on Huawei Cloud Ascend: Qwen3-VL OpenAI-Compatible Server + LangChain Tooling

This post walks through an end-to-end, reproducible setup:

- Deploy a **Qwen3-VL** multimodal inference server on **Huawei Cloud Ascend (910B)** using a Docker-based development environment.
- Expose an **OpenAI-compatible API** (`/v1/models`, `/v1/chat/completions`) for downstream integrations.
- Build a **multimodal agent** with **LangChain**, and then make it practical with a simple but effective cache:
  `hash(image) -> sqlite/redis -> structured analysis -> recommendation`.

---

## 1. Why an OpenAI-Compatible API on Ascend

Most application ecosystems (LangChain, OpenAI SDKs, internal orchestration services) already speak “OpenAI-style” chat endpoints. If the model server looks like OpenAI:

- you can reuse existing clients,
- you can plug it into agent frameworks quickly,
- you avoid custom protocol drift.

On Ascend, the key requirement is: **run inference inside an environment that provides the Ascend runtime + torch-npu stack**. In practice, that’s typically a prebuilt container image.

---

## 2. High-Level Architecture

- **Server (inside Ascend container)**
  - Loads Qwen3-VL weights from a mounted host path.
  - Runs a FastAPI app exposing OpenAI-style endpoints.
  - Uses NPU device via `torch_npu` (when available).

- **Client / Agent (on host)**
  - Calls the server’s OpenAI-compatible endpoints.
  - Optional: LangChain for tool wiring and “agent-like” orchestration.

---

## 3. Server Deployment in Docker (Ascend Environment)

### 3.1 Prerequisites

- Ascend 910B host with NPU driver installed.
- Docker installed on the host.
- Qwen3-VL weights accessible on the host filesystem.
- A container image that provides Ascend runtime/toolkit (example image name omitted here; use your internal approved image).

### 3.2 Start the Container + Server

In this repo we use a launcher script that:

- starts the container with required `/dev/davinci*` devices mounted,
- mounts model weights and code into the container,
- starts the server process in the container.

Key points:

- **Use only 2 NPUs** for the server when you need to share the machine:
  - set `ASCEND_RT_VISIBLE_DEVICES=0,1`
  - and mount only `/dev/davinci0` and `/dev/davinci1`.

- In some environments, `torch.npu.is_available()` may be false unless the container has additional privileges. We used:
  - `--privileged`
  - `--ipc=host`

These can be over-permissive for production; tighten them once you confirm what your runtime actually needs.

### 3.3 Health Checks

After startup:

```bash
curl -s http://127.0.0.1:9000/health
curl -s http://127.0.0.1:9000/v1/models
```

A working server should return `{"status":"ok"}` on `/health` and a model list on `/v1/models`.

---

## 4. Troubleshooting: CPU Fallback vs NPU

A common failure mode during early bring-up is:

- the server starts,
- but inference is accidentally running on CPU,
- and you hit errors like missing NPU ops (e.g. RMSNorm kernels) because tensors are on the wrong device.

What to verify:

- The container sees NPUs (`torch.npu.is_available()` true).
- You set `DEVICE=npu` (or auto-detect is correct).
- You restrict visible NPUs via `ASCEND_RT_VISIBLE_DEVICES`.

Once the server is confirmed to run on NPU, baseline requests should succeed.

---

## 5. Building the Multimodal Agent (LangChain)

### 5.1 Why LangChain Here

For a simple “describe an image” request, calling `/v1/chat/completions` directly is fastest.

LangChain becomes useful when you want:

- tool wiring (catalog search, product retrieval, rules),
- multi-step orchestration,
- structured composition of different backends.

### 5.2 A Critical Reliability Fix: Force Image Analysis First

A ReAct-style agent can answer before calling tools. With non-zero sampling temperature, this can produce occasional wrong early guesses.

We fixed this by:

- **forcing an image analysis tool call first**,
- then feeding the analysis result to the agent for recommendation.

This eliminates the agent “guessing the image” without looking.

---

## 6. Adding a Practical Cache: `hash(image) -> sqlite -> analysis -> recommendation`

### 6.1 Why Cache Matters

Multimodal inference is expensive. In many ecommerce workflows:

- the same product image is processed repeatedly,
- users ask follow-up questions,
- systems re-run analysis for every downstream step.

A cache gives you:

- lower latency,
- higher throughput,
- lower NPU cost.

### 6.2 Cache Design

We implemented a minimal SQLite cache:

- Compute SHA-256 of the image file.
- Store the LLM’s analysis text keyed by the hash.
- On repeated requests, return cached analysis immediately.

This pattern generalizes cleanly to Redis:

- key: `sha256(image_bytes)`
- value: JSON (`brand`, `category`, `attributes`, `language`, `created_at`)

### 6.3 Handling Language / Cache Content

One subtle issue: **whatever language the first analysis is generated in is what gets cached**.

If you switched prompts from one language to another, you must also:

- separate caches by “analysis prompt version / language”, or
- clear old cache entries.

In this demo we cleared the cache table via SQLite DDL (portable even when `rm` is restricted).

---

## 7. Measuring Latency

To make performance visible, we added a timing log around image analysis:

- `Image analysis took X.XXs`
- `Image analysis failed after X.XXs`

This is essential when comparing:

- “direct API call” vs “agent pipeline”,
- cached vs uncached runs.

---

## 8. How to Run the Agent Demo

From a host Python environment that can run LangChain:

```bash
python agent_chain_demo.py -f /path/to/image.webp
```

Repeat a run (and observe cache hits):

```bash
python agent_chain_demo.py -f /path/to/image.webp -i 2
```

---

## 9. Next Steps (Production Hardening)

- Convert analysis output to a stable schema:
  - `{brand, category, attributes, confidence}`.
- Replace SQLite with Redis for multi-process / multi-host caching.
- Put the agent into a deterministic workflow:
  - 1 tool call for analysis, 1 tool call for retrieval, 1 final response.
- Add request IDs and structured logs end-to-end.

---

## Appendix: `agent_chain_demo.py`

```python
#!/usr/bin/env python3
"""
LangChain Agent Full Call Chain Demo
Shows: User → Agent → LLM → Tool → Response
Supports image input: -f /path/to/image.jpg
"""

import argparse
import base64
import io
import os
import sys
import json
import time
import warnings
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from PIL import Image

# Suppress warnings for cleaner demo output (even if libraries override filters)
warnings.simplefilter("ignore")
warnings.showwarning = lambda *args, **kwargs: None

# Environment variables
os.environ["VISION_MODEL_URL"] = "http://127.0.0.1:9000/v1"
os.environ["VISION_API_KEY"] = "EMPTY"
os.environ["VISION_MODEL_NAME"] = "/mnt/model_weights"

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
    from langchain.agents import initialize_agent, Tool, AgentType
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import AgentAction, AgentFinish
    from langchain.callbacks.base import BaseCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LangChain import failed: {e}")
    print("Install: pip install 'langchain<0.2.0' 'langchain-openai<0.2.0'")
    LANGCHAIN_AVAILABLE = False
    sys.exit(1)


# ============================================
# Image utilities
# ============================================

def encode_image_to_base64(image_path: str, max_pixels: int = 2621440) -> str:
    """Encode image to base64."""
    img = Image.open(image_path).convert("RGB")
    
    # Downscale if too large
    if img.width * img.height > max_pixels:
        img = img.copy()
        img.thumbnail((int(max_pixels**0.5),) * 2)
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================
# Custom callback handler for visualizing the call chain
# ============================================

class CallChainVisualizer(BaseCallbackHandler):
    """Visualize LangChain call chain."""
    
    def __init__(self):
        self.step = 0
        self.indent = "  "
    
    def _print_step(self, title: str, content: str = "", emoji: str = "▶️"):
        """Print step info."""
        self.step += 1
        print(f"\n{emoji} Step {self.step}: {title}")
        if content:
            for line in content.split('\n'):
                print(f"{self.indent}{line}")
    
    def on_agent_action(self, action: AgentAction, **kwargs):
        """Triggered when the agent decides to act."""
        self._print_step(
            "Agent decides to call a tool",
            f"Tool: {action.tool}\n"
            f"Input: {action.tool_input}",
            "🔧"
        )
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Triggered when tool starts."""
        tool_name = serialized.get("name", "Unknown")
        self._print_step(
            f"Tool start: {tool_name}",
            f"Input: {input_str}",
            "🛠️"
        )
    
    def on_tool_end(self, output: str, **kwargs):
        """Triggered when tool ends."""
        self._print_step(
            "Tool output",
            f"Output: {output[:200]}..." if len(output) > 200 else f"Output: {output}",
            "✅"
        )
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Triggered when LLM starts."""
        model = serialized.get("name", "Unknown")
        self._print_step(
            f"Calling LLM: {model}",
            f"Prompt length: {len(prompts[0]) if prompts else 0} chars",
            "🤖"
        )
    
    def on_llm_end(self, response, **kwargs):
        """Triggered when LLM ends."""
        content = response.generations[0][0].text if response.generations else ""
        self._print_step(
            "LLM output",
            f"Content: {content[:200]}..." if len(content) > 200 else f"Content: {content}",
            "💬"
        )
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        """Triggered when agent finishes."""
        self._print_step(
            "Agent final answer",
            f"Answer: {finish.return_values.get('output', '')[:200]}...",
            "🎯"
        )


# ============================================
# Product catalog tool
# ============================================

class ProductCatalogTool:
    """Product catalog search tool."""
    
    def __init__(self, catalog_path: str = "/mnt/vision_agents/products.json"):
        self.catalog_path = Path(catalog_path)
        self.products = self._load_catalog()
    
    def _load_catalog(self) -> List[Dict]:
        """Load product catalog."""
        if not self.catalog_path.exists():
            return [
                {"id": "1", "name": "Nike Air Max 90", "brand": "Nike", "category": "Shoes", "price": 120},
                {"id": "2", "name": "iPhone 14 Pro", "brand": "Apple", "category": "Electronics", "price": 999},
                {"id": "3", "name": "Acer Aspire 5", "brand": "Acer", "category": "Electronics", "price": 599},
                {"id": "4", "name": "Sony WH-1000XM5", "brand": "Sony", "category": "Electronics", "price": 399},
                {"id": "5", "name": "Adidas Ultraboost", "brand": "Adidas", "category": "Shoes", "price": 180},
            ]
        try:
            with open(self.catalog_path, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def search(self, query: str) -> str:
        """Search products."""
        query_lower = query.lower()
        results = []
        
        for product in self.products:
            score = 0
            if query_lower in product.get('name', '').lower():
                score += 3
            if query_lower in product.get('brand', '').lower():
                score += 2
            if query_lower in product.get('category', '').lower():
                score += 1
            
            if score > 0:
                results.append((score, product))
        
        results.sort(reverse=True, key=lambda x: x[0])
        
        if not results:
            return f"No products found for '{query}'"
        
        output = f"Found {len(results)} related products:\n"
        for _, prod in results[:3]:
            output += f"- {prod['name']} ({prod['brand']}): ${prod['price']}\n"
        
        return output


# ============================================
# Image analysis tool
# ============================================

class ImageAnalysisTool:
    """Image analysis tool - calls Qwen3-VL."""
    
    def __init__(self, cache_db: str = "/mnt/vision_agents/langchain_multimodal/image_cache.sqlite"):
        self.endpoint = "http://127.0.0.1:9000/v1/chat/completions"
        self.model = "/mnt/model_weights"
        self.cache_db = cache_db
        self._init_cache()

    def _init_cache(self) -> None:
        Path(self.cache_db).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS image_cache ("
                "hash TEXT PRIMARY KEY, "
                "analysis TEXT NOT NULL, "
                "created_at INTEGER NOT NULL)"
            )
            conn.commit()

    def _get_cache(self, image_hash: str) -> Optional[str]:
        with sqlite3.connect(self.cache_db) as conn:
            cur = conn.execute(
                "SELECT analysis FROM image_cache WHERE hash = ?",
                (image_hash,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def _set_cache(self, image_hash: str, analysis: str) -> None:
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO image_cache (hash, analysis, created_at) VALUES (?, ?, ?)",
                (image_hash, analysis, int(time.time())),
            )
            conn.commit()

    def _hash_file(self, image_path: str) -> str:
        h = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    
    def analyze(self, image_path: str) -> str:
        """Analyze image."""
        import requests
        
        start_time = time.time()
        if not Path(image_path).is_file():
            return f"Analysis failed: image not found {image_path}"

        # Cache lookup by image hash
        image_hash = self._hash_file(image_path)
        cached = self._get_cache(image_hash)
        if cached:
            print("[Cache] Image analysis cache hit")
            elapsed = time.time() - start_time
            print(f"[Timing] Image analysis took {elapsed:.2f}s")
            return cached

        # Read image
        image_b64 = encode_image_to_base64(image_path)
        
        # Build request
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the product in this image. Identify brand, category, and key features. Keep it concise."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }],
            "max_tokens": 512,
            "temperature": 0.2
        }
        
        try:
            resp = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120
            )
            data = resp.json()
            analysis = data["choices"][0]["message"]["content"]
            self._set_cache(image_hash, analysis)
            elapsed = time.time() - start_time
            print(f"[Timing] Image analysis took {elapsed:.2f}s")
            return analysis
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[Timing] Image analysis failed after {elapsed:.2f}s")
            return f"Analysis failed: {e}"


# ============================================
# Agent setup
# ============================================

def create_agent_with_visualization():
    """Create agent with visualization callbacks."""
    
    print("=" * 80)
    print("Initializing LangChain Agent")
    print("=" * 80)
    
    # 1. Initialize LLM
    print("\n[1/4] Initializing LLM (ChatOpenAI)...")
    llm = ChatOpenAI(
        model="/mnt/model_weights",
        temperature=0.2,
        max_tokens=2000,
        openai_api_key="EMPTY",
        openai_api_base="http://127.0.0.1:9000/v1",
    )
    print("  ✓ LLM initialized")
    
    # 2. Initialize tools
    print("\n[2/4] Initializing tools...")
    catalog_tool = ProductCatalogTool()
    image_tool = ImageAnalysisTool()
    
    tools = [
        Tool(
            name="ProductSearch",
            func=catalog_tool.search,
            description="Search product catalog by brand or category. Input: brand or category, e.g. 'Nike' or 'Electronics'"
        ),
        Tool(
            name="ImageAnalysis",
            func=image_tool.analyze,
            description="Analyze image content and identify product brand and features. Input: full image path, e.g. '/mnt/test_data/pictures/test1.webp'"
        ),
    ]
    print(f"  ✓ Tools initialized: {[t.name for t in tools]}")
    
    # 3. Initialize memory
    print("\n[3/4] Initializing memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    print("  ✓ Memory initialized")
    
    # 4. Create agent
    print("\n[4/4] Creating agent (initialize_agent)...")
    
    # Visualization callbacks
    visualizer = CallChainVisualizer()
    
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        callbacks=[visualizer],
        max_iterations=10,
    )
    print("  ✓ Agent created")
    
    print("\n" + "=" * 80)
    print("Agent ready!")
    print("=" * 80)
    
    return agent_executor


# ============================================
# Demo scenarios
# ============================================

def demo_with_image(agent_executor, image_path: str):
    """Analyze image and recommend similar products."""
    print("\n" + "=" * 80)
    print("Scenario: Analyze image and recommend similar products")
    print("=" * 80)
    print(f"Image: {image_path}")
    print("-" * 80)
    print("User: Analyze the product in this image and recommend similar products")
    
    if not Path(image_path).exists():
        print(f"\nError: image not found: {image_path}")
        return
    
    try:
        # Force image analysis first
        image_tool = ImageAnalysisTool()
        analysis = image_tool.analyze(image_path)
        print("\n[Image Analysis Result]")
        print(analysis)

        # Recommend based on analysis (avoid agent guessing first)
        query = (
            "Based on the following image analysis, recommend similar products.\n"
            f"Image analysis: {analysis}"
        )
        result = agent_executor.invoke({"input": query})
        print("\nFinal answer:")
        print(result.get("output", result))
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


def demo_simple_chat(agent_executor):
    """Simple chat."""
    print("\n" + "=" * 80)
    print("Scenario: Simple chat (direct response, no tools)")
    print("=" * 80)
    print("User: Hi, what can you do?")
    print("-" * 80)
    
    try:
        result = agent_executor.invoke({"input": "Hi, what can you do?"})
        print("\nFinal answer:")
        print(result.get("output", result))
    except Exception as e:
        print(f"\nError: {e}")


def demo_tool_usage(agent_executor):
    """Use tools (search products)."""
    print("\n" + "=" * 80)
    print("Scenario: Use tool (search products)")
    print("=" * 80)
    print("User: Find Nike products")
    print("-" * 80)
    
    try:
        result = agent_executor.invoke({"input": "Find Nike products"})
        print("\nFinal answer:")
        print(result.get("output", result))
    except Exception as e:
        print(f"\nError: {e}")


# ============================================
# Main
# ============================================

def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description="LangChain Agent Full Call Chain Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze image and recommend similar products
  python agent_chain_demo.py -f /mnt/test_data/pictures/test1.webp

  # Run all demo scenarios
  python agent_chain_demo.py --all

  # Simple chat
  python agent_chain_demo.py --chat
        """
    )
    
    parser.add_argument("-f", "--file", help="Image path, e.g. /mnt/test_data/pictures/test1.webp")
    parser.add_argument("-i", "--iterations", type=int, default=1, help="Repeat count for image analysis, default 1")
    parser.add_argument("--all", action="store_true", help="Run all demo scenarios")
    parser.add_argument("--chat", action="store_true", help="Simple chat mode")
    parser.add_argument("--search", help="Search products, e.g. Nike")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           LangChain Agent Full Call Chain Demo                               ║
║   Shows: User → Agent → LLM → Tool → Response                                ║
║   Supports image input and tool calls                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    if not LANGCHAIN_AVAILABLE:
        print("Please install: pip install 'langchain<0.2.0' 'langchain-openai<0.2.0'")
        return
    
    # Create agent
    try:
        agent_executor = create_agent_with_visualization()
    except Exception as e:
        print(f"Failed to create agent: {e}")
        return
    
    # Run scenarios based on args
    if args.file:
        # Image analysis mode
        iterations = max(1, args.iterations)
        for i in range(iterations):
            if iterations > 1:
                print(f"\n==== Run {i + 1}/{iterations} ====")
            demo_with_image(agent_executor, args.file)
    elif args.search:
        # Search mode
        print("\n" + "=" * 80)
        print(f"Search: {args.search}")
        print("=" * 80)
        try:
            result = agent_executor.invoke({"input": f"Find {args.search} products"})
            print(f"\nResult: {result.get('output', result)}")
        except Exception as e:
            print(f"\nError: {e}")
    elif args.chat:
        # Simple chat
        demo_simple_chat(agent_executor)
    elif args.all:
        # Run all scenarios
        print("\n" + "=" * 80)
        print("Starting demo scenarios")
        print("=" * 80)
        
        demo_simple_chat(agent_executor)
        time.sleep(1)
        
        demo_tool_usage(agent_executor)
        time.sleep(1)
        
        # If default image exists, run image demo
        default_image = "/mnt/test_data/pictures/test1.webp"
        if Path(default_image).exists():
            demo_with_image(agent_executor, default_image)
    else:
        # Default: show help
        parser.print_help()
        print("\nTip: use -f to specify image path")
        print("  Example: python agent_chain_demo.py -f /mnt/test_data/pictures/test1.webp")


if __name__ == "__main__":
    main()

```
