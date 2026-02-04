---
author: Robin
pubDatetime: 2026-02-04T16:35:00-03:00
title: "Huawei OmniInfer PD Disaggregation on Ascend: Qwen3-VL Single-Node Deployment (1P1D)"
description: "Practical guide to deploying Qwen3-VL with Prefill-Decode separation on Ascend NPUs using OmniInfer. Covers minimal 1P1D setup, KV cache transfer, troubleshooting, and complete working scripts."
tags:
  - huawei-cloud
  - ascend
  - omniinfer
  - qwen3-vl
  - vllm
  - prefill-decode
  - pd-separation
  - multimodal
  - inference-optimization
  - production-deployment
featured: true
draft: false
---

# Huawei OmniInfer PD Disaggregation on Ascend (Single Node, 1P1D) for Qwen3-VL

This post documents a practical, reproducible way to run **Prefill/Decode (PD) disaggregated inference** on a single Ascend host using the open-source **OmniInfer** project.

The goal is to expose an OpenAI-compatible API endpoint for **Qwen3-VL** (image + text) with **PD separation**:

- **Prefill node (P)**: vision encoder + prompt prefill
- **Decode node (D)**: autoregressive text generation

We keep the setup minimal (no Ansible, no multi-node), and reuse an existing Ascend vLLM container image.

---

## Why OmniInfer for PD

Native Ascend PD separation can be fragile in plain vLLM deployments (timeouts / engine-dead failures under certain workloads). OmniInfer provides a PD-oriented orchestration layer and scripts that generate ranktables and start Prefill/Decode services with KV-transfer configured.

In this guide we use OmniInfer’s **PD single-node script path** (`run_model_qwen.py` → `start_api_servers.py`) and run it inside an Ascend vLLM container.

---

## Environment

- OS: Linux (aarch64)
- NPUs: Ascend (single host)
- Model weights (HF format): mounted under `/mnt/model_weights`
- Test data: under `/mnt/test_data`

> Note: All paths in this post use `/mnt/...` to keep the examples generic.

---

## Step 1: Get OmniInfer Source

```bash
cd /mnt/GUI
git clone https://gitee.com/omniai/omniinfer.git
```

We only need the repo for its PD scripts and adaptor modules.

---

## Step 2: Choose the Runtime Container Image

We reuse an existing Ascend vLLM image already present on the host.

Verify images:

```bash
docker images | head
```

In this setup, we use `swr.cn-southwest-2.myhuaweicloud.com/ei_ascendcloud_devops/llm_inference` as the runtime image (vllm included).

---

## Step 3: PD Topology (1P1D, Minimal NPUs)

To avoid occupying the entire machine, we run:

- Prefill on **NPU 0**
- Decode on **NPU 1**

Ports:

- Prefill OpenAI API: `6660`
- Decode OpenAI API: `6760`

We intentionally **skip the global proxy** and call the prefill endpoint directly during validation.

---

## Step 4: Start the PD Services (Containerized)

We use a single host-side script to:

1. Start a container with required Ascend device mounts
2. Mount:
   - Model: `/mnt/model_weights` → `/data/models/qwen3-vl-8b`
   - OmniInfer source: `/mnt/GUI/omniinfer` → `/workspace/omniinfer`
   - Logs: `/mnt/GUI/omniinfer_logs` → `/workspace/omniinfer/tools/scripts/apiserverlog`
3. Run OmniInfer’s PD launcher (`run_model_qwen.py`) with `pd_separate`

Start it:

```bash
cd /mnt/GUI
./run_omniinfer_pd_container.sh
```

### Verify ports

```bash
docker exec -u root omniinfer_pd_qwen3_vl sh -c 'ss -lntp | egrep ":6660|:6760" || true'
```

Expected: `6660` and `6760` should be in LISTEN state.

---

## Step 5: Smoke Test with an Image

### Important: Model ID

In this setup the served model name is **`qwen3-vl-8b`**.

Your client must send `"model": "qwen3-vl-8b"`.

### Benchmark client (single image)

```bash
export VLLM_MODEL_NAME=qwen3-vl-8b
python /mnt/GUI/benchmark_client.py   -f /mnt/test_data/pictures/test1.webp   --url http://127.0.0.1:6660/v1/chat/completions
```

If the backend is working, you should see HTTP `200 OK`.

---

## Step 6: Attribute Extraction Eval (Limit 2)

```bash
export VLLM_MODEL_NAME=qwen3-vl-8b
python /mnt/GUI/attribute_extraction_eval.py   --input /mnt/test_data/attribute-extraction.jsonl   --base-url http://127.0.0.1:6660   --concurrency 2   --temperature 0.0   --max-tokens 512   --report-top-keys 20   --min-key-occ 5   --limit 2
```

---

## Notes / Practical Learnings

- **Don’t assume a global proxy exists.** Many OmniInfer proxy scripts expect an Nginx layout in `/usr/local/nginx`. If the image doesn’t include it, proxy startup will fail. For early validation, calling Prefill (`6660`) directly is simpler.
- **KV transfer connector differences matter.** Some vLLM builds do not include OmniInfer’s `AscendHcclConnectorV1`. In that case, a portable workaround is to use vLLM’s built-in `SharedStorageConnector` (local path KV handoff) for PD.
- **Always pass the correct model id.** For `vllm serve --served-model-name qwen3-vl-8b`, clients must send `model=qwen3-vl-8b`.

---

# Appendix: Files

Below are the exact scripts used in this walkthrough (paths adjusted to `/mnt/...` for documentation).

## `run_omniinfer_pd_container.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# OmniInfer PD (1P1D) local deployment using existing vllm-npu image
# Prefill: NPU 0-3, Decode: NPU 4-7, Proxy: port 9000

IMAGE=${IMAGE:-dab8b2fea9fa}
CONTAINER_NAME=${CONTAINER_NAME:-omniinfer_pd_qwen3_vl}
MODEL_PATH=${MODEL_PATH:-/mnt/model_weights}
CONTAINER_MODEL_PATH=${CONTAINER_MODEL_PATH:-/data/models/qwen3-vl-8b}
CODE_PATH=${CODE_PATH:-/mnt/GUI/omniinfer}
LOG_PATH=${LOG_PATH:-/mnt/GUI/omniinfer_logs}

PREFILL_DEVICES=${PREFILL_DEVICES:-0}
DECODE_DEVICES=${DECODE_DEVICES:-1}

SERVICE_PORT=${SERVICE_PORT:-6660}
HTTPS_PORT=${HTTPS_PORT:-9000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}

# Auto-detect host interface/IP for OmniInfer
if [ -z "${NET_IFACE:-}" ]; then
  NET_IFACE="$(ip -4 -o addr show scope global | awk 'NR==1{print $2}' | sed 's/:$//')"
fi
if [ -z "${HOST_IP:-}" ] && [ -n "${NET_IFACE}" ]; then
  HOST_IP="$(ip -4 addr show dev "${NET_IFACE}" | awk '/inet /{print $2}' | cut -d/ -f1 | head -n1)"
fi
HOST_IP=${HOST_IP:-127.0.0.1}

mkdir -p "${LOG_PATH}"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# Build device args
DEVICE_ARGS=""
IFS=',' read -ra PDEV <<< "${PREFILL_DEVICES}"
IFS=',' read -ra DDEV <<< "${DECODE_DEVICES}"
ALL_DEV=("${PDEV[@]}" "${DDEV[@]}")
for d in "${ALL_DEV[@]}"; do
  DEVICE_ARGS+=" --device=/dev/davinci${d}"
done

# Run container

docker run -itd \
  --name "${CONTAINER_NAME}" \
  --net=host \
  --privileged \
  --ipc=host \
  -u root \
  ${DEVICE_ARGS} \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/log/npu/:/usr/slog \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  -v "${MODEL_PATH}:${CONTAINER_MODEL_PATH}" \
  -v "${CODE_PATH}:/workspace/omniinfer" \
  -v "${LOG_PATH}:/workspace/omniinfer/tools/scripts/apiserverlog" \
  "${IMAGE}" \
  /bin/bash -lc "set -e; \
    mkdir -p /tmp/omniinfer_pd /workspace/omniinfer/tools/scripts/apiserverlog; \
    cd /tmp/omniinfer_pd; \
    export PATH=/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin:\${PATH}; \
    export PYTHONPATH=/workspace/omniinfer:\${PYTHONPATH}; \
    export VLLM_PLUGINS=ascend_vllm; \
    export VLLM_USE_V1=1; \
    export HCCL_OP_EXPANSION_MODE=AIV; \
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True; \
    OMNI_SKIP_PROXY=1 python /workspace/omniinfer/tools/scripts/run_model_qwen.py \
      --model-path ${CONTAINER_MODEL_PATH} \
      --deploy-mode pd_separate \
      --graph-true 'false' \
      --model-name qwen3-vl-8b \
      --network-interface ${NET_IFACE} \
      --host-ip ${HOST_IP} \
      --prefill-server-list ${PREFILL_DEVICES} \
      --decode-server-list ${DECODE_DEVICES} \
      --service-port ${SERVICE_PORT} \
      --https-port ${HTTPS_PORT} \
      --max-model-len ${MAX_MODEL_LEN} \
      --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
      --max-num-seqs ${MAX_NUM_SEQS} \
      --log-path /workspace/omniinfer/tools/scripts/apiserverlog; \
    tail -f /dev/null"

echo "[INFO] OmniInfer PD started in container ${CONTAINER_NAME}"

```

## `omniinfer/tools/scripts/run_model_qwen.py`

```python
# import 
import os
import sys
import time
import json
import fcntl
import socket
import struct
import argparse
import warnings
import subprocess

def get_path_before_omniinfer():
    """Get the base path before the 'omniinfer' directory in the current script's path.
    
    Returns:
        str: The path segment before 'omniinfer' directory.
    Raises:
        ValueError: If 'omniinfer' directory is not found in the path.
    """
    # Get absolute path of the currently executing script
    script_path = os.path.abspath(sys.argv[0])
    
    # Split path into components using OS-specific separator
    path_parts = script_path.split(os.sep)
    
    # Find the index of 'omniinfer' in the path components
    try:
        omni_index = path_parts.index('omniinfer')
    except ValueError:
        raise ValueError("'omniinfer' directory not found in path")
    
    # Reconstruct path up to (but not including) 'omniinfer'
    before_omni = os.sep.join(path_parts[:omni_index])
    
    return before_omni

def get_network_interfaces():
    """
    Retrieves primary network interface information excluding loopback.
    Returns a dictionary with interface name and its IP address.
    Falls back to 'eth0' if no interfaces found.
    """
    # List all network interfaces except loopback (lo)
    if_names = [name for name in os.listdir('/sys/class/net') if name != 'lo']
    
    # Select first available interface or default to 'eth0'
    if_name = if_names[0] if if_names else 'eth0'

    try:
        # Get IP address for selected interface
        ip = get_ip_address(if_name)

        # Compose result dictionary
        interfaces = {
            'if_name': if_name,  # Network interface name
            'ip': ip             # IPv4 address of the interface
        }   
    except Exception as e:
        print(f"Error getting network interfaces: {if_name}:{e}")
        interfaces = {}  # Return empty dict on error
    
    return interfaces

def get_ip_address(if_name):
    """
    Retrieves the IPv4 address of a network interface using ioctl.
    Args:
        if_name: Name of the network interface (e.g., 'eth0')
    Returns:
        IPv4 address as string
    Raises:
        RuntimeError on failure
    """
    # Create UDP socket for ioctl operations
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # SIOCGIFADDR = 0x8915 (get interface address)
        # Pack interface name into byte structure (max 15 chars)
        packed_ifname = struct.pack('256s', if_name[:15].encode('utf-8'))
        
        # Perform ioctl call to get interface info
        # [20:24] slices the IP address from the returned structure
        ip_bytes = fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR constant
            packed_ifname
        )[20:24]
        
        # Convert packed binary IP to dotted-quad string
        return socket.inet_ntoa(ip_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to get IP address for interface {if_name}: {e}")


def run_default_mode(args):
    """Run in mixed deployment mode"""

    if (args.network_interface is not None and args.host_ip is None) or \
        (args.network_interface is None and args.host_ip is not None):
        warnings.warn(
            "For best results, please specify both --network-interface AND --host-ip "
            "together. Falling back to auto-detection for missing values.",
            RuntimeWarning
        )
    # Get network interface
    if args.network_interface:
        intf = {'if_name': args.network_interface, 'ip': get_ip_address(args.network_interface)}
    else:
        intf = get_network_interfaces()
        if not intf:
            raise RuntimeError("No network interface found and none specified")

    # Override IP if host-ip was specified
    if args.host_ip:
        intf['ip'] = args.host_ip


    env = os.environ.copy()
    # Network config for distributed training
    env['GLOO_SOCKET_IFNAME'] = intf['if_name']
    env['TP_SOCKET_IFNAME'] = intf['if_name']

    # Hardware and framework settings
    env['ASCEND_RT_VISIBLE_DEVICES'] = args.server_list  # Use first 8 NPUs
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'fork'   # Process spawning method
    env['OMNI_USE_QWEN'] = '1'  # Enable custom model support
    env['VLLM_USE_V1'] = '1'
    env['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
    env['VLLM_LOGGING_LEVEL'] = 'INFO'

    env['HCCL_OP_EXPANSION_MODE'] = 'AIV'
    env['TNG_HOST_COPY'] = '1'
    env['TASK_QUEUE_ENABLE'] = '2'
    env['CPU_AFFINITY_CONF'] = '2'

    if args.graph_true.lower() == 'false':
          # Base command for API server
        cmd = [
            'python',  os.path.join(args.code_path, 'omniinfer/tools/scripts/start_api_servers.py'),
            '--num-servers', '1',
            '--model-path', args.model_path,
            '--master-ip', intf['ip'],       # Coordinator IP
            '--master-port', args.master_port,         # Coordinator port
            '--tp', str(len(args.server_list.split(','))) ,   # tensor parallelism
            '--served-model-name', args.model_name,
            '--base-api-port', args.https_port,        # HTTP service port
            '--log-dir', args.log_path,  # Log directory
            '--extra-args', '--enforce-eager '   # Disable graph execution
        ]

        if hasattr(args, 'additional_config') and args.additional_config:
            cmd.extend(['--additional-config', args.additional_config])

    # Graph mode specific optimizations
    elif args.graph_true.lower() == 'true':
        # Base command for API server
        additional_config = args.additional_config if args.additional_config else \
                '{"graph_model_compile_config": {"level":1, "use_ge_graph_cached":false, "block_num_floating_range":50}, "enable_hybrid_graph_mode": false}'
        cmd = [
            'python',  os.path.join(args.code_path,'omniinfer/tools/scripts/start_api_servers.py'),
            '--num-servers', '1',
            '--model-path', args.model_path,
            '--master-ip', intf['ip'],       # Coordinator IP
            '--master-port', args.master_port,    # Coordinator port
            '--tp', str(len(args.server_list.split(','))),     # tensor parallelism
            '--served-model-name', args.model_name,
            '--base-api-port', args.https_port,        # HTTP service port
            '--log-dir', args.log_path,  # Log directory
            '--gpu-util', '0.9',  # Target NPU utilization
            '--max-model-len', args.max_model_len,
            '--extra-args', f'--max-num-batched-tokens {args.max_num_batched_tokens} --max-num-seqs {args.max_num_seqs} ',
            '--additional-config', additional_config
        ]
    
    print(f'Starting with NIC: {intf["if_name"]}, IP: {intf["ip"]}')

    subprocess.run(cmd, env=env)


def set_common_env_vars(intf, env):
    """Set common environment variables for all servers"""
    
    # Ascend NPU library path
    env['PYTHONPATH'] = '/usr/local/Ascend:' + env.get('PYTHONPATH', '')
    
    # Network configuration for distributed communication
    env['LOCAL_DECODE_SERVER_IP_LIST'] = intf['ip']  # Local decoder IP
    env['GLOBAL_DECODE_SERVER_IP_LIST'] = intf['ip'] # Global decoder IP
    env['GLOO_SOCKET_IFNAME'] = intf['if_name']      # Gloo communication interface
    env['TP_SOCKET_IFNAME'] = intf['if_name']        # Tensor parallelism interface
    
    # Framework configuration
    env['VLLM_USE_V1'] = '1'                       # Use vLLM v1 API
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'fork'    # Process spawning method
    env['VLLM_LOGGING_LEVEL'] = 'INFO'             # Log verbosity level
    env['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
    
    # Custom model support
    env['OMNI_USE_QWEN'] = '1'  # Enable QWEN model optimizations

    # Pod configuration
    env['PREFILL_POD_NUM'] = '1'       # Prefill pod count
    env['DECODE_POD_NUM'] = '1'        # Decoder pod count



def start_perfill_api_servers(intf, args):
    """Start prefill API servers with specialized configuration"""
    ip = intf['ip']

    env = os.environ.copy()
    set_common_env_vars(intf, env)  # Apply common network settings
    
    # Specialized environment for prefill servers
    env['VLLM_LLMDATADIST_ZMQ_PORT'] = '5570'  # ZeroMQ port for data distribution
    env['ASCEND_RT_VISIBLE_DEVICES'] = args.prefill_server_list  # NPUs for prefill
    
    prefill_server_list_list = args.prefill_server_list.split(',')
    prefill_rank_table_suffix = ''.join(prefill_server_list_list)

    # Ranktable paths for distributed training
    env['RANK_TABLE_PATH'] = os.path.join(args.code_path, 'omniinfer/tools/scripts/global_path/')
    env['GLOBAL_RANK_TABLE_FILE_PATH'] = os.path.join(args.code_path, 'omniinfer/tools/scripts/global_path/global_ranktable_merge.json')
    env['RANK_TABLE_FILE_PATH'] = os.path.join(args.code_path, f'omniinfer/tools/scripts/perfill-ranktable/local_ranktable_{ip}_{prefill_rank_table_suffix}.json')
    env['ROLE'] = 'prefill'  # Server role identifier

    # HCCL communication settings
    env['HCCL_INTRA_ROCE_ENABLE'] = '1'  # Enable RoCE communication
    env['HCCL_INTRA_PCIE_ENABLE'] = '0'  # Disable PCIe communication
    env['HCCL_DETERMINISTIC'] = 'true'  # Enable deterministic behavior
    env['CLOSE_MATMUL_K_SHIFT'] = '1'  # Optimization flag
    env['HCCL_OP_EXPANSION_MODE'] = 'AIV'

    # Prefill-specific optimizations
    env['ENABLE_PREFILL_TND'] = '1'

    # KV transfer configuration for attention
    kv_transfer_config = {
        "kv_connector": "SharedStorageConnector",
        "kv_buffer_device": "npu",
        "kv_role": "kv_producer",  # Prefill produces KV cache
        "kv_rank": 0,
        "engine_id": "0",
        "kv_parallel_size": 1,
        "kv_connector_extra_config": {
            "shared_storage_path": "/tmp/vllm_kv",
            "prefill": {"dp_size": 1, "tp_size": int(len(args.prefill_server_list.split(',')))},
            "decode": {"dp_size": 1, "tp_size": int(len(args.decode_server_list.split(',')))},
        },
    }

    # Command to start prefill API servers
    cmd = [
        'python', os.path.join(args.code_path,'omniinfer/tools/scripts/start_api_servers.py'),
        '--num-servers', '1',
        '--model-path', args.model_path,
        '--master-ip', intf['ip'],  # Coordinator IP
        '--master-port', args.master_port,    # Coordinator port
        '--base-api-port', args.service_port,  # API service port
        '--tp', str(len(args.prefill_server_list.split(','))),                # 8-way tensor parallelism
        '--served-model-name', args.model_name,
        '--max-model-len', args.max_model_len, # Max context length
        '--log-dir', args.log_path + '/prefill/',  # Log directory
        '--no-enable-prefix-caching',  # Disable caching
        '--gpu-util', '0.9',        # Target NPU utilization
        '--extra-args', f'--max-num-batched-tokens {args.max_num_batched_tokens} --max-num-seqs {args.max_num_seqs} --enforce-eager',  # Perf mance flags
        '--kv-transfer-config', json.dumps(kv_transfer_config)  # KV transfer settings
    ]

    subprocess.Popen(cmd, env=env)  # Start as background process

def start_decoder_api_servers(intf, args):
    """Start decoder API servers with specialized configuration"""
    ip = intf['ip']

    env = os.environ.copy()
    set_common_env_vars(intf, env)  # Apply common network settings
    
    # Specialized environment for decoder servers
    env['VLLM_LLMDATADIST_ZMQ_PORT'] = '5569'  # Different ZeroMQ port
    env['ASCEND_RT_VISIBLE_DEVICES'] = args.decode_server_list  # Different NPU set
    
    deocde_server_list_list = args.decode_server_list.split(',')
    decode_rank_table_suffix = ''.join(deocde_server_list_list)

    # Ranktable paths for distributed training
    env['RANK_TABLE_PATH'] = os.path.join(args.code_path, 'omniinfer/tools/scripts/global_path/')
    env['GLOBAL_RANK_TABLE_FILE_PATH'] = os.path.join(args.code_path, 'omniinfer/tools/scripts/global_path/global_ranktable_merge.json')
    env['RANK_TABLE_FILE_PATH'] = os.path.join(args.code_path, f'omniinfer/tools/scripts/decode-ranktable/local_ranktable_{ip}_{decode_rank_table_suffix}.json')
    env['ROLE'] = 'decode'  # Server role identifier

    # Advanced HCCL settings
    env['HCCL_INTRA_ROCE_ENABLE'] = '1'  # Enable RoCE communication
    env['HCCL_INTRA_PCIE_ENABLE'] = '0'  # Disable PCIe communication
    env['HCCL_BUFFSIZE'] = '1000'       # Communication buffer size
    env['HCCL_OP_EXPANSION_MODE'] = 'AIV'  # Operation expansion mode
    env['VLLM_ENABLE_MC2'] = '1'        # Memory optimization

    # Debugging and profiling flags
    env['DUMP_GE_GRAPH'] = '2'
    env['DUMP_GRAPH_LEVEL'] = '3'

    # Decoder-specific optimizations
    env['DECODE_DP_SIZE'] = '1'         # Data parallelism size
    env['MOE_DISPATCH_COMBINE'] = '1'   # Mixture-of-Experts optimization
    env['HCCL_DETERMINISTIC'] = 'true'  # Enable deterministic behavior
    env['CLOSE_MATMUL_K_SHIFT'] = '1'   # Optimization flag

    
    # Server offset handling
    try:
        server_offset = env['SERVER_OFFSET']
    except KeyError:
        server_offset = '0'

    # KV transfer configuration for attention
    kv_transfer_config = {
        "kv_connector": "SharedStorageConnector",
        "kv_buffer_device": "npu",
        "kv_role": "kv_consumer",  # Decoder consumes KV cache
        "kv_rank": 1,
        "engine_id": "0",
        "kv_parallel_size": 1,
        "kv_connector_extra_config": {
            "shared_storage_path": "/tmp/vllm_kv",
            "prefill": {"dp_size": 1, "tp_size": int(len(args.prefill_server_list.split(',')))},
            "decode": {"dp_size": 1, "tp_size": int(len(args.decode_server_list.split(',')))},
        },
    }

    if args.graph_true.lower() == 'false':
          # Base command for API server
        # Command to start decoder API servers
        cmd = [
            'python', os.path.join(args.code_path,'omniinfer/tools/scripts/start_api_servers.py'),
            '--num-servers', '1',
            '--server-offset', server_offset,  # Server offset parameter
            '--num-dp', env['DECODE_DP_SIZE'],  # Data parallelism degree
            '--model-path', args.model_path,
            '--master-ip', intf['ip'],      # Coordinator IP
            '--master-port', args.master_port,        # Coordinator port
            '--base-api-port', str(int(args.service_port) + 100),      # API service port
            '--tp', str(len(args.decode_server_list.split(','))),                    # 8-way tensor parallelism
            '--served-model-name', args.model_name,
            '--max-model-len', args.max_model_len,    # Max context length
            '--log-dir', args.log_path + '/decode/',  # Log directory
            '--no-enable-prefix-caching',  # Disable caching
            '--extra-args', f'--max-num-batched-tokens {args.max_num_batched_tokens} --max-num-seqs {args.max_num_seqs}',  # Performance flag
            '--kv-transfer-config', json.dumps(kv_transfer_config)  # KV transfer settings
        ]

        if hasattr(args, 'additional_config') and args.additional_config:
            cmd.extend(['--additional-config', args.additional_config])

    # Graph mode specific optimizations
    elif args.graph_true.lower() == 'true':
        additional_config = args.additional_config if args.additional_config else \
                '{"graph_model_compile_config":{"level":1,"use_ge_graph_cached":false, "block_num_floating_range":50}, "decode_gear_list": [64]}'
        # Command to start decoder API servers
        cmd = [
            'python', os.path.join(args.code_path,'omniinfer/tools/scripts/start_api_servers.py'),
            '--num-servers', '1',
            '--server-offset', server_offset,  # Server offset parameter
            '--num-dp', env['DECODE_DP_SIZE'],  # Data parallelism degree
            '--model-path', args.model_path,
            '--master-ip', intf['ip'],      # Coordinator IP
            '--master-port', args.master_port,        # Coordinator port
            '--base-api-port', str(int(args.service_port) + 100),      # API service port
            '--tp', str(len(args.decode_server_list.split(','))),           # 8-way tensor parallelism
            '--served-model-name', args.model_name,
            '--max-model-len', args.max_model_len,    # Max context length
            '--log-dir', args.log_path + '/decode/',  # Log directory
            '--no-enable-prefix-caching',  # Disable caching
            '--extra-args', f'--max-num-batched-tokens {args.max_num_batched_tokens} --max-num-seqs {args.max_num_seqs} ',  # Performance flag
            '--additional-config', additional_config,  # Graph mode config
            '--kv-transfer-config', json.dumps(kv_transfer_config)  # KV transfer settings
        ]

    subprocess.Popen(cmd, env=env)  # Start as background process

def start_global_proxy(intf, args):
    """Start global proxy for routing requests"""
    env = os.environ.copy()
    env['PATH'] = '/usr/local/nginx:' + env.get('PATH', '')  # Ensure nginx in PATH

    # Start proxy script
    cmd = [
        'bash', os.path.join(args.code_path, 'omniinfer/omni/accelerators/sched/global_proxy/global_proxy.sh'),
        '--listen-port', args.https_port,          # Proxy listening port
        '--prefill-servers-list', intf['ip'] + ':' + args.service_port,  # Prefill server endpoints
        '--decode-servers-list', intf['ip'] + ':' + str(int(args.service_port) + 100),    # Decoder server endpoints
    ]

    subprocess.run(cmd, env=env)

def kill_all_processes():
    """Terminate all related processes"""
    # Kill processes by pattern matching
    subprocess.run("kill -9 $(ps aux | grep 'start_decode.sh' | awk '{print $2}')", shell=True)
    subprocess.run("kill -9 $(ps aux | grep 'start_prefill.sh' | awk '{print $2}')", shell=True)
    subprocess.run("kill -9 $(ps aux | grep 'run_benchmark.sh' | awk '{print $2}')", shell=True)
    subprocess.run("kill -9 $(ps aux | grep 'python' | awk '{print $2}')", shell=True)
    subprocess.run("kill -9 $(ps aux | grep 'python3.11' | awk '{print $2}')", shell=True)
    subprocess.run("kill -9 $(ps aux | grep 'python3.10' | awk '{print $2}')", shell=True)
    subprocess.run("kill -9 $(ps aux | grep 'start_do_16.sh' | awk '{print $2}')", shell=True)

def pd_ranktable(intf, args):
    """Generate ranktable configuration for distributed training"""
    ip = intf['ip']
    
    # Prefill ranktable generation
    # target_path_perfill = os.path.join(args.code_path, 'omniinfer/tools/scripts/perfill-ranktable/')
    # if not os.path.exists(target_path_perfill):
    #     print(f'Path {target_path_perfill} does not exist, creating it...')
    cmd_p = [
        'python', os.path.join(args.code_path, 'omniinfer/tools/scripts/pd_ranktable_tools.py'),
        '--mode', 'gen',
        '--prefill-server-list', args.prefill_server_list,  # NPU IDs for prefill
        '--api-server',             # API server flag
        '--save-dir', './perfill-ranktable',
    ]
    subprocess.run(cmd_p)
    # else:
    #     print(f'Path {target_path_perfill} already exists, skipping creation...')

    # Decoder ranktable generation
    # target_path_decode = os.path.join(args.code_path,'omniinfer/tools/scripts/decode-ranktable/')
    # if not os.path.exists(target_path_decode):
    #     print(f'Path {target_path_decode} does not exist, creating it...')
    cmd_d = [
        'python', os.path.join(args.code_path, 'omniinfer/tools/scripts/pd_ranktable_tools.py'),
        '--mode', 'gen',
        '--decode-server-list', args.decode_server_list,  # NPU IDs for decoder
        '--save-dir', './decode-ranktable',
    ]
    subprocess.run(cmd_d)
    # else:
    #     print(f'Path {target_path_decode} already exists, skipping creation...')

    # Global ranktable merge
    # target_path_global = os.path.join(args.code_path, 'omniinfer/tools/scripts/global_path/')
    # if not os.path.exists(target_path_global):
    #     print(f'Path {target_path_global} does not exist, creating it...')

    prefill_server_list_list = args.prefill_server_list.split(',')
    prefill_rank_table_suffix = ''.join(prefill_server_list_list)

    deocde_server_list_list = args.decode_server_list.split(',')
    decode_rank_table_suffix = ''.join(deocde_server_list_list)

    cmd_global = [
        'python', os.path.join(args.code_path, 'omniinfer/tools/scripts/pd_ranktable_tools.py'),
        '--mode', 'merge-all',  # Merge all ranktables
        '--api-server-list', f'perfill-ranktable/local_ranktable_{ip}_host.json',
        '--prefill-server-list', f'perfill-ranktable/local_ranktable_{ip}_{prefill_rank_table_suffix}.json',
        '--decode-server-list', f'decode-ranktable/local_ranktable_{ip}_{decode_rank_table_suffix}.json',
        '--save-dir', 'global_path'
    ]
    subprocess.run(cmd_global)
    # else:
    #     print(f'Path {target_path_global} already exists, skipping creation...')

def run_pd_separate_mode(args):
    """Run pipeline parallel (prefill/decoder separate) mode"""

    if (args.network_interface is not None and args.host_ip is None) or \
        (args.network_interface is None and args.host_ip is not None):
        warnings.warn(
            "For best results, please specify both --network-interface AND --host-ip "
            "together. Falling back to auto-detection for missing values.",
            RuntimeWarning
        )
    # Get network interface
    if args.network_interface:
        intf = {'if_name': args.network_interface, 'ip': get_ip_address(args.network_interface)}
    else:
        intf = get_network_interfaces()
        if not intf:
            raise RuntimeError("No network interface found and none specified")

    # Override IP if host-ip was specified
    if args.host_ip:
        intf['ip'] = args.host_ip

    # Setup distributed training configuration
    pd_ranktable(intf, args)

    # Start both server types
    start_perfill_api_servers(intf, args)
    time.sleep(1)  # Brief pause for servers to initialize
    start_decoder_api_servers(intf, args)

    time.sleep(2)  # Brief pause for servers to initialize

    if os.environ.get("OMNI_SKIP_PROXY", "0") == "1":
        print("OMNI_SKIP_PROXY=1 set. Skipping global proxy startup.")
        return

    # User control loop
    while True:
        user_input = input("\nEnter 'yes' to start global proxy, 'q' to quit: ").strip().lower()
        
        if user_input == 'yes' or user_input == 'y' or user_input == 'Y' or user_input == 'YES':
            start_global_proxy(intf, args)  # Start proxy after servers
            print("Global proxy started successfully!")
            break
        elif user_input == 'q' or user_input == 'Q' or user_input == 'quit' or user_input == 'exit':
            kill_all_processes()  # Cleanup before exit
            print("All processes terminated. Exiting program.")
            break
        else:
            print("Invalid input. Please enter 'yes' to proceed or 'q' to quit.")
           
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="OmniInfer Deployment Script - Launches model servers in different deployment modes"
        )
    parser.add_argument('--model-path', type=str, required=True,
                        help="Absolute path to the model checkpoint directory (required)")
    parser.add_argument('--deploy-mode', type=str, default='default',
                        help="Deployment strategy: 'default'  or 'pd_separate'  (default: default)")
    parser.add_argument('--graph-true', type=str, default='false',
                        help="Enable graph optimization mode: 'true' for optimized execution, 'false' for standard mode (default: false)")
    
    parser.add_argument('--network-interface', type=str, default=None,
                        help="Network interface name for distributed communication (default: auto-detect)")
    parser.add_argument('--host-ip', type=str, default=None,
                        help="Local machine's IP address for service binding (default: auto-detect from network interface)")
    parser.add_argument('--model-name', type=str, default='default_model',
                        help="Model identifier used for API endpoints (default: default_model)")
    parser.add_argument('--max-model-len', type=str, default='65536',
                        help="Maximum context length supported by the model in tokens (default: 65536)")
    parser.add_argument('--max-num-batched-tokens', type=str, default='32768',
                        help="Maximum context length supported by the model in tokens (default: 32768)")
    parser.add_argument('--max-num-seqs', type=str, default='64',
                        help="Maximum number of sequences supported by the model in tokens (default: 64)")
    parser.add_argument('--log-path', type=str, default='./apiserverlog',
                        help="Directory path for storing service logs (default: ./apiserverlog)")

    parser.add_argument('--server-list', type=str, default='0,1,2,3,4,5,6,7',
                        help="default mode: NPU device IDs for parallel processing (default: 0-7)")
    parser.add_argument('--prefill-server-list', type=str, default='0,1,2,3,4,5,6,7',
                        help="pd-separated:NPU device IDs dedicated to prompt prefill processing (default: '0,1,2,3,4,5,6,7')")
    parser.add_argument('--decode-server-list', type=str, default='8,9,10,11,12,13,14,15',
                        help="pd-separated:NPU device IDs dedicated to token decoding processing (default: '8,9,10,11,12,13,14,15')")
    
    parser.add_argument('--service-port', type=str, default='6660',
                        help="- In 'pd' mode: Prefill service port (Decoder uses this port + offset)\n"
                            "Global proxy will connect to these ports (default: 6660)")
    parser.add_argument('--master-port', type=str, default='8888',
                        help="The --master-port parameter in your command specifies the central coordination port used" \
                            " for distributed communication between different components of the inference system.")
    parser.add_argument('--https-port', type=str, default='8001',
                        help="Port for accepting HTTPS requests (default: 8001)")

    parser.add_argument('--additional-config', type=str, default=None,
                        help="JSON format advanced config, e.g. '{\"enable_graph_mode\":true}'")


    args = parser.parse_args()
    args.code_path = get_path_before_omniinfer()  # Get base path before 'omniinfer'
    
    # Validate critical paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
     # Validate execution mode
    if args.deploy_mode not in ['pd_separate', 'default']:
        raise ValueError(f"Invalid operations mode: {args.deploy_mode}")

    # Deployment mode routing
    if args.deploy_mode == 'default':
        run_default_mode(args)
    elif args.deploy_mode == 'pd_separate':
        run_pd_separate_mode(args)

    

```

## `benchmark_client.py`

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
    model: str = os.getenv("VLLM_MODEL_NAME", "/opt/models/qwen3-vl-8b")
    max_tokens: int = 256
    temperature: float = 0.0
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    num_npus: int = 2
    batch_size: int = 2
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
        default=os.getenv("VLLM_MODEL_NAME", "/opt/models/qwen3-vl-8b"),
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

## `attribute_extraction_eval.py`

```python
#!/usr/bin/env python3
"""
Attribute Extraction Evaluator for OpenAI-Compatible Multimodal APIs

This script evaluates a JSONL dataset where each line is a list of OpenAI-style
messages (system/user + assistant ground truth). It:

1) Sends ONLY system+user messages to /v1/chat/completions
2) Parses the model output as JSON (best-effort)
3) Compares prediction JSON against assistant ground truth JSON
4) Reports simple metrics + writes per-sample results to JSONL
"""

import argparse
import os
import asyncio
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from tqdm.asyncio import tqdm


def _now_ts() -> int:
    return int(time.time())


def _normalize_str(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


_MULTI_SEP_RE = re.compile(r"[,\n;\|]+")


def _split_multi_value(s: str) -> List[str]:
    """
    Split a string field that may represent multiple values, e.g.:
    "Superior,Inferior" -> ["Superior", "Inferior"]
    """
    raw = _normalize_str(s)
    parts = [p.strip() for p in _MULTI_SEP_RE.split(raw) if p.strip()]
    return parts if len(parts) > 1 else [raw]


def _try_parse_number(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = _normalize_str(v).lower()
        s = s.replace(",", ".")
        m = re.fullmatch(r"[-+]?\d+(\.\d+)?", s)
        if not m:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort: extract the first JSON object from a possibly noisy output.
    We prefer strict JSON if possible; otherwise we try to locate the first {...}.
    """
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None

    # Common: fenced code block
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()

    # Strict JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find first {...} and try to parse progressively
    start = text.find("{")
    if start < 0:
        return None
    # Greedy to last brace, then shrink until parse works
    end = text.rfind("}")
    if end <= start:
        return None
    candidate = text[start : end + 1]
    for cut in range(0, min(2000, len(candidate))):
        try_text = candidate[: len(candidate) - cut]
        if not try_text.endswith("}"):
            continue
        try:
            obj = json.loads(try_text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _as_messages(line_obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(line_obj, list):
        raise ValueError("Each line must be a JSON list of messages.")
    msgs: List[Dict[str, Any]] = []
    for m in line_obj:
        if not isinstance(m, dict):
            continue
        if "role" not in m or "content" not in m:
            continue
        msgs.append({"role": m["role"], "content": m["content"]})
    return msgs


def _split_input_and_gt(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    input_msgs = [m for m in messages if m.get("role") in ("system", "user")]
    gt = None
    for m in messages:
        if m.get("role") == "assistant":
            gt = m.get("content")
            if isinstance(gt, str):
                gt = gt.strip()
            else:
                gt = json.dumps(gt, ensure_ascii=False)
            break
    return input_msgs, gt


def _value_match(a: Any, b: Any) -> bool:
    # Match JSON scalars with light normalization and multi-value awareness
    if a is None and b is None:
        return True

    # numeric equality (handles numeric-like strings)
    na = _try_parse_number(a)
    nb = _try_parse_number(b)
    if na is not None and nb is not None:
        return na == nb

    # booleans
    if isinstance(a, bool) and isinstance(b, bool):
        return a is b

    # lists: compare as sets of normalized strings
    if isinstance(a, list) and isinstance(b, list):
        sa = {_normalize_str(str(x)).lower() for x in a if str(x).strip()}
        sb = {_normalize_str(str(x)).lower() for x in b if str(x).strip()}
        return sa == sb

    # strings: compare either as normalized scalar, or as multi-value sets
    if isinstance(a, str) and isinstance(b, str):
        pa = _split_multi_value(a)
        pb = _split_multi_value(b)
        if len(pa) > 1 or len(pb) > 1:
            sa = {_normalize_str(x).lower() for x in pa if x.strip()}
            sb = {_normalize_str(x).lower() for x in pb if x.strip()}
            return sa == sb
        return _normalize_str(a).lower() == _normalize_str(b).lower()

    # fallback
    return a == b


@dataclass
class EvalConfig:
    base_url: str
    timeout_s: float
    concurrency: int
    max_tokens: int
    temperature: float
    model: str


@dataclass
class SampleResult:
    index: int
    ok: bool
    latency_s: float
    error: Optional[str]
    gt_json_ok: bool
    pred_json_ok: bool
    key_coverage: float
    field_accuracy: float
    exact_match: bool
    pred_raw: str
    pred_json: Optional[Dict[str, Any]]


@dataclass
class KeyStats:
    gt_count: int = 0
    pred_count: int = 0
    correct_count: int = 0
    mismatch_count: int = 0
    missing_count: int = 0


async def _call_chat(
    client: httpx.AsyncClient,
    cfg: EvalConfig,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "stream": False,
    }
    r = await client.post(
        f"{cfg.base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=cfg.timeout_s,
    )
    r.raise_for_status()
    return r.json()


def _get_content(resp: Dict[str, Any]) -> str:
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(resp, ensure_ascii=False)


def _compute_metrics(gt_obj: Optional[Dict[str, Any]], pred_obj: Optional[Dict[str, Any]]) -> Tuple[float, float, bool]:
    if not isinstance(gt_obj, dict) or not gt_obj:
        return 0.0, 0.0, False
    if not isinstance(pred_obj, dict) or not pred_obj:
        return 0.0, 0.0, False
    gt_keys = set(gt_obj.keys())
    pred_keys = set(pred_obj.keys())
    inter = gt_keys & pred_keys
    key_coverage = (len(inter) / len(gt_keys)) if gt_keys else 0.0
    if not inter:
        return key_coverage, 0.0, False
    correct = 0
    for k in inter:
        if _value_match(gt_obj.get(k), pred_obj.get(k)):
            correct += 1
    field_accuracy = correct / len(inter)
    exact = key_coverage == 1.0 and field_accuracy == 1.0 and len(pred_keys) == len(gt_keys)
    return key_coverage, field_accuracy, exact


def _update_key_stats(stats: Dict[str, KeyStats], gt_obj: Optional[Dict[str, Any]], pred_obj: Optional[Dict[str, Any]]) -> None:
    if not isinstance(gt_obj, dict):
        return
    if not isinstance(pred_obj, dict):
        pred_obj = {}
    for k, gt_v in gt_obj.items():
        s = stats.setdefault(k, KeyStats())
        s.gt_count += 1
        if k in pred_obj:
            s.pred_count += 1
            if _value_match(gt_v, pred_obj.get(k)):
                s.correct_count += 1
            else:
                s.mismatch_count += 1
        else:
            s.missing_count += 1


async def evaluate_file(
    input_path: Path,
    out_path: Path,
    cfg: EvalConfig,
    limit: int,
    report_top_keys: int,
    min_key_occ: int,
) -> None:
    lines: List[str] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if limit > 0 and len(lines) >= limit:
                break

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(max(1, cfg.concurrency))
        results: List[SampleResult] = []
        key_stats: Dict[str, KeyStats] = {}
        failure_reasons = Counter()

        async def worker(i: int, line: str) -> None:
            async with sem:
                t0 = time.time()
                try:
                    line_obj = json.loads(line)
                    msgs = _as_messages(line_obj)
                    in_msgs, gt_text = _split_input_and_gt(msgs)
                    gt_obj = _extract_json_object(gt_text or "")
                    resp = await _call_chat(client, cfg, in_msgs)
                    pred_raw = _get_content(resp)
                    pred_obj = _extract_json_object(pred_raw)
                    _update_key_stats(key_stats, gt_obj, pred_obj)
                    key_cov, field_acc, exact = _compute_metrics(gt_obj, pred_obj)
                    results.append(
                        SampleResult(
                            index=i,
                            ok=True,
                            latency_s=time.time() - t0,
                            error=None,
                            gt_json_ok=isinstance(gt_obj, dict),
                            pred_json_ok=isinstance(pred_obj, dict),
                            key_coverage=key_cov,
                            field_accuracy=field_acc,
                            exact_match=exact,
                            pred_raw=pred_raw,
                            pred_json=pred_obj,
                        )
                    )
                except Exception as e:
                    failure_reasons[str(e)[:200]] += 1
                    results.append(
                        SampleResult(
                            index=i,
                            ok=False,
                            latency_s=time.time() - t0,
                            error=str(e),
                            gt_json_ok=False,
                            pred_json_ok=False,
                            key_coverage=0.0,
                            field_accuracy=0.0,
                            exact_match=False,
                            pred_raw="",
                            pred_json=None,
                        )
                    )

        tasks = [asyncio.create_task(worker(i, line)) for i, line in enumerate(lines)]
        for t in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
            await t

    # Write per-sample results
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda r: r.index)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "index": r.index,
                        "ok": r.ok,
                        "latency_s": r.latency_s,
                        "error": r.error,
                        "gt_json_ok": r.gt_json_ok,
                        "pred_json_ok": r.pred_json_ok,
                        "key_coverage": r.key_coverage,
                        "field_accuracy": r.field_accuracy,
                        "exact_match": r.exact_match,
                        "pred_raw": r.pred_raw,
                        "pred_json": r.pred_json,
                        "ts": _now_ts(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Summary
    ok = [r for r in results if r.ok]
    n = len(results)
    n_ok = len(ok)
    avg_lat = sum(r.latency_s for r in ok) / n_ok if n_ok else 0.0
    avg_cov = sum(r.key_coverage for r in ok) / n_ok if n_ok else 0.0
    avg_acc = sum(r.field_accuracy for r in ok) / n_ok if n_ok else 0.0
    exact = sum(1 for r in ok if r.exact_match)

    print("\n====================")
    print("ATTRIBUTE EXTRACTION EVAL SUMMARY")
    print("====================")
    print(f"Samples:            {n}")
    print(f"Successful calls:   {n_ok}")
    print(f"Success rate:       {(n_ok / n * 100.0) if n else 0.0:.1f}%")
    print(f"Avg latency:        {avg_lat:.2f}s")
    print(f"Avg key coverage:   {avg_cov:.3f}")
    print(f"Avg field accuracy: {avg_acc:.3f}")
    print(f"Exact matches:      {exact} ({(exact / n_ok * 100.0) if n_ok else 0.0:.1f}%)")
    print(f"Results:            {out_path}")

    if report_top_keys > 0 and key_stats:
        rows = []
        for k, s in key_stats.items():
            if s.gt_count < min_key_occ:
                continue
            acc = (s.correct_count / s.pred_count) if s.pred_count else 0.0
            miss = (s.missing_count / s.gt_count) if s.gt_count else 0.0
            rows.append((acc, miss, s.gt_count, s.pred_count, k))
        rows.sort(key=lambda x: (x[0], -x[2], x[4]))
        print("\n--------------------")
        print(f"Worst Keys (min_occ={min_key_occ})")
        print("--------------------")
        for acc, miss, gt_c, pred_c, k in rows[:report_top_keys]:
            print(f"{k:32s} acc={acc:.3f} miss={miss:.3f} gt={gt_c} pred={pred_c}")

    if failure_reasons:
        print("\n--------------------")
        print("Top Failure Reasons")
        print("--------------------")
        for reason, cnt in failure_reasons.most_common(5):
            print(f"{cnt:4d}  {reason}")


def _discover_model_id(base_url: str, timeout_s: float) -> Optional[str]:
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        models = data.get("data") or []
        if models and isinstance(models, list) and isinstance(models[0], dict):
            return models[0].get("id")
    except Exception:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate attribute-extraction.jsonl against an OpenAI-compatible server.")
    ap.add_argument("--input", default="/mnt/test_data/attribute-extraction.jsonl", help="Input JSONL path")
    ap.add_argument("--out", default="/mnt/GUI/attribute_extraction_results.jsonl", help="Output JSONL path")
    ap.add_argument("--base-url", default="http://127.0.0.1:9000", help="Server base URL (no /v1)")
    ap.add_argument("--timeout", type=float, default=180.0, help="Request timeout seconds")
    ap.add_argument("--concurrency", type=int, default=4, help="Max concurrent requests")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=512, help="max_tokens for chat/completions")
    ap.add_argument("--temperature", type=float, default=0.0, help="temperature for chat/completions")
    ap.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL_NAME", "/opt/models/qwen3-vl-8b"),
        help="Model id (default: VLLM_MODEL_NAME or /opt/models/qwen3-vl-8b)",
    )
    ap.add_argument("--report-top-keys", type=int, default=20, help="Print worst keys (0 disables)")
    ap.add_argument("--min-key-occ", type=int, default=5, help="Min occurrences for key report")

    args = ap.parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)

    model = args.model.strip()
    if not model:
        discovered = _discover_model_id(args.base_url, args.timeout)
        model = discovered or "/opt/models/qwen3-vl-8b"

    cfg = EvalConfig(
        base_url=args.base_url,
        timeout_s=args.timeout,
        concurrency=max(1, args.concurrency),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        model=model,
    )

    asyncio.run(evaluate_file(input_path, out_path, cfg, args.limit, args.report_top_keys, args.min_key_occ))


if __name__ == "__main__":
    main()

```
