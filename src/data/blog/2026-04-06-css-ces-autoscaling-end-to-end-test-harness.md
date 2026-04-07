---
author: Robin
pubDatetime: 2026-04-06T09:15:00-03:00
title: "Building a CSS + CES Auto-Scaling End-to-End Test Harness"
description: "A practical guide to validating Huawei Cloud CSS auto-scaling with CES metrics, realistic OpenSearch load, and a repeatable end-to-end test harness."
tags:
  - huawei-cloud
  - css
  - ces
  - autoscaling
  - opensearch
  - testing
  - e2e
featured: false
draft: false
---

# Building a CSS + CES Auto-Scaling End-to-End Test Harness

## Introduction

This post documents a practical approach for testing automatic scale-out and scale-in on a Huawei Cloud CSS cluster by combining:

- CSS cluster runtime APIs
- CES (Cloud Eye) metrics
- External OpenSearch search load
- An autoscaler daemon driven by Cloud Eye thresholds

The goal was straightforward:

1. Deploy and verify the autoscaler
2. Generate realistic search pressure against a CSS cluster
3. Confirm that Cloud Eye metrics rise enough to trigger automatic client-node scale-out
4. Remove the load
5. Confirm that the autoscaler eventually scales the cluster back down

The implementation described here avoids exposing any customer-specific identifiers, credentials, cluster names, project IDs, or region-specific operational details. The focus is purely on the engineering process.

## Architecture

The test setup used four moving parts:

- A CSS cluster with `ess-client` nodes
- Cloud Eye as the metric source
- An autoscaler daemon that polls Cloud Eye and calls CSS scale APIs
- An external E2E driver that creates test data, generates search traffic, and observes cluster state transitions

The control loop looked like this:

1. The E2E script starts the autoscaler daemon with a dedicated test configuration
2. The script creates a temporary test index and bulk-loads documents
3. The script generates sustained concurrent search traffic
4. Cloud Eye reports elevated search queue / CPU metrics
5. The autoscaler detects a scale-up condition and adds one client node
6. After the new node stabilizes, the E2E script removes search pressure
7. Cloud Eye metrics fall below scale-down thresholds
8. The autoscaler removes one client node
9. The E2E script verifies the cluster returned to its original client-node count

## Deployment and Test Strategy

### 1. Separate production config from test config

A dedicated test config was critical. The production autoscaling policy used larger windows and more conservative thresholds. That is a good default for production, but it slows down validation and makes E2E tests unnecessarily expensive.

The test config lowered:

- scale-up thresholds
- scale-down wait time
- autoscaler polling interval
- cooldown interval
- client-node upper bound

The last point matters: for a scale-out / scale-in validation, the test should prove `+1` and `-1`, not allow a second scale-up while the first expansion is still settling.

### 2. Use external search load, not internal system stress

The easiest way to test `ess-client` autoscaling was not host-level CPU stress inside CSS nodes. It was external search traffic against the cluster's OpenSearch endpoint.

That approach had two advantages:

- It exercised the exact request path handled by client nodes
- It mapped cleanly to Cloud Eye metrics such as search queue and CPU usage

### 3. Let the E2E script manage the autoscaler lifecycle

The test driver started and stopped the autoscaler itself. That made the workflow repeatable and removed hidden preconditions from the test:

- no manual daemon startup
- no accidental reuse of a stale PID file
- no mixing production and E2E logs

To support that cleanly, the startup script was updated to accept environment-variable overrides for:

- config file path
- PID file path
- log file path

That small change made the daemon reusable in both normal and test modes.

## Key Issues Found During Validation

### 1. CSS auth failures were not really AK/SK failures

The first visible symptom looked like an authentication or authorization problem, but the actual root causes were implementation bugs:

- wrong region and project mapping
- SDK region imports using the wrong module path
- CSS response parsing assuming fields that did not exist in the installed SDK
- expansion request classes referenced without proper imports

Once region, project, endpoint, and SDK field handling were corrected, cluster access and direct scale operations worked normally.

### 2. CES request formatting was wrong

The next major blocker was Cloud Eye metric retrieval.

The original CES call path failed because it used request fields that do not match the installed Huawei Cloud Python SDK model. In this SDK:

- metric dimensions are passed as `dim_0`, `dim_1`, etc.
- time range uses `_from`, not `from_`

The autoscaler initially failed every Cloud Eye query because of that mismatch.

After fixing the request model, the API stopped returning `400` errors.

### 3. Logical metric names did not match actual Cloud Eye metric names

Even after the CES request shape was corrected, metric reads still returned empty datapoints. The reason was simpler: the application used logical names such as:

- `cpu_usage`
- `jvm_heap_usage`
- `thread_pool_search_queue`

But the cluster's actual Cloud Eye metrics were exposed under names such as:

- `avg_cpu_usage`
- `avg_jvm_heap_usage`
- `avg_thread_pool_search_queue`

The fix was to keep logical metric names in the autoscaler but map them internally to provider-specific metric names before querying CES.

### 4. The cluster stabilized slower than the autoscaler's decision loop

After automatic scale-out started working, a second problem appeared: repeated scale-up attempts during cluster convergence.

The sequence looked like this:

1. The autoscaler triggered `1 -> 2`
2. The new client node appeared, but its instance status remained non-ready for a while
3. Search traffic was still active
4. The autoscaler continued polling metrics
5. High CPU / queue metrics still matched scale-up conditions
6. The autoscaler tried to perform `2 -> 3`
7. CSS rejected the request with a conflict because another operation was still in progress

This was not a CSS bug. It was a control-loop design issue.

The fix used two layers:

- The autoscaler now refuses any scaling decision while target node-type instances are not fully stable
- The E2E config caps the test at exactly one extra client node

That eliminated second-step expansion attempts during the test.

## Final Test Flow

With the fixes in place, the final test flow was:

1. Verify cluster visibility and stable initial state
2. Start autoscaler daemon with a dedicated E2E config
3. Create a temporary search test index
4. Bulk-load test documents
5. Start concurrent search workers
6. Poll CSS runtime state until client-node count increases by one and all client instances become stable
7. Stop search load
8. Wait for Cloud Eye metrics to fall below the scale-down thresholds
9. Let the autoscaler remove one client node
10. Poll until client-node count returns to the initial value and the remaining client instances are stable
11. Delete the temporary index
12. Stop the autoscaler daemon

## Practical Lessons

### SDK model assumptions are dangerous

Do not assume a Huawei Cloud SDK request or response model from memory. Generated SDKs often differ in:

- field names
- query parameter names
- response layout
- nested vs flattened objects

The fastest way to validate an assumption is to inspect the installed SDK directly.

### Cloud Eye metric names should be discovered, not guessed

If a service exposes `ListMetrics`, use it. It is the safest way to confirm:

- namespace
- dimension key
- metric names
- aggregation naming pattern

### Stable state must be part of autoscaling logic

Node count alone is not enough. A cluster can report the target number of instances while still being in a transitional state.

The autoscaler must treat “instance count reached” and “instance status stable” as separate conditions.

### E2E tests should enforce their own boundaries

An E2E scale test should intentionally constrain the policy:

- short windows
- low thresholds
- short cooldown
- explicit max node count

That keeps the test deterministic and prevents over-scaling during verification.

## Conclusion

The final result was a repeatable CSS + CES autoscaling validation workflow that:

- uses real Cloud Eye metrics
- drives realistic OpenSearch search pressure
- verifies automatic scale-out
- verifies automatic scale-in
- isolates test settings from production settings
- avoids repeated scale-up attempts during cluster convergence

This kind of harness is valuable not only for one-time validation, but also for regression testing future autoscaling changes.

## Appendix: E2E Test Script

```python
#!/usr/bin/env python3
"""
CSS auto-scaling end-to-end test.

This script starts the autoscaler daemon with a dedicated test config,
creates a temporary OpenSearch index and load, waits for automatic
scale-out, then removes the load and waits for automatic scale-in.
"""

import argparse
import os
import random
import string
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from opensearchpy import OpenSearch, helpers

from sdk_client import CSSClientWrapper
from utils import load_config, setup_logging


class AutoscalingE2ETest:
    """End-to-end load-driven autoscaling verification."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path).resolve()
        self.config = load_config(str(self.config_path))
        self.logger = setup_logging(self.config.get('logging', {}))
        self.css_client = CSSClientWrapper(self.config, self.logger)

        self.cluster_id = self.config['cluster']['cluster_id']
        self.cluster_name = self.config['cluster'].get('cluster_name', self.cluster_id)
        self.node_type = self.config['cluster'].get('node_type', 'ess-client')
        self.opensearch_endpoint = self.config['cluster']['opensearch_endpoint']
        self.e2e_conf = self.config.get('e2e_test', {})

        self.script_dir = self.config_path.parent
        self.start_script = self.script_dir / 'start.sh'
        self.pid_file = self.script_dir / 'autoscaler-e2e.pid'
        self.log_file = self.script_dir / 'autoscaler-e2e.log'

        self.index_name = self.e2e_conf.get('index_name', 'autoscaling_e2e_test')
        self.doc_count = int(self.e2e_conf.get('doc_count', 5000))
        self.bulk_batch_size = int(self.e2e_conf.get('bulk_batch_size', 500))
        self.search_workers = int(self.e2e_conf.get('search_workers', 24))
        self.search_request_timeout = int(self.e2e_conf.get('search_request_timeout', 10))
        self.scale_up_timeout_seconds = int(self.e2e_conf.get('scale_up_timeout_minutes', 20)) * 60
        self.scale_down_timeout_seconds = int(self.e2e_conf.get('scale_down_timeout_minutes', 20)) * 60
        self.cleanup_index = bool(self.e2e_conf.get('cleanup_index', True))

        self.load_stop_event = threading.Event()
        self.load_threads = []
        self.load_errors = []
        self.search_count = 0
        self.search_count_lock = threading.Lock()
        self.autoscaler_started = False

    def _subprocess_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env['CONFIG_FILE'] = str(self.config_path)
        env['PID_FILE'] = str(self.pid_file)
        env['LOG_FILE'] = str(self.log_file)
        return env

    def _build_opensearch_client(self) -> OpenSearch:
        endpoint = self.opensearch_endpoint
        if '://' in endpoint:
            scheme, rest = endpoint.split('://', 1)
        else:
            scheme, rest = 'http', endpoint

        if ':' in rest:
            host, port_text = rest.rsplit(':', 1)
            port = int(port_text)
        else:
            host = rest
            port = 9200

        return OpenSearch(
            hosts=[{'host': host, 'port': port}],
            use_ssl=(scheme == 'https'),
            verify_certs=False,
            timeout=self.search_request_timeout,
        )

    def _summarize_runtime_state(self, state: Optional[Dict[str, Any]]) -> str:
        if not state:
            return 'state unavailable'

        instance_summary = ', '.join(
            f"{instance['name']}({instance['type']}:{instance['status']})"
            for instance in state.get('instances', [])
        ) or 'no instances'

        return (
            f"cluster_status={state.get('status')}, "
            f"client_nodes={state.get('client_node_count')}, "
            f"instances=[{instance_summary}]"
        )

    def _wait_for_client_nodes(self, target_count: int, timeout_seconds: int, phase_name: str) -> bool:
        deadline = time.time() + timeout_seconds
        last_state = None

        while time.time() < deadline:
            last_state = self.css_client.get_cluster_runtime_state(self.cluster_id)
            self.logger.info(f"{phase_name} poll: {self._summarize_runtime_state(last_state)}")
            if last_state:
                client_instances = last_state.get('client_instances', [])
                statuses = [instance.get('status') for instance in client_instances]
                if (
                    last_state.get('client_node_count') == target_count and
                    last_state.get('status') == '200' and
                    statuses and
                    all(status == '200' for status in statuses)
                ):
                    return True
            time.sleep(30)

        self.logger.error(
            f"{phase_name} timed out after {timeout_seconds}s; "
            f"last state: {self._summarize_runtime_state(last_state)}"
        )
        return False

    def _assert_cluster_ready(self) -> int:
        if self.node_type != 'ess-client':
            raise RuntimeError(f"Unsupported node_type for E2E test: {self.node_type}")

        state = self.css_client.get_cluster_runtime_state(self.cluster_id)
        if not state:
            raise RuntimeError("Failed to fetch cluster runtime state")
        if state.get('status') != '200':
            raise RuntimeError(f"Cluster not ready: {self._summarize_runtime_state(state)}")
        if not self.css_client.is_cluster_stable(self.cluster_id, node_type=self.node_type):
            raise RuntimeError("Cluster or ess-client instances are not stable")

        initial_count = state.get('client_node_count', 0)
        if initial_count < 1:
            raise RuntimeError(f"Expected at least 1 ess-client node, found {initial_count}")

        self.logger.info(
            "Cluster ready for E2E test: "
            f"cluster={self.cluster_name} cluster_id={self.cluster_id} "
            f"initial_client_nodes={initial_count}"
        )
        return initial_count

    def _ensure_autoscaler_not_running(self):
        if self.pid_file.exists():
            raise RuntimeError(f"E2E autoscaler PID file already exists: {self.pid_file}")

    def _run_start_script(self, command: str):
        subprocess.run(
            [str(self.start_script), command],
            cwd=str(self.script_dir),
            env=self._subprocess_env(),
            check=True,
        )

    def _start_autoscaler(self):
        self._ensure_autoscaler_not_running()
        self.logger.info("Starting autoscaler daemon for E2E test")
        self._run_start_script('start')
        self.autoscaler_started = True

    def _stop_autoscaler(self):
        if not self.autoscaler_started:
            return
        self.logger.info("Stopping autoscaler daemon for E2E test")
        try:
            self._run_start_script('stop')
        finally:
            self.autoscaler_started = False

    def _random_text(self, length: int = 64) -> str:
        alphabet = string.ascii_lowercase + string.digits
        return ''.join(random.choices(alphabet, k=length))

    def _prepare_index(self, client: OpenSearch):
        if client.indices.exists(index=self.index_name):
            self.logger.info(f"Deleting stale test index: {self.index_name}")
            client.indices.delete(index=self.index_name)

        self.logger.info(f"Creating test index: {self.index_name}")
        client.indices.create(
            index=self.index_name,
            body={
                'settings': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                },
                'mappings': {
                    'properties': {
                        'title': {'type': 'text'},
                        'category': {'type': 'keyword'},
                        'payload': {'type': 'text'},
                        'counter': {'type': 'integer'},
                    }
                },
            },
        )

        actions = []
        self.logger.info(f"Indexing {self.doc_count} documents into {self.index_name}")
        for i in range(self.doc_count):
            actions.append({
                '_index': self.index_name,
                '_id': i,
                '_source': {
                    'title': f'autoscaling test document {i}',
                    'category': f'group-{i % 20}',
                    'payload': ' '.join(self._random_text(16) for _ in range(16)),
                    'counter': i,
                }
            })

        helpers.bulk(
            client,
            actions,
            chunk_size=self.bulk_batch_size,
            request_timeout=max(60, self.search_request_timeout),
        )
        client.indices.refresh(index=self.index_name)

    def _cleanup_index(self, client: OpenSearch):
        if not self.cleanup_index:
            self.logger.info("Cleanup disabled; preserving test index")
            return
        if client.indices.exists(index=self.index_name):
            self.logger.info(f"Deleting test index: {self.index_name}")
            client.indices.delete(index=self.index_name)

    def _search_worker(self, worker_id: int, client: OpenSearch):
        query = {
            'size': 25,
            'query': {
                'function_score': {
                    'query': {'match_all': {}},
                    'random_score': {}
                }
            },
            'aggs': {
                'categories': {'terms': {'field': 'category', 'size': 20}}
            },
            'sort': [
                {'counter': 'desc'}
            ]
        }

        while not self.load_stop_event.is_set():
            try:
                client.search(index=self.index_name, body=query, request_timeout=self.search_request_timeout)
                with self.search_count_lock:
                    self.search_count += 1
            except Exception as exc:
                self.load_errors.append((worker_id, str(exc)))
                time.sleep(1)

    def _start_search_load(self, client: OpenSearch):
        self.logger.info(f"Starting search load with {self.search_workers} workers")
        self.load_stop_event.clear()
        self.load_threads = []
        self.load_errors = []
        self.search_count = 0

        for worker_id in range(self.search_workers):
            thread = threading.Thread(
                target=self._search_worker,
                args=(worker_id, client),
                daemon=True,
            )
            thread.start()
            self.load_threads.append(thread)

    def _stop_search_load(self):
        self.logger.info("Stopping search load")
        self.load_stop_event.set()
        for thread in self.load_threads:
            thread.join(timeout=5)
        self.load_threads = []
        self.logger.info(f"Total successful search requests: {self.search_count}")
        if self.load_errors:
            self.logger.warning(f"Search load observed {len(self.load_errors)} errors")

    def run(self) -> int:
        opensearch_client = self._build_opensearch_client()
        initial_count = 0

        try:
            initial_count = self._assert_cluster_ready()
            self._start_autoscaler()
            self._prepare_index(opensearch_client)
            self._start_search_load(opensearch_client)

            if not self._wait_for_client_nodes(
                target_count=initial_count + 1,
                timeout_seconds=self.scale_up_timeout_seconds,
                phase_name='Scale-up'
            ):
                return 1

            self._stop_search_load()

            if not self._wait_for_client_nodes(
                target_count=initial_count,
                timeout_seconds=self.scale_down_timeout_seconds,
                phase_name='Scale-down'
            ):
                return 1

            self.logger.info("Autoscaling E2E test completed successfully")
            return 0

        finally:
            self._stop_search_load()
            try:
                self._cleanup_index(opensearch_client)
            except Exception as exc:
                self.logger.error(f"Failed to clean up test index: {exc}")
            self._stop_autoscaler()


def main() -> int:
    parser = argparse.ArgumentParser(description='CSS autoscaling end-to-end test')
    parser.add_argument(
        '--config', '-c',
        default='config.e2e.yaml',
        help='Path to E2E configuration file (default: config.e2e.yaml)'
    )
    args = parser.parse_args()

    if not os.getenv('HUAWEICLOUD_SDK_AK') or not os.getenv('HUAWEICLOUD_SDK_SK'):
        print("Error: Environment variables HUAWEICLOUD_SDK_AK and HUAWEICLOUD_SDK_SK must be set")
        return 1

    runner = AutoscalingE2ETest(args.config)
    return runner.run()


if __name__ == '__main__':
    sys.exit(main())
```
