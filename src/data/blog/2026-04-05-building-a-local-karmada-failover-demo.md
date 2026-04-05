---
author: Robin
pubDatetime: 2026-04-05T15:45:00-03:00
title: "Building a Local Karmada Failover Demo: Multi-Cluster Kubernetes Traffic Switching on One Host"
description: "A step-by-step guide to building a single-host Karmada failover demo with two kind member clusters, a stable HAProxy entrypoint, and visible traffic cutover between Kubernetes clusters."
tags:
  - kubernetes
  - karmada
  - multi-cluster
  - failover
  - kind
  - haproxy
  - devops
  - demo
featured: true
draft: false
---

# Building a Local Karmada Failover Demo: Multi-Cluster Kubernetes Traffic Switching on One Host

## Introduction

In this article I will walk through a complete local Karmada demo environment that runs on a single Linux host and demonstrates Kubernetes workload failover between two member clusters.

The goal is not just to install Karmada, but to build something visual enough to prove that traffic really moves between clusters:

- a Karmada control plane
- two minimal Kubernetes member clusters
- a stateless demo application
- a fixed browser entrypoint on the host
- automatic traffic cutover from `cluster 1` to `cluster 2`

I used this setup to demonstrate a simple but useful high-availability story:

1. traffic starts on `member1`
2. the browser always uses the same URL on the host
3. `member1` fails
4. traffic switches to `member2`
5. the page shows that the backend changed, while the probe timeline keeps moving

This is not a stateful migration demo. There is no shared database and no persistent session replication. Instead, this is a **stateless failover demo** designed to make cluster switching easy to observe.

The final blog output and demo assets are based on a live environment under:

- repository: `/root/karmada`
- demo assets: `/root/karmada/samples/failover-demo`

---

## What We Are Building

The final topology is:

- `karmada-host`: the host Kubernetes cluster that runs the Karmada control plane
- `member1`: the first member cluster, which serves traffic as `cluster 1`
- `member2`: the second member cluster, which serves traffic as `cluster 2`
- a small HTTP demo application propagated by Karmada to both member clusters
- an HAProxy container on the host that exposes a fixed URL:
  - `http://127.0.0.1:8088/`

The browser only talks to the host proxy. The proxy sends traffic to `member1` first and keeps `member2` as a hot standby. When `member1` fails, traffic is switched to `member2`.

---

## Why Karmada for This Demo

Karmada is useful here because it gives us a real multi-cluster control plane:

- it propagates the application resources to multiple clusters
- it keeps the workload definitions consistent
- it lets us converge placement when we want to finalize the migration

In this demo, **Karmada manages workload distribution**, while **HAProxy provides the single stable ingress point** for the user.

That division of responsibility is intentional:

- Karmada solves multi-cluster resource orchestration
- the host proxy solves request routing and fast traffic switching

This makes the demo honest and easy to reason about.

---

## Host Environment

The environment used for this demo was a single Rocky Linux host with Docker enabled. The host already had the Karmada repository checked out at:

```bash
/root/karmada
```

The following tools were required:

- `docker`
- `go`
- `kubectl`
- `kind`
- `git`

For the Karmada repository version used in this environment, Go `1.25.7` was installed.

---

## A Critical Fix for kind on This Host

Before I could reliably run Karmada on kind clusters, I hit a host-level issue:

- `kube-proxy` and `CoreDNS` inside the kind nodes were failing
- the failures looked like service DNS instability and API watch issues
- the root cause was low `inotify` limits on the host

I fixed that by adding:

```conf
# /etc/sysctl.d/99-karmada-kind.conf
fs.inotify.max_user_instances = 8192
fs.inotify.max_user_watches = 1048576
```

The actual file on this host is:

- `/etc/sysctl.d/99-karmada-kind.conf`

Apply it immediately with:

```bash
sysctl --system
```

If you skip this and kind networking behaves strangely, fix this first.

---

## Step 1: Clone the Karmada Repository

```bash
git clone https://github.com/karmada-io/karmada.git /root/karmada
cd /root/karmada
```

This demo uses Karmada’s own repository both for control plane deployment and for the sample application assets.

---

## Step 2: Create Three kind Clusters

I did not use the default `local-up-karmada.sh` end to end because I wanted a tighter environment:

- one host cluster for Karmada
- two member clusters
- no extra third member

The cluster names are:

- `karmada-host`
- `member1`
- `member2`

The Karmada repository already includes kind configs and helper logic in `hack/`.

The rough flow is:

```bash
export PATH=/usr/local/go/bin:/root/go/bin:/root/.local/bin:$PATH
cd /root/karmada

source hack/util.sh

mkdir -p /root/.kube

util::create_cluster "karmada-host" "/root/.kube/karmada.config" "${DEFAULT_CLUSTER_VERSION}" "/tmp/karmada-manual"
util::create_cluster "member1" "/root/.kube/member-tmp-member1.config" "${DEFAULT_CLUSTER_VERSION}" "/tmp/karmada-manual" "/root/karmada/artifacts/kindClusterConfig/member1.yaml"
util::create_cluster "member2" "/root/.kube/member-tmp-member2.config" "${DEFAULT_CLUSTER_VERSION}" "/tmp/karmada-manual" "/root/karmada/artifacts/kindClusterConfig/member2.yaml"
```

Then wait for all three clusters:

```bash
util::check_clusters_ready "/root/.kube/karmada.config" "karmada-host"
util::check_clusters_ready "/root/.kube/member-tmp-member1.config" "member1"
util::check_clusters_ready "/root/.kube/member-tmp-member2.config" "member2"
```

At this point you should have:

- `/root/.kube/karmada.config`
- `/root/.kube/member-tmp-member1.config`
- `/root/.kube/member-tmp-member2.config`

---

## Step 3: Connect Pod Networks Between Clusters

kind clusters on one machine do not magically understand each other’s Pod CIDRs. Karmada needs connectivity between clusters, so routes must be added between them.

I added routes between:

- `member1` and `member2`
- `karmada-host` and `member1`
- `karmada-host` and `member2`

The helper function from `hack/util.sh` is:

```bash
util::add_routes <src_cluster> <kubeconfig> <context>
```

The concrete calls were:

```bash
util::add_routes member1 /root/.kube/member-tmp-member2.config member2
util::add_routes member2 /root/.kube/member-tmp-member1.config member1
util::add_routes karmada-host /root/.kube/member-tmp-member1.config member1
util::add_routes member1 /root/.kube/karmada.config karmada-host
util::add_routes karmada-host /root/.kube/member-tmp-member2.config member2
util::add_routes member2 /root/.kube/karmada.config karmada-host
```

Then merge the member kubeconfigs:

```bash
export KUBECONFIG=/root/.kube/member-tmp-member1.config:/root/.kube/member-tmp-member2.config
kubectl config view --flatten > /root/.kube/members.config
```

Final kubeconfig files:

- Karmada host/control plane:
  - `/root/.kube/karmada.config`
- member clusters:
  - `/root/.kube/members.config`

---

## Step 4: Deploy the Karmada Control Plane

With `karmada-host` ready, deploy the control plane using the Karmada repository’s helper:

```bash
cd /root/karmada
export PATH=/usr/local/go/bin:/root/go/bin:/root/.local/bin:$PATH
./hack/deploy-karmada.sh /root/.kube/karmada.config karmada-host
```

This installs:

- `karmada-apiserver`
- `karmada-controller-manager`
- `karmada-scheduler`
- `karmada-descheduler`
- `karmada-webhook`
- `karmada-search`
- `karmada-metrics-adapter`
- supporting CRDs and APIService objects

Verify:

```bash
kubectl --kubeconfig=/root/.kube/karmada.config --context=karmada-host -n karmada-system get deploy
```

The Karmada API context generated by the deploy script is:

- `karmada-apiserver`

---

## Step 5: Join the Member Clusters

Now join `member1` and `member2` to Karmada:

```bash
export KUBECONFIG=/root/.kube/karmada.config

/root/go/bin/karmadactl join \
  --karmada-context=karmada-apiserver \
  member1 \
  --cluster-kubeconfig=/root/.kube/members.config \
  --cluster-context=member1

/root/go/bin/karmadactl join \
  --karmada-context=karmada-apiserver \
  member2 \
  --cluster-kubeconfig=/root/.kube/members.config \
  --cluster-context=member2
```

Then deploy Karmada scheduler estimators:

```bash
./hack/deploy-scheduler-estimator.sh /root/.kube/karmada.config karmada-host /root/.kube/members.config member1
./hack/deploy-scheduler-estimator.sh /root/.kube/karmada.config karmada-host /root/.kube/members.config member2
```

And metrics-server to both member clusters:

```bash
./hack/deploy-k8s-metrics-server.sh /root/.kube/members.config member1
./hack/deploy-k8s-metrics-server.sh /root/.kube/members.config member2
```

Verify:

```bash
kubectl --kubeconfig=/root/.kube/karmada.config --context=karmada-apiserver get clusters
```

Expected:

- `member1` is `READY=True`
- `member2` is `READY=True`

---

## Step 6: Build the Stateless Demo Application

The demo application lives here:

- `/root/karmada/samples/failover-demo/app/main.go`

It is intentionally tiny:

- `GET /` returns the demo page
- `GET /status` returns JSON with:
  - `cluster`
  - `instance`
  - `timestamp`
  - `message`
- `GET /healthz` returns `200 OK`

The important part is not the backend logic. The frontend is where the observability lives:

- successful probe count
- consecutive success count
- heartbeat timeline
- recent backend switch events

That makes failover visible without requiring a shared database.

Build and load the image into both member clusters:

```bash
cd /root/karmada/samples/failover-demo
./scripts/build-and-load.sh
```

This script:

- builds `karmada/failover-demo:v1`
- loads it into `member1`
- loads it into `member2`

---

## Step 7: Apply the Karmada Manifests

The manifest directory is:

- `/root/karmada/samples/failover-demo/manifests`

It contains:

- `deployment.yaml`
- `service.yaml`
- `deployment-propagationpolicy.yaml`
- `service-propagationpolicy.yaml`
- `overridepolicy.yaml`

The pattern is:

- one Deployment propagated to both clusters
- one NodePort Service propagated to both clusters
- one OverridePolicy that makes `member1` return `cluster 1`
- one OverridePolicy rule that makes `member2` return `cluster 2`

Deploy them:

```bash
./scripts/deploy.sh
```

That script applies everything to the Karmada API server and then waits until the Deployment exists and rolls out on both member clusters.

Verify directly:

```bash
kubectl --kubeconfig=/root/.kube/members.config --context=member1 get deploy,svc,pod -n default
kubectl --kubeconfig=/root/.kube/members.config --context=member2 get deploy,svc,pod -n default
```

At this point:

- `member1` serves `cluster 1`
- `member2` serves `cluster 2`

Both are ready, but users still need a single host-side entrypoint.

---

## Step 8: Start the Host Proxy

The proxy script is:

- `/root/karmada/samples/failover-demo/scripts/start-proxy.sh`

It starts an HAProxy container on the host and exposes:

```bash
http://127.0.0.1:8088/
```

The proxy design is:

- `member1` is primary
- `member2` is backup
- health checks run against `/healthz`

Start it:

```bash
./scripts/start-proxy.sh
```

The script creates a config in:

- `/tmp/karmada-failover-demo/haproxy.cfg`

Then launches:

- container: `karmada-failover-proxy`

Verify:

```bash
curl http://127.0.0.1:8088/status
```

If the proxy is healthy and `member1` is primary, the response should look like:

```json
{"cluster":"cluster 1","instance":"...","timestamp":"...","message":"serving from cluster 1"}
```

Open the page in a browser:

```text
http://127.0.0.1:8088/
```

You should see:

- current backend
- successful probes
- consecutive successful probes
- heartbeat timeline
- recent events

This is much clearer than a simple local counter because it directly answers the question: **Did requests keep succeeding while the backend changed?**

---

## Step 9: Simulate Cluster Failover

To simulate failure of the primary cluster, stop `member1`:

```bash
./scripts/fail-member1.sh
```

That script simply runs:

```bash
docker stop member1-control-plane
```

Because HAProxy is already health-checking both backends:

- `member1` is marked down
- traffic is automatically redirected to `member2`

Verify from the host:

```bash
curl http://127.0.0.1:8088/status
```

Now you should get:

```json
{"cluster":"cluster 2","instance":"...","timestamp":"...","message":"serving from cluster 2"}
```

The browser page should continue updating and should record a switch event such as:

- `switched from cluster 1 to cluster 2`

This is the key moment in the demo.

---

## Step 10: Converge Placement to member2

Traffic switching alone is not the full story. After the host proxy fails over, we can also ask Karmada to converge the workload placement to `member2` only.

That script is:

- `/root/karmada/samples/failover-demo/scripts/promote-member2.sh`

Run it:

```bash
./scripts/promote-member2.sh
```

This script:

- patches the Karmada-side Deployment replicas to `1`
- patches the Deployment PropagationPolicy to only target `member2`
- patches the Service PropagationPolicy to only target `member2`

In other words:

- traffic is already on `member2`
- Karmada now makes `member2` the final desired placement

This creates a nice two-step story:

1. fast traffic failover
2. control plane convergence

---

## Step 11: Switch Traffic Back to member1

The reverse direction is also useful for demos.

To bring `member1` back:

```bash
docker start member1-control-plane
```

Wait for it to become Ready again:

```bash
kubectl --kubeconfig=/root/.kube/members.config --context=member1 wait --for=condition=Ready node --all --timeout=180s
kubectl --kubeconfig=/root/.kube/karmada.config --context=karmada-apiserver wait --for=condition=Ready cluster member1 --timeout=180s
```

Then patch Karmada placement back to both clusters and make the replica count `2` again. That is exactly what I did in the live demo environment:

```bash
kubectl --kubeconfig=/root/.kube/karmada.config --context=karmada-apiserver patch deployment failover-demo -n default --type merge -p '{"spec":{"replicas":2}}'
kubectl --kubeconfig=/root/.kube/karmada.config --context=karmada-apiserver patch propagationpolicy failover-demo-deployment -n default --type merge -p '{"spec":{"placement":{"clusterAffinity":{"clusterNames":["member1","member2"]},"replicaScheduling":{"replicaDivisionPreference":"Weighted","replicaSchedulingType":"Divided","weightPreference":{"staticWeightList":[{"targetCluster":{"clusterNames":["member1"]},"weight":1},{"targetCluster":{"clusterNames":["member2"]},"weight":1}]}}}}}'
kubectl --kubeconfig=/root/.kube/karmada.config --context=karmada-apiserver patch propagationpolicy failover-demo-service -n default --type merge -p '{"spec":{"placement":{"clusterAffinity":{"clusterNames":["member1","member2"]}}}}'
```

Finally, restart the host proxy configuration so `member1` becomes primary again:

```bash
./scripts/start-proxy.sh
```

After that:

```bash
curl http://127.0.0.1:8088/status
```

Should return `cluster 1` again.

---

## How the Demo Proves Continuity

The original version of the page had a local browser counter. That was not convincing enough because a local counter does not prove the backend remained reachable.

The improved page demonstrates continuity in three better ways:

### 1. Successful probe count

The count only increments when `/status` succeeds.

### 2. Heartbeat timeline

Each probe is visualized:

- green: success
- amber: success but slow
- red: failure

If the timeline stays green through the switch, the audience immediately understands that the service did not visibly drop.

### 3. Switch events

The page records backend transitions:

- connected to `cluster 1`
- switched from `cluster 1` to `cluster 2`
- switched from `cluster 2` to `cluster 1`

That is much more intuitive in a live demo than reading raw JSON from the terminal.

---

## Why This Demo Is Stateless

This demo does **not** synchronize backend state between clusters.

That is intentional.

The backend only exposes:

- which cluster is serving
- which Pod instance is serving
- when the response was generated

Everything else needed for the visual experience is browser-local.

This means:

- no Redis
- no shared SQL database
- no replicated session store
- no cross-cluster data consistency problem

That keeps the demo focused on **traffic failover**, not state migration.

If you later want a stateful version, the next step is to add a shared backend such as Redis or PostgreSQL and move the continuity signal from the browser into the shared store.

---

## Useful Commands

### Check Karmada

```bash
export PATH=/usr/local/go/bin:/root/go/bin:/root/.local/bin:$PATH
kubectl --kubeconfig=/root/.kube/karmada.config --context=karmada-apiserver get clusters
kubectl --kubeconfig=/root/.kube/karmada.config --context=karmada-apiserver get deployment failover-demo -n default
kubectl --kubeconfig=/root/.kube/karmada.config --context=karmada-apiserver get propagationpolicy -n default
kubectl --kubeconfig=/root/.kube/karmada.config --context=karmada-apiserver get resourcebinding -n default
```

### Check Member Clusters

```bash
kubectl --kubeconfig=/root/.kube/members.config --context=member1 get deploy,svc,pod -n default
kubectl --kubeconfig=/root/.kube/members.config --context=member2 get deploy,svc,pod -n default
```

### Check the Unified Host Entry Point

```bash
curl http://127.0.0.1:8088/status
```

### Stop the Proxy

```bash
cd /root/karmada/samples/failover-demo
./scripts/stop-proxy.sh
```

---

## Lessons Learned

### 1. Keep the demo honest

Karmada is not a traffic router. It is a multi-cluster orchestration layer. In this demo, HAProxy owns ingress and failover speed, while Karmada owns workload distribution and convergence.

That separation makes the demo clearer and easier to debug.

### 2. A visual demo needs strong signals

A local counter is not enough. A probe timeline plus switch events is much easier for an audience to trust.

### 3. kind is convenient, but the host still matters

The biggest infrastructure issue in this environment was not Karmada itself. It was the host’s `inotify` limit, which broke kind networking components in subtle ways.

### 4. Start with stateless failover first

If the goal is to teach multi-cluster switching, a stateless demo gets you there faster and with fewer moving parts.

---

## Conclusion

A single machine is enough to build a realistic Karmada failover lab:

- one kind cluster for the Karmada control plane
- two kind member clusters
- one small stateless application
- one fixed host-side entrypoint

With that setup, you can demonstrate:

- multi-cluster propagation with Karmada
- cluster-specific behavior through OverridePolicy
- traffic failover from `member1` to `member2`
- control plane convergence after the switch
- switch-back from `member2` to `member1`

The result is a compact but convincing multi-cluster Kubernetes demo that is practical for local development, workshops, and technical deep dives.