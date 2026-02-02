---
author: Robin
pubDatetime: 2026-02-02T08:40:00-03:00
title: "Huawei Cloud Terraform Quickstart: VPC + ECS (Destroyable MVP)"
description: "A practical, reproducible quickstart to provision a Huawei Cloud VPC, subnet, security group, EIP, and an ECS VM using Terraform—plus how to tear it all down cleanly."
tags:
  - huawei-cloud
  - terraform
  - iaas
  - devops
  - quickstart
featured: true
draft: true
---

## TL;DR

This post gives you a **copy-paste Terraform MVP** that creates:
- VPC + Subnet
- Security Group (SSH)
- Elastic IP
- 1 ECS VM

…and a clean `terraform destroy` to remove everything.

> Goal: turn “I can’t use it” into “I can reproduce it in 30 minutes”.

---

## Why this matters

Many cloud evaluations fail not because the product is weak, but because teams can’t quickly achieve:
- **Reproducibility** (one command, same result)
- **Portability** (clear mapping vs AWS/Azure patterns)
- **Operability** (known pitfalls + troubleshooting)

Terraform-based quickstarts are the fastest way to build **engineering certainty**.

---

## Prerequisites

- Terraform >= 1.5
- A Huawei Cloud account (IAM user)
- Region/project selected (example: `ap-southeast-1` — replace with yours)

### Credentials (recommended)

Use environment variables (avoid committing secrets):

```bash
export HWC_ACCESS_KEY="..."
export HWC_SECRET_KEY="..."
export HWC_REGION_NAME="..."
export HWC_PROJECT_ID="..."
```

> Notes:
> - If you use a different auth mechanism (agency/STS), document it here.

---

## Repo structure (recommended)

Create a small repo like this:

```text
huawei-terraform-quickstarts/
  vpc-ecs/
    main.tf
    variables.tf
    outputs.tf
    versions.tf
    README.md
```

---

## Step-by-step

### 1) Initialize

```bash
terraform init
```

### 2) Plan

```bash
terraform plan \
  -var='name_prefix=demo' \
  -var='az=...'
```

### 3) Apply

```bash
terraform apply \
  -var='name_prefix=demo' \
  -var='az=...'
```

### 4) Verify

- Confirm VPC/Subnet are created
- Check ECS is running
- SSH using the EIP

```bash
ssh -i ~/.ssh/<your_key> root@<EIP>
```

---

## Common pitfalls

### Pitfall 1: Wrong project/region
Symptoms: resources not found / API errors.

Fix:
- verify `HWC_REGION_NAME`
- verify `HWC_PROJECT_ID`

### Pitfall 2: Security group rules
Symptoms: can’t SSH.

Fix:
- allow inbound TCP/22 from your public IP

---

## Cost hygiene

- Use small flavors
- Always destroy after testing

```bash
terraform destroy -auto-approve
```

---

## AWS/Azure mental mapping (quick)

- AWS VPC ≈ Huawei VPC
- AWS Subnet ≈ Huawei Subnet
- AWS Security Group ≈ Huawei Security Group
- AWS EIP ≈ Huawei EIP
- AWS EC2 ≈ Huawei ECS

---

## Next: make it production-ready

- Remote state (S3/OBS backend)
- Naming & tags
- Multi-env (dev/stage/prod)
- CI pipeline (plan/apply with approvals)

---

## What I need to finalize this post

To convert this draft into a fully runnable quickstart, I’ll fill in:
- Exact provider blocks and resource names
- A tested minimal `main.tf`/`variables.tf` set

If you tell me your **region** and whether you prefer **Ubuntu** or **openEuler**, I’ll publish a v1 that is fully runnable.
