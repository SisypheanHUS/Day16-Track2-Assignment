# Lab 16: Cloud AI Environment Setup — CPU Fallback Path (GCP)

## Overview

This project deploys a cloud-based ML infrastructure on **Google Cloud Platform** using **Terraform** (Infrastructure as Code). Since GPU quota is not available on new GCP accounts, the CPU fallback path was used: deploying a **LightGBM** gradient boosting model on a CPU instance instead of vLLM + Gemma on a GPU instance.

The full pipeline covers: Terraform IaC -> Cloud VM provisioning -> ML training -> Inference benchmarking -> Billing verification -> Resource cleanup.

## Architecture

```
Internet
   |
   v
[GCP External Load Balancer] (port 80)
   |
   v
[Cloud NAT + Cloud Router]
   |
   v
[Private Subnet: 10.0.0.0/24]
   |
   v
[e2-standard-8 VM] (8 vCPU, 32 GB RAM, Debian 12)
   - LightGBM benchmark
   - Credit Card Fraud Detection dataset (284,807 transactions)
```

Key infrastructure components:
- **VPC** with private subnet (no public IP on compute node)
- **Cloud NAT** for outbound internet access (Docker/model downloads)
- **IAP SSH tunneling** for secure access (no bastion host needed)
- **Firewall rules** restricted to IAP and health check IP ranges
- **Service Account** with least-privilege (logging + monitoring only)

## How I Did It

### 1. GCP Account Setup
- Created a new GCP account with $300 free trial credit
- Created project `day16-494216`
- Enabled Compute Engine and IAM APIs

### 2. GPU Quota Issue
GCP blocks GPU quota (default = 0) for new accounts. Requesting an increase can take hours to days and is often denied for accounts without billing history. Instead of waiting, I switched to the **CPU fallback path** as described in the lab instructions.

### 3. Terraform Modifications for CPU Path
Modified `terraform-gcp/` to remove GPU dependencies:
- Changed `machine_type` from `n1-standard-4` to `e2-standard-8` (8 vCPU, 32 GB RAM)
- Removed `guest_accelerator` block (no GPU attached)
- Changed `on_host_maintenance` from `TERMINATE` to `MIGRATE`
- Switched boot image from Deep Learning VM to standard `debian-12`

### 4. Infrastructure Deployment
```bash
cd terraform-gcp
terraform init
terraform apply -var="project_id=day16-494216" -var="hf_token=dummy"
```
All 16 resources created in under 3 minutes (VPC, subnet, NAT, router, firewall rules, VM, instance group, health check, backend service, URL map, HTTP proxy, forwarding rule, service account, IAM bindings).

### 5. ML Benchmark Execution
SSHed into the VM via IAP tunnel:
```bash
gcloud compute ssh ai-gpu-node --zone=us-central1-a --tunnel-through-iap
```

Installed dependencies and ran the LightGBM benchmark on the Credit Card Fraud Detection dataset (Kaggle).

### 6. Resource Cleanup
Destroyed all infrastructure immediately after collecting results:
```bash
terraform destroy
```

## Benchmark Results (GCP `e2-standard-8`)

| Metric | Result |
|---|---|
| Data load time | 1.531 s |
| Training time | 1.263 s |
| Best iteration | 45 |
| AUC-ROC | 0.899812 |
| Accuracy | 0.962852 |
| F1-Score | 0.071930 |
| Precision | 0.037580 |
| Recall | 0.836735 |
| Inference latency (1 row) | 0.4347 ms |
| Inference throughput (1000 rows) | 0.9432 ms |

## CPU vs GPU: Why CPU Was Used

- **GPU quota unavailable**: New GCP accounts have 0 GPU quota by default. The approval process is slow and unreliable for accounts without billing history.
- **Cost comparison**: `e2-standard-8` costs ~$0.27/hr vs GPU path (`n1-standard-4` + T4) at ~$0.54/hr — the CPU path is actually cheaper.
- **Workload fit**: LightGBM (gradient boosting) is inherently a CPU workload. Training completed in 1.26 seconds on 284K rows — adding a GPU would provide no benefit for this algorithm.
- **Availability**: CPU instances are available immediately without special quota requests, making them practical for lab environments with time constraints.

## Project Structure

```
.
├── README.md                  # This file
├── README_aws.md              # Lab instructions (AWS version)
├── README_gcp.md              # Lab instructions (GCP version)
├── benchmark.py               # LightGBM benchmark script
├── benchmark_result.json      # Benchmark output from GCP run
├── Screenshot VM.png          # Terminal output of benchmark
├── Screenshot Billing.png     # GCP Billing showing costs
├── .env.example               # Environment variable template
├── terraform/                 # AWS Terraform (GPU path)
│   ├── main.tf
│   ├── variables.tf
│   ├── providers.tf
│   ├── outputs.tf
│   └── user_data.sh
└── terraform-gcp/             # GCP Terraform (CPU path — used)
    ├── main.tf
    ├── variables.tf
    ├── providers.tf
    ├── outputs.tf
    └── user_data.sh
```

## Prerequisites

- [Terraform](https://www.terraform.io/downloads) >= 1.0
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud` CLI)
- GCP account with billing enabled
- Python 3.x, LightGBM, scikit-learn, pandas (on the VM)
- Kaggle account (for dataset download)

## Quick Start (Reproduce)

```bash
# 1. Authenticate with GCP
gcloud auth login
gcloud auth application-default login
gcloud config set project <YOUR_PROJECT_ID>

# 2. Enable APIs
gcloud services enable compute.googleapis.com iam.googleapis.com

# 3. Deploy infrastructure
cd terraform-gcp
terraform init
terraform apply -var="project_id=<YOUR_PROJECT_ID>" -var="hf_token=dummy"

# 4. SSH into VM
gcloud compute ssh ai-gpu-node --zone=us-central1-a --tunnel-through-iap

# 5. On the VM: install deps, download data, run benchmark
sudo apt-get update -y && sudo apt-get install -y python3 python3-pip
pip3 install --break-system-packages lightgbm scikit-learn pandas kaggle
mkdir -p ~/ml-benchmark && cd ~/ml-benchmark
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p .
python3 benchmark.py

# 6. IMPORTANT: Destroy resources when done
terraform destroy
```
