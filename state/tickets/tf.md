# tf — Terraform GKE

## Issue
The Helm chart's gpu profile targets a GKE node pool that only exists in someone's head: no VPC/subnet with GKE IP ranges, no cluster, no L4 GPU pool with taint, no Artifact Registry, no Workload Identity SA. The chart cannot be installed anywhere.

## Question
Can `deploy/terraform` (root) + `deploy/terraform/gke/` (child-dir files of the same root module) pass `terraform init -backend=false && terraform validate` and `tflint` cleanly, while creating: VPC + subnet (with GKE pod/service IP ranges), GKE Standard cluster (Workload Identity on), L4 GPU node pool (autoscale 0-2, NO_SCHEDULE taint), Artifact Registry repo, and a Workload Identity service account? Refutation: validate fails, or any of the six named resources is missing/wrong shape in the .tf.

## Evidence
- Observed: `deploy/terraform/` does not exist; `tools/check.sh:64-72` expects `deploy/terraform` as the terraform root (init -backend=false + validate + tflint). Terraform 1.x and tflint in `.tools/bin`.
- Check: `cd deploy/terraform && terraform init -backend=false -input=false && terraform validate && tflint --init && tflint` — expected exit 0. Log: `/work/.agent-evidence/tf-check.log`.
- Known-answer checks: (1) `terraform validate` is the reference validator itself (HCL+schema check by the HashiCorp binary); (2) structural grep must show autoscaling 0-2, `nvidia-l4` machine, taint, `workload_identity_config`, secondary ranges (pods+services), `artifact_registry_repository`, `google_service_account` — expected all present.
- Plausible range: tflint issues 0; validate "Success! The configuration is valid."

## Done when
`cd deploy/terraform && terraform init -backend=false -input=false && terraform validate` exits 0; `tflint` exits 0 with no issues; the grep checklist above passes. `make check`'s terraform section green. No plan/apply ever run. Regression: check.sh already gates it in make check.

## Stop when
Out of scope: plan/apply, real GCP credentials, other clouds, Terraform modules for CPU-only nodes. Stop/split: if the registry (hashicorp/google provider) is unreachable from the sandbox after two attempts, record blocker + consider a filesystem provider mirror or pinning; do not loop on network.

## Plan
1. Root at deploy/terraform/ (versions/variables/outputs), resources in deploy/terraform/gke/ (auto-loaded as same root module).
2. Run the acceptance check once (expect failure: no files), then write configs, re-run to green.
3. Structural grep checklist. NOTES.md + commit + merge master.
