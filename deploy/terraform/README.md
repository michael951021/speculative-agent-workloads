# Terraform: GKE for draftlab

GCP infrastructure as code for the cluster the `deploy/helm` gpu profile targets.
**Never plan/apply from the sandbox** — validated only
(`terraform init -backend=false && terraform validate` + `tflint`, wired into `make check`).

## Layout

- Root (`.`): `versions.tf` (provider, pinned `hashicorp/google ~> 5.0`),
  `variables.tf` / `outputs.tf` (pass-through to the module), `main.tf`.
- `gke/` (child module): `network.tf` (VPC + regional subnet with `pods`/`services`
  secondary ranges), `cluster.tf` (GKE Standard, regional, VPC-native, Workload
  Identity on, `initial_node_count = 0` + L4 GPU node pool: `nvidia-l4`, autoscale
  0–2, `NO_SCHEDULE` taint `draftlab.io/gpu`), `registry.tf` (Artifact Registry
  docker repo, keep-last-10 cleanup), `workload_identity.tf` (GCP SA +
  `roles/artifactregistry.reader`).

## Apply (host only, with credentials)

```sh
cd deploy/terraform
export GOOGLE_CREDENTIALS="$(gcloud auth print-access-token ...)"  # or ADC
terraform init && terraform apply -var project_id=<proj>
```

After apply:

- kubectl context: `gcloud container clusters get-credentials <cluster> --region <region>`
- Bind the workload identity SA:
  `kubectl annotate sa default -n kube-system iam.gke.io/gcp-service-account=<output workload_identity_sa_email>`
  (the chart does this for its own SAs).
- Push images to `<output registry_repository>` and point the helm
  `image.registry` values at it.

## Key knobs (root variables, all have defaults except project_id)

| variable | default | purpose |
| --- | --- | --- |
| `project_id` | — | GCP project |
| `region` | us-central1 | subnet + cluster region |
| `cluster_name` | draftlab-gke | GKE cluster name |
| `vpc_name` / `subnet_name` | draftlab-vpc / draftlab-subnet | network names |
| `subnet_cidr` / `pods_cidr` / `services_cidr` | 10.10.0.0/22, 10.128.0.0/14, 10.144.0.0/20 | IP plan |
| `gpu_machine_type` | g2-standard-12 | smallest g2 that hosts an L4 |
| `gpu_count` | 1 | L4s per node |
| `gpu_node_min` / `gpu_node_max` | 0 / 2 | pool autoscaling bounds |
| `gpu_taint_key` / `gpu_taint_value` | draftlab.io/gpu / true | NO_SCHEDULE taint |
| `service_account_id` | draftlab-wi | Workload Identity GCP SA |
| `registry_repo_id` | draftlab | Artifact Registry repo id |

Outputs: cluster name/location/endpoint/CA, `workload_identity_namespace` +
`workload_identity_sa_email`, `registry_repository`, network names, pool name.
