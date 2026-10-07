# Root exposes every input as a pass-through to the ./gke module.
# project_id has no default; everything else below has a working default.

variable "project_id" {
  description = "GCP project to build the cluster in."
  type        = string
}

variable "region" {
  description = "GCP region for the VPC subnet and (regional) GKE cluster."
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "Name of the GKE Standard cluster."
  type        = string
  default     = "speculative-agent-workloads-gke"
}

variable "node_pool_name" {
  description = "Name of the L4 GPU node pool."
  type        = string
  default     = "gpu-l4"
}

variable "vpc_name" {
  description = "VPC network name."
  type        = string
  default     = "speculative-agent-workloads-vpc"
}

variable "subnet_name" {
  description = "Subnet in the VPC that the cluster's pods/services draw IPs from."
  type        = string
  default     = "speculative-agent-workloads-subnet"
}

variable "subnet_cidr" {
  description = "Primary CIDR for the subnet."
  type        = string
  default     = "10.10.0.0/22"
}

variable "pods_cidr" {
  description = "Secondary CIDR reserved for cluster pod IPs."
  type        = string
  default     = "10.128.0.0/14"
}

variable "services_cidr" {
  description = "Secondary CIDR reserved for cluster service (LB) IPs."
  type        = string
  default     = "10.144.0.0/20"
}

variable "gpu_machine_type" {
  description = "Node machine type for the L4 GPU (L4 needs g2-standard-12+)."
  type        = string
  default     = "g2-standard-12"
}

variable "gpu_count" {
  description = "Number of nvidia-l4 accelerators per GPU node."
  type        = number
  default     = 1
}

variable "gpu_node_min" {
  description = "Autoscaling lower bound for the GPU pool (0 lets it scale away)."
  type        = number
  default     = 0
}

variable "gpu_node_max" {
  description = "Autoscaling upper bound for the GPU pool."
  type        = number
  default     = 2
}

variable "gpu_taint_key" {
  description = "Taint key applied to GPU nodes so non-GPU pods stay off them."
  type        = string
  default     = "speculative-agent-workloads.io/gpu"
}

variable "gpu_taint_value" {
  description = "Taint value paired with gpu_taint_key."
  type        = string
  default     = "true"
}

variable "service_account_id" {
  description = "ID for the Workload Identity service account (lowercase, <=28 chars)."
  type        = string
  default     = "speculative-agent-workloads-wi"
}

variable "registry_repo_id" {
  description = "Artifact Registry repository id for container images."
  type        = string
  default     = "speculative-agent-workloads"
}
