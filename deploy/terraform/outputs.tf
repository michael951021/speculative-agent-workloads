# Re-exposed from the ./gke module (see gke/outputs.tf for documentation).
output "cluster_name" {
  description = "GKE cluster name."
  value       = module.gke.cluster_name
}

output "cluster_location" {
  description = "GKE cluster region."
  value       = module.gke.cluster_location
}

output "cluster_ca_certificate" {
  description = "Cluster API CA cert (base64)."
  value       = module.gke.cluster_ca_certificate
  sensitive   = true
}

output "cluster_endpoint" {
  description = "Cluster API endpoint."
  value       = module.gke.cluster_endpoint
  sensitive   = true
}

output "workload_identity_namespace" {
  description = "identity_namespace for workload identity binding."
  value       = module.gke.workload_identity_namespace
}

output "workload_identity_sa_email" {
  description = "GCP SA email for the iam.gke.io/gcp-service-account annotation."
  value       = module.gke.workload_identity_sa_email
}

output "registry_repository" {
  description = "Artifact Registry docker repository URL (push base)."
  value       = module.gke.registry_repository
}

output "vpc_name" {
  description = "VPC network name."
  value       = module.gke.vpc_name
}

output "subnet_name" {
  description = "Subnet hosting the cluster."
  value       = module.gke.subnet_name
}

output "node_pool_name" {
  description = "L4 GPU node pool name."
  value       = module.gke.node_pool_name
}
