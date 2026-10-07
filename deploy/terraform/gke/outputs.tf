output "cluster_name" {
  description = "GKE cluster name."
  value       = google_container_cluster.gke.name
}

output "cluster_location" {
  description = "GKE cluster region."
  value       = google_container_cluster.gke.location
}

output "cluster_ca_certificate" {
  description = "Cluster API CA cert (base64)."
  value       = google_container_cluster.gke.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "cluster_endpoint" {
  description = "Cluster API endpoint."
  value       = google_container_cluster.gke.endpoint
  sensitive   = true
}

output "workload_identity_namespace" {
  description = "Workload pool for workload identity binding."
  value       = "${var.project_id}.svc.id.goog"
}

output "workload_identity_sa_email" {
  description = "GCP SA email to put in the iam.gke.io/gcp-service-account annotation."
  value       = google_service_account.speculative_agent_workloads.email
}

output "registry_repository" {
  description = "Artifact Registry docker push base (e.g. us-central1-docker.pkg.dev/<project>/speculative-agent-workloads)."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}

output "vpc_name" {
  description = "VPC network name."
  value       = google_compute_network.vpc.name
}

output "subnet_name" {
  description = "Subnet hosting the cluster."
  value       = google_compute_subnetwork.subnet.name
}

output "node_pool_name" {
  description = "L4 GPU node pool name."
  value       = google_container_node_pool.gpu.name
}
