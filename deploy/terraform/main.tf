# Root passes all inputs through to the ./gke module (VPC, cluster, GPU pool,
# Artifact Registry, Workload Identity SA) and re-exposes its outputs.
module "gke" {
  source             = "./gke"
  project_id         = var.project_id
  region             = var.region
  cluster_name       = var.cluster_name
  node_pool_name     = var.node_pool_name
  vpc_name           = var.vpc_name
  subnet_name        = var.subnet_name
  subnet_cidr        = var.subnet_cidr
  pods_cidr          = var.pods_cidr
  services_cidr      = var.services_cidr
  gpu_machine_type   = var.gpu_machine_type
  gpu_count          = var.gpu_count
  gpu_node_min       = var.gpu_node_min
  gpu_node_max       = var.gpu_node_max
  gpu_taint_key      = var.gpu_taint_key
  gpu_taint_value    = var.gpu_taint_value
  service_account_id = var.service_account_id
  registry_repo_id   = var.registry_repo_id
}
