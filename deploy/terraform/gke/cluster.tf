# GKE Standard cluster, regional, VPC-native (pods+services IP ranges),
# Workload Identity enabled. initial_node_count = 0 so no default node pool;
# the only pool is the explicit L4 GPU pool below (autoscale 0-max, tainted).

resource "google_container_cluster" "gke" {
  name               = var.cluster_name
  location           = var.region
  initial_node_count = 0

  network       = google_compute_network.vpc.name
  subnetwork    = google_compute_subnetwork.subnet.name
  network_policy {
    enabled = true
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Workload Identity: pods' GCP SA is bound to a K8s SA via this pool.
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Autoscaling is driven per node pool (min 0); enable the cluster flag.
  cluster_autoscaling {
    enabled = true
  }

  vertical_pod_autoscaling {
    enabled = true
  }

  # GPU pool is tainted NO_SCHEDULE, so allow the cluster to be scale-to-zero.
  deletion_protection = false

  release_channel {
    channel = "REGULAR"
  }

  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  private_cluster_config {
    enable_private_nodes = true
    master_global_access_config {
      enabled = true
    }
  }

  logging_config {
    enable_components = ["SYSTEM_COMPONENTS"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
  }
}

# L4 GPU node pool: autoscales 0..max and carries a NO_SCHEDULE taint so only
# pods that tolerate it (the vLLM/engine replicas) land on GPUs.
resource "google_container_node_pool" "gpu" {
  name     = var.node_pool_name
  location = var.region
  cluster  = google_container_cluster.gke.name

  autoscaling {
    min_node_count = var.gpu_node_min
    max_node_count = var.gpu_node_max
    location_policy = "ANY"
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = var.gpu_machine_type
    image_type   = "COS"
    disk_size_gb = 500
    disk_type    = "pd-ssd"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    taint {
      key    = var.gpu_taint_key
      value  = var.gpu_taint_value
      effect = "NO_SCHEDULE"
    }

    guest_accelerator {
      type  = "nvidia-l4"
      count = var.gpu_count
    }

    labels = {
      "speculative-agent-workloads.io/pool" = "gpu"
    }
  }
}
