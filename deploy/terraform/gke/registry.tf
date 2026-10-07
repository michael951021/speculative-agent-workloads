# Container images (gateway, engine, vllm) are pushed here by the host build.
resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = var.registry_repo_id
  description   = "speculative-agent-workloads container images"
  format        = "DOCKER"

  cleanup_policies {
    id = "keep-last-10"
    most_recent_versions {
      keep_count = 10
    }
  }
}
