# Workload Identity service account. A K8s ServiceAccount is bound to it by
# the Helm chart via the annotation:
#   iam.gke.io/gcp-service-account: <email from output workload_identity_sa_email>
# The cluster's identity namespace (see workload_identity_config) makes that
# binding impersonate this GCP SA without workload-identity user setup.

resource "google_service_account" "speculative_agent_workloads" {
  project      = var.project_id
  account_id   = var.service_account_id
  display_name = "speculative-agent-workloads Workload Identity"
}

# Let the pool pull (and push, for image sync) from the Artifact Registry.
resource "google_project_iam_member" "registry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.speculative_agent_workloads.email}"
}
