terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# GCP credentials via the GOOGLE_CREDENTIALS env var (or ADC on the host).
# This config is validated only (init -backend=false + validate + tflint);
# it is never plan/apply'd from the sandbox.
provider "google" {
  project = var.project_id
  region  = var.region
}
