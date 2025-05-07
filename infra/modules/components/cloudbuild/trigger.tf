resource "google_cloudbuild_trigger" "trigger" {
  project  = var.project_id
  name     = "on-${var.repo_branch_to_run_on}-push"
  location = "global"

  github {
    owner = var.repo_owner
    name  = var.repo_name
    push {
      branch = "^${var.repo_branch_to_run_on}$"
    }
  }

  build {
    step {
      name       = "gcr.io/cloud-builders/gsutil"
      args       = ["cp", "gs://mybucket/remotefile.zip", "localfile.zip"]
      timeout    = "120s"
      secret_env = ["MY_SECRET"]
    }

    step {
      name   = "ubuntu"
      script = "echo hello" # using script field
    }
  }

  filename = var.repo_filename

  service_account = google_service_account.cloudbuild_sa.email
  substitutions = {
    _MY_SUBSTITUTION = "my_value"
  }
}