resource "google_cloudbuild_trigger" "terrgrunt_trigger" {
  project  = var.project_id
  name     = "terragrunt-on-${var.github_repo_branch_to_run_on}-push"
  location = "global"

  source_to_build {
    ref       = "refs/heads/${var.github_repo_branch_to_run_on}"
    repo_type = "GITHUB"
  }

  github {
    owner = var.github_repo_owner
    name  = var.github_repo_name
    pull_request {
      branch = "^${var.github_repo_branch_to_run_on}$"
    }
    push {
      branch = "^${var.github_repo_branch_to_run_on}$"
    }
  }

  build {
    step {
      name = "gcr.io/cloud-builders/docker"
      args = ["pull", "alpine/terragrunt:1.11.4"]
      id   = "pull-terragrunt"
    }

    step {
      name       = "alpine/terragrunt:1.11.4"
      entrypoint = "terragrunt"
      args       = ["run-all", "init", "-reconfigure"]
      dir        = "$${_TERRAGRUNT_DIR}"
      id         = "terragrunt-init-reconfigure"
      wait_for   = ["pull-terragrunt"]
    }

    step {
      name       = "alpine/terragrunt:1.11.4"
      entrypoint = "terragrunt"
      args       = ["run-all", "plan", "-out=plan.tfplan", "-no-color"]
      dir        = "$${_TERRAGRUNT_DIR}"
      id         = "terragrunt-plan"
      wait_for   = ["terragrunt-init-reconfigure"]
    }

    step {
      name       = "alpine/terragrunt:1.11.4"
      entrypoint = "bash"
      args = [
        "-c",
        <<-EOF
        echo "Checking condition: Branch='$BRANCH_NAME'"

        if [ "$BRANCH_NAME" = "${var.github_repo_branch_to_run_on}" ]; then
          echo "Condition met (Branch=$BRANCH_NAME). Running 'terragrunt apply'..."
          terragrunt run-all apply --terragrunt-non-interactive plan.tfplan
        else
          echo "Skipping apply: Branch is '$BRANCH_NAME', not 'main'. Showing plan instead."
          terragrunt run-all show plan.tfplan --no-color
        fi
        EOF
      ]
      dir      = "$${_TERRAGRUNT_DIR}"
      id       = "terragrunt-apply"
      wait_for = ["terragrunt-plan"]
    }
  }
  // The trigger will only run if there is any changes in the respective folder. (e.g. infra/deployments/hub/dev)
  // This is to avoid running the trigger for any changes in the other folders.
  included_files = [
    "$${_TERRAGRUNT_DIR}/**}"
  ]

  substitutions = {
    _TERRAGRUNT_DIR = "./infra/${var.github_repo_path_to_folder}"
  }

  service_account = google_service_account.cloudbuild_sa.email
}