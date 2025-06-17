resource "google_cloudbuild_trigger" "terrgrunt_trigger" {
  project     = var.project_id
  name        = "terragrunt-on-${var.github_repo_deploy_branch}-push-for-${var.project_id}"
  location    = var.location
  description = "Trigger for terragrunt apply on push to ${var.github_repo_deploy_branch} branch for project ${var.project_id}."

  approval_config {
    approval_required = var.require_manual_approval
  }

  repository_event_config {
    repository = google_cloudbuildv2_repository.matrix_repo.id
    push {
      branch = "^${var.github_repo_deploy_branch}$"
    }
  }

  // The trigger will only run if there is any changes in the respective folder. (e.g. infra/deployments/hub/dev)
  // This is to avoid running the trigger for any changes in the other folders.
  included_files = [
    "${var.github_repo_path_to_folder}/**",
    "infra/modules/**"
  ]


  build {
    options {
      logging = "CLOUD_LOGGING_ONLY"
    }

    available_secrets {
      secret_manager {
        env          = "GIT_CRYPT_KEY"
        version_name = google_secret_manager_secret_version.gitcrypt_key_version.id
      }
      secret_manager {
        env          = "GITHUB_TOKEN"
        version_name = google_secret_manager_secret_version.github_token_version.id
      }
    }
    // The following steps are executed in the order they are defined.
    // The first step initializes the git submodules
    step {
      name       = var.terragrunt_container_image
      entrypoint = "bash"
      args = [
        "-c",
        <<-EOF
        git config --global url."https://$$${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
        git submodule update --init --recursive
        EOF
      ]
      id         = "git-submodule-init"
      secret_env = ["GITHUB_TOKEN"]
    }
    // The second step unlocks the git-crypt key
    step {
      name       = var.terragrunt_container_image
      entrypoint = "bash"
      args = [
        "-c",
        <<-EOF
        apt-get update && apt-get install -y git-crypt
        echo "$$GIT_CRYPT_KEY" | base64 -d > TMP_KEY
        git-crypt unlock TMP_KEY && rm TMP_KEY
        EOF
      ]
      dir        = "./infra/secrets"
      id         = "git-crypt-unlock"
      secret_env = ["GIT_CRYPT_KEY"]
      wait_for   = ["git-submodule-init"]
    }
    // The third step initializes the terragrunt configuration
    // and reconfigures it to ensure that the latest changes are applied.
    step {
      name       = var.terragrunt_container_image
      entrypoint = "terragrunt"
      args       = ["run-all", "init", "-reconfigure"]
      dir        = "$${_TERRAGRUNT_DIR}"
      id         = "terragrunt-init-reconfigure"
      wait_for   = ["git-crypt-unlock"]
    }
    // The fourth step runs the terragrunt plan command
    step {
      name       = var.terragrunt_container_image
      entrypoint = "terragrunt"
      args       = ["run-all", "plan", "-out=plan.tfplan", "-no-color"]
      dir        = "$${_TERRAGRUNT_DIR}"
      id         = "terragrunt-plan"
      wait_for   = ["terragrunt-init-reconfigure"]
    }
    // The fifth step checks the branch condition and runs the terragrunt apply command if the condition is met.
    step {
      name       = var.terragrunt_container_image
      entrypoint = "bash"
      args = [
        "-c",
        <<-EOF
        echo "Checking condition: Branch='$BRANCH_NAME'"
        if [ "$BRANCH_NAME" = "${var.github_repo_deploy_branch}" ]; then
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

  substitutions = {
    _TERRAGRUNT_DIR = "${var.github_repo_path_to_folder}"
  }

  service_account = google_service_account.cloudbuild_sa.id
  // this is to avoid sending the build logs to Github.
  include_build_logs = "INCLUDE_BUILD_LOGS_UNSPECIFIED"
}