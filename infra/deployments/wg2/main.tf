module "cloudbuild" {
  source                     = "../../modules/components/cloudbuild"
  github_repo_path_to_folder = var.github_repo_path_to_folder
  project_id                 = var.project_id
  github_app_installation_id = var.github_app_installation_id
  github_repo_owner          = var.github_repo_owner
  github_repo_name           = var.github_repo_name
  github_repo_token          = var.github_classic_token_for_cloudbuild
  github_repo_deploy_branch  = var.github_branch_to_run_on
  gitcrypt_key               = var.gitcrypt_key
  slack_webhook_url          = var.slack_webhook_url
}