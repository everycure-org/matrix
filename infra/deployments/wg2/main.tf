module "cloudbuild" {
  source                     = "../../modules/components/cloudbuild"
  github_repo_path_to_folder = path.module
  project_id                 = var.project_id
  github_app_installation_id = var.github_app_installation_id
  github_repo_owner          = var.github_repo_owner
  github_repo_name           = var.github_repo_name
  github_repo_token          = var.gitops_repo_creds
  github_repo_deploy_branch  = var.github_branch_to_run_on
}