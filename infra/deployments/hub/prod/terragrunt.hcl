include "root" {
  path = find_in_parent_folders("root.hcl")
}

include "hub" {
  path = find_in_parent_folders("hub.hcl")
}

inputs = {
  gitops_repo_url                     = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/matrix/github.yaml")).repo
  gitops_repo_creds                   = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/matrix/github.yaml")).creds
  k8s_secrets                         = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/matrix/prod_k8s_secrets.yaml"))
  github_app_installation_id          = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/matrix/github.yaml")).github_app_installation_id
  github_repo_owner                   = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/matrix/github.yaml")).repo_owner
  github_repo_name                    = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/matrix/github.yaml")).repo_name
  github_branch_to_run_on             = "main"
  github_classic_token_for_cloudbuild = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/matrix/github.yaml")).github_classic_token_for_cloudbuild
  github_repo_path_to_folder          = get_path_from_repo_root()
  gitcrypt_key                        = filebase64("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/git-crypt.key")
  slack_webhook_url                   = file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/slack_webhook_url.txt")
}