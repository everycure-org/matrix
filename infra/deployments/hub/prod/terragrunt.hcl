include "root" {
  path = find_in_parent_folders("root.hcl")
}

include "hub" {
  path = find_in_parent_folders("hub.hcl")
}

inputs = {
  gitops_repo_url            = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/matrix/github.yaml")).repo
  gitops_repo_creds          = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/matrix/github.yaml")).creds
  k8s_secrets                = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/matrix/prod_k8s_secrets.yaml"))
  github_app_installation_id = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/matrix/github.yaml")).github_app_installation_id
  github_repo_owner          = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/matrix/github.yaml")).repo_owner
  github_repo_name           = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/matrix/github.yaml")).repo_name
  github_branch_to_run_on    = "infra-prod-debug"
}