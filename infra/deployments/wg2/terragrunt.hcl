include "root" {
  # Needed to look up everything in root terragrunt.hcl. Else it will simply be a local tf project
  path = find_in_parent_folders("root.hcl")
}

inputs = {
  github_token               = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/dev/matrix/github.yaml")).read_only_token
  gitops_repo_url            = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/dev/matrix/github.yaml")).repo
  gitops_repo_creds          = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/dev/matrix/github.yaml")).creds
  k8s_secrets                = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/dev/matrix/dev_k8s_secrets.yaml"))
  github_app_installation_id = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/dev/matrix/github.yaml")).github_app_installation_id
  github_repo_owner          = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/dev/matrix/github.yaml")).repo_owner
  github_repo_name           = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/dev/matrix/github.yaml")).repo_name
  github_branch_to_run_on    = "infra"
}
