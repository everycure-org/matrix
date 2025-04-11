include "root" {
  # Needed to look up everything in root terragrunt.hcl. Else it will simply be a local tf project
  path = find_in_parent_folders("root.hcl")
}

inputs = {
  github_token      = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/common/github.yaml")).read_only_token
  gitops_repo_url   = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/common/github.yaml")).repo
  gitops_repo_creds = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/common/github.yaml")).creds
  k8s_secrets       = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/dev/dev_k8s_secrets.yaml"))
}
