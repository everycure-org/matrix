include "root" {
  path = find_in_parent_folders("root.hcl")
}

include "hub" {
  path = find_in_parent_folders("hub.hcl")
}

inputs = {
  gitops_repo_url   = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/common/github.yaml")).repo
  gitops_repo_creds = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/common/github.yaml")).creds
  k8s_secrets       = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}/secrets/prod/prod_k8s_secrets.yaml"))
}