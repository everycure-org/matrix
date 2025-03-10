include "root" {
  # Needed to look up everything in root terragrunt.hcl. Else it will simply be a local tf project
  path = find_in_parent_folders("root.hcl")
}

inputs = {
  gitops_repo_url   = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}//secrets/github.yaml")).repo
  gitops_repo_creds = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}//secrets/github.yaml")).creds
  k8s_secrets       = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}//secrets/k8s_secrets.yaml"))
}
