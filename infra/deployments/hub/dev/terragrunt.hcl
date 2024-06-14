include "root" {
  # Needed to look up everything in root terragrunt.hcl. Else it will simply be a local tf project
  path = find_in_parent_folders()
}

inputs = {
  # TODO still need to add git-crypt here
  gitops_repo_url   = yamldecode(file("${dirname(find_in_parent_folders())}//secrets/github.yaml")).repo
  gitops_repo_creds = yamldecode(file("${dirname(find_in_parent_folders())}//secrets/github.yaml")).creds
}