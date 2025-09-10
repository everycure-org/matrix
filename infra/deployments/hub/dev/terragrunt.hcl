include "root" {
  # Needed to look up everything in root terragrunt.hcl. Else it will simply be a local tf project
  path = find_in_parent_folders()
}

inputs = {
  gitops_repo_url   = yamldecode(file("${dirname(find_in_parent_folders())}//secrets/github.yaml")).repo
  gitops_repo_creds = yamldecode(file("${dirname(find_in_parent_folders())}//secrets/github.yaml")).creds
  k8s_secrets       = yamldecode(file("${dirname(find_in_parent_folders())}//secrets/k8s_secrets.yaml"))
}

generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
provider "google" {
  project          = "mtrx-hub-dev-3of"
  billing_project  = "core-422020"
}
provider "google-beta" {
  project          = "mtrx-hub-dev-3of"
  billing_project  = "core-422020"
}
EOF
}