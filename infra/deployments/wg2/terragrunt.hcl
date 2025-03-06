include "root" {
  # Needed to look up everything in root terragrunt.hcl. Else it will simply be a local tf project
  path = find_in_parent_folders("root.hcl")
}

generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
provider "google" {
  project          = "mtrx-wg2-modeling-dev-9yj"
  billing_project  = "core-422020"
}
provider "google-beta" {
  project          = "mtrx-wg2-modeling-dev-9yj"
  billing_project  = "core-422020"
}
EOF
}

inputs = {
  github_token = yamldecode(file("${dirname(find_in_parent_folders("root.hcl"))}//secrets/github.yaml")).read_only_token
}
