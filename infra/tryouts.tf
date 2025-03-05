generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
provider "google" {
  project          = "${local.project_id}"
  billing_project  = "${local.billing_project}"
}

provider "google-beta" {
  project          = "${local.project_id}"
  billing_project  = "${local.billing_project}"
}
EOF
}



#env_vars = read_terragrunt_config(find_in_parent_folders("auto.tfvars"))
#project_id          = lookup(local.env_vars.locals, "project_id")


# generate "provider" {
#   path      = "provider.tf"
#   if_exists = "overwrite_terragrunt"
#   contents  = <<EOF
# provider "google" {
#   project          = "mtrx-hub-dev-3of"
#   billing_project  = "core-422020"
# }
# provider "google-beta" {
#   project          = "mtrx-hub-dev-3of"
#   billing_project  = "core-422020"
# }
# EOF
# }