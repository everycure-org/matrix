# ---------------------------------------------------------------------------------------------------------------------
# TERRAGRUNT CONFIGURATION
# Terragrunt is a thin wrapper for Terraform that provides extra tools for working with multiple Terraform modules,
# remote state, and locking: https://github.com/gruntwork-io/terragrunt
# ---------------------------------------------------------------------------------------------------------------------

locals {
  # Automatically load level specific variables
  # initiative_vars = read_terragrunt_config(find_in_parent_folders("initiative.hcl"))
  //   wg_vars = read_terragrunt_config(find_in_parent_folders("workinggroup.hcl"))

  # Extract the variables we need for easy access
  # initiative_name = local.initiative_vars.locals.initiative_name
  # wg_name   = local.account_vars.locals.wg_vars.wg_name
  # ---------------------------------------------------------------------------------------------------------------------
  # GLOBAL PARAMETERS
  # These variables apply to all configurations in this subfolder. These are automatically merged into the child
  # `terragrunt.hcl` config via the include block.
  # ---------------------------------------------------------------------------------------------------------------------
  globals = {
    org_id          = "207929716073"
    billing_account = "01980D-0D4096-C8CEA4"
    default_region  = "us-central1"
    super_admins    = ["gcp-admins@everycure.org"]
  }
  root_directory  = get_terragrunt_dir()
  environment     = basename(get_terragrunt_dir()) # dev or prod
  deployment_path = get_original_terragrunt_dir()
  env_vars        = read_terragrunt_config("${local.deployment_path}/env.hcl")
}

# Configure root level variables that all resources can inherit. This is especially helpful with multi-account configs
# where terraform_remote_state data sources are placed directly into the modules.
inputs = merge(
  local.globals,
  { "environment" = local.environment },
  local.env_vars.locals
)

# generate a variables file for the above
generate default_variables {
  path      = "default_variables.gen.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
variable "org_id" {}
variable "billing_account" {}
variable "default_region" {}
variable "super_admins" {}
variable "environment" {}
variable "project_id" {}
variable "billing_project" {}
variable "storage_bucket_name" {}
variable "gitops_repo_url" {}
variable "gitops_repo_creds" {}
variable "github_app_installation_id" {}
variable "github_repo_owner" {}
variable "github_repo_name" {}
variable "github_branch_to_run_on" {}
variable "infra_bucket_name" {}
variable "github_classic_token_for_cloudbuild" {}
variable "github_repo_path_to_folder" {}
variable "gitcrypt_key" {}
variable "slack_webhook_url" {}
EOF
}


# FIXME may need to be moved into another location 

# Configure Terragrunt to automatically store tfstate files 
remote_state {
  backend = "gcs"
  config = {
    bucket               = local.env_vars.locals.infra_bucket_name
    skip_bucket_creation = true
  }
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite_terragrunt"
  }
}

generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
provider "google" {
  project          = var.project_id
  billing_project  = var.billing_project
}
provider "google-beta" {
  project          = var.project_id
  billing_project  = var.billing_project
}
EOF
}
