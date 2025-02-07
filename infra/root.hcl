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
  root_directory = get_terragrunt_dir()
}

# Configure root level variables that all resources can inherit. This is especially helpful with multi-account configs
# where terraform_remote_state data sources are placed directly into the modules.
inputs = merge(
  local.globals
)

# generate a variables file for the above
generate default_variables {
  path      = "default_variables.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
variable "org_id" {}
variable "billing_account" {}
variable "default_region" {}
variable "super_admins" {}
EOF
}


# FIXME may need to be moved into another location 

# Configure Terragrunt to automatically store tfstate files 
remote_state {
  backend = "gcs"
  config = {
    bucket  = "mtrx-us-central1-hub-dev-storage"
    prefix  = "terragrunt/core/${path_relative_to_include()}/"
    skip_bucket_creation = true
  }
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite_terragrunt"
  }
}

