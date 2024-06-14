include "root" {
  # Needed to look up everything in root terragrunt.hcl. Else it will simply be a local tf project
  path = find_in_parent_folders()
}

inputs = {
  parent_folder = dependency.folders.folder_ids["matrix_prod"]
}