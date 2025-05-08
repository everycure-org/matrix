variable "project_id" {
  description = "The GCP project ID."
  type        = string
}

variable "github_app_installation_id" {
  description = "The GitHub App installation ID."
  type        = string
}

variable "github_repo_owner" {
  description = "The owner of the GitHub repository."
  type        = string
}

variable "github_repo_name" {
  description = "The name of the GitHub repository."
  type        = string
}

variable "github_repo_token" {
  description = "The Personal Access Token."
  type        = string
}

variable "github_repo_branch_to_run_on" {
  description = "The branch to run the trigger on."
  type        = string
}

variable "github_repo_path_to_folder" {
  description = "The path to the folder where the terraform files are located."
  type        = string
}
