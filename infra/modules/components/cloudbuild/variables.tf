variable "project_id" {
  description = "The GCP project ID."
  type        = string
}

variable "github_app_installation_id" {
  description = "The GitHub App installation ID."
  type        = string
}

variable "repo_owner" {
  description = "The owner of the GitHub repository."
  type        = string
}

variable "repo_name" {
  description = "The name of the GitHub repository."
  type        = string
}

variable "PAT" {
  description = "The Personal Access Token."
  type        = string
  default     = ""
}

variable "repo_branch_to_run_on" {
  description = "The branch to run the trigger on."
  type        = string
}

variable "repo_filename" {
  description = "The filename of the trigger."
  type        = string
  default     = "./infra/cloudbuild.yaml"
}