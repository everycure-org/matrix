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

variable "github_repo_deploy_branch" {
  description = "The branch to listen to for deployment."
  type        = string
}

variable "github_repo_path_to_folder" {
  description = "The path to the folder where the terraform files are located."
  type        = string
}

variable "location" {
  default = "us-central1"
}

variable "gitcrypt_key" {
  description = "The gitcrypt key used to unlock the secrets repository."
  type        = string
}

variable "slack_webhook_url" {
  description = "The Slack webhook URL for notifications."
  type        = string
}