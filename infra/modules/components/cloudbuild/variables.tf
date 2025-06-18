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

variable "terragrunt_container_image" {
  description = "The Docker image to use for the Terragrunt container."
  type        = string
  default     = "ghcr.io/devops-infra/docker-terragrunt:aws-gcp-tf-1.11.4-tg-0.78.4"
}

variable "require_manual_approval" {
  description = "Whether manual approval is required for the Cloud Build trigger."
  type        = bool
  default     = false
}