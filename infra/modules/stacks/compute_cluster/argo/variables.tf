variable "namespace" {
  default = "argocd"
  type    = string
}

variable "repo_url" {
  type = string
}

variable "repo_creds" {
  type = string
}

variable "repo_path" {
  description = "Path in repo to deploy by ArgoCD."
  type        = string
  default     = "infra/argo/app-of-apps/"
}

variable "repo_revision" {
  description = "Revision to deploy by ArgoCD."
  type        = string
}

variable "environment" {
  type = string
}

variable "project_id" {
  type = string
}

variable "bucket_name" {
  type = string
}

variable "aip_oauth_client_id" {
  type = string
}

variable "metrics_bucket_name" {
  type = string
}