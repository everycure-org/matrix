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