variable "namespace" {
  description = "Name of Kubernetes namespace for external-secrets."
  type        = string
  default     = "external-secrets"
}

variable "sa_annotations" {
  description = "Annotations for the service account. useful for workload identity"
  type        = map(string)
  default     = {}
}