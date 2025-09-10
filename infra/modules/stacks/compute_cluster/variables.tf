variable "name" {
  type    = string
  default = "compute-cluster"
}

variable "project_id" {
  type = string
}

variable "default_region" {
  type = string
}

variable "network" {
  type = string
}

variable "subnetwork" {
  type = string
}

variable "zones" {
  type = list(string)
  default = [
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-central1-f"
  ]
}

variable "pod_ip_range" { type = string }
variable "svc_ip_range" { type = string }
variable "environment" { type = string }

variable "gitops_repo_url" {
  type = string
}
variable "gitops_repo_creds" {
  type      = string
  sensitive = true
}

variable "k8s_secrets" {
  type      = map(string)
  sensitive = true
}

variable "bucket_name" {
  type = string
}

variable "dns_zone" {
  type = string

}