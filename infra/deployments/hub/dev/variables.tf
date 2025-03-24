variable "gitops_repo_url" {

}
variable "gitops_repo_creds" {
  sensitive = true
}
variable "k8s_secrets" {
  sensitive = true
  type      = map(string)
}

variable "storage_bucket_name" {
  default = "mtrx-us-central1-hub-dev-storage"
}

variable "project_id" {
}


variable "network_name" {
}

variable "k8s_subnetwork" {
}

variable "k8s_pod_ip_range" {
}

variable "k8s_svc_ip_range" {
}
