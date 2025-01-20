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
  default = "mtrx-hub-dev-3of"
}


variable "network_name" {
  default = "matrix-hub-dev-nw"
}

variable "k8s_subnetwork" {
  default = "sn-hub-dev-us"
}

variable "k8s_pod_ip_range" {
  default = "sn-hub-dev-us-pod-range"
}

variable "k8s_svc_ip_range" {
  default = "sn-hub-dev-us-svc-range"
}