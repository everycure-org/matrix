variable "gitops_repo_url" {

}
variable "gitops_repo_creds" {
  sensitive = true
}
variable "k8s_secrets" {
  sensitive = true
  type      = map(string)
}

variable "network_name" {
}

variable "k8s_subnetwork" {
}

variable "k8s_pod_ip_range" {
}

variable "k8s_svc_ip_range" {
}
