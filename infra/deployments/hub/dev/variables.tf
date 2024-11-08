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
