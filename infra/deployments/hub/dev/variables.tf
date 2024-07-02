
variable "gitops_repo_url" {

}
variable "gitops_repo_creds" {
    sensitive = true
}
variable "k8s_secrets" {
    sensitive = true
    type = map(string)
}