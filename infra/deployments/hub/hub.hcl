# generate a variables file for the above
generate hub_variables {
  path      = "hub_variables.gen.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
variable "network_name" {}
variable "repo_revision" {}
variable "k8s_subnetwork" {}
variable "k8s_pod_ip_range" {}
variable "k8s_svc_ip_range" {}
variable "k8s_secrets" {
  sensitive = true
  type      = map(string)
}
EOF
}
