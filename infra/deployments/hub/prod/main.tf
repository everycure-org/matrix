module "compute_cluster" {
  source            = "../../../modules/stacks/compute_cluster/"
  default_region    = var.default_region
  project_id        = var.project_id
  network           = var.network_name
  subnetwork        = var.k8s_subnetwork
  pod_ip_range      = var.k8s_pod_ip_range
  svc_ip_range      = var.k8s_svc_ip_range
  environment       = var.environment
  gitops_repo_url   = var.gitops_repo_url
  gitops_repo_creds = var.gitops_repo_creds
  k8s_secrets       = var.k8s_secrets
  bucket_name       = var.storage_bucket_name
  repo_revision     = var.repo_revision
  docker_registry   = var.docker_registry

}


module "dns" {
  source      = "../../../modules/components/dns"
  environment = var.environment
}