module "compute_cluster" {
  source            = "../../../modules/stacks/compute_cluster/"
  default_region    = var.default_region
  project_id        = var.project_id
  network           = var.network_name
  subnetwork        = var.k8s_subnetwork
  pod_ip_range      = var.k8s_pod_ip_range
  svc_ip_range      = var.k8s_svc_ip_range
  environment       = "dev"
  gitops_repo_url   = var.gitops_repo_url
  gitops_repo_creds = var.gitops_repo_creds
  dns_zone          = module.dns.dns_zone
  k8s_secrets       = var.k8s_secrets
  bucket_name       = "mtrx-us-central1-hub-dev-storage"
}


module "dns" {
  source      = "../../../modules/components/dns"
  environment = var.environment
}

module "data_release_zone" {
  source                = "../../../modules/stacks/data_release_zone"
  project_id            = var.project_id
  region                = var.default_region
  dns_managed_zone_name = module.dns.dns_zone
  environment           = "dev"
}
