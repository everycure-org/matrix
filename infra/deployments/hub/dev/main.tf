module "bootstrap_data" {
  source      = "../../../modules/components/bootstrap_file_content/"
  bucket_name = "mtrx-us-central1-hub-dev-storage"
}

module "compute_cluster" {
  source            = "../../../modules/stacks/compute_cluster/"
  default_region    = var.default_region
  project_id        = module.bootstrap_data.content.project_id
  network           = module.bootstrap_data.content.network.network_name
  subnetwork        = module.bootstrap_data.content.k8s_config.subnetwork
  pod_ip_range      = module.bootstrap_data.content.k8s_config.pod_ip_range
  svc_ip_range      = module.bootstrap_data.content.k8s_config.svc_ip_range
  environment       = "dev"
  gitops_repo_url   = var.gitops_repo_url
  gitops_repo_creds = var.gitops_repo_creds
}

module "matrix" {
  source = "../../../modules/stacks/matrix"
  default_region    = var.default_region
  project_id        = module.bootstrap_data.content.project_id
  environment       = "dev"
}