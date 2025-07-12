module "compute_cluster" {
  source              = "../../../modules/stacks/compute_cluster/"
  default_region      = var.default_region
  project_id          = var.project_id
  network             = var.network_name
  subnetwork          = var.k8s_subnetwork
  pod_ip_range        = var.k8s_pod_ip_range
  svc_ip_range        = var.k8s_svc_ip_range
  environment         = var.environment
  gitops_repo_url     = var.gitops_repo_url
  gitops_repo_creds   = var.gitops_repo_creds
  k8s_secrets         = var.k8s_secrets
  bucket_name         = var.storage_bucket_name
  repo_revision       = var.repo_revision
  aip_oauth_client_id = var.aip_oauth_client_id

}


module "dns" {
  source      = "../../../modules/components/dns"
  environment = var.environment
}

module "data_release_zone" {
  source                = "../../../modules/stacks/data_release_zone"
  project_id            = var.project_id
  region                = var.default_region
  dns_managed_zone_name = module.dns.dns_zone_name
  dns_name              = module.dns.dns_name
  environment           = var.environment
  enable_cdn            = false # Set to false for dev environment to avoid CDN costs
}


module "cloudbuild" {
  source                     = "../../../modules/components/cloudbuild"
  github_repo_path_to_folder = var.github_repo_path_to_folder
  project_id                 = var.project_id
  github_app_installation_id = var.github_app_installation_id
  github_repo_owner          = var.github_repo_owner
  github_repo_name           = var.github_repo_name
  github_repo_token          = var.github_classic_token_for_cloudbuild
  github_repo_deploy_branch  = var.github_branch_to_run_on
  slack_webhook_url          = var.slack_webhook_url
  gitcrypt_key               = var.gitcrypt_key
  require_manual_approval    = true
}