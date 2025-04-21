# NOTE: Don't call this ArgoCD, this will clash with the Helm naming
# and give errors.
module "argo" {
  depends_on          = [module.gke]
  source              = "./argo"
  repo_url            = var.gitops_repo_url
  repo_creds          = var.gitops_repo_creds
  repo_path           = "infra/argo/"
  environment         = var.environment
  repo_revision       = var.repo_revision
  project_id          = var.project_id
  bucket_name         = var.bucket_name
  aip_oauth_client_id = var.aip_oauth_client_id
}
