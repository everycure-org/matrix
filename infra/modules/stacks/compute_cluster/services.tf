# NOTE: Don't call this ArgoCD, this will clash with the Helm naming
# and give errors.
module "argo" {
  depends_on = [module.gke]
  source     = "./argo"
  repo_url   = var.gitops_repo_url
  repo_creds = var.gitops_repo_creds
  repo_path  = "infra/argo/"

  repo_revision = "infra"
}
