module "service_accounts" {
  source     = "terraform-google-modules/service-accounts/google"
  version    = "~> 4.0"
  project_id = var.project_id
  prefix     = "sa-"
  names      = ["gke-worker-node-group"]
  project_roles = [
    # TODO: restrict roles further here
    "${var.project_id}=>roles/owner",
  ]
}
