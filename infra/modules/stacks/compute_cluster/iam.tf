locals {
  k8s_sa_roles = [
    "roles/container.defaultNodeServiceAccount",
    "roles/storage.objectAdmin",
    "roles/storage.admin", # FUTURE more fine grained, but else it can't list buckets. surprisingly hard to find a predefined role that has storage.buckets.get
    "roles/bigquery.dataEditor",
    "roles/bigquery.user",
    "roles/dns.admin",
    "roles/bigquery.jobUser",
    "roles/aiplatform.user",
    "roles/artifactregistry.admin",    # Required for deleting images in cleanup workflows (includes reader/writer permissions)
    "roles/compute.loadBalancerAdmin", # Required for Gateway API to manage load balancers
    "roles/compute.securityAdmin",     # Required for Gateway API to manage SSL certificates
  ]
}

resource "google_project_iam_member" "k8s_sa_bindings" {
  for_each = toset(local.k8s_sa_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${module.gke.service_account}"

}