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

resource "google_bigquery_dataset" "dataset" {
  for_each       = toset(["rtx-kg2", "robokop"])
  project        = module.bootstrap_data.content.project_id
  dataset_id     = "kg-${each.key}"
  description    = "Dataset with nodes and edges for ${each.key}"
  location       = "EU"

  labels = {
    env = "dev",
    kg = each.key
  }
}

