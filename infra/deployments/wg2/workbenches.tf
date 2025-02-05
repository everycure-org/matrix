locals {
  users = {
    # "amy" = "amy@everycure.org"
    "pascal" = "pascal@everycure.org"
  }
}

module "workbenches" {
  source   = "./modules/workbench"
  for_each = local.users

  username = each.key
  email    = each.value

  network = var.shared_network_name
  subnet  = var.shared_subnetwork_name

  service_account = "vertex-ai-workbench-sa@${var.project_id}.iam.gserviceaccount.com"
  project_id      = var.project_id

  # Optional: Override defaults if needed
  # machine_type = "e2-standard-8"
  # boot_disk_size_gb = 200
  # data_disk_size_gb = 150
  post_startup_script = local.gcs_path
}

locals {
  gcs_path = "gs://${var.storage_bucket_name}/post_startup_script.sh"
}

resource "google_storage_bucket_object" "post_startup_script" {
  name   = "post_startup_script.sh"
  bucket = var.storage_bucket_name
  source = "./setup_vertex_workbench.sh"
}


