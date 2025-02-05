locals {
  users = {
    # "amy" = "amy@everycure.org"
    "pascal" = "pascal@everycure.org"
  }
}
#   project_id        = module.bootstrap_data.content.project_id
#   network           = module.bootstrap_data.content.network.network_name
#   subnetwork        = module.bootstrap_data.content.k8s_config.subnetwork
#   pod_ip_range      = module.bootstrap_data.content.k8s_config.pod_ip_range
#   svc_ip_range      = module.bootstrap_data.content.k8s_config.svc_ip_range

module "workbenches" {
  source   = "./modules/workbench"
  for_each = local.users

  username = each.key
  email    = each.value

  network = var.shared_network_name
  subnet  = var.shared_subnetwork_name

  service_account = "vertex-ai-workbench-sa@mtrx-wg2-modeling-dev-9yj.iam.gserviceaccount.com"
  project_id      = "mtrx-wg2-modeling-dev-9yj"

  # Optional: Override defaults if needed
  # machine_type = "e2-standard-8"
  # boot_disk_size_gb = 200
  # data_disk_size_gb = 150
}

