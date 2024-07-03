data "google_client_config" "default" {
}

# docs here https://registry.terraform.io/modules/terraform-google-modules/kubernetes-engine/google/latest/submodules/private-cluster
module "gke" {
  source              = "terraform-google-modules/kubernetes-engine/google//modules/private-cluster"
  version             = "31.0.0"
  project_id          = var.project_id
  name                = var.name
  deletion_protection = var.environment == "dev" ? false : true
  region              = var.default_region
  zones               = var.zones
  # disables workload identity and thus, IAM managed in GCP. Instead we use good old K8S auth
  # disabled this due to the complexity involved with managing K8S SA identity in GCP IAM
  identity_namespace         = null
  network                    = var.network
  subnetwork                 = var.subnetwork
  ip_range_pods              = var.pod_ip_range
  ip_range_services          = var.svc_ip_range
  http_load_balancing        = true
  network_policy             = false
  master_ipv4_cidr_block     = "172.16.0.0/28"
  horizontal_pod_autoscaling = true
  filestore_csi_driver       = false
  dns_cache                  = false
  remove_default_node_pool   = true
  enable_private_nodes       = true
  enable_private_endpoint    = false # FUTURE: switch this to true
  create_service_account     = true
  service_account_name       = "sa-k8s-node"
  node_metadata              = "UNSPECIFIED"

  # FUTURE: Refine mode pools
  node_pools = [
    {
      name            = "default-node-pool"
      machine_type    = "e2-medium"
      node_locations  = "us-central1-a"
      min_count       = 1
      max_count       = 10
      local_ssd_count = 0
      spot            = false
      disk_size_gb    = 100
      disk_type       = "pd-standard"
      image_type      = "COS_CONTAINERD"
      enable_gcfs     = false
      enable_gvnic    = false
      logging_variant = "DEFAULT"
      auto_repair     = true
      auto_upgrade    = true
      # service_account    = module.k8s_sa.email
      preemptible        = false
      initial_node_count = 3
      # accelerator_count           = 1
      # accelerator_type            = "nvidia-l4"
      # gpu_driver_version          = "LATEST"
      # gpu_sharing_strategy        = "TIME_SHARING"
      # max_shared_clients_per_gpu = 2
    },
    {
      name = "stable-cpu-pool"
      # machine_type       = "n4-standard-8"
      machine_type    = "e2-standard-8"
      node_locations  = "us-central1-a,us-central1-c"
      min_count       = 0
      max_count       = 20
      local_ssd_count = 0
      disk_size_gb    = 200
      # disk_type = "hyperdisk-balanced"
      enable_gcfs  = true
      enable_gvnic = true
      # service_account    = module.k8s_sa.email
      initial_node_count = 1
    },
    # {
    #   name = "stable-cpu-pool-2"
    #   # machine_type       = "n4-standard-8"
    #   machine_type       = "e2-standard-8"
    #   min_count          = 0
    #   max_count          = 20
    #   local_ssd_count    = 0
    #   disk_size_gb       = 200
    #   enable_gcfs        = true
    #   enable_gvnic       = true
    #   service_account    = module.k8s_sa.email
    #   initial_node_count = 1
    # },
  ]

  # https://cloud.google.com/artifact-registry/docs/access-control#gke
  # node_pools_oauth_scopes = {
  #   all = [
  #     "https://www.googleapis.com/auth/logging.write",
  #     "https://www.googleapis.com/auth/monitoring",
  #     "https://www.googleapis.com/auth/cloud-platform",
  #     "https://www.googleapis.com/auth/devstorage.read_write"
  #   ]
  # }

}
