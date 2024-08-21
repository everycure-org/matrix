data "google_client_config" "default" {
}

locals {
  base_node_pool = [
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
  ]
  cpu_node_pools = [for size in [8, 16, 32] : {
    name = "e2-standard-${size}-nodes"
    # machine_type       = "n4-standard-8"
    machine_type       = "e2-standard-${size}"
    node_locations     = "us-central1-a,us-central1-c"
    min_count          = 0
    max_count          = 20
    local_ssd_count    = 0
    disk_size_gb       = 200
    enable_gcfs        = true
    enable_gvnic       = true
    initial_node_count = 0
    }
  ]

  mem_node_pools = [for size in [4, 8, 16, 32, 48, 64] : {
    name               = "n2-standard-${size}-nodes"
    machine_type       = "n2-standard-${size}"
    node_locations     = "us-central1-a,us-central1-c"
    min_count          = 0
    max_count          = 20
    local_ssd_count    = 0
    disk_size_gb       = 200
    enable_gcfs        = true
    enable_gvnic       = true
    initial_node_count = 0
    }
  ]
  gpu_node_pools = [
    # FUTURE add GPU pools here
  ]
  node_pools_combined = concat(local.base_node_pool, local.cpu_node_pools, local.gpu_node_pools, local.mem_node_pools)
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
  node_pools = local.node_pools_combined

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
