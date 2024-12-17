data "google_client_config" "default" {
}

locals {
  default_node_locations = "us-central1-a,us-central1-c"
  mem_node_pools = [for size in [8, 16] : {
    name               = "e2-highmem-${size}-nodes"
    machine_type       = "e2-highmem-${size}"
    node_locations     = local.default_node_locations
    min_count          = 0
    max_count          = 20
    local_ssd_count    = 0
    disk_size_gb       = 200
    enable_gcfs        = true
    enable_gvnic       = true
    initial_node_count = 0
    spot               = false
    }
  ]
  n2d_node_pools = [for size in [32, 48, 64] : {
    name               = "n2d-highmem-${size}-nodes"
    machine_type       = "n2d-highmem-${size}"
    node_locations     = local.default_node_locations
    min_count          = 0
    max_count          = 5
    local_ssd_count    = 0
    disk_type          = "pd-ssd"
    disk_size_gb       = 400
    enable_gcfs        = true
    enable_gvnic       = true
    initial_node_count = 0
    }
  ]

  standard_node_pools = [for size in [4, 8, 16, 32, 48, 64] : {
    name               = "n2-standard-${size}-nodes"
    machine_type       = "n2-standard-${size}"
    node_locations     = local.default_node_locations
    min_count          = 0
    max_count          = 20
    local_ssd_count    = 0
    disk_type          = size > 32 ? "pd-ssd" : "pd-standard"
    disk_size_gb       = 200
    enable_gcfs        = true
    enable_gvnic       = true
    initial_node_count = 0
    }
  ]
  gpu_node_pools = [
    {
      name               = "g2-standard-16-l4-nodes" # 1 GPU, 16vCPUs, 64GB RAM
      machine_type       = "g2-standard-16"
      node_locations     = local.default_node_locations
      min_count          = 0
      max_count          = 20
      local_ssd_count    = 0
      disk_size_gb       = 200
      disk_type          = "pd-ssd"
      enable_gcfs        = true
      enable_gvnic       = true
      initial_node_count = 0
      accelerator_count  = 1
      accelerator_type   = "nvidia-l4"
      gpu_driver_version = "LATEST"
    },
  ]
  node_pools_combined = concat(local.standard_node_pools, local.mem_node_pools, local.gpu_node_pools, local.n2d_node_pools)
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
  identity_namespace  = null
  network             = var.network
  subnetwork          = var.subnetwork
  ip_range_pods       = var.pod_ip_range
  ip_range_services   = var.svc_ip_range
  http_load_balancing = true
  # necessary for https://cloud.google.com/kubernetes-engine/docs/concepts/gateway-api#shared_gateway_per_cluster
  gateway_api_channel             = "CHANNEL_STANDARD"
  network_policy                  = false
  master_ipv4_cidr_block          = "172.16.0.0/28"
  horizontal_pod_autoscaling      = true
  filestore_csi_driver            = true
  dns_cache                       = false
  remove_default_node_pool        = true
  enable_private_nodes            = true
  enable_private_endpoint         = false # FUTURE: switch this to true
  enable_vertical_pod_autoscaling = true
  create_service_account          = true
  service_account_name            = "sa-k8s-node"
  node_metadata                   = "UNSPECIFIED"

  # FUTURE: Refine node pools
  node_pools = local.node_pools_combined

  node_pools_labels = {
    for pool in local.node_pools_combined : pool.name => {
      gpu_node = can(pool.accelerator_count) ? "true" : "false"
    }
  }
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
