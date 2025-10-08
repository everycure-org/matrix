data "google_client_config" "default" {
}

locals {
  default_node_locations = "us-central1-c"                             # Single location for simplicity, can be expanded to multiple zones if needed
  gpu_node_locations     = "us-central1-a,us-central1-b,us-central1-c" # GPU nodes in multiple zones for availability

  # NOTE: Debugging node group scaling can be done using the GCP cluster logs, we create
  # node groups in 2 node locations, hence why the total amount of node groups.
  # https://console.cloud.google.com/kubernetes/clusters/details/us-central1/compute-cluster/logs/autoscaler_logs?chat=true&inv=1&invt=Abp5KQ&project=mtrx-hub-dev-3of
  n2d_node_pools = [for size in [8, 16, 32, 48, 64] : {
    name               = "n2d-highmem-${size}-nodes"
    machine_type       = "n2d-highmem-${size}"
    node_locations     = local.default_node_locations
    min_count          = 0
    max_count          = 10
    disk_type          = "pd-ssd"
    disk_size_gb       = 200
    enable_gcfs        = true
    enable_gvnic       = true
    initial_node_count = 0
    location_policy    = "ANY"
    }
  ]

  gpu_node_pools = [
    {
      name               = "g2-standard-16-l4-nodes" # 1 GPU, 16vCPUs, 64GB RAM
      machine_type       = "g2-standard-16"
      node_locations     = local.gpu_node_locations
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
      location_policy    = "ANY"
    },
    {
      name               = "g2-standard-32-nodes"
      machine_type       = "g2-standard-32" # 32 vCPUs, 128GB RAM
      node_locations     = local.gpu_node_locations
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
      location_policy    = "ANY"
    }
  ]

  # Dedicated management node pool for ArgoCD, Prometheus, MLflow, etc.
  management_node_pools = [
    {
      name               = "management-nodes"
      machine_type       = "n2-standard-16" # 8 vCPUs, 32GB RAM
      node_locations     = "us-central1-c"  # Single location.
      min_count          = 1                # Single instance, no HA
      max_count          = 1                # Single instance, no HA
      local_ssd_count    = 0
      disk_type          = "pd-standard" # Cost-effective for management workloads
      disk_size_gb       = 200
      enable_gcfs        = true
      enable_gvnic       = true
      initial_node_count = 1
      location_policy    = "ANY"
    }
  ]

  # Spot node pools for cost-effective compute workloads
  n2d_spot_node_pools = [for size in [8, 16, 32, 48, 64] : {
    name               = "n2d-highmem-${size}-spot-nodes"
    machine_type       = "n2d-highmem-${size}"
    node_locations     = local.default_node_locations
    min_count          = 0
    max_count          = 20 # Higher max count for spot instances
    disk_type          = "pd-ssd"
    disk_size_gb       = 200
    enable_gcfs        = true
    enable_gvnic       = true
    initial_node_count = 0
    spot               = true
    location_policy    = "ANY"
    }
  ]

  # Spot GPU node pools for cost-effective GPU workloads
  gpu_spot_node_pools = [
    {
      name               = "g2-standard-16-l4-spot-nodes" # 1 GPU, 16vCPUs, 64GB RAM
      machine_type       = "g2-standard-16"
      node_locations     = local.gpu_node_locations
      min_count          = 0
      max_count          = 30 # Higher max count for spot instances
      local_ssd_count    = 0
      disk_size_gb       = 200
      disk_type          = "pd-ssd"
      enable_gcfs        = true
      enable_gvnic       = true
      initial_node_count = 0
      accelerator_count  = 1
      accelerator_type   = "nvidia-l4"
      gpu_driver_version = "LATEST"
      spot               = true
      location_policy    = "ANY"
    },
    {
      name               = "g2-standard-32-spot-nodes"
      machine_type       = "g2-standard-32" # 32 vCPUs, 128GB RAM
      node_locations     = local.gpu_node_locations
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
      spot               = true
      location_policy    = "ANY"
    }
  ]

  # GitHub Actions runner node pools for CI/CD workloads
  github_runner_node_pools = [
    {
      name               = "github-runner-standard-nodes"
      machine_type       = "e2-standard-8" # 8 vCPUs, 32GB RAM - good for CI/CD and Docker in Docker (dind)
      node_locations     = local.default_node_locations
      min_count          = 0
      max_count          = 50
      local_ssd_count    = 0
      disk_size_gb       = 100 # Smaller disk for CI runners
      disk_type          = "pd-ssd"
      enable_gcfs        = true
      enable_gvnic       = true
      initial_node_count = 0
      spot               = false
      location_policy    = "ANY"
    }
  ]

  # Combine all node pools
  node_pools_combined = concat(
    local.n2d_node_pools,
    local.gpu_node_pools,
    local.management_node_pools,
    local.n2d_spot_node_pools,
    local.gpu_spot_node_pools,
    local.github_runner_node_pools
  )

  # Define node pools that should have the large memory taint
  large_memory_pools = concat(
    [for size in [8, 16, 32, 48, 64] : "n2d-highmem-${size}-nodes"],
    [for size in [8, 16, 32, 48, 64] : "n2d-highmem-${size}-spot-nodes"],
    [for size in [16, 32, 48, 64] : "n2-standard-${size}-nodes"],
    ["g2-standard-16-l4-nodes"],
    ["g2-standard-16-l4-spot-nodes"]
  )

  # Create a map of node pool taints
  node_pools_taints_map = {
    # Default empty taints for all pools
    all = []

    # Add GPU taint to GPU node pool
    "g2-standard-16-l4-nodes" = [
      {
        key    = "nvidia.com/gpu"
        value  = "present"
        effect = "NO_SCHEDULE"
      },
      {
        key    = "workload"
        value  = "true"
        effect = "NO_SCHEDULE"
      }
    ],

    # Add GPU and spot taints to GPU spot node pool
    "g2-standard-16-l4-spot-nodes" = [
      {
        key    = "nvidia.com/gpu"
        value  = "present"
        effect = "NO_SCHEDULE"
      },
      {
        key    = "spot"
        value  = "true"
        effect = "NO_SCHEDULE"
      },
      {
        key    = "workload"
        value  = "true"
        effect = "NO_SCHEDULE"
      }
    ],

    # Small standard and highmem pools (restrict general scheduling)
    "n2-standard-4-nodes" = [
      {
        key    = "workload"
        value  = "true"
        effect = "NO_SCHEDULE"
      }
    ]

    "n2-standard-8-nodes" = [
      {
        key    = "workload"
        value  = "true"
        effect = "NO_SCHEDULE"
      }
    ]

    "n2d-highmem-8-nodes" = [
      {
        key    = "workload"
        value  = "true"
        effect = "NO_SCHEDULE"
      }
    ]

    # GitHub Actions runner node pools
    "github-runner-standard-nodes" = [
      {
        key    = "github-runner"
        value  = "true"
        effect = "NO_SCHEDULE"
      }
    ]
  }

  # Add large memory taints for the appropriate node pools
  node_pools_taints = merge(
    local.node_pools_taints_map,
    {
      for pool in local.large_memory_pools :
      pool => [
        {
          key    = "node-memory-size"
          value  = "large"
          effect = "NO_SCHEDULE"
        },
        {
          key    = "workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ] if !contains(keys(merge(local.node_pools_taints_map)), pool)
    },
    {
      # Add spot taints for spot node pools
      for pool in concat(
        [for size in [8, 16, 32, 48, 64] : "n2d-highmem-${size}-spot-nodes"]
      ) :
      pool => [
        {
          key    = "node-memory-size"
          value  = "large"
          effect = "NO_SCHEDULE"
        },
        {
          key    = "spot"
          value  = "true"
          effect = "NO_SCHEDULE"
        },
        {
          key    = "workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ] if !contains(keys(merge(local.node_pools_taints_map)), pool)
    }
  )
}

# docs here https://registry.terraform.io/modules/terraform-google-modules/kubernetes-engine/google/latest/submodules/private-cluster
module "gke" {
  source              = "terraform-google-modules/kubernetes-engine/google//modules/private-cluster"
  version             = "35.0.1"
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
  enable_cost_allocation          = true
  create_service_account          = true
  # see instructions here: https://cloud.google.com/kubernetes-engine/docs/how-to/google-groups-rbac
  authenticator_security_group = "gke-security-groups@everycure.org"
  service_account_name         = "sa-k8s-node"
  node_metadata                = "UNSPECIFIED"
  gke_backup_agent_config      = false

  node_pools = local.node_pools_combined

  # Apply taints through the node_pools_taints parameter
  node_pools_taints = local.node_pools_taints

  node_pools_labels = {
    for pool in local.node_pools_combined : pool.name => merge(
      {
        gpu_node  = can(pool.accelerator_count) ? "true" : "false"
        spot_node = lookup(pool, "spot", false) ? "true" : "false"
        # Billing labels for cost tracking
        cost-center       = pool.name == "management-nodes" ? "infrastructure-management" : contains(["github-runner-standard-nodes", "github-runner-spot-nodes"], pool.name) ? "ci-cd-infrastructure" : "compute-workloads"
        workload-category = pool.name == "management-nodes" ? "platform-services" : contains(["github-runner-standard-nodes", "github-runner-spot-nodes"], pool.name) ? "ci-cd" : "data-science"
        environment       = var.environment
      },
      pool.name == "management-nodes" ? {
        workload-type    = "management"
        billing-category = "infrastructure"
        service-tier     = "management"
        } : contains(["github-runner-standard-nodes", "github-runner-spot-nodes"], pool.name) ? {
        workload-type    = "ci-cd"
        billing-category = lookup(pool, "spot", false) ? "github-runner-spot" : "github-runner-standard"
        service-tier     = "ci-cd"
        } : can(pool.accelerator_count) ? {
        workload-type    = "compute"
        billing-category = lookup(pool, "spot", false) ? "gpu-compute-spot" : "gpu-compute"
        service-tier     = "compute"
        } : {
        workload-type    = "compute"
        billing-category = lookup(pool, "spot", false) ? "cpu-compute-spot" : "cpu-compute"
        service-tier     = "compute"
      }
    )
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
