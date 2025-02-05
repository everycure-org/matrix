# NOTE: This configuration was partially generated using AI assistance.

locals {
  # Sanitize email for labels: lowercase, replace @ and . with dash, max 63 chars
  sanitized_email = lower(replace(replace(var.email, "@", "-"), ".", "-"))
}

# Create a workbench instance for the user
resource "google_notebooks_instance" "user_workbench" {
  name          = "${var.username}-workbench"
  location      = var.location
  machine_type  = var.machine_type
  desired_state = "STOPPED"

  lifecycle {
    ignore_changes = [
      desired_state,
      machine_type,
      install_gpu_driver,
      accelerator_config,
      data_disk_size_gb,
      create_time,
      update_time,
    ]
  }

  vm_image {
    project      = "deeplearning-platform-release"
    image_family = "common-cpu"
  }

  instance_owners = [var.email]

  boot_disk_type    = "PD_STANDARD"
  boot_disk_size_gb = var.boot_disk_size_gb

  data_disk_type    = "PD_STANDARD"
  data_disk_size_gb = var.data_disk_size_gb

  no_public_ip    = true
  no_proxy_access = false

  network = var.network
  subnet  = var.subnet

  service_account = var.service_account

  metadata = {
    framework               = "workbench"
    proxy-mode              = "service_account"
    enable-oslogin          = "TRUE"
    enable-guest-attributes = "TRUE"
    title                   = "workbench"
  }


  labels = merge({
    consumer-project-id = var.project_id
    notebooks-product   = "workbench-instances"
    resource-name       = "${var.username}-workbench"
    email               = local.sanitized_email
  }, var.labels)
} 