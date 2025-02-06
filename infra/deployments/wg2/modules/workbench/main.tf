# NOTE: This configuration was partially generated using AI assistance.

terraform {
  required_providers {
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 6.19.0"
    }
  }
}

locals {
  # Sanitize email for labels: lowercase, replace @ and . with dash, max 63 chars
  sanitized_email = lower(replace(replace(var.email, "@", "-"), ".", "-"))
}

# Create a workbench instance for the user
resource "google_workbench_instance" "user_workbench" {
  provider = google-beta
  name     = "${var.username}-dev-workbench"
  location = var.location
  project  = var.project_id

  gce_setup {
    machine_type = var.machine_type
    service_accounts {
      email = var.service_account
    }
    boot_disk {
      disk_type    = "PD_STANDARD"
      disk_size_gb = var.boot_disk_size_gb
    }

    data_disks {
      disk_type    = "PD_STANDARD"
      disk_size_gb = var.data_disk_size_gb
    }

    vm_image {
      project = "cloud-notebooks-managed"
      family  = "workbench-instances"
    }
    network_interfaces {
      network = var.network
      subnet  = var.subnet
    }
    disable_public_ip = true
    metadata = {
      idle-timeout-seconds = "10800"
      post-startup-script  = var.post_startup_script
    }
  }


  instance_owners = [var.email]




  lifecycle {
    ignore_changes = [
      create_time,
      update_time,
      desired_state,
    ]
  }
  desired_state = "STOPPED"

  labels = merge({
    consumer-project-id = var.project_id
    notebooks-product   = "workbench-instances"
    resource-name       = "${var.username}-workbench"
    email               = local.sanitized_email
  }, var.labels)
}
