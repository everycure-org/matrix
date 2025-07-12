
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
      idle-timeout-seconds = "3600" # 60 minutes
      post-startup-script  = var.post_startup_script
    }
  }



  instance_owners = [var.email]
  lifecycle {
    ignore_changes = [
      desired_state,
      gce_setup[0].metadata.resource-url,
      gce_setup[0].data_disks,
      gce_setup[0].machine_type,
      gce_setup[0].accelerator_configs,
      gce_setup[0].boot_disk[0].disk_size_gb,
    ]
  }
  # when we create, they are started, from this point onward, we ignore the state -> leave running ones running
  desired_state = "ACTIVE"

  labels = merge({
    consumer-project-id = var.project_id
    notebooks-product   = "workbench-instances"
    email               = local.sanitized_email
  }, var.labels)
}
