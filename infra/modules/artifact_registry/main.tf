# This local variable converts the number of days into the required seconds format.
locals {
  delete_older_than_seconds = var.delete_older_than_days > 0 ? format("%ds", var.delete_older_than_days * 86400) : null
}

resource "google_artifact_registry_repository" "this" {
  project       = var.project_id
  location      = var.location
  repository_id = var.repository_id
  format        = var.format
  description   = var.description

  # Dynamic block to create the DELETE policy only if delete_older_than_days is greater than 0.
  dynamic "cleanup_policies" {
    for_each = local.delete_older_than_seconds != null ? [1] : []
    content {
      id     = "delete-old-artifacts"
      action = "DELETE"
      condition {
        older_than = local.delete_older_than_seconds
        tag_state  = "ANY"
      }
    }
  }

  # Dynamic block to create the KEEP policy only if keep_count is greater than 0.
  dynamic "cleanup_policies" {
    for_each = var.keep_count > 0 ? [1] : []
    content {
      id     = "keep-minimum-recent-versions"
      action = "KEEP"
      condition {
        tag_state = var.tag_state
      }
    }
  }
}
