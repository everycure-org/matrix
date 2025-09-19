resource "google_artifact_registry_repository" "this" {
  project                = var.project_id
  location               = var.location
  repository_id          = var.repository_id
  format                 = var.format
  description            = var.description
  cleanup_policy_dry_run = false
  # Dynamic block to create the DELETE policy only if delete_older_than_days is greater than 0.
  dynamic "cleanup_policies" {
    for_each = var.delete_older_than_days != null ? [1] : []
    content {
      id     = "delete-old-artifacts"
      action = "DELETE"
      condition {
        older_than = var.delete_older_than_days
        tag_state  = "ANY"
      }
    }
  }
  dynamic "cleanup_policies" {
    for_each = var.keep_number_of_last_images > 0 ? [1] : []
    content {
      id     = "keep-last-images"
      action = "KEEP"
      most_recent_versions {
        keep_count = var.keep_number_of_last_images
      }
    }
  }
}
