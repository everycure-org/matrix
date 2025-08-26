resource "google_artifact_registry_repository" "this" {
  project                = var.project_id
  location               = var.location
  repository_id          = var.repository_id
  format                 = var.format
  description            = var.description
  cleanup_policy_dry_run = true
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

  # Policy to delete sample-run images after 1 day (configurable)
  dynamic "cleanup_policies" {
    for_each = length(var.sample_run_tag_prefixes) > 0 ? [1] : []
    content {
      id     = "delete-sample-run-images"
      action = "DELETE"
      condition {
        tag_prefixes = var.sample_run_tag_prefixes
        older_than   = "1d" # This is a fixed value for sample-run images
      }
    }
  }
}
