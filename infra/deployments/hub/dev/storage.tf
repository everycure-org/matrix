resource "google_storage_bucket_iam_member" "object_user" {
  bucket = var.storage_bucket_name
  role   = "roles/storage.objectUser"
  member = "group:matrix-all@everycure.org"

  condition {
    title      = "only_edit_raw"
    expression = "resource.name.startsWith('projects/_/buckets/${var.storage_bucket_name}/objects/data/01_RAW')"
  }
}