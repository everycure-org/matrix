resource "google_storage_bucket_iam_member" "storage_members" {
  bucket = var.storage_bucket_name
  role = "roles/storage.user"
  member = "group:matrix-all@everycure.org"

  condition {
    title = "only_write_raw"
    expression = "resource.name.startsWith('projects/_/buckets/${var.storage_bucket_name}/objects/data/01_RAW')"
  }
}