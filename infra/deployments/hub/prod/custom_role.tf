resource "google_project_iam_custom_role" "read_and_no_delete_or_overwrite_storage_role" {
  project     = var.project_id
  role_id     = "ReadAndNoDeleteOrOverwriteStorageRole"
  title       = "Read and No Delete or Overwrite Storage Role"
  description = "Custom role with read access to storage and metadata, without delete or overwrite permissions."

  permissions = [
    "storage.folders.create",
    "storage.folders.list",
    "storage.folders.get",
    "storage.managedFolders.create",
    "storage.managedFolders.list",
    "storage.managedFolders.get",
    "storage.multipartUploads.create",
    "storage.multipartUploads.abort",
    "storage.multipartUploads.listParts",
    "storage.multipartUploads.list",
    "storage.objects.create",
    "storage.objects.get",
    "storage.objects.list",
    "storage.buckets.get",
    "storage.managedFolders.get",
    "storage.managedFolders.list",
    "storage.multipartUploads.list",
  ]
}

resource "google_project_iam_custom_role" "bigquery_read_write_no_delete" {
  role_id     = "bigQueryReadWriteNoDelete"
  title       = "BigQuery Read/Write Without Delete"
  description = "Allows read and insert access to BigQuery tables without delete/overwrite."
  project     = var.project_id

  permissions = [
    "bigquery.datasets.get",
    "bigquery.tables.get",
    "bigquery.tables.list",
    "bigquery.tables.create",
    "bigquery.tables.updateData",
    "bigquery.jobs.create",
    "bigquery.readsessions.create",
    "bigquery.tables.getData"
  ]
}