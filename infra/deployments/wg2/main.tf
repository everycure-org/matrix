module "bootstrap_data" {
  source      = "../../modules/components/bootstrap_file_content/"
  bucket_name = var.storage_bucket_name
}
