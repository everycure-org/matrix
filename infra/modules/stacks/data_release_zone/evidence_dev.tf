// NOTE: This configuration was partially generated using AI assistance.

module "evidence_dev_website" {
  source = "github.com/gruntwork-io/terraform-google-static-assets//modules/cloud-storage-static-website?ref=v0.6.0"

  project               = var.project_id
  website_domain_name   = "data.${data.google_dns_managed_zone.dev_zone.dns_name}"
  website_location      = var.region
  website_storage_class = "STANDARD"

  enable_versioning                = true
  force_destroy_website            = var.environment != "prod"
  force_destroy_access_logs_bucket = var.environment != "prod"

  index_page     = "index.html"
  not_found_page = "404.html"

  # Default ACLs for public access
  website_acls = [
    "READER:allUsers"
  ]

  # Enable CORS for the website
  enable_cors          = true
  cors_origins         = ["*"]
  cors_methods         = ["GET", "HEAD", "OPTIONS"]
  cors_max_age_seconds = 3600

  # Configure access logs
  access_logs_expiration_time_in_days = 30

  # Add DNS entry
  create_dns_entry      = true
  dns_managed_zone_name = var.dns_managed_zone_name
  dns_record_ttl        = 300

  custom_labels = {
    environment = var.environment
    component   = "evidence-dev"
  }
}

data "google_dns_managed_zone" "dev_zone" {
  name = var.dns_managed_zone_name
}
