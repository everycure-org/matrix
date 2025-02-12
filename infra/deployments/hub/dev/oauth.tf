resource "google_iap_client" "matrix_dev_cli_client" {
  display_name = "Matrix Dev CLI"
  brand        = google_iap_brand.dev_brand.name
}

resource "google_iap_brand" "dev_brand" {
  support_email     = "gcp-admins@everycure.org"
  application_title = "Every Cure Dev Platform"
  project           = var.project_id
}