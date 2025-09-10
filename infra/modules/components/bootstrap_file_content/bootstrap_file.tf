variable "bucket_name" {}

data "google_client_config" "current" {}

data "google_storage_bucket_object" "object" {
  name   = "tf_bootstrap.json"
  bucket = var.bucket_name
}

data "http" "object" {
  url = format("%s?alt=media", data.google_storage_bucket_object.object.self_link)

  # Optional request headers
  request_headers = {
    "Authorization" = "Bearer ${data.google_client_config.current.access_token}"
  }
}

output "content" {
  value = jsondecode("${data.http.object.response_body}")
}