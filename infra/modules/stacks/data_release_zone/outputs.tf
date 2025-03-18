output "website_url" {
  description = "The URI of the Evidence.dev website bucket"
  value       = google_storage_bucket.website.url
}
