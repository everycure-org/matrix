output "repository_name" {
  description = "The full name of the created repository."
  value       = google_artifact_registry_repository.this.name
}

output "repository_url" {
  description = "The URL of the repository."
  value       = "${var.location}-docker.pkg.dev/${var.project_id}/${var.repository_id}"
}