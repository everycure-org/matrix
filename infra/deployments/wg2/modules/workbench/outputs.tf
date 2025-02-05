# NOTE: This file was partially generated using AI assistance.

output "workbench_instances" {
  description = "Map of created workbench instances"
  value       = google_notebooks_instance.user_workbench
}

output "instance_name" {
  description = "Name of the workbench instance"
  value       = google_notebooks_instance.user_workbench.name
}

output "instance_url" {
  description = "URL of the workbench instance"
  value       = google_notebooks_instance.user_workbench.proxy_uri
} 