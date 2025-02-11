output "workbench_instance" {
  description = "Map of created workbench instances"
  value       = google_workbench_instance.user_workbench
}

output "instance_name" {
  description = "Name of the workbench instance"
  value       = google_workbench_instance.user_workbench.name
}

output "instance_url" {
  description = "URL of the workbench instance"
  value       = google_workbench_instance.user_workbench.proxy_uri
} 