output "dns_zone" {
  value = google_dns_managed_zone.dev_zone.name
}