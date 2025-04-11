output "dns_zone_name" {
  value = google_dns_managed_zone.dns_zone.name
}

output "dns_name" {
  value = google_dns_managed_zone.dns_zone.dns_name
}