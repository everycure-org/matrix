

resource "google_certificate_manager_dns_authorization" "dns_authorization" {
  name        = "platform-dns-auth"
  location    = "global"
  description = "dns authorization for platform"
  domain      = "platform.dev.everycure.org"
}

resource "google_dns_record_set" "dns_authorization" {
  name         = google_certificate_manager_dns_authorization.dns_authorization.dns_resource_record.0.name
  managed_zone = google_dns_managed_zone.dev_zone.name
  type         = google_certificate_manager_dns_authorization.dns_authorization.dns_resource_record.0.type
  rrdatas         = [google_certificate_manager_dns_authorization.dns_authorization.dns_resource_record.0.data]
  ttl          = 300
}

resource "google_certificate_manager_certificate_map" "default" {
  name        = "cert-map"
  labels      = {
    "terraform" : true,
  }
}

resource "google_certificate_manager_certificate" "default" {
  name        = "dns-cert"
  description = "The default cert"
  scope       = "ALL_REGIONS"
  labels = {
    env = "dev"
  }
  managed {
    domains = [
      google_certificate_manager_dns_authorization.dns_authorization.domain,
      ]
    dns_authorizations = [
      google_certificate_manager_dns_authorization.dns_authorization.id,
      ]
  }
}