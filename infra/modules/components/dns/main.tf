# FUTURE need to also have one for non DEV
resource "google_dns_managed_zone" "dev_zone" {
  name        = "dev-zone"
  dns_name    = "dev.everycure.org."
  description = "DNS zone for EveryCure development"
  labels = {
    environment = "dev"
  }
}

resource "google_dns_record_set" "docs_txt_validation" {
  name         = "docs.${google_dns_managed_zone.dev_zone.dns_name}"
  managed_zone = google_dns_managed_zone.dev_zone.name
  type         = "TXT"
  ttl          = 300
  rrdatas      = ["google-site-verification=n2nDhJm8oiCTDYqg5zi_eq_IbZw1cawmyuEvjPMJ2w8"]
}

resource "google_dns_record_set" "a" {
  name         = "docs.${google_dns_managed_zone.dev_zone.dns_name}"
  managed_zone = google_dns_managed_zone.dev_zone.name
  type         = "A"
  ttl          = 300

  rrdatas = [
    # TODO document where these come from (Google gives em to us in AppEngine)
    "216.239.32.21",
    "216.239.34.21",
    "216.239.36.21",
    "216.239.38.21"
  ]
}

resource "google_dns_record_set" "aaaa" {
  name         = "docs.${google_dns_managed_zone.dev_zone.dns_name}"
  managed_zone = google_dns_managed_zone.dev_zone.name
  type         = "AAAA"
  ttl          = 300

  rrdatas = [
    "2001:4860:4802:32::15",
    "2001:4860:4802:34::15",
    "2001:4860:4802:36::15",
    "2001:4860:4802:38::15"
  ]
}