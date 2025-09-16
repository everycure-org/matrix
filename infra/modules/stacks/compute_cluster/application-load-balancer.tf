# LiteLLM External Load Balancer Resources
# Creates static IP and managed certificate that Kubernetes Ingress will use

# Reserve a static IP address for LiteLLM
resource "google_compute_global_address" "litellm_static_ip" {
  name    = "litellm-static-ip-${var.environment}"
  project = var.project_id
}

# Create a Google-managed SSL certificate for LiteLLM
resource "google_compute_managed_ssl_certificate" "litellm_ssl_cert" {
  name    = "litellm-ssl-cert"
  project = var.project_id

  managed {
    domains = ["litellm.application.${var.environment}.everycure.org"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Get the DNS managed zone
data "google_dns_managed_zone" "environment_zone" {
  name    = "${var.environment}-zone"
  project = var.project_id
}

# Add DNS record for LiteLLM pointing to the load balancer IP
resource "google_dns_record_set" "litellm_dns" {
  name         = "litellm.application.${data.google_dns_managed_zone.environment_zone.dns_name}"
  managed_zone = data.google_dns_managed_zone.environment_zone.name
  project      = var.project_id
  type         = "A"
  ttl          = 300
  rrdatas      = [google_compute_global_address.litellm_static_ip.address]
}

# Output the external IP address
output "litellm_external_ip" {
  description = "External IP address for LiteLLM load balancer"
  value       = google_compute_global_address.litellm_static_ip.address
}

output "litellm_domain" {
  description = "LiteLLM domain name"
  value       = "litellm.application.${var.environment}.everycure.org"
}