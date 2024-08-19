data "google_dns_managed_zone" "cluster_zone" {
  name = var.dns_zone
}
# Reserve a global IP address
resource "google_compute_global_address" "ingress_ip" {
  name = "${var.name}-ingress-ip"
}

# Create a ConfigMap with the IP and DNS zone information
resource "kubernetes_config_map" "dns_config" {
  metadata {
    name      = "dns-config"
    namespace = "dev-pascal" # TODO change to traefik
  }

  data = {
    ingress_ip        = google_compute_global_address.ingress_ip.address
    dns_zone_name     = data.google_dns_managed_zone.cluster_zone.name
    dns_zone_dns_name = data.google_dns_managed_zone.cluster_zone.dns_name
  }
}