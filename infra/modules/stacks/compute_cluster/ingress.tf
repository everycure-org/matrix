data "google_dns_managed_zone" "cluster_zone" {
  name = var.dns_zone
}
# TODO remove, we use Gateway class instead
# # Reserve a global IP address
# resource "google_compute_global_address" "ingress_ip" {
#   name = "${var.name}-ingress-ip"
# }

# Create a ConfigMap with the IP and DNS zone information
# resource "kubernetes_config_map" "dns_config" {
#   metadata {
#     name      = "dns-config"
#     namespace = "dev-pascal" # TODO change to traefik
#   }
# 
#   data = {
#     ingress_ip        = google_compute_global_address.ingress_ip.address
#     dns_zone_name     = data.google_dns_managed_zone.cluster_zone.name
#     dns_zone_dns_name = data.google_dns_managed_zone.cluster_zone.dns_name
#   }
# }

resource "google_compute_subnetwork" "proxy_only_subnet" {
  name          = "proxy-only-subnet"
  ip_cidr_range = "10.2.0.0/16" # TODO not hard coded
  region        = var.default_region
  network       = var.network
  purpose       = "REGIONAL_MANAGED_PROXY" # Set the purpose of the subnet
  role          = "ACTIVE"                 # Set the role of the subnet
}