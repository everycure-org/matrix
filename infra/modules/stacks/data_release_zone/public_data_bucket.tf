locals {
  website_domain_name = "data.${trimsuffix(var.dns_name, ".")}"
}

# Create the main bucket for hosting the website
resource "google_storage_bucket" "website" {
  name                        = local.website_domain_name
  project                     = var.project_id
  location                    = var.region
  storage_class               = "STANDARD"
  force_destroy               = var.environment != "prod"
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  website {
    main_page_suffix = "index.html"
    not_found_page   = "404.html"
  }

  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "OPTIONS"]
    response_header = ["*"]
    max_age_seconds = 3600
  }

  labels = {
    environment = var.environment
    component   = "public-data-bucket"
  }
}


# Make bucket public
resource "google_storage_bucket_iam_member" "public_read" {
  bucket     = google_storage_bucket.website.name
  role       = "roles/storage.objectViewer"
  member     = "allUsers"
  depends_on = [time_sleep.wait_for_org_policies]
}

# Create a Google-managed SSL certificate
resource "google_compute_managed_ssl_certificate" "default" {
  name    = "public-data-ssl-cert"
  project = var.project_id

  managed {
    domains = [local.website_domain_name]
  }
}

# Create backend bucket configuration
resource "google_compute_backend_bucket" "static_website" {
  name        = "public-data-backend-bucket-${var.environment}"
  project     = var.project_id
  bucket_name = google_storage_bucket.website.name
  enable_cdn  = var.enable_cdn
}

# Create URL map
resource "google_compute_url_map" "static_website" {
  name            = "public-data-url-map-${var.environment}"
  project         = var.project_id
  default_service = google_compute_backend_bucket.static_website.self_link
}

# Create HTTPS proxy
resource "google_compute_target_https_proxy" "static_website" {
  name             = "public-data-https-proxy-${var.environment}"
  project          = var.project_id
  url_map          = google_compute_url_map.static_website.self_link
  ssl_certificates = [google_compute_managed_ssl_certificate.default.self_link]
}

# Reserve a static IP address
resource "google_compute_global_address" "static_website" {
  name    = "public-data-static-ip-${var.environment}"
  project = var.project_id
}

# Create forwarding rule for HTTPS
resource "google_compute_global_forwarding_rule" "static_website_https" {
  name       = "public-data-https-forwarding-rule-${var.environment}"
  project    = var.project_id
  target     = google_compute_target_https_proxy.static_website.self_link
  port_range = "443"
  ip_address = google_compute_global_address.static_website.address
}

# Create forwarding rule for HTTP (optional - redirects to HTTPS)
resource "google_compute_target_http_proxy" "static_website" {
  name    = "public-data-http-proxy-${var.environment}"
  project = var.project_id
  url_map = google_compute_url_map.static_website.self_link
}

resource "google_compute_global_forwarding_rule" "static_website_http" {
  name       = "public-data-http-forwarding-rule-${var.environment}"
  project    = var.project_id
  target     = google_compute_target_http_proxy.static_website.self_link
  port_range = "80"
  ip_address = google_compute_global_address.static_website.address
}

# Add A record for the domain pointing to the load balancer IP
resource "google_dns_record_set" "website" {
  name         = "${local.website_domain_name}."
  managed_zone = var.dns_managed_zone_name
  project      = var.project_id
  type         = "A"
  ttl          = 300
  rrdatas      = [google_compute_global_address.static_website.address]
}

# Add a dummy index.html file for testing
resource "google_storage_bucket_object" "hello_world" {
  name         = "test/hello/index.html"
  content      = "<html><body><h1>Hello, World!</h1></body></html>"
  bucket       = google_storage_bucket.website.name
  content_type = "text/html"
}
