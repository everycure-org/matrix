variable "project_id" {
  description = "The ID of the Google Cloud project where resources will be created"
  type        = string
}

variable "region" {
  description = "The region where resources will be created"
  type        = string
}

variable "environment" {
  description = "The environment (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "dns_managed_zone_name" {
  description = "The name of the Cloud DNS managed zone for creating DNS records"
  type        = string
}

variable "enable_cdn" {
  description = "Whether to enable the CDN for the website"
  type        = bool
  default     = true
}
