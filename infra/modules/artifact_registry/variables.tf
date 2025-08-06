variable "project_id" {
  description = "The GCP project ID."
  type        = string
}

variable "location" {
  description = "The location for the Artifact Registry repository."
  type        = string
}

variable "repository_id" {
  description = "The ID of the repository. Should be separated by dashes (e.g., 'my-repo')."
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,63}$", var.repository_id))
    error_message = "repository_id must start with a lowercase letter, followed by lowercase letters, digits, or dashes, and be between 5 and 63 characters long."
  }
  type = string
}

variable "format" {
  description = "The format of the repository (e.g., DOCKER, MAVEN, NPM)."
  type        = string
  default     = "DOCKER"
  validation {
    condition     = contains(["DOCKER", "MAVEN", "NPM"], var.format)
    error_message = "format must be one of: DOCKER, MAVEN, NPM."
  }
}

variable "description" {
  description = "A description for the repository."
  type        = string
  default     = "Managed by Terraform."
}

variable "delete_older_than_days" {
  description = "The number of days after which to delete images. Set to 0 to disable. Currently set to 3 days."
  type        = string
  validation {
    condition     = can(regex("^[0-9]+[dD]$", var.delete_older_than_days))
    error_message = "delete_older_than_days must be a number followed by 'd' (e.g., '3d' for 3 days)."
  }
  default = "3d" # Default value is set to 3 days
}

variable "keep_count" {
  description = "The minimum number of recent versions to keep. Disabled by default."
  type        = number
  default     = 0
}

variable "sample_run_tag_prefixes" {
  description = "The tag prefixes for sample-run images."
  type        = list(string)
  default     = []

}