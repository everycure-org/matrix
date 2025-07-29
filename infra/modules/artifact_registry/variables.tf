variable "project_id" {
  description = "The GCP project ID."
  type        = string
}

variable "location" {
  description = "The location for the Artifact Registry repository."
  type        = string
}

variable "repository_id" {
  description = "The ID of the repository."
  type        = string
}

variable "format" {
  description = "The format of the repository (e.g., DOCKER, MAVEN, NPM)."
  type        = string
  default     = "DOCKER"
}

variable "description" {
  description = "A description for the repository."
  type        = string
  default     = "Managed by Terraform."
}

variable "delete_older_than_days" {
  description = "The number of days after which to delete images. Set to 0 to disable. Currently set to 3 days."
  type        = number
  default     = 3
}

variable "keep_count" {
  description = "The minimum number of recent versions to keep. Disabled by default."
  type        = number
  default     = 0
}

variable "tag_state" {
  description = "The tag state to apply to the cleanup policies. Defaults to ANY."
  type        = string
  default     = "ANY"

}