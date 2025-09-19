variable "project_id" {
  type        = string
  description = "The project ID for the Google Cloud resources"
}

variable "host" {
  type        = string
  description = "The host for the PostgreSQL database"
  default     = "localhost"
}