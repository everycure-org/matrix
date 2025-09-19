variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "secret_name" {
  description = "The GCP secret name"
  type        = string
}

variable "schema_name" {
  description = "The name of the Postgres schema to create"
  type        = string
}