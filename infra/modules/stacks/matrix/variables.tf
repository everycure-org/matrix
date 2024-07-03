variable "project_id" {
  type = string
}

variable "default_region" {
  type = string
}

variable "environment" { type = string }

variable "bucket_name" {
  type = string
}

variable "kg_sources" {
  type    = list(string)
  default = ["rtx-kg2", "robokop"]
}