# NOTE: This file was partially generated using AI assistance.

variable "username" {
  description = "User to create workbench for"
  type        = string
  validation {
    condition     = can(regex("^[a-zA-Z]+$", var.username))
    error_message = "Username must contain only letters and no spaces."
  }
}

variable "email" {
  description = "Email of the user"
  type        = string
}

variable "location" {
  description = "The location where workbenches will be created"
  type        = string
  default     = "us-central1-a"
}

variable "machine_type" {
  description = "The machine type for the workbench instances"
  type        = string
  default     = "e2-standard-8"
}

variable "network" {
  description = "The VPC network to use for the workbench instances"
  type        = string
}

variable "subnet" {
  description = "The subnet to use for the workbench instances"
  type        = string
}

variable "service_account" {
  description = "The service account email to use for the workbench instances"
  type        = string
}

variable "project_id" {
  description = "The project ID where the workbenches will be created"
  type        = string
}

variable "boot_disk_size_gb" {
  description = "Size of the boot disk in GB"
  type        = number
  default     = 150
}

variable "data_disk_size_gb" {
  description = "Size of the data disk in GB"
  type        = number
  default     = 100
}

variable "labels" {
  description = "Additional labels to apply to the workbench instances"
  type        = map(string)
  default     = {}
}

variable "post_startup_script" {
  description = "The post startup script to run on the workbench instances"
  type        = string
}
