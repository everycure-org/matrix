variable "storage_bucket_name" {
  default = "mtrx-us-central1-wg2-modeling-dev-storage"

}

variable "shared_network_name" {
  default = "projects/mtrx-hub-dev-3of/global/networks/matrix-hub-dev-nw"
}

variable "shared_subnetwork_name" {
  default = "projects/mtrx-hub-dev-3of/regions/us-central1/subnetworks/sn-hub-dev-us"
}

variable "project_id" {
  default = "mtrx-wg2-modeling-dev-9yj"
}

variable "location" {
  default = "us-central1"
}

variable "github_token" {
  description = "GitHub token for cloning the matrix repository"
  type        = string
  sensitive   = true
}
