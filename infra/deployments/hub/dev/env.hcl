locals {
  project_id          = "mtrx-hub-dev-3of"
  storage_bucket_name = "mtrx-us-central1-hub-dev-storage"
  billing_project     = "core-422020"
  k8s_svc_ip_range    = "sn-hub-dev-us-svc-range"
  k8s_pod_ip_range    = "sn-hub-dev-us-pod-range"
  k8s_subnetwork      = "sn-hub-dev-us"
  network_name        = "matrix-hub-dev-nw"
  repo_revision       = "infra"
}
