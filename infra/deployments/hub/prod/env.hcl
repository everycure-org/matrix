locals {
  project_id          = "mtrx-hub-prod-sms"
  storage_bucket_name = "mtrx-us-central1-hub-prod-storage"
  billing_project     = "core-422020"
  k8s_svc_ip_range    = "sn-hub-prod-us-svc-range"
  k8s_pod_ip_range    = "sn-hub-prod-us-pod-range"
  k8s_subnetwork      = "sn-hub-prod-us"
  network_name        = "matrix-hub-prod-nw"
  repo_revision       = "main"
  aip_oauth_client_id = "864737999815-il01aht59rcsg9nal2bcik66mujd8hv5.apps.googleusercontent.com"
  infra_bucket_name   = "mtrx-hub-prod-infra"
}
