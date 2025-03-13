locals {
  project_id          = "mtrx-hub-prod-sms"
  storage_bucket_name = "mtrx-us-central1-hub-prod-storage"
  billing_project     = "core-422020"
  k8s_svc_ip_range    = "sn-hub-prod-us-svc-range"
  k8s_pod_ip_range    = "sn-hub-prod-us-pod-range"
  k8s_subnetwork      = "sn-hub-prod-us"
  network_name        = "matrix-hub-prod-nw"
  repo_revision       = "infra-prod-debug"
  docker_registry     = "us-central1-docker.pkg.prod/mtrx-hub-prod-3of"

}

