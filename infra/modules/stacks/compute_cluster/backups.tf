resource "google_gke_backup_backup_plan" "rpo_daily_window" {
  name     = "rpo-daily-window"
  cluster  = module.gke.cluster_id
  location = "us-central1"
  retention_policy {
    backup_delete_lock_days = 7
    backup_retain_days      = 14
  }
  backup_schedule {
    paused = false
    rpo_config {
      target_rpo_minutes = 1440
    }
  }
  backup_config {
    include_volume_data = true
    include_secrets     = false
    all_namespaces      = true
  }
}