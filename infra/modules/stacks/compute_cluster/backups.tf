resource "google_gke_backup_backup_plan" "rpo_daily_window" {
  name     = "rpo-daily-window"
  cluster  = module.gke.cluster_id
  location = var.default_region
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

resource "google_gke_backup_restore_plan" "mlflow_and_neo4j" {
  name        = "restore-neo-mlflow-ns"
  location    = var.default_region
  backup_plan = google_gke_backup_backup_plan.rpo_daily_window.id
  cluster     = module.gke.cluster_id
  restore_config {
    selected_namespaces {
      namespaces = ["neo4j", "mlflow"]
    }
    namespaced_resource_restore_mode = "MERGE_REPLACE_ON_CONFLICT"
    volume_data_restore_policy       = "RESTORE_VOLUME_DATA_FROM_BACKUP"
    cluster_resource_restore_scope {
      no_group_kinds = true
    }
    cluster_resource_conflict_policy = "USE_EXISTING_VERSION"
  }
}