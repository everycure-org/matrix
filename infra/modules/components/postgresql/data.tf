# Get the application user password
data "google_secret_manager_secret_version" "postgresql_app" {
  secret  = "postgresql_super_user_db_password"
  project = var.project_id
}