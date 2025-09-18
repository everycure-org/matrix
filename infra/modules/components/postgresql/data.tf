# Get the application user password
data "google_secret_manager_secret_version" "postgresql_app" {
  secret  = "postgresql_super_user_db_password"
  project = var.project_id
}

# Get the Litellm Postgres password
data "google_secret_manager_secret_version" "litellm_postgres_password" {
  secret  = "litellm_postgres_password"
  project = var.project_id
}