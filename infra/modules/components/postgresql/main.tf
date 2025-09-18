provider "postgresql" {
  host             = "localhost"
  port             = "5432"
  database         = "app"
  username         = "postgres"
  password         = data.google_secret_manager_secret_version.postgresql_app.secret_data
  connect_timeout  = 15
  superuser        = true
  expected_version = "17.6"
}

# You would need to port-forward the Kubernetes service to your localhost on port 5434 for this to work
# Example command:
# kubectl port-forward -n postgresql svc/postgresql-cloudnative-pg-cluster-pooler-rw 5432:5432

resource "postgresql_role" "litellm" {
  name             = "litellm"
  password         = data.google_secret_manager_secret_version.litellm_postgres_password.secret_data
  login            = true
  superuser        = false
  inherit          = true
  replication      = false
  connection_limit = -1
  create_database  = true
  create_role      = true
}