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

# Create dedicated schema for litellm
resource "postgresql_schema" "litellm" {
  name     = "litellm"
  database = "app"
  owner    = postgresql_role.litellm.name

  depends_on = [postgresql_role.litellm]
}

# Grant ALL privileges on the litellm schema
resource "postgresql_grant" "litellm_schema_all" {
  database    = "app"
  role        = postgresql_role.litellm.name
  schema      = postgresql_schema.litellm.name
  object_type = "schema"
  privileges  = ["USAGE", "CREATE"]

  depends_on = [postgresql_schema.litellm]
}

# Grant ALL privileges on all existing tables in litellm schema
resource "postgresql_grant" "litellm_tables" {
  database    = "app"
  role        = postgresql_role.litellm.name
  schema      = postgresql_schema.litellm.name
  object_type = "table"
  privileges  = ["ALL"]

  depends_on = [postgresql_schema.litellm]
}

# Grant ALL privileges on all existing sequences in litellm schema
resource "postgresql_grant" "litellm_sequences" {
  database    = "app"
  role        = postgresql_role.litellm.name
  schema      = postgresql_schema.litellm.name
  object_type = "sequence"
  privileges  = ["ALL"]

  depends_on = [postgresql_schema.litellm]
}

# Grant default privileges for future tables in litellm schema
resource "postgresql_default_privileges" "litellm_future_tables" {
  database    = "app"
  role        = postgresql_role.litellm.name
  schema      = postgresql_schema.litellm.name
  owner       = postgresql_role.litellm.name
  object_type = "table"
  privileges  = ["ALL"]

  depends_on = [postgresql_schema.litellm]
}

# Grant default privileges for future sequences in litellm schema
resource "postgresql_default_privileges" "litellm_future_sequences" {
  database    = "app"
  role        = postgresql_role.litellm.name
  schema      = postgresql_schema.litellm.name
  owner       = postgresql_role.litellm.name
  object_type = "sequence"
  privileges  = ["ALL"]

  depends_on = [postgresql_schema.litellm]
}