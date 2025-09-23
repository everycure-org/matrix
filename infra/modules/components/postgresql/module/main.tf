resource "postgresql_role" "role" {
  name             = "${var.schema_name}-role"
  password         = data.google_secret_manager_secret_version.secret.secret_data
  login            = true
  superuser        = false
  inherit          = true
  replication      = false
  connection_limit = -1
  create_database  = true
  create_role      = true
}

# Create dedicated schema for litellm
resource "postgresql_schema" "schema" {
  name     = var.schema_name
  database = "app"
  owner    = postgresql_role.role.name

  depends_on = [postgresql_role.role]
}

# Grant ALL privileges on the schema
resource "postgresql_grant" "schema_all" {
  database    = "app"
  role        = postgresql_role.role.name
  schema      = postgresql_schema.schema.name
  object_type = "schema"
  privileges  = ["USAGE", "CREATE"]

  depends_on = [postgresql_schema.schema]
}

# Grant ALL privileges on all existing tables in schema
resource "postgresql_grant" "schema_tables" {
  database    = "app"
  role        = postgresql_role.role.name
  schema      = postgresql_schema.schema.name
  object_type = "table"
  privileges  = ["ALL"]

  depends_on = [postgresql_schema.schema]
}

# Grant ALL privileges on all existing sequences in schema
resource "postgresql_grant" "schema_sequences" {
  database    = "app"
  role        = postgresql_role.role.name
  schema      = postgresql_schema.schema.name
  object_type = "sequence"
  privileges  = ["ALL"]

  depends_on = [postgresql_schema.schema]
}

# Grant default privileges for future tables in schema
resource "postgresql_default_privileges" "schema_future_tables" {
  database    = "app"
  role        = postgresql_role.role.name
  schema      = postgresql_schema.schema.name
  owner       = postgresql_role.role.name
  object_type = "table"
  privileges  = ["ALL"]

  depends_on = [postgresql_schema.schema]
}

# Grant default privileges for future sequences in schema
resource "postgresql_default_privileges" "schema_future_sequences" {
  database    = "app"
  role        = postgresql_role.role.name
  schema      = postgresql_schema.schema.name
  owner       = postgresql_role.role.name
  object_type = "sequence"
  privileges  = ["ALL"]

  depends_on = [postgresql_schema.schema]
}