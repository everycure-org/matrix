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

module "litellm_postgresql" {
  source      = "./module"
  project_id  = var.project_id
  secret_name = "litellm_postgres_password"
  schema_name = "litellm"
}

module "agro_workflow_postgresql" {
  source      = "./module"
  project_id  = var.project_id
  secret_name = "Argo_workflows_postgres_password"
  schema_name = "agro_workflow"
}