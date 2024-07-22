
resource "google_bigquery_dataset" "dataset" {
  for_each    = toset(var.kg_sources)
  project     = var.project_id
  dataset_id  = "kg_${replace(each.key, "-", "_")}"
  description = "Dataset with nodes and edges for ${each.key}"
  location    = "US"

  labels = {
    env = var.environment
    kg  = each.key
  }
}

resource "google_bigquery_dataset" "data_api_dataset" {
  project     = var.project_id
  dataset_id  = "data_api"
  description = "Dataset with EveryCure data API components"
  location    = "US"

  labels = {
    env   = var.environment
    owner = "everycure"
  }
}