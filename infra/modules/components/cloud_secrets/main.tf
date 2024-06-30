locals {
  secrets = yamldecode(file("${path.module}/cloud_secrets.yaml"))

  secrets_output = {
      for k, _ in local.secrets : k => google_secret_manager_secret.secrets[k].name
  }

}

resource "google_secret_manager_secret" "secrets" {
  for_each  = local.secrets
  secret_id = each.key
  project = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "secret_values" {
  for_each    = local.secrets
  secret      = google_secret_manager_secret.secrets[each.key].id
  secret_data = each.value
}
