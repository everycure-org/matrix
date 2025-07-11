locals {
  secrets_output = {
    for k, _ in var.k8s_secrets : k => google_secret_manager_secret.secrets[k].name
  }
  # wrapping as nonsensitive to be able to apply for_each. This requires careful treatment of the secrets to avoid them being leaked as for_each key
  secret_keys = nonsensitive(toset(keys(var.k8s_secrets)))
}

resource "google_secret_manager_secret" "secrets" {
  for_each  = local.secret_keys
  secret_id = each.key
  project   = var.project_id

  replication {
    auto {}
  }
  labels = {
    purpose = "k8s"
    cluster = var.name
  }
}

resource "google_secret_manager_secret_version" "secret_values" {
  for_each       = local.secret_keys
  secret         = google_secret_manager_secret.secrets[each.key].id
  secret_data_wo = var.k8s_secrets[each.key]
}
