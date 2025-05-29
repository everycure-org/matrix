locals {
  group_sa_map = {
    internal_data_science = {
      group      = "group:data-science@everycure.org"
      account_id = "sa-internal-data-science"
    }
    external_subcon_standard = {
      group      = "group:ext.subcontractors.standard@everycure.org"
      account_id = "sa-subcon-standard"
    }
    external_subcon_embiology = {
      group      = "group:ext.subcontractors.embiology@everycure.org"
      account_id = "sa-subcon-embiology"
    }
  }
}

resource "google_service_account" "sa" {
  for_each     = local.group_sa_map
  account_id   = each.value.account_id
  display_name = "Service account for ${each.key}"
}

resource "google_service_account_iam_member" "allow_impersonation_service_account_user" {
  for_each = local.group_sa_map

  service_account_id = google_service_account.sa[each.key].name
  role               = "roles/iam.serviceAccountUser"
  member             = each.value.group
}

resource "google_service_account_iam_member" "allow_impersonation_service_account_token_creator" {
  for_each = local.group_sa_map

  service_account_id = google_service_account.sa[each.key].name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = each.value.group
}
