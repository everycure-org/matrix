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
