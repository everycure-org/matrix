output "google_service_accounts" {
  value = {
    for sa in google_service_account.sa :
    sa.account_id => {
      email        = sa.email
      display_name = sa.display_name
    }
  }
}