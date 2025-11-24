locals {
  # Base paths
  embiology_path_raw                 = "data/01_RAW/embiology"
  embiology_path                     = "projects/_/buckets/${var.storage_bucket_name}/objects/${local.embiology_path_raw}/"
  dev_bucket_name                    = "mtrx-us-central1-hub-dev-storage"
  orchard_dev_project_id             = "ec-orchard-dev"
  orchard_prod_project_id            = "ec-orchard-prod"
  prod_k8s_sas                       = "serviceAccount:sa-k8s-node@mtrx-hub-prod-sms.iam.gserviceaccount.com"         # Kubernetes node SA for cluster workloads
  matrix_hub_dev_github_sa_rw_member = "serviceAccount:sa-github-actions-rw@mtrx-hub-dev-3of.iam.gserviceaccount.com" # GitHub Actions RW SA for dev


  # Allowed path prefixes for access
  allowed_paths = [
    "projects/_/buckets/${var.storage_bucket_name}/objects/data/",
    "projects/_/buckets/${var.storage_bucket_name}/objects/data_releases/",
    "projects/_/buckets/${var.storage_bucket_name}/objects/kedro/data/",
  ]

  # OR condition for IAM expression (allowed prefixes)
  path_or_expression = join(" || ", [
    for p in local.allowed_paths : "resource.name.startsWith(\"${p}\")"
  ])

  # Expression: allow all except embiology
  path_or_exclude_embiology = "(${local.path_or_expression}) && !resource.name.startsWith(\"${local.embiology_path}\")"

  # Expression: include allowed + embiology
  path_or_include_embiology = join(" || ", concat(local.allowed_paths, [local.embiology_path]))

  # Service Account to Group Mapping
  # This map defines the service accounts and their corresponding groups
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

  # Common binding members (dynamically resolved in usage)
  binding_members = [
    google_service_account.sa["internal_data_science"].member,
    google_service_account.sa["external_subcon_standard"].member,
    google_service_account.sa["external_subcon_embiology"].member,
  ]
}