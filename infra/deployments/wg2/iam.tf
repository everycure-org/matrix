# Giving hub-dev SA access to this project to allow it to deploy via CI/CD

module "project_iam_bindings" {
  source   = "terraform-google-modules/iam/google//modules/projects_iam"
  projects = [var.project_id]
  version  = "~> 8.0"

  mode = "additive"

  bindings = {
    "roles/viewer"                      = ["serviceAccount:sa-github-actions-ro@mtrx-hub-dev-3of.iam.gserviceaccount.com"]
    "roles/owner"                       = ["serviceAccount:sa-github-actions-rw@mtrx-hub-dev-3of.iam.gserviceaccount.com"]
    "roles/compute.osLoginExternalUser" = ["group:matrix-all@everycure.org"]
  }

}
