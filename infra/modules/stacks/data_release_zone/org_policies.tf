# explicitly setting a 
provider "google" {
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true
}

# Disable org policy to allow sharing with people outside of the org
resource "google_org_policy_policy" "domain_restricted_sharing" {
  name   = "projects/${var.project_id}/policies/iam.allowedPolicyMemberDomains"
  parent = "projects/${var.project_id}"

  spec {
    rules {
      allow_all = "TRUE"
    }
  }
}
# 
# # allow public buckets
resource "google_org_policy_policy" "public_access_prevention" {
  depends_on = [google_org_policy_policy.domain_restricted_sharing]
  name       = "projects/${var.project_id}/policies/storage.publicAccessPrevention"
  parent     = "projects/${var.project_id}"

  spec {
    rules {
      enforce = "FALSE"
    }
  }
}

# wait for the org policies to propagate
resource "time_sleep" "wait_for_org_policies" {
  create_duration = "30s"
  triggers = {
    public_access_prevention  = google_org_policy_policy.public_access_prevention.name
    domain_restricted_sharing = google_org_policy_policy.domain_restricted_sharing.name
  }
}