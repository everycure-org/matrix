---
title: GCP Foundations
---

- infra was set up using the [terraform-example-foundation](https://github.com/terraform-google-modules/terraform-example-foundation) repository.

## OOTB Features

- networking
- `project_budget` for projectsG

## CI for Foundations
Can be found here:
https://console.cloud.google.com/cloud-build/builds;region=us-central1?project=prj-b-cicd-lvhz

TODO we will move this to Github Actions

Repo here: [everycure-org/core](https://github.com/everycure-org/core)

## Networking setup

We opted for a [shared VPC](https://cloud.google.com/vpc/docs/shared-vpc) setup from GCP for all our MATRIX projects. In this setup, we have a hub project that contains any shared resources (e.g. compute cluster, shared datasets) and a number of spoke projects (working groups) that connect to the hub project's VPC. 

## Creation log

### Using the helper
- used the foundation-deployer helper to create the infrastructure
- used `core-422020` project as the "starter" project and set all permissions to the `gcp-admins@everycure.org` group with `pascal@everycure.org` as member.
- selected the simpler networking structure & went for the google cloud build based CI/CD pipeline rather than Github Actions (which was default & should be less cost constrained)
- created a `Makefile` in `infra/` repo to codify all steps used to set it up
- had some issues with getting the backend.tf to work with the helper, so had to re-run a few times to get 0-bootstrap to work
- then ran into issues with the SA not having the right permissions to edit groups
  ```
  # Request a Super Admin to Grant 'Group Admin' role in the
  # Admin Console of the Google Workspace to the Bootstrap service account:
  # sa-terraform-bootstrap@prj-b-seed-77e7.iam.gserviceaccount.com
  
  # See: https://cloud.google.com/identity/docs/how-to/setup#assigning_an_admin_role_to_the_service_account
  # for additional information
  
  # Press Enter to continue
  ```
  - found the `roleID` for the Group Administrator via [this page](https://developers.google.com/admin-sdk/directory/reference/rest/v1/roles/list?apix=true&apix_params={"customer":"C02dmw33l","maxResults":50}) to be `49856820306509826`
  ```json
  {
    "assignedTo": "111866437155274016643",
    "roleId": "49856820306509826",
    "scopeType": "CUSTOMER",
    "kind": "admin#directory#roleAssignment"
  }
- this added the SA to be able to do roleAssignments from within Cloud Build which makes the SA also a super admin in a way
- stages org & environments were created without issues through cloud build
- net failed. got [stuck on this](https://github.com/terraform-google-modules/terraform-example-foundation/issues/923)
- adding `roles/iam.serviceAccountUser` and `roles/iam.serviceAccountTokenCreator` to the `gcp-admins` group (which I am a member of) at root level fixed the issue. 
- ran out of projects so wanted to remove BU2. But need to terraform destroy everything first, not just delete folder. Else it just disappears but resources remain.  
- So I commented out the resources created in BU2 first, then ran terraform apply again. Why not terraform destroy? Well because the `tf-wrapper.sh` doesn't have a destroy command.
- running the destroy commands from my machine kept giving me errors as well
  ```
  Error: Error when reading or editing BillingBudget
  "billingAccounts/01980D-0D4096-C8CEA4/budgets/c74f3114-fed7-4946-a691-92e2ab7d94c2":
  googleapi: Error 403: Your application is authenticating by using local Application
  Default Credentials. The billingbudgets.googleapis.com API requires a quota project,
  which is not set by default. To learn how to set your quota project, see
  https://cloud.google.com/docs/authentication/adc-troubleshooting/user-creds .
  ```
  - setting the quota project in the gcloud config did not fix the issue though.
  - resolution? 

     ```bash
     export GOOGLE_IMPERSONATE_SERVICE_ACCOUNT=sa-terraform-proj@prj-b-seed-77e7.iam.gserviceaccount.com
     ```

- Decided to swap the entire "projects" layer out for a new one using terragrunt. this whole tf-wrapper.sh is just too much of a home baked solution. Bad Google! Bad!

### Using Terragrunt

Switching over to terragrunt

```bash
# create root
touch terragrunt.hcl
# create folders
mkdir -p matrix/{hub,data_wg,modelling_wg,validation_wg}/{development,production}/
touch matrix/initiative.hcl
touch  matrix/{hub,data_wg,modelling_wg,validation_wg}//workinggroup.hcl
touch  matrix/{hub,data_wg,modelling_wg,validation_wg}/{development,production}/terragrunt.hcl

# Define the multiline string using a here document
multiline_string=$(cat <<EOF
# Include the root `terragrunt.hcl` configuration. The root configuration contains settings that are common across all
# components and environments, such as how to configure remote state.
include "root" {
 path = find_in_parent_folders()
}
#
# Include the envcommon configuration for the component. The envcommon configuration contains settings that are common
# for the component across all environments.
# include "envcommon" {
#  path = "${dirname(find_in_parent_folders())}/_envcommon/dns.hcl"
# }
EOF
)

# Use brace expansion to generate file paths and append the multiline string to each one
for file in matrix/{hub,data_wg,modelling_wg,validation_wg}/{development,production}/terragrunt.hcl; do
    # Append the multiline string to each generated file path
    cat <<EOF >> "$file"
$multiline_string
EOF
done

```

Rolling out groups, folders, projects, and budgets using terragrunt. Way easier!

### Remove networking & environments layers

- We really don't want to deal with networking and the folders were also not what we want. 