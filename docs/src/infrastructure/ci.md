# CI 

We currently have a self-hosted-runner active on GCP which is not codified in terraform but executes our CI tests. This should likely be replaced with a more efficient setp

## Testing

We leverage Github Actions to test and build the pipeline. For testing, we leverage a mix
of unit and integration tests. The integration tests leverage a set of fabricated
datasets to test the end-to-end pipeline without the need to access real data. 

Check the `.github/workflows/matrix-ci.yml` folder for details on how we test our pipeline.

## Building executables

We execute our pipeline on Kubernetes. For this we build container images with github
actions and push them to the artifact registry of GCP. See the Github actions pipeline
for how this is implemented. 

## OIDC setup for GitHub Actions

We leverage OIDC to authenticate GitHub actions to deploy to our environment

### JWT Token of GitHub

Because debugging this can be painful, here is the JWT token sent from GH Actions to GCP for reference

```json
{
  "actor": "pascalwhoop",
  "actor_id": "2284951",
  "aud": "https://github.com/everycure-org",
  "base_ref": "",
  "event_name": "push",
  "exp": 1725377667,
  "head_ref": "",
  "iat": 1725377367,
  "iss": "https://token.actions.githubusercontent.com",
  "job_workflow_sha": "e75edcc5b6cb88fc32f8d65d1da2655576b905ca",
  "jti": "3fb4ff7d-3582-4a13-b419-5727d5d94efa",
  "nbf": 1725376767,
  "ref": "refs/heads/main",
  "ref_protected": "true",
  "ref_type": "branch",
  "repository": "everycure-org/matrix",
  "repository_id": "797215326",
  "repository_owner": "everycure-org",
  "repository_owner_id": "162148032",
  "repository_visibility": "private",
  "run_attempt": "1",
  "run_id": "10685993555",
  "run_number": "154",
  "runner_environment": "github-hosted",
  "sha": "e75edcc5b6cb88fc32f8d65d1da2655576b905ca",
  "sub": "repo:everycure-org/matrix:ref:refs/heads/main",
  "workflow": "Infrastructure Deploy",
  "workflow_ref": "everycure-org/matrix/.github/workflows/main-deploy.yml@refs/heads/main",
  "workflow_sha": "e75edcc5b6cb88fc32f8d65d1da2655576b905ca"
}
```

## License handling

To make sure we do not use any software that gets us into trouble later on, we have
`trivy` scan for licenses of software we use. However, we don't yet have this implemented
in our CI. This will have to be added to our release pipeline when we build it. For now,
the Makefile in our `matrix` pipeline holds the command to scan for licenses.
