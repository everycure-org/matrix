# Continuous Integration

## Testing

We leverage Github Actions to test and build the pipeline. For testing, we leverage a mix
of unit and integration tests. The integration tests leverage a set of fabricated
datasets to test the end-to-end pipeline without the need to access real data. 

Check the `.github/workflows/matrix-ci.yml` folder for details on how we test our pipeline.

## Building executables

We execute our pipeline on Kubernetes. For this we build container images with github
actions and push them to the artifact registry of GCP. See the Github actions pipeline
for how this is implemented. 

## License handling

To make sure we do not use any software that gets us into trouble later on, we have
`trivy` scan for licenses of software we use. However, we don't yet have this implemented
in our CI. This will have to be added to our release pipeline when we build it. For now,
the Makefile in our `matrix` pipeline holds the command to scan for licenses.

## Secrets files

As mentioned in the [GCP Foundations](../infrastructure/gcp.md) page, we use
`git-crypt`. Because the CI doesn't have a public-private GPG key to share with us, we
exported the symmetric key and added it to github actions' secrets. 

```
git-crypt export-key /tmp/key && cat /tmp/key | base64 && rm /tmp/key
```
