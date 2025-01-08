# Adding a new service via HTTPS

Many of the services we deploy on the cluster should be deployed and made accessible via https. The below steps explain how this is done

## 1. Create a new application in ArgoCD

- Add an application to the  `infra/argo/app-of-apps/templates` file which creates a new application in ArgoCD.
- make sure this points at a separate dev branch so you can work on this deployment without flooding the infra branch with PRs

## 2. Develop the application

- iterate until the application is working as expected
- add an `HttpRoute` which directs the traffic to the new service
- when the service does not respond via 200 at the service port (`/` root path) a `HealthCheckPolicy` is necessary.


## 3. Merge the application to the infra branch

- overwrite the `targetRevision` with `{{ .Values.spec.source.targetRevision }}` to point back at the infra branch
- merge the application to the infra branch

