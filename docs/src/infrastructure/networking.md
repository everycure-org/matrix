---
title: Networking
---

The networking layer of our software engineering project is designed to strike a balance between minimal maintenance requirements and optimal control and security. To achieve this, we employ a hub-and-spoke architecture, where the central hub network provides a Virtual Private Cloud (VPC) to our spoke projects, which represent our working group projects.

To ensure segregation of development and production environments, we maintain two separate networks for each. This allows for effective isolation and management of our infrastructure.

The below graphic visualizes this hub/spoke setup
![](../assets/img/mtrx_network.drawio.svg)

## Firewall Configuration

To facilitate secure communication between our networks and the outside world, we have defined firewall routes to permit the following traffic:

1. HTTP and HTTPS traffic for accessing web interfaces
1. SSH traffic for remote access to our instances and tunneling, only permitted on `dev`

This configuration enables secure and controlled access to our infrastructure, while maintaining the integrity of our network architecture.

## Terraform

All networking is configured in terraform and changes can be proposed through Pull Requests.

## DNS

Our root DNS is registered with Namecheap. We delegated the SOA records for
`dev.everycure.org` to the Google DNS servers which means we can control this entire
subdomain via Google DNS.  Next we defined that domain as a zone in GCP to manage records
here. Now we can create further subdomains (e.g. `docs` to host this page) in this zone.
We use this for domain ownership validation and it is also needed for SSL certificates.

The high level flow of the DNS setup is visualized below:

1. We define the DNS records via terraform as part of our infrastructure rollout
2. The DNS records propagate to the global DNS network and as users enter a domain, their
browsers make DNS lookup calls to their configured DNS server which points their browser
at the correct IP address for the given domain they try to access.

![](../assets/img/mtrx_dns.drawio.svg)

??? info "Primer video on DNS"
    If you need a primer on DNS, this short video may help:
    
    <iframe width="560" height="315" src="https://www.youtube.com/embed/UVR9lhUGAyU?si=KAdxf24jYOzasIwf" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Docs Page

The docs page is hosted via AppEngine. Please check the [their
documentation](https://cloud.google.com/appengine/docs/standard/securing-custom-domains-with-ssl?hl=en)
on how to set up AppEngine with SSL and DNS for a custom domain.


<!-- TODO improve documentation on this -->

we are leveraging the Gateway API from Kubernetes which is relatively new

## Reading material:

- Using Gatweay for Ingress: https://gateway-api.sigs.k8s.io/guides/
- External DNS & Gateway: https://kubernetes-sigs.github.io/external-dns/v0.13.1/tutorials/gateway-api/
- https://cert-manager.io/docs/configuration/acme/
  - Cert manager and Gateway: https://cert-manager.io/docs/usage/gateway/
- how it works on GKE: https://cloud.google.com/kubernetes-engine/docs/concepts/gateway-api
- securing it with IaP : https://cloud.google.com/kubernetes-engine/docs/how-to/secure-gateway


Example with whoami service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: whoami
spec:
  selector:
    app: whoami
  ports:
    - name: http
      port: 80
      targetPort: 80
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: whoami
spec:
  replicas: 1
  selector:
    matchLabels:
      app: whoami
  template:
    metadata:
      labels:
        app: whoami
    spec:
      containers:
        - name: whoami
          image: containous/whoami
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: 100m
              memory: 100Mi
---
kind: HTTPRoute
apiVersion: gateway.networking.k8s.io/v1beta1
metadata:
  name: whoami-route
spec:
  parentRefs: # here we reference the gateway that we want to leverage to route traffic
  - kind: Gateway
    name: external-http
  hostnames:
  - "whoami-test.platform.dev.everycure.org" # this is the subdomain that we want to route to
  rules:
  - backendRefs:
    - name: whoami
      port: 80
```