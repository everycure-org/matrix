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






## Gateway API Setup

Our project utilizes the Kubernetes Gateway API, which represents the next generation of Kubernetes Ingress, Load Balancing, and Service Mesh APIs. It's designed to be generic, expressive, and role-oriented[^1]. The core idea of the setup is visualized in this diagram well by Google:

![](https://cloud.google.com/static/kubernetes-engine/images/model-gateway-per-cluster.svg)

### Components

#### 1. External DNS

We've configured External DNS to work with Gateway API resources:

```yaml
external-dns:
  provider:
    name: google
  extraArgs:
    - --source=gateway-httproute
  rbac:
    additionalPermissions:
      - apiGroups: ["gateway.networking.k8s.io"]
        resources: ["gateways","httproutes","grpcroutes","tlsroutes","tcproutes","udproutes"] 
        verbs: ["get","watch","list"]
      - apiGroups: [""]
        resources: ["namespaces"]
        verbs: ["get","watch","list"]
```

This configuration allows External DNS to manage DNS records based on Gateway API resources, particularly HTTPRoutes.

#### 2. Cert Manager

We've enabled Cert Manager to work with Gateway API:

```yaml
cert-manager:
  config:
    enableGatewayAPI: true
```

This allows Cert Manager to provision and manage TLS certificates for Gateway resources[^2].

#### Example: Whoami Service

We've set up a simple "whoami" service to demonstrate the use of Gateway API:

```yaml
kind: HTTPRoute
apiVersion: gateway.networking.k8s.io/v1beta1
metadata:
  name: whoami-route
spec:
  parentRefs:
  - kind: Gateway
    name: external-http
  hostnames:
  - "whoami-test.platform.dev.everycure.org"
  rules:
  - backendRefs:
    - name: whoami
      port: 80
```

This HTTPRoute resource:
- Associates with a Gateway named "external-http"
- Routes traffic for the hostname "whoami-test.platform.dev.everycure.org"
- Directs traffic to the "whoami" service on port 80

### Key Concepts

1. **GatewayClass**: Defines a set of Gateways with a common configuration and behavior.
2. **Gateway**: Describes how traffic can be translated to Services within the cluster.
3. **HTTPRoute**: Describes how HTTP requests should be routed by a Gateway.

Our setup leverages these concepts to provide a flexible and powerful routing solution.


### Additional Resources

- **Using Gateway for Ingress**: [https://gateway-api.sigs.k8s.io/guides/](https://gateway-api.sigs.k8s.io/guides/)
- **External DNS & Gateway**: [https://kubernetes-sigs.github.io/external-dns/v0.13.1/tutorials/gateway-api/](https://kubernetes-sigs.github.io/external-dns/v0.13.1/tutorials/gateway-api/)
- **Cert Manager Configuration**:
  - **ACME**: [https://cert-manager.io/docs/configuration/acme/](https://cert-manager.io/docs/configuration/acme/)
  - **Cert Manager and Gateway**: [https://cert-manager.io/docs/usage/gateway/](https://cert-manager.io/docs/usage/gateway/)
- **Gateway API on GKE**:
  - **How it Works**: [https://cloud.google.com/kubernetes-engine/docs/concepts/gateway-api](https://cloud.google.com/kubernetes-engine/docs/concepts/gateway-api)
  - **Securing with IAP**: [https://cloud.google.com/kubernetes-engine/docs/how-to/secure-gateway](https://cloud.google.com/kubernetes-engine/docs/how-to/secure-gateway)

[^1]: https://gateway-api.sigs.k8s.io/
[^2]: https://gateway-api.sigs.k8s.io/guides/