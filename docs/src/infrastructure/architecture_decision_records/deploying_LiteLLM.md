Architecture Decision Record: Adding LiteLLM to Matrix Infrastructure Kubernetes Stack
Date: 04-09-2025
Status: Proposed
Author: Platform Engineering Team

# Context

The MATRIX platform at EveryCure requires a robust LLM gateway to manage multiple AI model providers efficiently. As we expand our AI-driven drug repurposing capabilities using Gemini 2.0 and other LLM providers, we need a unified interface that provides:

- Consistent API interface across multiple LLM providers
- Cost tracking and budget management
- Rate limiting and load balancing
- Caching for improved performance and cost reduction
- Audit logging and compliance
- High availability and scalability

Our current infrastructure runs on Google Kubernetes Engine (GKE) and we need a solution that integrates seamlessly with our existing Kubernetes stack.

# Decision

We will deploy LiteLLM Proxy Server as our LLM gateway within our matrix infrastructure Kubernetes stack with the following components:

LiteLLM Proxy deployed via Helm chart
PostgreSQL for persistent storage and user management
Redis for caching and distributed state management
Istio as the service mesh for traffic management

# Implementation Details

1. **LiteLLM Configuration**: We will deploy it through Helm chart via ArgoCD in a namespace called `lite-llm`
2. **PostgreSQL**: Deployment of an in-cluster database in a different namespace. Database users and tables to be managed through. Terraform.
3. **Redis**: In-cluster Deployment caching in a different namespace.
4. **Istio**: In-cluster Deployment service mesh in a different namespace.

# Alternatives Considered

# Istio

**Nginx**: Too basic for service mesh requirements (primarily an ingress controller)
**Envoy**: Requires manual configuration, lacks control plane
**Cilium**: While performant with eBPF, less mature for enterprise service mesh needs
**Kong + Custom Plugin**: More complex, requires custom development
**AWS API Gateway**: Vendor lock-in, limited LLM features
**Custom Solution**: High maintenance, long development time
**Portkey.ai**: External dependency, data sovereignty concerns
