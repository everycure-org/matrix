# Glossary

- `kubectl` - command line tool for interacting with kubernetes clusters. Acts as intermediary between user and the Kubernetes cluster.
- `kind` - Kuberntes IN Docker - is a tool that allows creation of local Kubernetes clusters directly in Docker containers. It’s particularly useful for testing, development and CI/CD pipelines.
- `Argo` is an open-source suite of tools designed to make it easier to deploy, manage, and operate containerized applications and workflows on Kubernetes.
    - `Argo Workflows` - A Kubernetes-native workflow engine for orchestrating parallel jobs or steps in a process. Often used for CI/CD pipelines, ETL jobs, data processing, and machine learning workflows.
    - `Argo CD` - A declarative, GitOps continuous delivery tool for Kubernetes. Ideal for managing Kubernetes applications using GitOps practices. It automatically syncs the desired state of Kubernetes resources defined in Git repositories with the cluster.
    - `Argo Rollouts` - A Kubernetes controller for advanced deployment strategies such as blue-green deployments, canary releases, and progressive delivery. Perfect for managing the release process of applications, allowing for gradual rollouts and quick rollbacks.
    - `Argo Events` - An event-driven automation tool that triggers actions based on specific events occurring within or outside the Kubernetes cluster. Commonly used to automate tasks in response to events, such as running a workflow when new data arrives or triggering a deployment when a webhook is received. Supports event sources like webhooks, S3 events, Kafka, and more. It also integrates with Argo Workflows and Argo CD to trigger complex pipelines.
- `k9s` - K9s is a terminal-based interface that provides a streamlined, interactive way to manage Kubernetes clusters. It’s designed to improve productivity by making it easier to navigate and manage Kubernetes resources without needing to type complex kubectl commands.

