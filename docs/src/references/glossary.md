# Glossary

<!-- Please add various terms and definitions here. -->

## Kubernetes & Container Orchestration

- `kubectl` - command line tool for interacting with kubernetes clusters. Acts as intermediary
  between user and the Kubernetes cluster.
- `kind` - Kuberntes IN Docker - is a tool that allows creation of local Kubernetes clusters
  directly in Docker containers. It's particularly useful for testing, development and CI/CD
  pipelines.
- `k9s` - K9s is a terminal-based interface that provides a streamlined, interactive way to manage
  Kubernetes clusters. It's designed to improve productivity by making it easier to navigate and
  manage Kubernetes resources without needing to type complex kubectl commands.
- `GKE` - Google Kubernetes Engine - A managed Kubernetes service provided by Google Cloud Platform
  that handles cluster management, scaling, and updates.

## Argo Suite

- `Argo` is an open-source suite of tools designed to make it easier to deploy, manage, and operate
  containerized applications and workflows on Kubernetes.
  - `Argo Workflows` - A Kubernetes-native workflow engine for orchestrating parallel jobs or steps
    in a process. Often used for CI/CD pipelines, ETL jobs, data processing, and machine learning
    workflows.
  - `Argo CD` - A declarative, GitOps continuous delivery tool for Kubernetes. Ideal for managing
    Kubernetes applications using GitOps practices. It automatically syncs the desired state of
    Kubernetes resources defined in Git repositories with the cluster.
  - `Argo Rollouts` - A Kubernetes controller for advanced deployment strategies such as blue-green
    deployments, canary releases, and progressive delivery. Perfect for managing the release process
    of applications, allowing for gradual rollouts and quick rollbacks.
  - `Argo Events` - An event-driven automation tool that triggers actions based on specific events
    occurring within or outside the Kubernetes cluster. Commonly used to automate tasks in response
    to events, such as running a workflow when new data arrives or triggering a deployment when a
    webhook is received. Supports event sources like webhooks, S3 events, Kafka, and more. It also
    integrates with Argo Workflows and Argo CD to trigger complex pipelines.

## Data & ML Infrastructure

- `MLflow` - An open-source platform for managing the complete machine learning lifecycle:

  - Tracks experiments, parameters, and metrics
  - Packages and versions ML code for reproducibility
  - Manages and deploys models from multiple ML libraries
  - [Learn more: MLflow Documentation](https://mlflow.org/docs/latest/index.html)

- `Neo4j` - A native graph database platform optimized for connected data:

  - Stores data in nodes and relationships instead of tables
  - Provides a powerful query language (Cypher)
  - Offers built-in graph algorithms and analytics
  - [Learn more: Neo4j Graph Database](https://neo4j.com/docs/getting-started/current/)

- `Kedro` - An open-source Python framework for creating reproducible data science code:

  - Enforces software engineering best practices
  - Enables modular, maintainable pipeline development
  - Provides data versioning and lineage tracking
  - [Learn more: Kedro Documentation](https://kedro.readthedocs.io/en/stable/)

- `PySpark` - The Python API for Apache Spark, a unified analytics engine:

  - Enables distributed data processing at scale
  - Provides high-level APIs for SQL, ML, and graph processing
  - Optimizes execution through DAG-based scheduling
  - [Learn more: PySpark Guide](https://spark.apache.org/docs/latest/api/python/)

- `Vertex AI` - Google Cloud's unified ML platform:

  - Provides end-to-end ML model development tools
  - Offers managed infrastructure for training and serving
  - Integrates with popular ML frameworks
  - [Learn more: Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)

- `Pandera` - A statistical data testing toolkit for Python:

  - Validates pandas DataFrames using expressive schemas
  - Enables statistical and logical data checks
  - Integrates with data science workflows
  - [Learn more: Pandera Documentation](https://pandera.readthedocs.io/)

- `Joblib` - A set of tools for lightweight pipelining in Python:

  - Provides transparent disk-caching of functions
  - Enables simple parallel computing
  - Optimizes NumPy array serialization
  - [Learn more: Joblib Documentation](https://joblib.readthedocs.io/)

- `FastAPI` - A modern web framework for building APIs with Python:

  - Offers automatic API documentation
  - Provides high performance through async support
  - Implements type hints and data validation
  - [Learn more: FastAPI Documentation](https://fastapi.tiangolo.com/)

- `Pydantic` - A data validation library using Python type annotations:
  - Enforces type hints at runtime
  - Provides powerful data parsing and serialization
  - Integrates with modern Python frameworks
  - [Learn more: Pydantic Documentation](https://docs.pydantic.dev/)

## Cloud & Infrastructure

- `GCP` - Google Cloud Platform - A suite of cloud computing services that runs on the same
  infrastructure that Google uses internally.
- `Terraform` - An infrastructure as code tool that enables you to safely and predictably create,
  change, and improve infrastructure.
- `Terragrunt` - A thin wrapper for Terraform that provides extra tools for working with multiple
  Terraform modules, remote state, and keeping configurations DRY.
- `IAP` - Identity-Aware Proxy - A Google Cloud service that provides a central authentication layer
  to applications accessed by HTTPS.

## Development Tools

- `git-crypt` - A tool that enables transparent encryption and decryption of files in a git
  repository, used for securing sensitive information in our codebase.
- `Docker` - A platform for developing, shipping, and running applications in containers, providing
  consistency across development and production environments.
- `Artifact Registry` - Google Cloud's fully managed service for storing, managing, and securing
  container images and other artifacts.

## Monitoring & Observability

- `Grafana` - An open-source analytics and monitoring solution, used for visualizing metrics, logs,
  and alerts from various data sources.
- `Prometheus` - An open-source monitoring and alerting toolkit, designed for reliability and
  scalability in collecting and querying metrics.

## Security & Access Control

- `Workload Identity` - A GCP feature that enables Kubernetes service accounts to act as GCP service
  accounts, providing secure access to GCP resources.
- `External Secrets` - A Kubernetes operator that integrates external secret management systems like
  GCP Secret Manager with Kubernetes secrets.
- `IAP (Identity-Aware Proxy)` - A Google Cloud service that provides a central authentication layer
  to applications accessed by HTTPS. It enables:
  - Single sign-on (SSO) with Google accounts
  - Fine-grained access control to web applications
  - Protection of applications without the need for VPNs
  - Integration with Google Cloud's security features

## CI/CD & DevOps Tools

- `GitHub Actions` - A continuous integration and delivery (CI/CD) platform that automates build,
  test, and deployment pipelines. In our stack, it's used for:
  - Building and testing pipeline images
  - Deploying documentation
  - Managing releases
  - Automating pipeline execution
  - Infrastructure deployment via Terraform
- `Helm` - A package manager for Kubernetes that helps define, install, and upgrade even the most
  complex Kubernetes applications. Used to manage our Kubernetes application deployments in a
  templated and versioned way.

## Documentation & Diagramming Tools

- `MkDocs Material` - A modern documentation static site generator based on Python and Material
  Design. Features include:
  - Responsive design
  - Built-in search functionality
  - Code highlighting
  - Version control integration
  - Navigation and table of contents
- `Excalidraw` - A virtual collaborative whiteboarding tool that lets you easily create diagrams
  that look hand-drawn. Used for creating simple, sketch-like diagrams in our documentation.
- `Draw.io` - A diagramming tool integrated into our documentation workflow for creating and
  maintaining technical diagrams, architecture drawings, and flowcharts. Supports both web-based and
  desktop usage.
