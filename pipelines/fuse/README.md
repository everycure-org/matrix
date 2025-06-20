# Kedro pipeline Parallel execution

Matrix uses Kedro as it's main data pipelining framework, ensuring reproducable and re-usable data science workflows. However, to execute pipelines<sup>1</sup> in a distributed setting we're using Argo Workflows. This document aims to describe the process of executing a Kedro pipeline on Argo workflows.

<sup>1</sup> To complicate the matters even further, these pipelines contain nodes with varying hardware requirements such as GPUs.

## Kedro vs Argo Workflows

Before diving into the process, it's important to highlight the differences between Kedro and Argo. See the table below.

| Feature / Aspect            | **Kedro**                                                | **Argo Workflows**                                          |
|----------------------------|----------------------------------------------------------|-------------------------------------------------------------|
| **Purpose**                | Data and ML pipeline development                         | Container-native workflow orchestration on Kubernetes       |
| **Execution Environment**  | Local, on-prem, CI/CD, or deployed to Airflow/KubeFlow   | Kubernetes (must run on a Kubernetes cluster)               |
| **Pipeline Definition**    | Python-based, using `nodes` and `pipelines` abstraction  | YAML-based DAGs of container steps with dependencies        |
| **Best For**               | ML/data engineering teams building reproducible pipelines| DevOps teams running scalable, Kubernetes-native workflows  |


> 💡 Kedro is for development, while Argo is used for pipeline execution. Moreover, Kedro orchestrates nodes using their dataset dependencies, whereas Argo ochestrates tasks using task dependencies.
 