---
title: First Cluster Run 
---

!!! warning
    Note that this section is heavily focusing on the infrastructure which can be only applicable to the Matrix Project & Matrix GCP. Therefore, this section is useful and applicable if you can access our infrastructure.
    
    If you intend to adapt Matrix Codebase & Infrastructure to your own cloud system, these instructions might be also helpful to give you an idea how we utilize the cluster however they might not be 1:1 comparable. 

After completing first steps & first full run, you should be able to:

- Understand and leverage our tech stack
- Understand how config and kedro parameters work well together
- Set up MATRIX environment on your machine 
- Set up Docker Containers on your machine 
- Run the Repurposing Pipeline with a fabricated, sampled, and real data

In this section, we will focus on running the same pipeline on the cluster i.e. within the `cloud` environment. We will discuss the technology that we use for parallelization and cloud computing, and we will guide you through how to run the pipeline on the cluster

[Cluster Set-up :material-skip-next:](./cluster_setup.md){ .md-button .md-button--primary }