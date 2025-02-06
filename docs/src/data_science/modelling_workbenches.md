!!! example "Proposal Document"
    This document is not yet describing completed work. Thus, it is meant as a proposal document and is open for discussion.

This documentation is intended for data scientists and model developers who want to focus
on building models using Every Cure's data, without needing to dive into the core data
integration and modeling pipelines.

If you prefer working with notebooks and specific datasets to build models, rather than
contributing to the main pipeline development, this guide will help you get started.

## Resources We Provide

We provide several key resources to support your model development:

### Data Access
- Access to our biomedical knowledge graph and associated datasets through BigQuery and direct file access via Google Cloud Storage
- Regular data updates and versioning through our release pipeline
- Documentation on data schemas and available fields

### Infrastructure
- Dedicated Vertex AI workbench instances with pre-configured development environments
- Ability to schedule the notebook on larger compute resources for model training and evaluation
- Cloud storage for model or data artifacts and experiment results
- A validation pipeline to systematically evaluate model predictions (currently manual process)

## Prerequisites and Expectations

To effectively utilize our workbenches, we expect:

### Technical Skills
- Working knowledge of:
  - Linux command line and basic system operations
  - Git version control
  - Python programming language and common data science libraries
- Basic familiarity with Google Cloud Platform (GCP) or time to self-learn using existing online resources

### Best Practices
- Thorough documentation of experiments, including:
  - Clear description of methodology and approach
  - Documentation of both successful and failed experiments to build institutional knowledge
  - Regular updates on progress and findings
  - Code comments and inline documentation

!!! note
    We are developing a more structured framework for experiment tracking, result reporting, and knowledge sharing. This will be implemented in future updates to streamline the experimentation process.

## Getting started

### Getting a workbench

<!-- TODO check the link after merging -->
1. Create a pull request to [this
   file](https://github.com/everycure-org/matrix/edit/infra/infra/deployments/wg2/workbenches.tf),
   adding your name and email to the list of users for which we create a workbench. Please
   create the PR _to the infra branch_ and not main.
2. After the PR was merged, navigate to [this
   page](https://console.cloud.google.com/vertex-ai/workbench/instances?inv=1&invt=AboxFQ&project=mtrx-wg2-modeling-dev-9yj)
   which lists all workbenches we have provisioned for you and others.
3. Click "Open Jupyterlab" to open the workbench with your name on it. Note you may not
   be able to access others' workbenches as we will work towards restricting access to avoid
   any credentials being exposed.

This workbench comes pre-installed with the correct version of java, python, uv and anything else you may need.

## Accessing the data 

You have several options:

1. Use the Google BigQuery integration in the workbench as [documented by Google here](https://cloud.google.com/vertex-ai/docs/workbench/instances/bigquery)
2. Pull the matrix repo and use the kedro catalog


### Using Google native tooling

You can dive into all the data releases [on BigQuery](https://console.cloud.google.com/bigquery?inv=1&invt=AboxrQ&project=mtrx-hub-dev-3of&ws=!1m0). Alternatively you can navigate to [our release history page](https://docs.dev.everycure.org/releases/release_history/) and select a release you are interested in and get the BigQuery URL from there. 

Next you can simply create a cell in the notebook and run

```python
%%bigquery nodes_upstream_sources_count
SELECT upstream_data_source, count(*) as count FROM `mtrx-hub-dev-3of.release_v0_3_0.nodes` GROUP BY upstream_data_source
```
and the result of the query will be stored in a pandas dataframe that you can use in the rest of your notebook.

!!! warning
    Our data is very large and the BigQuery queries can take a while to run. Also pandas is not the greatest tool for the job when it comes to big data. See below for alternatives to get the full data.


!!! tip
    Feel free to change the release version to see the data for different releases.

    If you want to see the list of all releases easily, you can easily see them in [BigQuery](https://console.cloud.google.com/bigquery?inv=1&invt=Abo2vQ&project=mtrx-hub-dev-3of&ws=!1m0).
    
![](../assets/img/bq_datasets.png)


### Using the matrix repo

We already prepared the repo and dependencies at the start of the workbench instance. However, you may have to update the dependencies to the latest version in the coming months as we continue to progress with the development.

1. :white_check_mark: clone repo
2. :white_check_mark: cd to `matrix/pipelines/matrix` and follow the [installation instructions](../onboarding/local-setup.md)
3. Open the notebook `example_notebook.ipynb` and follow the instructions in the notebook.

## How it works behind the scenes

The workbench infrastructure is managed through Terraform in the `infra/deployments/wg2` directory. Here's what's set up:

### Core Components

1. **Vertex AI Workbenches**: Individual workbenches are provisioned for each data scientist through a Terraform module. Each workbench:
   - Has its own service account for authentication
   - Is connected to the shared VPC network
   - Runs a post-startup script to configure the environment, install dependencies and clone the matrix repo

2. **Idle Instance Management**: To help manage costs, we've implemented an automated system to detect and notify about idle instances:

   - A PubSub topic `ds-workbench-machine-state-events` receives monitoring alerts
   - A Cloud Monitoring alert policy watches for low CPU utilization:
     - Triggers when CPU usage is below 10% for 3 hours
     - Sends notifications through a PubSub notification channel
   - A Cloud Function processes these alerts and may take additional actions such as shutting down or deleting unused instances
     <!-- - shuts down idle instances -->
     <!-- - alerts about the deletion of instances after 30 days of inactivity -->
     <!-- - deletes the instances (including their disk storage) after 35 days of inactivity -->

### Alert Flow

1. Cloud Monitoring continuously checks workbench VM instances for CPU utilization
2. If a VM's CPU stays below a defined threshold (currently 10% for 3 hours), an alert is triggered
3. The alert is sent to a PubSub topic via the notification channel
4. A serverless Cloud Function receives these messages and logs them for tracking

The same as above is done but for testing whether an instance has been used at all in order to create deletion alerts.

### Infrastructure as Code

All components are defined in Terraform:

- Workbenches are created through a reusable module
- Monitoring and alerting policies are defined declaratively
- The notification system uses Cloud Pub/Sub for reliable message delivery
- A Python-based Cloud Function handles the alert processing

This setup ensures we can:

- Track idle instances to optimize resource usage
- Maintain consistent workbench configurations across the team
- Scale the infrastructure as the team grows
- Keep infrastructure changes version controlled and reviewable


## Open Work

We still have a few open pieces of work left:

- [AIP-97: Enable people to schedule notebooks on larger instance types and GPU backed machines while restricting such use for interactive notebooks](https://linear.app/everycure/issue/AIP-97/enable-people-to-schedule-notebooks-on-larger-instance-types-and-gpu)
- [AIP-138: Block conda from injecting itself into `.bashrc` all the time](https://linear.app/everycure/issue/AIP-138/block-conda-from-injecting-itself-into-bashrc-all-the-time)
- [AIP-118: Automate stopping and deletion of instances in Vertex](https://linear.app/everycure/issue/AIP-118/automate-stopping-and-deletion-of-instances-in-vertex)
- [AIP-96: Block non admins from changing instance types in vertex AI via IAM](https://linear.app/everycure/issue/AIP-96/block-non-admins-from-changing-instance-types-in-vertex-ai-via-iam)