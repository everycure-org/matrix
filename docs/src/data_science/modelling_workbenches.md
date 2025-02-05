- This doc is for those that do not want to dive fully into the
  development and contribution to the main Every Cure data integration and modelling pipelines
  but instead want to solely focus on building a good model based on our data. 
- I.e. if you yourself to be a model developer and you prefer working in notebooks and with specific datasets, this page is for you


## What we provide you
- Data
- Compute resources
- Storage
- [currently manual] A validation pipeline to test the model predictions 

## What we expect from you 

- basic knowledge of Linux, Git, Python.
- good practices on experimentation and documentation, especially when the experiment fails. We still want to learn from what did not work.
- basic understanding of cloud platforms like GCP or the ability to learn this or work with a colleague who is familiar with GCP

## Getting started

### Getting a workbench

<!-- TODO update the link after merging -->
1. Create a pull request to [this file](https://github.com/everycure-org/matrix/pull/1102/files#diff-e92de7fa0b983c12a1b3bd7e1d7ab6d13df3bb1df37b75f1e1d5d63a9e4d8b41), adding your name and email to the list of users for which we create a workbench
2. After the PR was merged, navigate to [this page](https://console.cloud.google.com/vertex-ai/workbench/instances?inv=1&invt=AboxFQ&project=mtrx-wg2-modeling-dev-9yj) which lists all workbenches we have provisioned for you and others.
3. Click "Open Jupyterlab" to open the workbench with your name on it. Note you may not be able to access others' workbenches as we will work towards restricting access to avoid any credentials being exposed.

This workbench comes pre-installed with the correct version of java, python, uv and anything else you may need.

### Accessing the data 

You have several options:

1. Use the Google BigQuery integration in the workbench as [documented by Google here](https://cloud.google.com/vertex-ai/docs/workbench/instances/bigquery)
2. Pull the matrix repo and use the kedro catalog


#### Using Google native tooling

You can dive into all the data releases [on BigQuery](https://console.cloud.google.com/bigquery?inv=1&invt=AboxrQ&project=mtrx-hub-dev-3of&ws=!1m0). Alternatively you can navigate to [our release history page](https://docs.dev.everycure.org/releases/release_history/) and select a release you are interested in and get the BigQuery URL from there. 

Next you can simply create a cell in the notebook and run

```python
%%bigquery upstream_sources
SELECT upstream_data_source, count(*) as count FROM `mtrx-hub-dev-3of.release_v0_3_0.nodes` GROUP BY upstream_data_source
```

and the result of the query will be stored in a pandas dataframe that you can use in the rest of your notebook.

!!! warning
    Our data is very large and the BigQuery queries can take a while to run. Also pandas is not the greatest tool for the job when it comes to big data. See below for alternatives to get the full data.

<!-- add explanation on how to get it into spark -->

![](../assets/bq_access.png)


#### Using the matrix repo

1. clone repo
2. cd to `matrix/pipelines/matrix` and follow the [installation instructions](../onboarding/local-setup.md)
3. create a notebook in this folder and paste the below code in the first cell

<!-- TODO add 1 cell trick on how to load kedro datasets as pyspark dataframes -->
```python

```