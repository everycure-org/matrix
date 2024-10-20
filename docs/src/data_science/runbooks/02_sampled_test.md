---
title: Sample of the EveryCure's Knowledge Graph
---

## Background

The whole of EveryCure's Knowledge Graph (KG) contain more than
5 million nodes and 18 million edges. The `test` pipeline
generates fake data in order to test the mechanical process
of the default Kedro pipeline (such as calculating node
embeddings and building Machine Learning models using
those embeddings). However, using fake generated data
does a limited test as the real content of the KG is not
accounted for. For example, some content in the KG could
make the pipeline fail since it could be incompatible with
some of the functions in the pipeline. A common example is a
case that content in the KG is registered as `NULL` which
could make processing functions to fail.
  
Since the whole KG is far too large to use for testing,
we have created a Kedro pipeline called `create_sample` and
an evironment called `sample` to use as a representative
sub-set of the KG to make a more realistic test.

## Creating a sample

**NOTE** The creation of a KG sample is only allowed by the
Engineering Team.

To create a sample, the `create_sample` pipeline is used:  
  

    kedro run -p create_sample -e sample
  
  
This pipeline will calculate a representative sample of the
KG and store it in the GPC bucket (bucket:/kedro/staging/data/sampled_test).
Users can use this sample in any part of the pipeline by using the
environment `-e sample`.  
  
## Using a sample for testing the default pipeline

The first step for using the sample in the default pipeline for
testing is downloading it from the cloud bucket and storing
it locally. Is better to have the sample stored locally as if
you are developing and changing the pipeline and want to test
your changes with real data this step will avoid reading from
the cloud at each step.  
  
To download the sample the `ingestion` pipeline has to be called
with the `sample` environment:  
  
    kedro run -p ingestion -e sample
  
This will save the sample in your local space (specifically in
matrix/pipelines/matrix/data/sampled_test/02_intermediate).  
  

Once the sample is stored locally, you can run the default pipeline
with the sample by running:  
  
    kedro run -e sample
  
This will save all intermediate data in the data/sampled_test folder.






