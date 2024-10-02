---
title: "Deep dive: Evaluation pipeline"
---

# Deep dive: Evaluation pipeline

The evaluation pipeline is a crucial component of our drug repurposing system, designed to assess the performance and reliability of our predictive models. This pipeline employs a variety of metrics and techniques to provide a comprehensive understanding of how well our models are performing in identifying potential new drug-disease associations.

In this deep dive, we'll explore the different aspects of our evaluation pipeline, including:

1. The types of evaluation metrics we use
2. How we generate negative test pairs
3. Implementation details in Kedro

## Overview of the suite

The primary use-case of our drug repurposing models is to rank millions of drug-disease pairs in order of priority. More precisely, the set of drug disease pairs consists of all combinations of drugs and diseases from curated lists and is referred to as the matrix. A secondary but important use case for drug repurposing system is diseases-specific ranking, that is, ranking a list of drugs in order of likelihood of treating a given disease. Furthermore, in order to avoid failed clinical trials, it is critical that the model does not rank highly pairs which look like they may represent a viable treatment (hence may enter a clinical trial) but really do not (hence would fail the clinical trial). 

These different aspects of model performance are measured by three classes evaluation metrics which are computed in very different ways.

The following table summarises the evaluation metrics that comprise the current version of our evaluation suite. 


| Full-matrix ranking | Disease specific ranking | Ground truth classification|
|--------------|-------------|----------|
| How likely are positive drug-disease pairs to appear near the top of the matrix output? | How well does the model rank drugs for a specific disease? | How accurately can the model distinguish between known positive and negative drug-disease pairs? |
| Recall@n, AUROC | Hit@k, MRR | Accuracy, F1 score |
| Computed using the full matrix | Computed using diseases in the known positive test set  | Computed the set of known positive and negative pairs |

## Time-split validation and recent clinical trials data

Time-split validation is a technique where we divide our dataset based on a temporal cutoff, using older data for training and newer data for testing, to simulate real-world scenarios and assess our model's ability to predict future drug-disease associations.

We implement time-split validation in our pipeline by using an additional ground truth test set coming from the results of recent clinical trials. 

## The performance metrics in detail



