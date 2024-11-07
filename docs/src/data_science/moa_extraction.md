---
title: "MOA extraction pipeline"
---

# MOA extraction pipeline

Mechanism of action (MOA) prediction models aim to predict the biological pathways by which a given drug acts on a given disease. 
The MOA extraction module is an example of a path-based approach to MOA prediction. It extracts paths in the the knowledge graph that are likely to be relevant to the MOA of a given drug-disease pair.
The MOA extraction module is loosely inspired by the [KGML-xDTD](https://github.com/kgml-xdt/kgml-xdt) MOA module which extracts paths in the KG using an adversarial actor-critic reinforcement learning algorithm. 
In contrast, we use a simpler approach of extracting paths using a supervised binary classifier. 

## Methodology

The main component of the MOA extraction system is a binary classifier that predicts whether a given path in the knowledge graph is likely to represent a mechanism of action for a given drug-disease pair. This is illustrated in the following diagram:

![Path classifier architecture](../assets/img/MOA_extraction/path_classifier.svg)

An example of such a binary classifier is a transformer encoder model with a linear output layer. In this case, the input path needs to be embedded as a sequence of vectors. More details on the embedding process are provided in the [design choices section](#design-choices).

A different binary classifier is trained for each length of path. Currently our system supports 2 or 3 hop paths, so two binary classifiers are trained.

Now, fix the number of hops to be $n$. To use the binary classier to predict $n$-hop MOAs for a given drug-disease pair, we perform the following process:
1. Extract all $n$-hop paths between the drug and disease in the knowledge graph. 
2. Assign a score to each path using the binary classifier and rank accordingly.
3. Return the top-k paths as the predicted MOAs for the given drug-disease pair.

Furthermore, our system supports explicit filters to be applied to the paths extracted in step 1, for instance, remove any path containing a drug-disease edge. The prediction process is illustrated by the following diagram:

![Path classifier architecture](../assets/img/MOA_extraction/path_inference.svg)

## Overview 

## Design choices 