---
title: Node Embeddings Pipeline
---
<!-- NOTE: This file was partially generated using AI assistance.  -->

# Generating Knowledge Graph Embeddings

In order to develop predictive models or thoroughly analyse our KG, we need to represent it in a numerical format. For this purpose we have embedding pipeline which at the moment supports two libraries for topological embedding calculation -  Neo4j Graph Data Science (GDS) and PyTorch Geometric (PyG) libraries. At the moment, we have the following models available

* GraphSAGE (GDS)
* Node2Vec (GDS)
* Node2Vec (Pyg)
* Metapath2Vec (PyG) (in-development)

## GDS vs PyG

Before selecting an embedding model for computations, it is important to understand the differences between GDS and PyG models.

GDS Models:
- Work directly with the Neo4j graph structure
- Training is handled internally by the GDS library but there is limited control over the training
- The model architecutre is fixed - we have no way of customizing it other than changing hyperparameters
- No need for data conversion or transfer
- Once embeddings are calculated, they are directly stored in Neo4j graph as node properties

PyTorch Geometric Models:
- Require conversion of graph data from Neo4j to PyTorch tensors
- We have full control over the training loop and can customize training functionalities (such as scheduler etc)
- Utilize `torch_utils` library to convert Neo4j graph to PyG friendly formatedge_index format
- We can generate custom architectures using Pytorch-based models
- Embeddings are generated as PyTorch tensors and need additional steps to writing embeddings back to Neo4j

## Selecting a specific Model
To select a specific model, you will need to specify its class within the `embeddings.topological_estimator.object` parameter. This is where you can also specify other parameters such as dimensions of embeddings, learning rate etc (note that if you dont specify the parameters customly, default PyG/GDS parameters will be taken for this model).

#### PyTorch Geometric Node2Vec
Node2Vec is a skip-gram based method (similar to word2vec) which approximates node embeddings based on the biased random walks. It **does not require attributes in the numerical format** thus if possible, you should skip the node attribute encoding step (usually done with PubMedBERT or OpenAI) if possible. 

Below you can find the parameters required to calculate Node2Vec embeddings. Note that at the moment PyG Node2Vec is not utilizing any special schedulers/optimizers/generalization techniques however these can be customly implemented within `graph_algorithms.py` and `torch_utils.py`. 

```yaml
embeddings.topological_estimator:
  object: matrix.pipelines.embeddings.graph_algorithms.PygNode2Vec
  embedding_dim: 512
  walk_length: 30
  walks_per_node: 10
  q: 1.0
  p: 1.0
  context_size: 10
  concurrency: 4
  num_negative_samples: 1
  num_workers: 0
  epochs: 10
  batch_size: 128
  random_seed: 42
  learning_rate: 0.01
  optimizer: Adam
  sparse: False
```

### GDS GraphSAGE
Node2Vec is a skip-gram based method (similar to word2vec) which approximates node embeddings based on the biased random walks. It **does not require attributes in the numerical format** thus if possible, you should skip the node attribute encoding step (usually done with PubMedBERT or OpenAI) if possible. 

Below are parameters needed to use GDS Node2Vec.
```yaml
embeddings.topological_estimator:
object: matrix.pipelines.embeddings.graph_algorithms.GDSGraphSage
concurrency: 4
iterations: 100
sample_sizes: [25, 10]
tolerance: 1e-8
embedding_dim: 512
batch_size: 5000
epochs: 10
search_depth: 100
learning_rate: 0.01
activation_function: ReLu
random_seed: 42
feature_properties: [_pca_property]
```

### Custom PyG model
Similar to PyTorch, PyTorch Geometric provides one with all building blocks to define their own NN architecture for graph encoding models. Therefore you should be able to define your own custom PyG model. To develop such model, ensure the model adjusts structurally to `GraphAlgorithm` class within `graph_algorithms.py` and that all building blocks have their own module (such as `torch_utils.py` or e.g. `model.py` for complex architectures)

## Node Attribute embedding calculation
Nodes' attributes in our KG contain useful information on nodes' properties which can be critical for predictive modelling. Some embedding models (e.g. GraphSAGE) require these attributes in a numerical format so that they can be utilized when calculating the embeddings. You can specify the model of interest for the embedding calculation within the `embeddings.ai_config` parameter. At the moment, three different LLMs can be utilized to calculate those encodings:

OpenAI models
```yaml
embeddings.ai_config:
  api_key: ${globals:openai.api_key} 
  batch_size: 200
  endpoint: ${globals:openai.endpoint}
  model: text-embedding-3-small
  attribute: &_property embedding
```

PubMedBERT models (note that these are very slow)
```yaml
embeddings.ai_config:
  api_key: ${globals:openai.api_key}
  batch_size: 200
  endpoint: ${globals:pubmedbert.endpoint}
  model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext #You can also choose "NeuML/pubmedbert-base-embeddings"
  attribute: &_property embedding
```

You can specify which attributes should be encoded here:
```yaml
git embeddings.node.features: ["category", "name"]
```