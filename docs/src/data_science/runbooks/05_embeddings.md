---
title: Node Embeddings Pipeline
---
<!-- NOTE: This file was partially generated using AI assistance.  -->

# Switching Between Embedding Models


Our pipeline supports multiple graph embedding models, including both Neo4j Graph Data Science (GDS) implementations and PyTorch Geometric implementations. You can easily switch between these models by modifying the `embeddings.topological_estimator` configuration in the `parameters.yml` file.

## Switching Models

To switch between models, simply uncomment the desired model configuration in the `parameters.yml` file and comment out the others. The available models are:

1. GDS Node2Vec
2. PyTorch Geometric Node2Vec
3. GDS GraphSAGE

### Example: Switching to PyTorch Geometric Node2Vec

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

### Example: Switching to GDS GraphSAGE

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

## Differences Between GDS and PyTorch Geometric Models

### Implementation

1. GDS Models:
   - Implemented using Neo4j's Graph Data Science library
   - Run directly on the Neo4j graph database
   - Optimized for large-scale graph processing

2. PyTorch Geometric Models:
   - Implemented using the PyTorch Geometric library
   - Run on PyTorch tensors, requiring graph data to be converted from Neo4j format
   - Offer more flexibility and customization options

### Data Handling

1. GDS Models:
   - Work directly with the Neo4j graph structure
   - No need for data conversion or transfer

2. PyTorch Geometric Models:
   - Require conversion of graph data from Neo4j to PyTorch tensors
   - Use the `prepare_graph_data()` function to convert Neo4j graph to edge_index format

### Training Process

1. GDS Models:
   - Training is handled internally by the GDS library
   - Limited control over the training loop

2. PyTorch Geometric Models:
   - Full control over the training loop
   - Custom training function (`train_model()`) implemented for more flexibility

### Embedding Storage

1. GDS Models:
   - Embeddings are typically stored directly in the Neo4j graph as node properties

2. PyTorch Geometric Models:
   - Embeddings are generated as PyTorch tensors
   - Require an additional step to write embeddings back to Neo4j (`write_embeddings()` function)

### Performance Considerations

- GDS Models are generally faster for large-scale graphs due to their native integration with Neo4j
- PyTorch Geometric Models offer more flexibility and may be preferred for custom implementations or when integrating with other PyTorch-based models

By understanding these differences, you can choose the most appropriate model for your specific use case and easily switch between them by modifying the configuration parameters.