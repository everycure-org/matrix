
### Embeddings

Our embeddings pipeline computes vectorized representations of the entities in the knowledge graph in two stages:

1. Node Attribute Embedding Computation - We use GenAI models (e.g. OpenAI's `text-embedding-3-small` embedding API, and domain specific models such as PubMedBERT), for efficient node embedding, leveraging batch processing to reduce runtime and integrating error handling for API limits. 
2. Topological Embedding Computation - We have implemented options for GraphSAGE, and Node2Vec algorithms for computation of topological embeddings. Dimensionality reduction (e.g. PCA) has been modularized to enable flexible experimentation. 

!!! info
    Our graph database, i.e., [Neo4J](https://neo4j.com/docs/graph-data-science/current/algorithms/) comes with out-of-the-box functionality to compute both node and topological embeddings in-situ. The Kedro pipeline orchestrates the computation of these. However, we aim to move away from this and compute the node embeddings in a separate sub-pipeline, leveraging open source libraries such as Ray and the Hugging Face transformer library to have more control and flexibility over our node embeddings (and to parallelize the computation with many GPUs).

