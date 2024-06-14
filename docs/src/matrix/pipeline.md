High level problem outline:

```mermaid
graph TB
    subgraph graph-version
        subgraph sources
            direction TB
            data-1(data_source<sub>1</sub>)
            data-2(data_source<sub>2</sub>)
            data-n(data_source<sub>n</sub>)
        end

        integration[integration]
    end

    subgraph product
        direction LR
        tabular[(tabular)]
        kg((graph))
    end

    subgraph embeddings-version
        embeddings[embeddings]

        subgraph auxiliary
            direction LR
            vectors[(vectors)]
        end
    end


    subgraph model-version
        modelling[modelling]

        subgraph models
            direction TB
            model-1(data_source<sub>1</sub>)
            model-2(data_source<sub>2</sub>)
            model-m(data_source<sub>n</sub>)
        end
    end
    
    data-1 --> integration
    data-2 --> integration
    data-n --> integration

    integration --> tabular
    integration --> kg

    tabular --> embeddings
    kg --> embeddings

    embeddings --> vectors

    kg --> modelling
    vectors --> modelling

    modelling --> model-1
    modelling --> model-2
    modelling --> model-m
```

```mermaid
graph TB
    subgraph legend
        direction TB
        datset(dataset)
        pipeline[pipeline]
        database[(database)]
        knowledge-graph((graph))
    end
```

We propose a versioned approach to these datasets, i.e.,

- Graph version
- Embeddings version
- Models version

This will yield a semantic versioning string of the following format:

```
v.<graph-version>.<embeddings-version>.<modeling-version>
```



