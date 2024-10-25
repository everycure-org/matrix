# Learnings from the path navigation experiments

There's various options to compute paths in the graph. Computing paths is a hard problem, mainly due to the fact that they tend to explode rather quickly given increasing depths.

## Enriching the graph with indexes

The goal of the problem is to generate potential MoA paths in a graph for a given table of proposed drug disease pairs (note that this is not the full drug/disease universe in our graph).

Conceptually, we're adding the following structure to the graph:

- A temporarily node label (and unique index) on the drug and disease nodes respectively, i.e., `is_drug`, `is_disease`
- A temporarily relationship between the drug-disease pairs for quick navigation

## Graph traversal approaches

### Direct path query

The most straight-forward approach is to leverage Neo4J for path generation, and thereafter leverage Spark to filter the paths in a distributed fashion.

```python
# TODO: Place further restrictions on intermediate labels
# NOTE: Feels like introducing additional constraints slows the query down
MATCH p=(drug:is_drug)-[*2..2]->(disease:is_disease)
RETURN [node IN nodes(p) | node.id] as path,
```

> Pros:
>   - Very straight forward and native 
> Cons:
>   - This generates all paths between a drug and any disease
>   - This might produce paths that visit the same nodes 

### Expand config

Neo4J provides an [ExpandConfig](https://neo4j.com/labs/apoc/4.1/overview/apoc.path/apoc.path.expandConfig/) procedure for path expansion between a given source and set of target nodes. The procedure allows for whitelisting specific node and edge labels. Alternatively, there is also an [Expand](https://neo4j.com/labs/apoc/4.1/overview/apoc.path/apoc.path.expand/)

```python
MATCH (source:is_drug), (source:is_drug)-[:is_treats]->(target:is_disease)
WITH id(source) as sourceId, collect(id(target)) as targets
CALL apoc.path.expandConfig(sourceId, {
    terminatorNodes: targets,
    relationshipFilter: '>',
    labelFilter: '-Drug-SmallMolecule/is_disease',
    minLevel: 2,
    maxLevel: 3,
    uniqueness: 'NODE_PATH'
})
YIELD path
WITH [n IN nodes(path) | n.id] as p
RETURN p
```

> Pros:
>   - Lots of different configuration options 
>   - Ensures deduplication of paths
> Cons:
>   - More complex to set up
>   - Slower?

### GDS

> Was currently unable to pull this off, idea would be to create a 
> relevant projection of the graph and run the BFS algortihm, however the output
> of the algortihm is not so suitable for path extraction.

```python
CALL gds.graph.project(
    'trav',
    '*', 
    '*'
)
YIELD graphName, nodeCount, relationshipCount;
```

```python
MATCH (drug:is_drug {id:'CHEMBL.COMPOUND:CHEMBL3989646'}), (disease: {id:'MONDO:0003781'}) 
WITH drug, COLLECT(disease) AS diseases
CALL gds.bfs.stream('trav', {
  sourceNode: drug,
  targetNodes: diseases,
  maxDepth: 2
})
YIELD path
RETURN path
LIMIT 1
```