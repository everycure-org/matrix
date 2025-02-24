---
title: Summary
---

<!-- TODO: rename merged_kg_nodes sources as misleading  -->
```sql edges_per_node
select 
    all_edges / all_nodes as edges_per_node
    , edges_without_hyperconnected_nodes / nodes_without_hyperconnected_nodes as edges_per_node_without_hyperconnected_nodes
from 
    bq.overall_metrics
```

<p>The nodes of the knowledge graph have <span style="font-weight: 600"><Value data={edges_per_node} column="edges_per_node" /></span> edges on average. If we remove the top 1,000 most connected nodes, we now obtain an average of <span style="font-weight: 600"><Value data={edges_per_node} column="edges_per_node_without_hyperconnected_nodes" /></span> edges per node.</p>

