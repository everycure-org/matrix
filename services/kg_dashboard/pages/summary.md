---
title: Summary
---

<!-- TODO: rename merged_kg_nodes sources as misleading  -->
```sql edges_per_node
select 
    all_edges / all_nodes as edges_per_node
    , edges_without_hyperconnected_nodes / nodes_without_hyperconnected_nodes as edges_per_node_without_hyperconnected_nodes
    , disease_edges / disease_nodes as disease_edges_per_node
    , drug_edges / drug_nodes as drug_edges_per_node
from 
    bq.overall_metrics
```

# Edge density
<br/>

<Grid col=2>
    <p class="text-center text-lg"><span class="font-semibold text-4xl"><Value data={edges_per_node} column="edges_per_node" /></span><br/>edges per node on average</p>
    <p class="text-center text-lg"><span class="font-semibold text-4xl"><Value data={edges_per_node} column="edges_per_node_without_hyperconnected_nodes" /></span><br/>edges per node excluding the top 1,000 most connected nodes</p>
</Grid>

# Disease list connections
<br/>

<p class="text-center text-lg"><span class="font-semibold text-4xl"><Value data={edges_per_node} column="disease_edges_per_node" /></span><br/>edges per disease node on average</p>

# Drug list connections
<br/>

<p class="text-center text-lg"><span class="font-semibold text-4xl"><Value data={edges_per_node} column="drug_edges_per_node" /></span><br/>edges per drug node on average</p>
