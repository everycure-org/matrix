---
title: Key Nodes
---

This page displays the key disease and drug nodes that are being tracked for detailed analysis. Click on any node to view comprehensive connectivity information including descendant statistics.

```sql key_nodes_list
SELECT * FROM bq.key_nodes_list
```

## Tracked Nodes

<Details title="About Key Nodes">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
Key nodes are specific diseases and drugs selected for detailed tracking and analysis. Statistics include
not only direct connections to each key node, but also all connections to their descendants (subtypes,
subclasses).
</div>
</Details>

<DataTable
    data={key_nodes_list}
    search=true
    link=link
    title="Key Nodes Overview">

    <Column id="name" title="Name" />
    <Column id="id" title="ID" />
    <Column id="category" title="Category" />
    <Column id="total_edges" title="Direct Edges" contentType="bar" fmt="num0" />
    <Column id="unique_neighbors" title="Unique Neighbors" contentType="bar" fmt="num0" />
</DataTable>
