---
title: Knowledge Graphs
---

This page provides an overview of the upstream knowledge graphs integrated into MATRIX. Each knowledge graph contributes nodes and edges that are normalized and merged into our unified knowledge graph.

```sql kg_summary
select
  knowledge_graph,
  display_name,
  '/Knowledge Graphs/' || knowledge_graph as link,
  node_count,
  edge_count,
  normalization_success_rate,
  provenance_rate,
  knowledge_assertion_rate,
  manual_curation_rate
from bq.kg_summary
order by edge_count desc
```

```sql totals
select
  sum(node_count) as total_nodes,
  sum(edge_count) as total_edges,
  count(*) as kg_count
from bq.kg_summary
```

## Overview

<Grid cols=3>
  <BigValue data={totals} value="kg_count" title="Knowledge Graphs" />
  <BigValue data={totals} value="total_nodes" title="Total Nodes" fmt="num0" />
  <BigValue data={totals} value="total_edges" title="Total Edges" fmt="num0" />
</Grid>

## Knowledge Graph Comparison

Click on a knowledge graph name to view detailed metrics including normalization pipeline, quality profile, and contribution summary.

<DataTable data={kg_summary} link=link search=true>
  <Column id="display_name" title="Knowledge Graph" />
  <Column id="node_count" title="Nodes" fmt="num0" contentType="bar" barColor="#93c5fd" backgroundColor="#e5e7eb" />
  <Column id="edge_count" title="Edges" fmt="num0" contentType="bar" barColor="#86efac" backgroundColor="#e5e7eb" />
  <Column id="normalization_success_rate" title="Normalization Rate" fmt="pct1" contentType="bar" barColor="#fcd34d" backgroundColor="#e5e7eb" />
  <Column id="provenance_rate" title="Provenance Rate" fmt="pct1" />
</DataTable>
