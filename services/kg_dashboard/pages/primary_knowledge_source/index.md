---
title: Primary Knowledge Source
---

This page presents the primary knowledge sources ingested into the Matrix graph, offering an overview of the individual sources that compose the graph, along with key metrics and detailed insights for each source.

```sql knowledge_source_provenance
SELECT primary_knowledge_source.*, catalog.name as name 
FROM bq.primary_knowledge_source
LEFT OUTER JOIN infores.catalog on infores.catalog.id = bq.primary_knowledge_source.source
```

<DataTable data={knowledge_source_provenance} />
