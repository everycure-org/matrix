---
title: Primary Knowledge Source
---

This page presents the primary knowledge sources ingested into the Matrix graph, offering an overview of the individual sources that compose the graph, along with key metrics and detailed insights for each source.

```sql knowledge_source_provenance
SELECT * FROM bq.primary_knowledge_source
```

<DataTable data={knowledge_source_provenance} />
