---
title: Matrix KG Dashboard
---

Dashboard pages on the left side of the screen are for exploring the data in the Matrix Knowledge Graph. The dropdowns at the top of the page allow you to filter the data by KG Source, Subject Prefix, Predicate, and Object Prefix. The dropdown at the bottom of the page allows you to limit the number of results displayed.

```sql edges_for_sankey
  select 
      replace(subject_category,'biolink:','') as subject_category,      
      replace(object_category,'biolink:','') || ' ' as object_category,
      sum(count) as count
  from bq.merged_kg_edges
  group by all
  order by count desc
  limit 50
```

<SankeyDiagram data={edges_for_sankey} 
  sourceCol='subject_category'
  targetCol='object_category'
  valueCol='count'
  linkLabels='full'
  linkColor='gradient'
  title='Top 50 associations in the Matrix KG by subject category + object category count'
/>

