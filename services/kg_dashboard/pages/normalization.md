```sql merged_normalization_categories
select replace(category,'biolink:','') as category
from bq.normalization
where normalization_set = 'merged'
  and no_normalization_change = false
```

```sql merged_normalization
select original_prefix, 
       prefix || ' ' as prefix, 
       replace(category,'biolink:','') as category, 
       sum(count) as count
from bq.normalization
where normalization_set = 'merged'
  and no_normalization_change = false
  and replace(category, 'biolink:', '') in ${inputs.category.value}
group by all
```

<Dropdown
  data={merged_normalization_categories}
  name=category
  value=category
  label=category
  title="Filter Node Category"
  multiple=true
  selectAllByDefault=true
  description="Filter normalized node categories"
/>

<SankeyDiagram data={merged_normalization} 
  sourceCol="original_prefix" 
  targetCol="prefix" 
  valueCol="count" 
  title="Normalized Prefixes" 
/>

