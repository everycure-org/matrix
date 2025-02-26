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

```sql failed_normalization
select original_prefix, 
       prefix || ' ' as prefix,
       normalization_set,      
       sum(count) as count
from bq.normalization
where normalization_success = false
  and normalization_set <> 'merged'
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
  title="Normalization Success"
  linkLabels='full'  
  linkColor='gradient' 
  chartAreaHeight={800}
/>

<BarChart 
  data={failed_normalization}
  x=prefix
  y=count
  series=normalization_set
  title="Normalization Failures"
  swapXY=true
/>