```sql normalization_categories
select distinct replace(category,'biolink:','') as category
from bq.normalization
where normalization_set = 'merged'
```



```sql edge_merged_normalization
select original_prefix, 
       prefix || ' ' as prefix,
       sum(count) as count
from bq.normalization
where normalization_set = 'merged'
  and no_normalization_change = false
  and replace(category, 'biolink:', '') in ${inputs.category.value}
group by all
```


<Dropdown
  data={normalization_categories}
  name=category
  value=category
  label=category
  title="Filter By Category"
  multiple=true
  selectAllByDefault=true
  description="Filter normalized node categories"
/>

The normalization statistics count distinct normalizations (any individual ID being normalized is counted once) from subject and object columns on associations. For a normalization to count, the ID needs to have been updated, and it needs to have been marked as a normalization success. 

<SankeyDiagram data={edge_merged_normalization} 
    sourceCol="original_prefix" 
    targetCol="prefix" 
    valueCol="count" 
    title="Normalization"
    linkLabels='full'  
    linkColor='gradient' 
    chartAreaHeight={300}
/>
