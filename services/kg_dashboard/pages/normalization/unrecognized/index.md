```sql edge_failed_normalization
select original_prefix, 
       normalization_set,
       '/normalization/unrecognized/' || normalization_set  || '/' || prefix as link,
       sum(count) as count
from bq.normalization
where normalization_success = false
  and normalization_set <> 'merged'
  and replace(category, 'biolink:', '') in ${inputs.category.value}
group by all
order by count desc
```

```sql normalization_categories
select distinct replace(category,'biolink:','') as category
from bq.normalization
where normalization_set = 'merged'

```

```sql normalization_datasets
select 
    normalization_set,
    '/normalization/unrecognized/' || normalization_set as link, 
    sum(count) as count
from bq.normalization
where normalization_set <> 'merged'
  and normalization_success = false
group by all
order by count desc
```


Unrecognized normalizations are counted when a subject or object column is marked with normalization_success `false`. For each prefix, examples of failed normalization are captured and can be accessed by clicking on the Examples link. 

<DataTable data={normalization_datasets}
  title="Normalization By Upstream Source"        
  link=link
  rows=20
>
    <Column id="normalization_set" title="Upstream Data Source" />
    <Column id="count" contentType="bar"/>
</DataTable>



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


<DataTable data={edge_failed_normalization}
    title="All Unrecognized Normalizations"
    search=true
    pagination=true
    link=true
    rows=20
>
    <Column id="normalization_set" title="Upstream Data Source" />
    <Column id="original_prefix" />
    <Column id="count" contentType="bar"/>
    <Column id="link" contentType="link" linkLabel="Examples ->" title=" " />
</DataTable>