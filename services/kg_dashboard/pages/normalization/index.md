```sql normalization_categories
select distinct replace(category,'biolink:','') as category
from bq.normalization
where normalization_set = 'merged'
```



```sql edge_merged_normalization
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

```sql normalization_datasets
select 
    normalization_set,
    '/normalization/failures/' || normalization_set as link, 
    sum(count) as count
from bq.normalization
where normalization_set <> 'merged'
group by all
order by count desc
```

```sql edge_failed_normalization
select original_prefix, 
       normalization_set,
       '/normalization/failures/' || normalization_set  || '/' || prefix as link,
       sum(count) as count
from bq.normalization
where normalization_success = false
  and normalization_set <> 'merged'
  and replace(category, 'biolink:', '') in ${inputs.category.value}
group by all
order by count desc
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
<Tabs>
  <Tab label="Normalization Success">

    The normalization statistics count distinct normalizations (any individual ID being normalized is counted once) from subject and object columns on associations. For a normalization to count, the ID needs to have been updated, and it needs to have been marked as a normalization success. 

    <SankeyDiagram data={edge_merged_normalization} 
        sourceCol="original_prefix" 
        targetCol="prefix" 
        valueCol="count" 
        title="Normalization Success"
        linkLabels='full'  
        linkColor='gradient' 
        chartAreaHeight={800}
    />
  </Tab>
  <Tab label="Normalization Failures">

    Normalization failures are counted when a subject or object column is marked as having a failed 
    normalization. For each prefix, examples of failed normalization are captured and can be accessed by clicking on the Examples link. 

    <DataTable data={normalization_datasets}
      title="Normalization Datasets"        
      link=link
      rows=20
    >
        <Column id="normalization_set" title="Upstream Data Source" />
        <Column id="count" contentType="bar"/>
    </DataTable>

    <DataTable data={edge_failed_normalization}
      title="Normalization Failures"
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

  </Tab>
</Tabs>

<a href="/normalization/failures/rtx_kg2">RTX-KG2 Failed Normalization Examples</a> 
<a href="/normalization/failures/rtx_kg2/MONDO">RTX-KG2 Mondo Failed Normalization Examples</a>
<a href="/normalization/failures/robokop">Robokop Failed Normalization Examples</a> 
<a href="/normalization/failures/robokop/MONDO">Robokop Mondo Failed Normalization Examples</a>
