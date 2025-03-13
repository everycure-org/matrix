```sql normalization_categories
select distinct replace(category,'biolink:','') as category
from bq.node_normalization
where normalization_set = 'merged'
```

```sql node_merged_normalization
select original_prefix, 
       prefix || ' ' as prefix, 
       replace(category,'biolink:','') as category, 
       sum(count) as count
from bq.node_normalization
where normalization_set = 'merged'
  and no_normalization_change = false
  and replace(category, 'biolink:', '') in ${inputs.category.value}
group by all
```

```sql node_failed_normalization
select original_prefix, 
       prefix || ' ' as prefix,
       normalization_set,
       '/node/prefix/failed/' || prefix as link,
       sum(count) as count
from bq.node_normalization
where normalization_success = false
  and normalization_set <> 'merged'
  and replace(category, 'biolink:', '') in ${inputs.category.value}
group by all
order by count desc
```

```sql edge_merged_normalization
select original_prefix, 
       prefix || ' ' as prefix, 
       replace(category,'biolink:','') as category, 
       sum(count) as count
from bq.edge_normalization
where normalization_set = 'merged'
  and no_normalization_change = false
  and replace(category, 'biolink:', '') in ${inputs.category.value}
group by all
```

```sql normalization_datasets
select distinct normalization_set
from bq.node_normalization
where normalization_set <> 'merged'
```

```sql edge_failed_normalization
select original_prefix, 
       normalization_set,
       '/node/prefix/failed/' || prefix as link,
       sum(count) as count
from bq.edge_normalization
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
  <Tab label="Edges">

    The edge normalization statistics count distinct normalizations (any individual ID being normalized is counted once) from subject and object columns on associations. For a normalization to count, the ID needs to have been updated, and it needs to have been marked as a normalization success. 

    <SankeyDiagram data={edge_merged_normalization} 
        sourceCol="original_prefix" 
        targetCol="prefix" 
        valueCol="count" 
        title="Normalization Success"
        linkLabels='full'  
        linkColor='gradient' 
        chartAreaHeight={800}
    />

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
  <Tab label="Nodes">

    The node normalization statistics count distinct normalizations on node datasets. This means than the ID of a node record is updated via normalization to a new ID. 

    <SankeyDiagram data={node_merged_normalization} 
      sourceCol="original_prefix" 
      targetCol="prefix" 
      valueCol="count" 
      title="Normalization Success"
      linkLabels='full'  
      linkColor='gradient' 
      chartAreaHeight={800}
    />
    <DataTable data={node_failed_normalization}
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


