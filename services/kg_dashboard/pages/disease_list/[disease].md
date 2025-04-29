# {params.disease}

```sql disease
select * from bq.disease_list
where id = '${params.disease}'
```


```sql edge_type_with_connected_category
select
      replace(subject_category,'biolink:','') || ' ' ||
      replace(predicate,'biolink:','') || ' ' ||
      replace(object_category,'biolink:','') as edge_type,
      case when category = subject_category
           then replace(object_category,'biolink:','')
           else replace(subject_category,'biolink:','') end as connected_category,
      sum(count) as count
from bq.disease_list_edges
    join bq.disease_list on bq.disease_list_edges.id = bq.disease_list.id
where disease_list_edges.id = '${params.disease}'
  and ('${inputs.pks_filter.value}' = 'All' or primary_knowledge_source = '${inputs.pks_filter.value}')
group by all
having count > 0
order by count desc
```

```sql primary_knowledge_source_counts
  select
      primary_knowledge_source,
      sum(count) as count
  from bq.disease_list_edges
  where id = '${params.disease}'
  group by all
  having count > 0
  order by count desc
```


# <strong><Value data={disease} column="name" /></strong>
<br>
{#if disease.description}
<Value data={disease} column="definition" />
{/if}

{#if primary_knowledge_source_counts.length !== 0}
<BarChart
    data={primary_knowledge_source_counts}
    x=primary_knowledge_source
    y=count
    title="Edge Counts by Primary Knowledge Source"
/>
{/if}

{#if primary_knowledge_source_counts.length !== 0}
<div>
    <Dropdown 
        data={primary_knowledge_source_counts}
        name=pks_filter
        value=primary_knowledge_source
        title="Filter by Primary Knowledge Source"
        defaultValue="All">
        <DropdownOption value="All">All</DropdownOption>
    </Dropdown>
</div>
{/if}

{#if edge_type_with_connected_category.length !== 0}
<DataTable
    data={edge_type_with_connected_category}
    title="Edge Types Connected to {params.disease} Nodes"
    groupBy=connected_category
    subtotals=true
    totalRow=true
    groupsOpen=false>
    
    <Column id="connected_category" title="Connected Category" />
    <Column id="edge_type" title="Edge Type" />
    <Column id="count" contentType="bar"/>
</DataTable>
{/if}