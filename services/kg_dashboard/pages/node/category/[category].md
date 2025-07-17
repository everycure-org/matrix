# {params.category}

<script context="module">
  import { getSeriesColors, sourceOrder } from '../../../_lib/colors';
  
  // Enhanced sortBySeries function that uses the color ordering
  export function sortBySeriesOrdered(data, seriesColumn) {
    // Use the existing sourceOrder from colors.js
    return data.sort((a, b) => {
      const aIndex = sourceOrder.indexOf(a[seriesColumn]);
      const bIndex = sourceOrder.indexOf(b[seriesColumn]);
      
      // Both are known sources
      if (aIndex !== -1 && bIndex !== -1) {
        return aIndex - bIndex;
      }
      
      // a is known, b is unknown - a comes first
      if (aIndex !== -1 && bIndex === -1) {
        return -1;
      }
      
      // a is unknown, b is known - b comes first
      if (aIndex === -1 && bIndex !== -1) {
        return 1;
      }
      
      // Both are unknown - sort alphabetically
      return a[seriesColumn].localeCompare(b[seriesColumn]);
    });
  }
</script>

```sql number_of_nodes
select coalesce(sum(count), 0) as count
from bq.merged_kg_nodes
where category = 'biolink:${params.category}'
```

```sql number_of_edges
select coalesce(sum(count), 0) as count
from bq.merged_kg_edges
where subject_category = 'biolink:${params.category}'
   or object_category = 'biolink:${params.category}'    
```

```sql nodes_by_prefix
  select 
      prefix,
      '/node/prefix/' || prefix as link,
      coalesce(sum(count), 0) as count
  from bq.merged_kg_nodes
  where category = 'biolink:${params.category}'
  group by all
  order by count desc  
```

```sql node_prefixes_by_upstream_data_source
select prefix, 
       upstream_data_source, 
       coalesce(sum(count), 0) as count 
from bq.merged_kg_nodes
where category = 'biolink:${params.category}'
group by all
having count > 0
order by count desc
limit 50
```

```sql edge_type_with_connected_prefix
select 
      replace(subject_category,'biolink:','') || ' ' ||
      replace(predicate,'biolink:','') || ' ' || 
      replace(object_category,'biolink:','') as edge_type,
      case when 'biolink:${params.category}' = subject_category 
           then object_prefix 
           else subject_prefix end as connected_prefix,
      sum(count) as count   
from bq.merged_kg_edges
where (subject_category = 'biolink:${params.category}'
   or object_category = 'biolink:${params.category}')
   and ('${inputs.pks_filter.value}' = 'All' or primary_knowledge_source = '${inputs.pks_filter.value}')
group by all
order by count desc
```

```sql primary_knowledge_source_counts
  select
      primary_knowledge_source,
      sum(count) as count
  from bq.merged_kg_edges
  where subject_category = 'biolink:${params.category}'
    or object_category = 'biolink:${params.category}'    
  group by all
  having count > 0
  order by count desc
```

{#if number_of_edges.length > 0 && number_of_nodes.length > 0}
<Grid col=2>
    <p class="text-center text-lg pt-4"><span class="font-semibold text-2xl"><Value data={number_of_edges} column="count" fmt="integer"/></span><br/>edges</p>
    <p class="text-center text-lg pt-4"><span class="font-semibold text-2xl"><Value data={number_of_nodes} column="count" fmt="integer"/></span><br/>nodes</p>
</Grid>
{/if}

{#if node_prefixes_by_upstream_data_source.length !== 0}
<BarChart 
    data={sortBySeriesOrdered(node_prefixes_by_upstream_data_source, 'upstream_data_source')}
    x=prefix
    y=count
    series=upstream_data_source
    seriesColors={getSeriesColors(node_prefixes_by_upstream_data_source, 'upstream_data_source')}
    swapXY=true    
    title="Prefix"
/>
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

{#if edge_type_with_connected_prefix.length !== 0}
<DataTable
    data={edge_type_with_connected_prefix}
    title="Edge Types Connected to {params.category} Nodes"
    groupBy=connected_prefix
    subtotals=true
    totalRow=true
    groupsOpen=false>
    
    <Column id="connected_prefix" title="Connected Prefix" />
    <Column id="edge_type" title="Edge Type" />
    <Column id="count" contentType="bar"/>
</DataTable>
{/if}