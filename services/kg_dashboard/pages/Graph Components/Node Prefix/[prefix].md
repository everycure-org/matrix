# {params.prefix}

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
select COALESCE(sum(count), 0) as count
from bq.merged_kg_nodes
where prefix = '${params.prefix}'
```

```sql number_of_edges
select COALESCE(sum(count), 0) as count
from bq.merged_kg_edges
where subject_prefix = '${params.prefix}'
   or object_prefix = '${params.prefix}'    
```

```sql nodes_by_category
  select 
      replace(category,'biolink:','') as category,
      '/Graph Components/Node Category/' || replace(category,'biolink:','') as link,
      sum(count) as count
  from bq.merged_kg_nodes
  where prefix = '${params.prefix}'
  group by all
  order by count desc  
```

```sql node_categories_by_upstream_data_source
select replace(category,'biolink:','') as category, 
       upstream_data_source, 
       sum(count) as count 
from bq.merged_kg_nodes
where prefix = '${params.prefix}'
group by all
having count > 0
order by count desc
limit 50
```

```sql edge_type_with_connected_category
select 
      replace(subject_category,'biolink:','') || ' ' ||
      replace(predicate,'biolink:','') || ' ' || 
      replace(object_category,'biolink:','') as edge_type,
      case when '${params.prefix}' = subject_prefix 
           then replace(object_category,'biolink:','') 
           else replace(subject_category,'biolink:','') end as connected_category,
      sum(count) as count   
from bq.merged_kg_edges
where (subject_prefix = '${params.prefix}'
   or object_prefix = '${params.prefix}')
   and ('${inputs.pks_filter.value}' = 'All' or primary_knowledge_source = '${inputs.pks_filter.value}')
group by all
order by count desc
```

```sql primary_knowledge_source_counts
  select
      primary_knowledge_source,
      sum(count) as count
  from bq.merged_kg_edges
  where subject_prefix = '${params.prefix}'
    or object_prefix = '${params.prefix}'
  group by all
  having count > 0
  order by count desc
```

<Grid col=2>
    <p class="text-center text-lg pt-4"><span class="font-semibold text-2xl"><Value data={number_of_nodes} column="count" fmt="integer"/></span><br/>nodes</p>
    <p class="text-center text-lg pt-4"><span class="font-semibold text-2xl"><Value data={number_of_edges} column="count" fmt="integer"/></span><br/>edges</p>
</Grid>

{#if node_categories_by_upstream_data_source.length !== 0}
<BarChart 
    data={sortBySeriesOrdered(node_categories_by_upstream_data_source, 'upstream_data_source')}
    x=category
    y=count
    series=upstream_data_source
    seriesColors={getSeriesColors(node_categories_by_upstream_data_source, 'upstream_data_source')}
    swapXY=true    
    title="Category"
/>
{/if}

{#if primary_knowledge_source_counts.length !== 0}
<BarChart
    data={primary_knowledge_source_counts}
    x=primary_knowledge_source
    y=count
    split=upstream_data_source
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
    title="Edge Types Connected to {params.prefix} Nodes"
    groupBy=connected_category
    subtotals=true
    totalRow=true
    groupsOpen=false>
    
    <Column id="connected_category" title="Connected Category" />
    <Column id="edge_type" title="Edge Type" />
    <Column id="count" contentType="bar"/>
</DataTable>
{/if}