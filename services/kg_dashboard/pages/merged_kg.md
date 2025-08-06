---
title: Merged KG Composition
---
<p class="text-md mt-2 mb-6">
  This dashboard provides an overview of the merged knowledge graph, integrating multiple upstream data sources.
  It highlights how different biological entities and relationships are represented across these sources.
</p>

<script context="module">
  import { getSeriesColors, sortDataBySource } from '../_lib/colors';
</script>

<!-- Node Queries -->

```sql node_categories_by_upstream_data_source
select category, upstream_data_source, sum(count) as count 
from bq.merged_kg_nodes
group by all
order by count desc
limit ${inputs.node_category_limit.value}
```
```sql node_prefix_by_upstream_data_source
select prefix, upstream_data_source, sum(count) as count
from bq.merged_kg_nodes
group by all
order by count desc
limit ${inputs.node_prefix_limit.value}
```

<!-- Edge Queries -->

```sql edges
select * from bq.merged_kg_edges
```

```sql predicates_by_upstream_data_source
select 
    predicate,
    upstream_data_source,
    sum(count) as count
from bq.merged_kg_edges
  where subject_prefix in ${inputs.subject_prefix.value}
    and subject_category in ${inputs.subject_category.value}
    and object_prefix in ${inputs.object_prefix.value}
    and object_category in ${inputs.object_category.value}
    and upstream_data_source in ${inputs.upstream_data_source.value}
    and predicate in ${inputs.predicate.value}
    and primary_knowledge_source in ${inputs.primary_knowledge_source.value}
group by all
order by count desc
limit ${inputs.edge_limit.value}
```

```sql edge_types_by_upstream_data_source
  select 
      replace(subject_category,'biolink:','') || ' ' ||
      replace(predicate,'biolink:','') || ' ' || 
      replace(object_category,'biolink:','') as edge_type,
      upstream_data_source,
      sum(count) as count
  from bq.merged_kg_edges
  where subject_prefix in ${inputs.subject_prefix.value}
    and object_prefix in ${inputs.object_prefix.value}
    and subject_category in ${inputs.subject_category.value}
    and object_category in ${inputs.object_category.value}
    and upstream_data_source in ${inputs.upstream_data_source.value}
    and predicate in ${inputs.predicate.value}
    and primary_knowledge_source in ${inputs.primary_knowledge_source.value}
  group by all
  order by count desc
  limit ${inputs.edge_limit.value}  
```

```sql edge_prefixes_by_primary_knowledge_source
select 
    subject_prefix || ' ' ||
    replace(predicate,'biolink:','') || ' ' ||
    object_prefix as edge_type,
    primary_knowledge_source,
    sum(count) as count
from bq.merged_kg_edges
    where subject_prefix in ${inputs.subject_prefix.value}
        and object_prefix in ${inputs.object_prefix.value}
        and upstream_data_source in ${inputs.upstream_data_source.value}
        and predicate in ${inputs.predicate.value}
        and primary_knowledge_source in ${inputs.primary_knowledge_source.value}
    group by all
    order by count desc    
    limit ${inputs.edge_limit.value}  
```

```sql primary_knowledge_source_by_predicate
select 
    predicate,
    primary_knowledge_source,
    sum(count) as count
from bq.merged_kg_edges
  where subject_prefix in ${inputs.subject_prefix.value}
    and object_prefix in ${inputs.object_prefix.value}
    and subject_category in ${inputs.subject_category.value}
    and object_category in ${inputs.object_category.value}
    and upstream_data_source in ${inputs.upstream_data_source.value}
    and predicate in ${inputs.predicate.value}
  group by all
  order by count desc  
  limit ${inputs.edge_limit.value}  
```

<Tabs>
    <Tab label="Nodes">
        <p class="text-sm mb-4">
          The charts below show how various biological categories and identifier prefixes are distributed across data sources in the graph.
          This helps assess the composition and provenance of nodes.
          See the <a class="underline text-blue-600" href="https://biolink.github.io/biolink-model/" target="_blank">Biolink Model documentation</a> 
          for more on  <a class="underline text-blue-600" href="https://biolink.github.io/biolink-model/#classes-visualization" target="_blank">categories</a> and 
          their associated Valid ID Prefixes.
        </p>
        <BarChart 
            data={sortDataBySource(node_categories_by_upstream_data_source, 'upstream_data_source')}
            x=category
            y=count
            series=upstream_data_source
            seriesColors={getSeriesColors(node_categories_by_upstream_data_source, 'upstream_data_source')}
            swapXY=true    
            title="Node Categories by Upstream Data Source"
        />
        <Dropdown name=node_category_limit
                  title="Limit">
            <DropdownOption value=10>10</DropdownOption>
            <DropdownOption value=20>20</DropdownOption>
            <DropdownOption value=50>50</DropdownOption>
        </Dropdown> 
        
        <BarChart 
            data={sortDataBySource(node_prefix_by_upstream_data_source, 'upstream_data_source')}
            x=prefix
            y=count
            series=upstream_data_source
            seriesColors={getSeriesColors(node_prefix_by_upstream_data_source, 'upstream_data_source')}
            swapXY=true
            title="Node Prefix by Upstream Data Source"
        />
        <Dropdown name=node_prefix_limit
                  title="Limit">
            <DropdownOption value=10>10</DropdownOption>
            <DropdownOption value=20>20</DropdownOption>
            <DropdownOption value=50>50</DropdownOption>
            <DropdownOption value=100>200</DropdownOption>
        </Dropdown> 
    </Tab>
    <Tab label="Edges"> 
        
        <p class="text-sm mb-4">
          These plots explore edge relationships in the graph, including the frequency of predicates, 
          biological edge types, and the knowledge sources contributing to each. 
          See the <a class="underline text-blue-600" href="https://biolink.github.io/biolink-model/" target="_blank">Biolink Model documentation</a> 
          for more on <a class="underline text-blue-600" href="https://biolink.github.io/biolink-model/#predicates-visualization" target="_blank">predicates</a> and 
          <a class="underline text-blue-600" href="https://biolink.github.io/biolink-model/#classes-visualization" target="_blank">categories</a>
        </p>

        <div>
            <Dropdown data={edges}
                    name=subject_prefix
                    value=subject_prefix
                    title="Subject Prefix"
                    multiple=true
                    selectAllByDefault=true
                    />
            <Dropdown data={edges}
                    name=subject_category
                    value=subject_category
                    title="Subject Category"
                    multiple=true
                    selectAllByDefault=true
                    />
            <br/>
            <Dropdown data={edges}
                    name=predicate
                    value=predicate
                    title="Predicate"
                    multiple=true
                    selectAllByDefault=true
                    />
            <br/>
            <Dropdown data={edges}
                    name=object_prefix
                    value=object_prefix
                    title="Object Prefix"
                    multiple=true
                    selectAllByDefault=true
                    />
            <Dropdown data={edges}
                    name=object_category
                    value=object_category
                    title="Object Category"
                    multiple=true
                    selectAllByDefault=true
                    />
            <br/>
            <Dropdown data={edges}
                    name=primary_knowledge_source
                    value=primary_knowledge_source
                    title="Primary Knowledge Source"
                    multiple=true
                    selectAllByDefault=true
                    />

        </div>
        <div>
            <Dropdown data={edges}
                    name=upstream_data_source
                    value=upstream_data_source
                    title="Upstream Data Source"
                    multiple=true
                    selectAllByDefault=true
                    />
        </div>
        <div>
            <Dropdown name=edge_limit
                    title="Limit">
                <DropdownOption value=10>10</DropdownOption>
                <DropdownOption value=20>20</DropdownOption>
                <DropdownOption value=50>50</DropdownOption>
                <DropdownOption value=100>100</DropdownOption>
                <DropdownOption value=500>500</DropdownOption>
            </Dropdown> 
        </div>

        <BarChart
            data={sortDataBySource(predicates_by_upstream_data_source, 'upstream_data_source')}
            x=predicate
            y=count
            series=upstream_data_source
            seriesColors={getSeriesColors(predicates_by_upstream_data_source, 'upstream_data_source')} 
            swapXY=true
            title="Predicates by Upstream Data Source"    
        />

        <BarChart 
            data={sortDataBySource(edge_types_by_upstream_data_source, 'upstream_data_source')}
            x=edge_type
            y=count 
            series=upstream_data_source
            seriesColors={getSeriesColors(edge_types_by_upstream_data_source, 'upstream_data_source')}
            swapXY=true
            title="Edge Types by Upstream Data Source"
        />

        <BarChart 
            data={primary_knowledge_source_by_predicate}
            x=primary_knowledge_source
            y=count
            series=predicate
            swapXY=true
            title="Primary Knowledge Source by Predicate"
        />

        <BarChart 
            data={edge_prefixes_by_primary_knowledge_source}
            x=edge_type
            y=count
            series=primary_knowledge_source
            swapXY=true
            title="Edge Prefixes by Primary Knowledge Source"
        />
    </Tab>
</Tabs>

