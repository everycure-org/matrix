---
title: Merged KG Dashboard
---

<!-- <Details title='How to edit this page'>

  This page can be found in your project at `/pages/index.md`. Make a change to the markdown file and save it to see the change take effect in your browser.
</Details> -->

<!-- Node Queries -->

```sql node_categories_by_upstream_data_source
select category, upstream_data_source, sum(count) as count 
from bq.merged_kg_nodes
group by all
limit ${inputs.node_category_limit.value}
```
```sql node_prefix_by_upstream_data_source
select prefix, upstream_data_source, sum(count) as count
from bq.merged_kg_nodes
group by all
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
    and object_prefix in ${inputs.object_prefix.value}
    and upstream_data_source in ${inputs.upstream_data_source.value}
    and predicate in ${inputs.predicate.value}
group by all
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
    and upstream_data_source in ${inputs.upstream_data_source.value}
    and predicate in ${inputs.predicate.value}
  group by all
  order by count desc
  limit ${inputs.edge_limit.value}  
```

<Tabs>
    <Tab label="Nodes">
        
        <BarChart 
            data={node_categories_by_upstream_data_source}
            x=category
            y=count
            series=upstream_data_source
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
            data={node_prefix_by_upstream_data_source}
            x=prefix
            y=count
            series=upstream_data_source
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

        <div>
            <Dropdown data={edges}
                    name=subject_prefix
                    value=subject_prefix
                    title="Subject Prefix"
                    multiple=true
                    selectAllByDefault=true
                    />
            <Dropdown data={edges}
                    name=predicate
                    value=predicate
                    title="Predicate"
                    multiple=true
                    selectAllByDefault=true
                    />
            <Dropdown data={edges}
                    name=object_prefix
                    value=object_prefix
                    title="Object Prefix"
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
            data={predicates_by_upstream_data_source}
            x=predicate
            y=count
            series=upstream_data_source
            swapXY=true
            title="Predicates by Upstream Data Source"    
        />


        <BarChart 
            data={edge_types_by_upstream_data_source}
            x=edge_type
            y=count 
            series=upstream_data_source
            swapXY=true
            title="Edge Types by Upstream Data Source"
        />

    </Tab>
</Tabs>




