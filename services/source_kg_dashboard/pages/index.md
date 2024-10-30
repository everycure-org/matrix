---
title: Matrix KG Dashboard
---

<!-- <Details title='How to edit this page'>

  This page can be found in your project at `/pages/index.md`. Make a change to the markdown file and save it to see the change take effect in your browser.
</Details> -->

```sql edges
select * from matrix.edges
```

<Dropdown data={edges}
          name=kg_source
          value=kg_source
          title="KG Source"
          multiple=true
          defaultValue={['robokop_kg','rtx_kg2']}/>

<Dropdown data={edges}
          name=subject_prefix
          value=subject_prefix
          title="Subject Prefix"
          multiple=true
          defaultValue={['CHEBI']}/>

<Dropdown data={edges}
          name=predicate
          value=predicate
          title="Predicate"
          multiple=true
          defaultValue={['biolink:treats']}/>

<Dropdown data={edges}
          name=object_prefix
          value=object_prefix
          title="Object Prefix"
          multiple=true
          defaultValue={['MONDO']}/>

<Dropdown name=limit
          title="Limit">
    <DropdownOption value=10>10</DropdownOption>
    <DropdownOption value=20>20</DropdownOption>
    <DropdownOption value=50>50</DropdownOption>
    <DropdownOption value=100>100</DropdownOption>
    <DropdownOption value=500>500</DropdownOption>
</Dropdown>


```sql predicates_by_kg_source
select 
    predicate,
    kg_source,
    sum(count) as count
from matrix.edges
where kg_source in ${inputs.kg_source.value}
  and subject_prefix in ${inputs.subject_prefix.value}
  and object_prefix in ${inputs.object_prefix.value}
  and predicate in ${inputs.predicate.value}
group by all
limit ${inputs.limit.value}
```


<BarChart 
    data={predicates_by_kg_source}
    x=predicate
    y=count
    series=kg_source
    swapXY=true    
/>




```sql edges_by_kg_source
  select 
      subject_prefix || ' ' || predicate || ' ' || object_prefix as edge_type,
      kg_source,
      sum(count) as count
  from matrix.edges
  where subject_prefix in ${inputs.subject_prefix.value}
    and object_prefix in ${inputs.object_prefix.value}
    and kg_source in ${inputs.kg_source.value}
    and predicate in ${inputs.predicate.value}
  group by all
  order by count desc
  limit ${inputs.limit.value}  
```

<BarChart 
    data={edges_by_kg_source}
    x=edge_type
    y=count 
    series=kg_source
    swapXY=true    
/>

<!-- ## What's Next?
- [Connect your data sources](settings)
- Edit/add markdown files in the `pages` folder
- Deploy your project with [Evidence Cloud](https://evidence.dev/cloud)

## Get Support
- Message us on [Slack](https://slack.evidence.dev/)
- Read the [Docs](https://docs.evidence.dev/)
- Open an issue on [Github](https://github.com/evidence-dev/evidence) -->
