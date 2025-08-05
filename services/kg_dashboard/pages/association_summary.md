---
title: Association Summary
---
<p>
This Sankey diagram shows the flow of associations in the knowledge graph, from selected 
<strong>subject categories</strong> <Info description="Subject categories represent the types of entities that appear as the starting point (subject) in an association, such as drugs, genes, or diseases." />
through 
<strong>predicates</strong> <Info description="A predicate represents the type of relationship between two entities (nodes) in the knowledge graph." /> 
to 
<strong>object categories</strong> <Info description="Object categories are the types of entities that appear as the endpoint (object) in an association, such as phenotypes, pathways, or other genes." />.
The width of each flow represents the number of edges connecting the selected nodes and relationships.
</p>

```sql subject_categories
SELECT DISTINCT
    replace(subject_category,'biolink:','') as category,
    concat(replace(subject_category,'biolink:',''), ' (', sum(count), ' connections)') as label
FROM bq.merged_kg_edges
GROUP BY category
ORDER BY sum(count) DESC
```

```sql predicates
SELECT DISTINCT
    replace(predicate,'biolink:','') as predicate,
    concat(replace(predicate,'biolink:',''), ' (', sum(count), ' connections)') as label
FROM bq.merged_kg_edges
GROUP BY predicate
ORDER BY sum(count) DESC
```

```sql object_categories
SELECT DISTINCT
    replace(object_category,'biolink:','') as category,
    concat(replace(object_category,'biolink:',''), ' (', sum(count), ' connections)') as label
FROM bq.merged_kg_edges
GROUP BY category
ORDER BY sum(count) DESC
```

<script>
  //This constructs a dictionary which sets the depth of each subject category value,
  //predicate value, and object category value (as they're generated in combined_sankey
  //below) to 0, 1, and 2 respectively. The one gotcha here is that the '[S] ' and '[O] '
  //prefixes are added to the subject and object categories are added here as well as in
  //the SQL query below. 
  let depthOverrides = {}
  
  if (subject_categories && Array.isArray(subject_categories)) {    
    subject_categories.forEach(sc => {
      depthOverrides[('[S] ' + sc.category)] = 0;
    });    
  }

  if (predicates && Array.isArray(predicates)) {
    predicates.forEach(p => {
      depthOverrides[p.predicate] = 1;
    });
  }

  if (object_categories && Array.isArray(object_categories)) {
    object_categories.forEach(oc => {
      depthOverrides[('[O] ' + oc.category)] = 2;
    });
  }

</script>



## Filter Knowledge Graph Categories
Use the filters below to refine your view of associations in the Matrix Knowledge Graph. You can limit the visualization to specific subject categories, predicates, or object categories.

<Grid columns=3>

  <div>
    <Dropdown
      data={subject_categories}
      name=selected_subjects
      value=category
      label=label
      title="Filter Subject Categories"
      multiple=true
      selectAllByDefault=true
      description="Filter knowledge graph by subject categories"
    />
  </div>

  <div>
    <Dropdown
      data={predicates}
      name=selected_predicates
      value=predicate
      label=label
      title="Filter Predicates"
      multiple=true
      selectAllByDefault=true
      description="Filter knowledge graph by predicates"
    />
  </div>

  <div>
    <Dropdown
      data={object_categories}
      name=selected_objects
      value=category
      label=label
      title="Filter Object Categories"
      multiple=true
      selectAllByDefault=true
      description="Filter knowledge graph by object categories"
    />
  </div>
</Grid>

## Knowledge Graph Overview

```sql combined_sankey
-- First level: Subject Category to Predicate
SELECT 
    concat('[S] ', replace(subject_category,'biolink:','')) as source,
    replace(predicate,'biolink:','') as target,
    sum(count) as count
FROM bq.merged_kg_edges
WHERE replace(subject_category,'biolink:','') IN ${inputs.selected_subjects.value}
  AND replace(predicate,'biolink:','') IN ${inputs.selected_predicates.value}
GROUP BY all

UNION ALL

-- Second level: Predicate to Object Category
SELECT 
    replace(predicate,'biolink:','') as source,
    concat('[O] ', replace(object_category,'biolink:','')) as target,
    sum(count) as count
FROM bq.merged_kg_edges
WHERE replace(predicate,'biolink:','') IN ${inputs.selected_predicates.value}
  AND replace(object_category,'biolink:','') IN ${inputs.selected_objects.value}
GROUP BY all
ORDER BY count DESC
```

<SankeyDiagram data={combined_sankey} 
  sourceCol='source'
  targetCol='target'
  valueCol='count'
  linkLabels='full'
  linkColor='gradient'
  chartAreaHeight={1400}
  depthOverride={depthOverrides}
/>

```sql edge_stats
SELECT 
    replace(subject_category,'biolink:','') as subject_category,
    replace(predicate,'biolink:','') as predicate,
    replace(object_category,'biolink:','') as object_category,
    primary_knowledge_source,
    sum(count) as count
FROM bq.merged_kg_edges
WHERE replace(subject_category,'biolink:','') IN ${inputs.selected_subjects.value}
  AND replace(predicate,'biolink:','') IN ${inputs.selected_predicates.value}
  AND replace(object_category,'biolink:','') IN ${inputs.selected_objects.value}
GROUP BY all
ORDER BY count DESC
```

## Edge Statistics

<DataTable 
    data={edge_stats} 
    search=true
    pagination=true 
/>

Example parameterized dashboards:
 - <a href="/node/prefix/MONDO">Mondo Dashboard</a>
 - <a href="/node/category/Disease">Disease Dashboard</a>
 - <a href="/normalization">Normalization Dashboard</a>



<!-- NOTE: This file was partially generated using AI assistance. -->