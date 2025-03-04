---
title: Matrix KG Dashboard
---

<script>
  const release_version = import.meta.env.VITE_release_version;
</script>

## Version: {release_version}

Dashboard pages on the left side of the screen are for exploring the data in the Matrix Knowledge Graph. Select categories from the dropdowns below to filter the knowledge graph visualization.

## Filter Knowledge Graph Categories

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
  title='Filtered Knowledge Graph Flow'
  subtitle='Flow from Selected Subject Categories through Selected Predicates to Selected Object Categories'
  chartAreaHeight={1400}
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

To see additional node breakdowns by category or prefix, check out the <a href="/node/explore">Node Explore</a> page.

<!-- NOTE: This file was partially generated using AI assistance. -->