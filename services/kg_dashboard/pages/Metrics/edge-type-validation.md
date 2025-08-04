---
title: Edge Type Validation
---

<script>
  const positiveColor = '#73C991';
  const negativeColor = '#5E81AC';
</script>


```sql edge_validation
WITH edge_validation AS (
  SELECT 
    edges.subject_category,
    edges.predicate,
    edges.object_category,
    edges.count AS edge_count,
    CASE 
      WHEN valid_types.subject_category IS NOT NULL THEN 'Recognized'
      ELSE 'Unrecognized'
    END AS validation_status
  FROM bq.merged_kg_edges AS edges
  LEFT JOIN valid_edge_types.valid_edge_types AS valid_types
    ON edges.subject_category = valid_types.subject_category
    AND edges.predicate = valid_types.predicate  
    AND edges.object_category = valid_types.object_category
)
SELECT 
  validation_status,
  COUNT(*) AS edge_type_combinations,
  SUM(edge_count) AS total_edges
FROM edge_validation
GROUP BY validation_status
ORDER BY validation_status
```

## Overview

<Details title="About Edge Type Validation">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
  This page analyzes the conformance of edges in the knowledge graph to recognized biolink edge types defined in the 
  data model. By comparing the subject category, predicate, and object category combinations of edges against the 
  valid edge types catalog, we can assess schema compliance and identify unrecognized edge patterns.
</div>
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
  Edge types that don't match the predefined combinations are labeled as "unrecognized" rather than invalid, 
  as they may represent valid but undocumented relationship patterns or emerging use cases that haven't yet been 
  formally defined in the biolink model.
</div>
</Details>

<Grid col=2 class="max-w-4xl mx-auto mb-8">
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: {positiveColor}">
      <Value data={edge_validation.filter(d => d.validation_status === 'Recognized')} column="total_edges" fmt="num2m" />
    </span><br/>
    <span class="text-xl">Recognized Edges</span>
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: {negativeColor}">
      <Value data={edge_validation.filter(d => d.validation_status === 'Unrecognized')} column="total_edges" fmt="num2m" />
    </span><br/>
    <span class="text-xl">Unrecognized Edges</span>
  </div>
</Grid>

<ECharts 
  style={{ height: '400px' }}
  config={{
    title: { text: 'Edge Type Validation Overview', left: 'center' },
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)'
    },
    legend: {
      orient: 'vertical',
      left: 'left'
    },
    color: [positiveColor, negativeColor],
    series: [{
      name: 'Edge Validation',
      type: 'pie',
      radius: ['40%', '70%'],
      avoidLabelOverlap: false,
      label: {
        show: true,
        formatter: function(params) {
          const value = params.value;
          const formattedValue = value >= 1000000 ? (value / 1000000).toFixed(1) + 'M' : value.toLocaleString();
          return `${params.name}: ${formattedValue} (${params.percent}%)`;
        }
      },
      emphasis: {
        label: {
          show: true,
          fontSize: '18',
          fontWeight: 'bold'
        }
      },
      labelLine: {
        show: true
      },
      data: edge_validation.map(d => ({
        name: d.validation_status,
        value: d.total_edges
      }))
    }]
  }}
/>

## Edge Type Schema Coverage

```sql edge_type_schema_coverage
with validation_status AS (
  SELECT 
    edges.subject_category, 
    edges.predicate, 
    edges.object_category,
    CASE 
      WHEN valid_types.subject_category IS NOT NULL THEN 'Recognized'
      ELSE 'Unrecognized'
    END AS validation_status,
    COUNT(*) AS edge_count
  FROM bq.merged_kg_edges AS edges
  LEFT JOIN valid_edge_types.valid_edge_types AS valid_types
    ON edges.subject_category = valid_types.subject_category
    AND edges.predicate = valid_types.predicate  
    AND edges.object_category = valid_types.object_category
  GROUP BY ALL
  ORDER BY edge_count DESC)
SELECT validation_status, count(*) as count
from validation_status
group by validation_status
```

<Grid col=2 class="max-w-4xl mx-auto mb-8">
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: {positiveColor}">
      <Value data={edge_type_schema_coverage.filter(d => d.validation_status === 'Recognized')} column="count" fmt="num0" />
    </span><br/>
    <span class="text-xl">Recognized Edge Types</span>
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: {negativeColor}">
      <Value data={edge_type_schema_coverage.filter(d => d.validation_status === 'Unrecognized')} column="count" fmt="num0" />
    </span><br/>
    <span class="text-xl">Unrecognized Edge Types</span>
  </div>
</Grid>

<ECharts 
  style={{ height: '400px' }}
  config={{
    title: { text: 'Edge Type Schema Coverage', left: 'center' },
    tooltip: {
      trigger: 'item',
      formatter: function(params) {
        const value = params.value;
        const formattedValue = value >= 1000000 ? (value / 1000000).toFixed(1) + 'M' : value.toLocaleString();
        return `${params.name}: ${formattedValue} edge types (${params.percent}%)`;
      }
    },
    legend: {
      orient: 'vertical',
      left: 'left'
    },
    color: [positiveColor, negativeColor],
    series: [{
      name: 'Edge Type Coverage',
      type: 'pie',
      radius: ['40%', '70%'],
      avoidLabelOverlap: false,
      label: {
        show: true,
        formatter: function(params) {
          const value = params.value;
          const formattedValue = value >= 1000000 ? (value / 1000000).toFixed(1) + 'M' : value.toLocaleString();
          return `${params.name}: ${formattedValue} (${params.percent}%)`;
        }
      },
      emphasis: {
        label: {
          show: true,
          fontSize: '18',
          fontWeight: 'bold'
        }
      },
      labelLine: {
        show: true
      },
      data: edge_type_schema_coverage.map(d => ({
        name: d.validation_status,
        value: d.count
      }))
    }]
  }}
/>

## Unrecognized Edge Types Breakdown

```sql unrecognized_edge_types
SELECT 
  REPLACE(edges.subject_category, 'biolink:', '') AS subject_category,
  REPLACE(edges.predicate, 'biolink:', '') AS predicate,
  REPLACE(edges.object_category, 'biolink:', '') AS object_category,
  SUM(edges.count) AS edge_count
FROM bq.merged_kg_edges AS edges
LEFT JOIN valid_edge_types.valid_edge_types AS valid_types
  ON edges.subject_category = valid_types.subject_category
  AND edges.predicate = valid_types.predicate  
  AND edges.object_category = valid_types.object_category
WHERE valid_types.subject_category IS NULL
GROUP BY edges.subject_category, edges.predicate, edges.object_category
ORDER BY edge_count DESC
```


<Details title="Understanding Unrecognized Edge Types">
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
  This section provides a detailed breakdown of edge type combinations that are not recognized in the biolink schema, 
  ordered by the number of edges using each unrecognized pattern.
</div>
<div class="max-w-3xl mx-auto text-sm leading-snug text-gray-700 mb-4">
  Use the dimension grid below to filter and explore specific combinations of subject categories, predicates, 
  and object categories. The table will update to show only the edge types matching your selections.
</div>
</Details>

<DimensionGrid 
  data={unrecognized_edge_types} 
  metric='sum(edge_count)' 
  name=selected_edge_types
/>


```sql filtered_unrecognized_edge_types
WITH unrecognized_cleaned AS (
  SELECT 
    REPLACE(edges.subject_category, 'biolink:', '') AS subject_category,
    REPLACE(edges.predicate, 'biolink:', '') AS predicate,
    REPLACE(edges.object_category, 'biolink:', '') AS object_category,
    SUM(edges.count) AS edge_count
  FROM bq.merged_kg_edges AS edges
  LEFT JOIN valid_edge_types.valid_edge_types AS valid_types
    ON edges.subject_category = valid_types.subject_category
    AND edges.predicate = valid_types.predicate  
    AND edges.object_category = valid_types.object_category
  WHERE valid_types.subject_category IS NULL
  GROUP BY edges.subject_category, edges.predicate, edges.object_category
)
SELECT *
FROM unrecognized_cleaned
WHERE ${inputs.selected_edge_types}
ORDER BY edge_count DESC
```

<DataTable data={filtered_unrecognized_edge_types} rows=20>
  <Column id="subject_category" title="Subject Category" />
  <Column id="predicate" title="Predicate" />
  <Column id="object_category" title="Object Category" />
  <Column id="edge_count" title="Edge Count" contentType=bar fmt="num0" />
</DataTable>

