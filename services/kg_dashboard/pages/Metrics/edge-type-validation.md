---
title: Edge Type Validation
---

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

<!-- Explanatory header -->
<div class="text-left text-md max-w-3xl mx-auto mb-6">
  This page analyzes the conformance of edges in the knowledge graph to recognized biolink edge types defined in the 
  data model. By comparing the subject category, predicate, and object category combinations of edges against the 
  valid edge types catalog, we can assess schema compliance and identify unrecognized edge patterns.
</div>
<div class="text-left text-md max-w-3xl mx-auto mb-6">
  Edge types that don't match the predefined combinations are labeled as "unrecognized" rather than invalid, 
  as they may represent valid but undocumented relationship patterns or emerging use cases that haven't yet been 
  formally defined in the biolink model.
</div>

<Grid col=2 class="max-w-4xl mx-auto mb-8">
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: #1e40af;">
      <Value data={edge_validation.filter(d => d.validation_status === 'Recognized')} column="total_edges" fmt="num2m" />
    </span><br/>
    <span class="text-xl">Recognized Edges</span>
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: #93c5fd;">
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
    color: ['#1e40af', '#93c5fd'],
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

<div class="text-left text-lg font-semibold mt-10 mb-2 max-w-3xl mx-auto">
  Edge Type Schema Coverage
  <div class="text-sm font-normal mt-1 leading-snug">
    This view shows the number of distinct edge type combinations (subject category, predicate, object category) 
    that are recognized versus unrecognized in the schema.
  </div>
</div>

<Grid col=2 class="max-w-4xl mx-auto mb-8">
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: #1e40af;">
      <Value data={edge_type_schema_coverage.filter(d => d.validation_status === 'Recognized')} column="count" fmt="num0" />
    </span><br/>
    <span class="text-xl">Recognized Edge Types</span>
  </div>
  <div class="text-center text-lg">
    <span class="font-semibold text-4xl" style="color: #93c5fd;">
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
    color: ['#1e40af', '#93c5fd'],
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

``` sql edge_types
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

