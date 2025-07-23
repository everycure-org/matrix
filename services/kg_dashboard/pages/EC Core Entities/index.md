---
title: EC Core Entities
---

<script context="module">
  import { getSourceColor } from '../_lib/colors';
</script>

<script>
  // NOTE: This function was partially generated using AI assistance.
  function createHistogramBins(data, binWidth) {
    if (!data || !Array.isArray(data) || data.length === 0) return [];
    
    // Extract the column values
    const values = data.filter(v => v !== null && v !== undefined);
    if (values.length === 0) return [];
    
    // Calculate min and max if not provided
    const min = 0;
    const max = Math.max(...values);
    
    // Calculate number of bins based on width
    const binCount = Math.ceil((max - min) / binWidth);
    const bins = [];
    
    for (let i = 0; i < binCount; i++) {
      const binStart = min + (i * binWidth);
      const binEnd = min + ((i + 1) * binWidth);
      
      const count = values.filter(value => 
        value >= binStart && (i === binCount - 1 ? value <= binEnd : value < binEnd)
      ).length;
      
      bins.push({
        count: count,
        start: binStart,
        end: binEnd
      });
    }
    
    return bins;
  }

  function getHistogramEchartsOptions(data, data_name, data_key, binWidth, color, title) {
    const bins = !data || !Array.isArray(data) || data.length === 0 ? [] : createHistogramBins(data.map(d => d[data_key]), binWidth)
    const xAxis = bins.map(d => d.start)

    // Create series data with bin labels included
    const seriesData = bins.map(bin => ({
      value: bin.count,
      binStart: bin.start,
      binEnd: bin.end - 1,
    }))
    return {
      grid: {
        top: '15%',
        bottom: '18%',
      },
      title: {
          text: title,
          left: 'left',
          top: '0',
          textStyle: {
            fontSize: 14,
            fontWeight: 'bold'
          }
        },
      xAxis: {
        data: xAxis,
        silent: false,
        splitLine: {
          show: false
        },
        splitArea: {
          show: false
        }
      },
      yAxis: {
        splitArea: {
          show: false
        }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        },
        formatter: function(params) {
          const binStart = params[0].data.binStart;
          const binEnd = params[0].data.binEnd;
          const count = params[0].value;
          return `${count} ${data_name}s have between ${binStart} and ${binEnd} neighbours`;
        }
     },
     toolbox: {
       feature: { restore: {} }
     },
     dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 2,
          minValueSpan: 50
        },
        {
          type: 'slider',
          start: 0,
          end: 2,
          minValueSpan: 50
        }
      ],
      series: [
        {
          type: 'bar',
          data: seriesData,
          itemStyle: {color: color}
        }
      ]
    }
  }
</script>


```sql edges_per_node
select 
    n_nodes
    , n_edges
    , n_edges / n_nodes as edges_per_node
    , n_edges_without_most_connected_nodes / n_nodes_without_most_connected_nodes as edges_per_node_without_most_connected_nodes
    , n_edges_from_disease_list / n_nodes_from_disease_list as disease_edges_per_node
    , n_edges_from_drug_list / n_nodes_from_drug_list as drug_edges_per_node
    , median_drug_node_degree
    , median_disease_node_degree
from 
    bq.overall_metrics
```


This page summarizes the connectivity of EC Core Entities by examining two key lists: the disease list and the drug list. 
  For each list, we report the mean and median number of direct neighbors (connections) per node, 
  providing a view into how densely linked these entities are within the broader knowledge graph. 
  Additional charts break down the typical categories connected to each node type, 
  helping highlight which biological or clinical concepts most frequently co-occur. 
  This analysis offers insight into the network structure surrounding diseases and drugs of interest, 
  supporting downstream interpretations such as enrichment patterns or pathway overlaps.

## Disease list connections

<br/>

<Grid col=2>
    <p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="disease_edges_per_node" fmt="num1"/></span><br/>mean neighbours per disease node</p>
    <p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="median_disease_node_degree" fmt="num0"/></span><br/>median neighbours per disease node</p>
</Grid>

<br/>


```sql disease_list_neighbour_counts
select 
  * 
from 
  bq.disease_list_neighbour_counts
```

<ECharts
    style={{ height: '400px' }}
    config={getHistogramEchartsOptions(disease_list_neighbour_counts, "disease", "unique_neighbours", 10, getSourceColor("disease_list"),
         "Disease nodes neighbours")}
/>

<br/>

## Drug list connections

<br/>

<Grid col=2>
    <p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="drug_edges_per_node" fmt="num1"/></span><br/>mean neighbours per drug node</p>
    <p class="text-center text-lg"><span class="font-semibold text-2xl"><Value data={edges_per_node} column="median_drug_node_degree" fmt="num0"/></span><br/>median neighbours per drug node</p>
</Grid>

<br/>


```sql drug_list_neighbour_counts
select 
  * 
from 
  bq.drug_list_neighbour_counts
```

<ECharts
    style={{ height: '500px' }}
    config={getHistogramEchartsOptions(drug_list_neighbour_counts, "drug", "unique_neighbours", 50, getSourceColor("drug_list"), 
    "Drug nodes neighbours")}
/>