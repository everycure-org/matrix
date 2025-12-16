<script>
  /**
   * Source filter component for key nodes showing KG sources and primary knowledge sources
   * Displays two horizontal clickable bar charts with breadcrumb selection UI
   * Emits filter events to parent for coordinated filtering across visualizations
   */
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  // Component props
  export let sourceData = [];
  export let keyNodeId = '';

  // Selection state
  let selectedKGs = [];
  let selectedSources = [];

  // Filter and aggregate data for the current key node
  $: nodeSourceData = sourceData.filter(row => row.key_node_id === keyNodeId);

  // Aggregate by KG source
  $: kgData = nodeSourceData.reduce((acc, row) => {
    const kg = row.upstream_data_source;
    if (!acc[kg]) acc[kg] = 0;
    acc[kg] += row.edge_count;
    return acc;
  }, {});

  $: kgChartData = Object.entries(kgData).map(([kg, count]) => ({
    source: kg,
    edges: count
  })).sort((a, b) => b.edges - a.edges);

  // Aggregate by primary source (top 20 + Other)
  $: primaryData = nodeSourceData.reduce((acc, row) => {
    const source = row.primary_knowledge_source;
    if (!acc[source]) acc[source] = 0;
    acc[source] += row.edge_count;
    return acc;
  }, {});

  $: {
    const sorted = Object.entries(primaryData).sort((a, b) => b[1] - a[1]);
    const top20 = sorted.slice(0, 20);
    const remaining = sorted.slice(20);

    primaryChartData = top20.map(([source, count]) => ({
      source,
      edges: count
    }));

    if (remaining.length > 0) {
      const otherCount = remaining.reduce((sum, [, count]) => sum + count, 0);
      primaryChartData.push({
        source: 'Other sources',
        edges: otherCount
      });
    }
  }
  let primaryChartData = [];

  // Handle KG selection
  function toggleKG(kg) {
    if (selectedKGs.includes(kg)) {
      selectedKGs = selectedKGs.filter(k => k !== kg);
    } else {
      selectedKGs = [...selectedKGs, kg];
    }
    emitFilterChange();
  }

  // Handle primary source selection
  function toggleSource(source) {
    if (selectedSources.includes(source)) {
      selectedSources = selectedSources.filter(s => s !== source);
    } else {
      selectedSources = [...selectedSources, source];
    }
    emitFilterChange();
  }

  // Remove specific filter
  function removeKG(kg) {
    selectedKGs = selectedKGs.filter(k => k !== kg);
    emitFilterChange();
  }

  function removeSource(source) {
    selectedSources = selectedSources.filter(s => s !== source);
    emitFilterChange();
  }

  // Clear all filters
  function clearAll() {
    selectedKGs = [];
    selectedSources = [];
    emitFilterChange();
  }

  // Emit filter change event
  function emitFilterChange() {
    dispatch('filterChange', {
      selectedKGs,
      selectedSources
    });
  }
</script>

<div class="source-filters">
  <!-- KG Sources Bar -->
  <div class="chart-section">
    <h4 class="chart-title">Knowledge Graph Sources</h4>
    {#if kgChartData.length > 0}
      {@const totalEdges = kgChartData.reduce((sum, item) => sum + item.edges, 0)}
      <div class="stacked-bar-container">
        {#each kgChartData as item}
          <button
            class="stacked-segment"
            class:selected={selectedKGs.includes(item.source)}
            style="width: {(item.edges / totalEdges) * 100}%"
            on:click={() => toggleKG(item.source)}
            title="{item.source}: {item.edges.toLocaleString()} edges ({Math.round((item.edges / totalEdges) * 100)}%)"
          >
            <span class="segment-label">{item.source}</span>
          </button>
        {/each}
      </div>
      <div class="legend">
        {#each kgChartData as item}
          <button
            class="legend-item"
            class:selected={selectedKGs.includes(item.source)}
            on:click={() => toggleKG(item.source)}
          >
            <span class="legend-color" style="background-color: {selectedKGs.includes(item.source) ? '#2563eb' : '#3b82f6'}"></span>
            <span class="legend-label">{item.source}</span>
            <span class="legend-value">{item.edges.toLocaleString()}</span>
          </button>
        {/each}
      </div>
    {:else}
      <div class="no-data">No source data available</div>
    {/if}
  </div>

  <!-- Primary Knowledge Sources Bar -->
  <div class="chart-section">
    <h4 class="chart-title">Primary Knowledge Sources (Top 20)</h4>
    {#if primaryChartData.length > 0}
      {@const totalEdges = primaryChartData.reduce((sum, item) => sum + item.edges, 0)}
      <div class="stacked-bar-container">
        {#each primaryChartData as item}
          <button
            class="stacked-segment"
            class:selected={selectedSources.includes(item.source)}
            style="width: {(item.edges / totalEdges) * 100}%"
            on:click={() => toggleSource(item.source)}
            title="{item.source}: {item.edges.toLocaleString()} edges ({Math.round((item.edges / totalEdges) * 100)}%)"
          >
            <span class="segment-label">{item.source}</span>
          </button>
        {/each}
      </div>
      <div class="legend">
        {#each primaryChartData as item}
          <button
            class="legend-item"
            class:selected={selectedSources.includes(item.source)}
            on:click={() => toggleSource(item.source)}
          >
            <span class="legend-color" style="background-color: {selectedSources.includes(item.source) ? '#2563eb' : '#3b82f6'}"></span>
            <span class="legend-label">{item.source}</span>
            <span class="legend-value">{item.edges.toLocaleString()}</span>
          </button>
        {/each}
      </div>
    {:else}
      <div class="no-data">No source data available</div>
    {/if}
  </div>
</div>

<style>
  .source-filters {
    width: 100%;
    max-width: 1200px;
    margin: 2rem auto;
  }

  .selection-breadcrumbs {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin-top: 1rem;
    padding: 0.75rem;
    background-color: #f3f4f6;
    border-radius: 6px;
  }

  .breadcrumb-label {
    font-size: 0.875rem;
    font-weight: 600;
    color: #4b5563;
  }

  .breadcrumb-chip {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem 0.75rem;
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 9999px;
    font-size: 0.875rem;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .breadcrumb-chip:hover {
    background-color: #2563eb;
  }

  .chip-label {
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .chip-close {
    font-size: 1.25rem;
    font-weight: bold;
    line-height: 1;
  }

  .chart-section {
    margin-bottom: 2rem;
  }

  .chart-title {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: #1f2937;
  }

  .stacked-bar-container {
    display: flex;
    height: 40px;
    width: 100%;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .stacked-segment {
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #3b82f6;
    border: none;
    border-right: 1px solid white;
    cursor: pointer;
    transition: all 0.2s;
    padding: 0.5rem 0.25rem;
    position: relative;
  }

  .stacked-segment:last-child {
    border-right: none;
  }

  .stacked-segment:hover {
    background-color: #2563eb;
    transform: translateY(-2px);
    z-index: 1;
  }

  .stacked-segment.selected {
    background-color: #1e40af;
    box-shadow: inset 0 0 0 2px #fff;
  }

  .segment-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.375rem 0.75rem;
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .legend-item:hover {
    border-color: #3b82f6;
    background-color: #f0f9ff;
  }

  .legend-item.selected {
    border-color: #3b82f6;
    background-color: #dbeafe;
  }

  .legend-color {
    width: 12px;
    height: 12px;
    border-radius: 2px;
  }

  .legend-label {
    font-size: 0.875rem;
    color: #1f2937;
  }

  .legend-value {
    font-size: 0.875rem;
    font-weight: 600;
    color: #4b5563;
    margin-left: 0.25rem;
  }

  .no-data {
    text-align: center;
    padding: 2rem;
    color: #9ca3af;
    font-style: italic;
  }
</style>
