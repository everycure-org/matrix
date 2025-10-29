<script>
  /**
   * Interactive chord diagram for visualizing key node connections to biolink categories
   *
   * Design decisions:
   * - Uses ECharts with direct initialization (not Evidence wrapper) to enable click event handling
   * - Oval layout reduces vertical space while maintaining readability
   * - Node sizes scaled by distinct_nodes, link widths scaled by total_edges
   * - Selection state managed in component, not via ECharts' built-in selection
   * - Drill-down table shows 10 example edges per primary_knowledge_source for diversity
   */
  import { onMount } from 'svelte';
  import * as echarts from 'echarts';
  import { DataTable, Column } from '@evidence-dev/core-components';
  import { LAYOUT_CONSTANTS, LABEL_CONFIG, SELECTION_STYLE } from '../_lib/key-node-chord/constants.js';
  import { calculateCategoryPositions, createCenterNode, createLinks, formatTooltip } from '../_lib/key-node-chord/chord-layout.js';
  import { getSourceColor, fallbackColors } from '../_lib/colors.js';

  // Component props
  export let categoryData = [];
  export let edgeData = [];
  export let keyNodeName = 'Key Node';
  export let keyNodeCategory = null;
  export let selectedKGs = [];
  export let selectedSources = [];
  export let sourceData = [];
  export let keyNodeId = '';

  // Selection state
  let selectedCategory = null;
  let chartInstance = null;

  // Tooltip state
  let tooltipVisible = false;
  let tooltipContent = '';
  let tooltipX = 0;
  let tooltipY = 0;

  function showTooltip(event, content) {
    tooltipContent = content;
    tooltipX = event.clientX;
    tooltipY = event.clientY;
    tooltipVisible = true;
  }

  function hideTooltip() {
    tooltipVisible = false;
  }

  function updateTooltipPosition(event) {
    if (tooltipVisible) {
      tooltipX = event.clientX;
      tooltipY = event.clientY;
    }
  }

  // Convert proxy to plain array if needed
  $: plainCategoryData = categoryData ? Array.from(categoryData) : [];
  $: plainEdgeData = edgeData ? Array.from(edgeData) : [];
  $: plainSourceData = sourceData ? Array.from(sourceData) : [];

  // SINGLE SOURCE OF TRUTH: Start with all edges, then apply all filters
  // This ensures chord, bar charts, and table all work from the same dataset

  // Step 1: Filter edges by source selections (KG and primary)
  $: sourceFilteredEdges = plainEdgeData.filter(edge => {
    // If no source filters selected, include all
    if (selectedKGs.length === 0 && selectedSources.length === 0) return true;

    // Check KG filter
    const kgMatch = selectedKGs.length === 0 || selectedKGs.includes(edge.upstream_data_source);

    // Check primary source filter (edge may have multiple sources comma-separated)
    const edgeSources = edge.primary_knowledge_sources ? edge.primary_knowledge_sources.split(', ') : [];
    const sourceMatch = selectedSources.length === 0 ||
                       selectedSources.some(s => edgeSources.includes(s));

    return kgMatch && sourceMatch;
  });

  // Step 2: Filter by category selection
  $: fullyFilteredEdges = selectedCategory
    ? sourceFilteredEdges.filter(edge => edge.parent_category === selectedCategory)
    : sourceFilteredEdges;

  // Filter sourceData (full aggregates) for bar charts
  // This ensures bar charts show accurate counts, not estimates from samples
  $: filteredSourceData = plainSourceData.filter(row => {
    if (row.key_node_id !== keyNodeId) return false;

    // Filter by category if selected
    if (selectedCategory && row.parent_category !== selectedCategory) return false;

    // Filter by KG if selected
    if (selectedKGs.length > 0 && !selectedKGs.includes(row.upstream_data_source)) return false;

    // Filter by primary source if selected
    if (selectedSources.length > 0 && !selectedSources.includes(row.primary_knowledge_source)) return false;

    return true;
  });

  // Compute KG source aggregates from full source data (for bar charts)
  $: kgData = filteredSourceData.reduce((acc, row) => {
    const kg = row.upstream_data_source;
    if (kg && kg !== '') {
      if (!acc[kg]) acc[kg] = 0;
      acc[kg] += row.edge_count;
    }
    return acc;
  }, {});

  $: kgChartData = Object.entries(kgData).map(([kg, count]) => ({
    source: kg,
    edges: count,
    color: getSourceColor(kg.toLowerCase())
  })).sort((a, b) => b.edges - a.edges);

  // Compute primary source aggregates from full source data (for bar charts)
  $: primaryData = filteredSourceData.reduce((acc, row) => {
    const source = row.primary_knowledge_source;
    if (source && source !== '') {
      if (!acc[source]) acc[source] = 0;
      acc[source] += row.edge_count;
    }
    return acc;
  }, {});

  // Generate deterministic color for primary sources
  function getPrimarySourceColor(source, index) {
    // Use getSourceColor first (in case it's a known KG source)
    const knownColor = getSourceColor(source.toLowerCase());
    if (knownColor !== fallbackColors[Math.abs(source.split('').reduce((acc, char) => ((acc << 5) - acc) + char.charCodeAt(0), 0)) % fallbackColors.length]) {
      return knownColor;
    }
    // Otherwise use fallback colors in order
    return fallbackColors[index % fallbackColors.length];
  }

  $: {
    const sorted = Object.entries(primaryData).sort((a, b) => b[1] - a[1]);
    const top20 = sorted.slice(0, 20);
    const remaining = sorted.slice(20);

    primaryChartData = top20.map(([source, count], index) => ({
      source,
      edges: count,
      color: getPrimarySourceColor(source, index)
    }));

    if (remaining.length > 0) {
      const otherCount = remaining.reduce((sum, [, count]) => sum + count, 0);
      primaryChartData.push({
        source: 'Other sources',
        edges: otherCount,
        color: '#9CA3AF' // gray for "Other"
      });
    }
  }
  let primaryChartData = [];

  // Calculate totals for stacked bars
  $: kgTotalEdges = kgChartData.reduce((sum, item) => sum + item.edges, 0);
  $: primaryTotalEdges = primaryChartData.reduce((sum, item) => sum + item.edges, 0);

  // Source filter handlers
  function toggleKG(kg) {
    if (selectedKGs.includes(kg)) {
      selectedKGs = selectedKGs.filter(k => k !== kg);
    } else {
      selectedKGs = [...selectedKGs, kg];
    }
  }

  function toggleSource(source) {
    if (selectedSources.includes(source)) {
      selectedSources = selectedSources.filter(s => s !== source);
    } else {
      selectedSources = [...selectedSources, source];
    }
  }

  function removeKG(kg) {
    selectedKGs = selectedKGs.filter(k => k !== kg);
  }

  function removeSource(source) {
    selectedSources = selectedSources.filter(s => s !== source);
  }

  function clearAllFilters() {
    selectedKGs = [];
    selectedSources = [];
    selectedCategory = null;
  }

  // Recalculate category data based on source-filtered edges (for chord diagram)
  // This makes the chord diagram reactive to source filters
  $: filteredCategoryData = (() => {
    if (selectedKGs.length === 0 && selectedSources.length === 0) {
      return plainCategoryData;
    }

    // Aggregate source-filtered edges by category
    const categoryCounts = sourceFilteredEdges.reduce((acc, edge) => {
      const cat = edge.parent_category;
      if (!acc[cat]) {
        acc[cat] = { edges: 0, nodes: new Set() };
      }
      acc[cat].edges += 1;
      acc[cat].nodes.add(edge.subject);
      acc[cat].nodes.add(edge.object);
      return acc;
    }, {});

    // Convert to category data format
    return Object.entries(categoryCounts).map(([category, data]) => ({
      connected_category: category,
      total_edges: data.edges,
      distinct_nodes: data.nodes.size
    }));
  })();

  // Reactive computations for the chart
  $: centerNode = createCenterNode(keyNodeName, keyNodeCategory);
  $: categoryNodes = calculateCategoryPositions(filteredCategoryData);
  $: links = createLinks(keyNodeName, categoryNodes);
  $: allNodes = [centerNode, ...categoryNodes];

  // Apply selection styling to nodes
  $: styledNodes = allNodes.map(node => {
    if (node.category === 'center') {
      return node; // Center node always fully visible
    }

    const isSelected = selectedCategory === node.name;
    const hasSelection = selectedCategory !== null;

    return {
      ...node,
      // Keep the symbolSize from the layout function (don't override)
      itemStyle: {
        ...node.itemStyle,
        borderWidth: isSelected ? SELECTION_STYLE.selectedBorderWidth : SELECTION_STYLE.normalBorderWidth,
        borderColor: SELECTION_STYLE.selectedBorderColor,
        opacity: hasSelection ? (isSelected ? SELECTION_STYLE.selectedOpacity : SELECTION_STYLE.unselectedOpacity) : 1.0
      },
      label: {
        show: true,
        position: 'right',
        fontSize: LABEL_CONFIG.fontSize,
        fontWeight: isSelected ? 'bold' : LABEL_CONFIG.fontWeight,
        color: LABEL_CONFIG.color,
        distance: LABEL_CONFIG.distance,
        opacity: hasSelection ? (isSelected ? 1.0 : 0.5) : 1.0
      }
    };
  });

  // Apply selection styling to links
  $: styledLinks = links.map(link => {
    const isSelected = selectedCategory === link.target;
    const hasSelection = selectedCategory !== null;

    return {
      ...link,
      lineStyle: {
        ...link.lineStyle,
        opacity: hasSelection ? (isSelected ? 0.8 : 0.1) : 0.6
      }
    };
  });

  // Initialize chart on mount
  onMount(() => {
    const chartDom = document.getElementById(`chord-chart-${keyNodeName.replace(/\s+/g, '-')}`);
    if (chartDom) {
      chartInstance = echarts.init(chartDom);

      // Add click event listener
      chartInstance.on('click', handleNodeClick);

      // Set initial option
      if (chartOption) {
        chartInstance.setOption(chartOption);
      }
    }

    return () => {
      if (chartInstance) {
        chartInstance.dispose();
      }
    };
  });

  // Handle node clicks
  function handleNodeClick(params) {
    if (params.dataType === 'node') {
      const clickedNode = params.data;

      if (clickedNode && clickedNode.category === 'category') {
        // Toggle selection
        if (selectedCategory === clickedNode.name) {
          selectedCategory = null;
        } else {
          selectedCategory = clickedNode.name;
        }
      } else if (clickedNode && clickedNode.category === 'center') {
        // Click on center node clears selection
        selectedCategory = null;
      }
    }
  }

  // Update chart when data or selection changes
  $: if (chartInstance && chartOption) {
    chartInstance.setOption(chartOption, true);
  }

  // ECharts configuration
  $: chartOption = {
    tooltip: {
      show: true,
      formatter: formatTooltip
    },
    xAxis: {
      show: false,
      type: 'value',
      min: 0,
      max: LAYOUT_CONSTANTS.CANVAS_WIDTH
    },
    yAxis: {
      show: false,
      type: 'value',
      min: 0,
      max: LAYOUT_CONSTANTS.CANVAS_HEIGHT
    },
    series: [{
      type: 'graph',
      layout: 'none',
      data: styledNodes,
      links: styledLinks,
      roam: false,
      draggable: false,
      label: {
        show: true
      },
      emphasis: {
        focus: 'adjacency',
        scale: 1.1,
        lineStyle: {
          width: 4
        }
      }
    }]
  };

  // Format edges for display in tables
  $: displayEdges = fullyFilteredEdges.map(edge => {
    // Format primary knowledge sources as links
    const sourcesWithLinks = edge.primary_knowledge_sources
      ? edge.primary_knowledge_sources
          .split(', ')
          .map(source => `<a href="../Knowledge Sources/infores:${source}" style="color: #2563eb; text-decoration: underline;">${source}</a>`)
          .join(', ')
      : '';

    return {
      ...edge,
      edge_triple: `${edge.subject_name || edge.subject} → ${edge.predicate} → ${edge.object_name || edge.object}`,
      knowledge_sources_links: sourcesWithLinks
    };
  });
</script>

<div class="key-node-chord-dashboard">
  <!-- Chord Visualization -->
  <div class="chord-container">
    {#if categoryData && categoryData.length > 0}
      <div id="chord-chart-{keyNodeName.replace(/\s+/g, '-')}" style="width: 100%; height: {LAYOUT_CONSTANTS.CANVAS_HEIGHT}px;"></div>
    {:else}
      <div>No category data available</div>
    {/if}
  </div>

  <!-- Knowledge Source Filters -->
  {#if kgChartData.length > 0 || primaryChartData.length > 0}
    <div class="source-filters">
      <!-- KG Sources -->
      {#if kgChartData.length > 0}
        <div class="filter-bar">
          <span class="filter-label">KG Sources</span>
          <div class="stacked-bar-container">
            {#each kgChartData as item}
              <button
                class="stacked-segment"
                class:selected={selectedKGs.includes(item.source)}
                style="width: {(item.edges / kgTotalEdges) * 100}%; background-color: {item.color};"
                on:click={() => toggleKG(item.source)}
                on:mouseenter={(e) => showTooltip(e, `${item.source}: ${item.edges.toLocaleString()} edges (${Math.round((item.edges / kgTotalEdges) * 100)}%)`)}
                on:mouseleave={hideTooltip}
                on:mousemove={updateTooltipPosition}
              >
                <span class="segment-label">{item.source}</span>
              </button>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Primary Knowledge Sources -->
      {#if primaryChartData.length > 0}
        <div class="filter-bar">
          <span class="filter-label">Primary Sources</span>
          <div class="stacked-bar-container">
            {#each primaryChartData as item}
              <button
                class="stacked-segment"
                class:selected={selectedSources.includes(item.source)}
                style="width: {(item.edges / primaryTotalEdges) * 100}%; background-color: {item.color};"
                on:click={() => toggleSource(item.source)}
                on:mouseenter={(e) => showTooltip(e, `${item.source}: ${item.edges.toLocaleString()} edges (${Math.round((item.edges / primaryTotalEdges) * 100)}%)`)}
                on:mouseleave={hideTooltip}
                on:mousemove={updateTooltipPosition}
              >
                <span class="segment-label">{item.source}</span>
              </button>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Active Filter Breadcrumbs -->
      {#if selectedCategory || selectedKGs.length > 0 || selectedSources.length > 0}
        <div class="selection-breadcrumbs">
          <span class="breadcrumb-label">Filtered by:</span>

          {#if selectedCategory}
            <button class="breadcrumb-chip category-chip" on:click={() => selectedCategory = null}>
              <span class="chip-label">Category: {selectedCategory}</span>
              <span class="chip-close">×</span>
            </button>
          {/if}

          {#each selectedKGs as kg}
            <button class="breadcrumb-chip kg-chip" on:click={() => removeKG(kg)}>
              <span class="chip-label">KG: {kg}</span>
              <span class="chip-close">×</span>
            </button>
          {/each}

          {#each selectedSources as source}
            <button class="breadcrumb-chip source-chip" on:click={() => removeSource(source)}>
              <span class="chip-label">Source: {source}</span>
              <span class="chip-close">×</span>
            </button>
          {/each}

          <button class="clear-all-button" on:click={clearAllFilters}>Clear all</button>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Filtered Edges Table -->
  {#if selectedCategory || selectedKGs.length > 0 || selectedSources.length > 0}
    <div class="drill-down-container">
      <h3 class="drill-down-title">
        {#if selectedCategory}
          Edges for: <span class="category-name">{selectedCategory}</span>
          <button class="clear-button" on:click={() => selectedCategory = null}>✕ Clear Category</button>
        {:else}
          Filtered Edges
        {/if}
      </h3>

      {#if displayEdges.length > 0}
        <div class="edge-count-info">
          Showing {displayEdges.length} example edge{displayEdges.length !== 1 ? 's' : ''}
          {#if selectedCategory && (selectedKGs.length > 0 || selectedSources.length > 0)}
            matching category and selected sources
          {:else if selectedCategory}
            (up to 10 per knowledge source)
          {:else}
            matching selected sources
          {/if}
        </div>

        {#key selectedCategory}
          <DataTable
            data={displayEdges}
            pagination=true
            pageSize={25}
            title="Connected Edges">

            {#if !selectedCategory}
              <Column id="parent_category" title="Category" />
            {/if}
            <Column id="edge_triple" title="Edge" wrap=true />
            <Column id="upstream_data_source" title="KG" />
            <Column id="knowledge_sources_links" title="Primary Knowledge Sources" wrap=true contentType="html" />

          </DataTable>
        {/key}
      {:else}
        <div class="no-data">
          No edges found matching the selected filters
        </div>
      {/if}
    </div>
  {:else}
    <div class="instruction-message">
      Click on a category node in the diagram above or filter by source to see detailed edge information
    </div>
  {/if}

  <!-- Custom Tooltip -->
  {#if tooltipVisible}
    <div
      class="custom-tooltip"
      style="left: {tooltipX + 10}px; top: {tooltipY + 10}px;"
    >
      {tooltipContent}
    </div>
  {/if}
</div>

<style>
  .key-node-chord-dashboard {
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
  }

  .chord-container {
    margin-bottom: 2rem;
  }

  .drill-down-container {
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 2rem;
  }

  .drill-down-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .category-name {
    color: #1e40af;
    font-weight: bold;
  }

  .clear-button {
    margin-left: auto;
    background-color: #ef4444;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .clear-button:hover {
    background-color: #dc2626;
  }

  .edge-count-info {
    font-size: 0.875rem;
    color: #6b7280;
    margin-bottom: 1rem;
    font-style: italic;
  }

  .instruction-message {
    text-align: center;
    padding: 3rem 1rem;
    color: #6b7280;
    font-size: 1rem;
    font-style: italic;
  }

  .no-data {
    text-align: center;
    padding: 2rem;
    color: #9ca3af;
    font-style: italic;
  }

  /* Source Filter Styles */
  .source-filters {
    margin: 1.5rem 0;
  }

  .filter-bar {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    margin-bottom: 1rem;
  }

  .filter-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.025em;
  }

  .stacked-bar-container {
    display: flex;
    height: 32px;
    flex: 1;
    border-radius: 4px;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
  }

  .stacked-segment {
    display: flex;
    align-items: center;
    justify-content: center;
    border: none;
    border-right: 1px solid rgba(255, 255, 255, 0.3);
    cursor: pointer;
    transition: all 0.2s;
    padding: 0.5rem 0.25rem;
    position: relative;
  }

  .stacked-segment:last-child {
    border-right: none;
  }

  .stacked-segment:hover {
    filter: brightness(0.85);
    transform: translateY(-2px);
    z-index: 1;
  }

  .stacked-segment.selected {
    box-shadow: inset 0 0 0 2px rgba(255, 255, 255, 0.5);
    filter: brightness(0.7);
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

  .selection-breadcrumbs {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.75rem;
    padding: 0.5rem 0.75rem;
    background-color: #fef3c7;
    border: 1px solid #fcd34d;
    border-radius: 4px;
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

  .breadcrumb-chip.category-chip {
    background-color: #8b5cf6;
  }

  .breadcrumb-chip.category-chip:hover {
    background-color: #7c3aed;
  }

  .breadcrumb-chip.kg-chip {
    background-color: #3b82f6;
  }

  .breadcrumb-chip.kg-chip:hover {
    background-color: #2563eb;
  }

  .breadcrumb-chip.source-chip {
    background-color: #10b981;
  }

  .breadcrumb-chip.source-chip:hover {
    background-color: #059669;
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

  .clear-all-button {
    padding: 0.375rem 0.75rem;
    background-color: #ef4444;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 0.875rem;
    cursor: pointer;
    transition: background-color 0.2s;
    margin-left: auto;
  }

  .clear-all-button:hover {
    background-color: #dc2626;
  }

  .custom-tooltip {
    position: fixed;
    background-color: rgba(31, 41, 55, 0.95);
    color: white;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 500;
    pointer-events: none;
    z-index: 9999;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    white-space: nowrap;
    max-width: 400px;
    line-height: 1.4;
  }
</style>
