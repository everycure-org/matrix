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

  // Component props
  export let categoryData = [];
  export let edgeData = [];
  export let keyNodeName = 'Key Node';
  
  // Selection state
  let selectedCategory = null;
  let chartInstance = null;

  // Convert proxy to plain array if needed
  $: plainCategoryData = categoryData ? Array.from(categoryData) : [];

  // Reactive computations for the chart
  $: centerNode = createCenterNode(keyNodeName);
  $: categoryNodes = calculateCategoryPositions(plainCategoryData);
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

  // Convert edge data proxy to plain array and filter by selected category
  $: plainEdgeData = edgeData ? Array.from(edgeData) : [];

  // Filter edges for the selected category and add formatted edge column
  $: selectedEdgeData = selectedCategory
    ? plainEdgeData
        .filter(edge => edge.parent_category === selectedCategory)
        .map(edge => {
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
        })
    : [];
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

  <!-- Drill-down Table -->
  {#if selectedCategory}
    <div class="drill-down-container">
      <h3 class="drill-down-title">
        Edges for: <span class="category-name">{selectedCategory}</span>
        <button class="clear-button" on:click={() => selectedCategory = null}>✕ Clear</button>
      </h3>

      {#if selectedEdgeData.length > 0}
        <div class="edge-count-info">
          Showing {selectedEdgeData.length} example edge{selectedEdgeData.length !== 1 ? 's' : ''} (up to 10 per knowledge source)
        </div>

        {#key selectedCategory}
          <DataTable
            data={selectedEdgeData}
            search=true
            pagination=true
            pageSize={25}
            title="Connected Edges for {selectedCategory}">

            <Column id="edge_triple" title="Edge" wrap=true />
            <Column id="knowledge_sources_links" title="Knowledge Sources" wrap=true contentType="html" />

          </DataTable>
        {/key}
      {:else}
        <div class="no-data">
          No edge data available for {selectedCategory}
        </div>
      {/if}
    </div>
  {:else}
    <div class="instruction-message">
      Click on a category node in the diagram above to see detailed edge information
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
</style>
