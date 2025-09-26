

<script>
  // THIS FILE IS LARGELY CLAUDE GENERATED
  import { ECharts } from '@evidence-dev/core-components';
  import { showQueries } from '@evidence-dev/component-utilities/stores';
  import { DEFAULT_LEVEL_CONFIG } from '../_lib/knowledge-source-flow/constants.js';
  import { formatTooltip } from '../_lib/knowledge-source-flow/utils.js';
  import { processNetworkData } from '../_lib/knowledge-source-flow/data.js';
  import { calculateLayout, calculatePositions } from '../_lib/knowledge-source-flow/layout.js';
  import { createNodes } from '../_lib/knowledge-source-flow/nodes.js';
  import { createDebugInfo, updateDebugInfo } from '../_lib/knowledge-source-flow/debug.js';
  import { calculateDynamicHeight } from '../_lib/knowledge-source-flow/height.js';
  import { createLinks } from '../_lib/knowledge-source-flow/links.js';
  
  // Component's parameters
  export let nodeData = [];
  export let linkData = [];
  export let title = 'Network Graph';
  export let topNPrimarySources = 25;
  export let height = '900px';
  export let levelConfig = DEFAULT_LEVEL_CONFIG;
  
  let networkOption = {};
  let dynamicHeight = height;
  let debugInfo = createDebugInfo();


  $: processedData = processNetworkData(nodeData, linkData, levelConfig, topNPrimarySources);

  $: layout = calculateLayout(processedData, levelConfig);
  
  $: positions = calculatePositions(processedData, layout, levelConfig);

  $: nodes = createNodes(processedData, layout, positions, levelConfig);

  $: dynamicHeight = calculateDynamicHeight(processedData.primaryNodes, height);

  $: debugInfo = updateDebugInfo(debugInfo, processedData, layout, positions, nodes, dynamicHeight, levelConfig);

  $: links = createLinks(processedData, nodes);


  // FInal Echarts configuration
  $: networkOption = {
      legend: {
        show: false
      },
      title: {
        text: `${title}`,
        left: 'center'
      },
      grid: {
        left: '5%',
        right: '5%',
        top: '10%',
        bottom: '5%'
      },
      xAxis: {
        show: false,
        type: 'value',
        min: 0,
        max: 600  // centerX(300) + radiusX(150) + aggregator offset(200) + margin = ~550
      },
      yAxis: {
        show: false,
        type: 'value',
        min: 0,
        max: parseInt(dynamicHeight) + 50 // Add padding to ensure all nodes visible
      },
      tooltip: {
        show: true,
        formatter: (params) => formatTooltip(params, links)
      },
      series: [{
        type: 'graph',
        layout: 'force',
        force: {
          repulsion: 50,
          gravity: 0.1,
          edgeLength: 100,
          layoutAnimation: false,
          friction: 0.9
        },
        data: nodes,
        links: links,
        roam: false, // Disable zoom/pan
        draggable: false, // Disable dragging
        itemStyle: {
          borderWidth: 0
        },
        lineStyle: {
          color: 'rgba(157, 121, 214, 0.6)',
          curveness: 0.1 // Less curve for cleaner look
        },
        label: {
          show: true,
          position: 'inside',
          fontSize: 10,
          color: '#000'
        },
        emphasis: {
          focus: 'adjacency',
          itemStyle: {
            borderWidth: 3
          },
          lineStyle: {
            width: 4
          }
        }
      }]
    };



</script>

<ECharts config={networkOption} data={nodeData} height={dynamicHeight} width="100%" />

<!-- Debug Information Display - Only visible when Evidence's "Show Queries" is enabled -->
{#if $showQueries}
<div style="margin-top: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; font-family: monospace; font-size: 12px;">
  <h4 style="margin: 0 0 10px 0; color: #333;">Layout Debug Info:</h4>
  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
    <div>
      <strong>Node Count:</strong> {debugInfo.nodeCount}<br/>
      <strong>Positioning:</strong> {debugInfo.positioning}<br/>
      <strong>Arc Size:</strong> {debugInfo.arcDegrees}°
    </div>
    <div>
      <strong>Center X:</strong> {debugInfo.centerX}px<br/>
      <strong>Center Y:</strong> {debugInfo.centerY}px<br/>
      <strong>Radius Y:</strong> {debugInfo.radiusY}px
    </div>
    <div>
      <strong>Algorithm:</strong> Equal Angular Spacing<br/>
      <strong>Y-Scaling:</strong> Dynamic (all counts)<br/>
      <strong>Radius X:</strong> 150px (fixed)
    </div>
  </div>
  <div style="margin-top: 10px; font-size: 11px;">
    <strong>Primary Y Spread:</strong> {debugInfo.primarySpread}px<br/>
    <strong>Primary X Spread:</strong> {debugInfo.primaryXSpread}px<br/>
    <strong>Primary X Range:</strong> {debugInfo.minMaxPrimaryX.min}px to {debugInfo.minMaxPrimaryX.max}px<br/>
    <strong>Aggregator Spacing:</strong> {debugInfo.aggregatorSpacing}px<br/>
    <strong>Primary Y Positions:</strong> [{debugInfo.primaryYPositions.join(', ')}]<br/>
    <strong>Primary X Positions:</strong> [{debugInfo.primaryXPositions.join(', ')}]<br/>
    <strong>Primary Angles (degrees):</strong> [{debugInfo.primaryAngles.join(', ')}]<br/>
    <strong>Aggregator Y Positions:</strong> [{debugInfo.aggregatorYPositions.join(', ')}] (Count: {debugInfo.aggregatorCount})<br/>
    <strong>Aggregator X Positions:</strong> [{debugInfo.aggregatorXPositions.join(', ')}]<br/>
    <strong>Unified Y Positions:</strong> [{debugInfo.unifiedYPositions.join(', ')}] (Count: {debugInfo.unifiedCount})<br/>
    <strong>Unified X Positions:</strong> [{debugInfo.unifiedXPositions.join(', ')}]<br/>
    <strong>Content Y Bounds:</strong> {debugInfo.contentBounds.minY}px to {debugInfo.contentBounds.maxY}px<br/>
    <strong>Dynamic Height:</strong> {debugInfo.dynamicHeight}px<br/>
    <strong>Height Calculation:</strong> {debugInfo.heightCalculation}<br/>
    <strong>Sample Node Data:</strong> {JSON.stringify(debugInfo.sampleNodeData, null, 2)}<br/>
    <strong>Actual Coordinate Bounds:</strong> {JSON.stringify(debugInfo.actualBounds, null, 2)}<br/>
    <strong>Axis Bounds:</strong> X: 0-600, Y: 0-{parseInt(dynamicHeight) + 50}
  </div>
</div>
{/if}