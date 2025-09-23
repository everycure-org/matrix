

<script>
  // THIS FILE IS LARGELY CLAUDE GENERATED
  import { getSourceColor } from '../_lib/colors';
  import { ECharts } from '@evidence-dev/core-components';
  import { showQueries } from '@evidence-dev/component-utilities/stores';
  
  export let networkData = [];
  export let title = 'Network Graph';
  export let topNPrimarySources = 25;
  export let height = '900px';
  
  let networkOption = {};
  let dynamicHeight = height; // Dynamic height based on content
  let primaryNodeCount = 0; // Store primary node count for tooltip positioning
  
  // Debug variables to display current calculations
  let debugInfo = {
    nodeCount: 0,
    arcDegrees: 0,
    radiusY: 0,
    centerY: 0,
    centerX: 0,
    positioning: 'unknown',
    primarySpread: 0,
    aggregatorSpacing: 0,
    primaryYPositions: [],
    primaryXPositions: [],
    primaryAngles: [],
    aggregatorYPositions: [],
    aggregatorXPositions: [],
    unifiedYPositions: [],
    unifiedXPositions: [],
    aggregatorCount: 0,
    unifiedCount: 0,
    minMaxPrimaryX: { min: 0, max: 0 },
    primaryXSpread: 0,
    contentBounds: { minY: 0, maxY: 0 },
    dynamicHeight: 0,
    heightCalculation: '',
    sampleNodeData: []
  };

  // === CONSTANTS ===
  const LAYOUT_CONSTANTS = {
    OVAL_RADIUS_X: 150,
    MIN_RADIUS_Y: 100,
    MAX_RADIUS_Y: 250,
    RADIUS_Y_SCALE_FACTOR: 15,
    CENTER_X: 300,
    MIN_CENTER_Y: 200,
    MAX_CENTER_Y: 400,
    CENTER_Y_BASE: 150,
    CENTER_Y_SCALE_FACTOR: 8,
    MAX_TOTAL_SPREAD: Math.PI * 0.4, // 72 degrees
    MIN_SPACING: Math.PI * 0.08,
    ARC_CENTER: Math.PI,
    AGGREGATOR_X_OFFSET: 200,
    UNIFIED_X_OFFSET: 150,
    AGGREGATOR_SPACING_FRACTION: 0.5,
    MIN_AGGREGATOR_SPACING: 12,
    UNIFIED_SPACING: 80
  };

  const NODE_SIZE_CONSTANTS = {
    PRIMARY_MIN_SIZE: 8,
    PRIMARY_MAX_SIZE: 35,
    PRIMARY_SCALE_DIVISOR: 100000,
    PRIMARY_SCALE_MULTIPLIER: 3,
    PRIMARY_BASE_SIZE: 8,
    UNIFIED_MIN_SIZE: 60,
    UNIFIED_MAX_SIZE: 100,
    UNIFIED_SCALE_DIVISOR: 1000000,
    UNIFIED_SCALE_MULTIPLIER: 0.8,
    UNIFIED_BASE_SIZE: 60,
    AGGREGATOR_MIN_SIZE: 45,
    AGGREGATOR_MAX_SIZE: 80,
    AGGREGATOR_SCALE_DIVISOR: 1000000,
    AGGREGATOR_SCALE_MULTIPLIER: 0.6,
    AGGREGATOR_BASE_SIZE: 45
  };

  const HEIGHT_CONSTANTS = {
    BASE_HEIGHT: 300,      // Minimum height for 1 source
    MAX_HEIGHT: 900,       // Target height for 25 sources
    MAX_SOURCES: 25        // Reference point for maximum scaling
  };

  const COLORS = {
    PRIMARY_NODE: '#88C0D0',
    PRIMARY_LABEL: '#333',
    NON_PRIMARY_LABEL: '#fff'
  };

  // === UTILITY FUNCTIONS ===
  function calculateAngleRangeForNodeCount(nodeCount) {
    if (nodeCount <= 2) {
      return {
        dynamicAngleRange: Math.PI * 0.5,
        positioning: 'small (≤2)'
      };
    } else if (nodeCount <= 5) {
      return {
        dynamicAngleRange: Math.PI * (2/3),
        positioning: 'small (3-5)'
      };
    } else if (nodeCount <= 10) {
      return {
        dynamicAngleRange: Math.PI * (5/6),
        positioning: 'medium (6-10)'
      };
    } else {
      return {
        dynamicAngleRange: Math.PI * 0.65,
        positioning: 'large (11+)'
      };
    }
  }

  function calculateNodePosition(nodeData, category, counters, layout, positions) {
    const { centerX, centerY, radiusX, radiusY, nodeCount, arcCenter } = layout;
    const { primarySpread } = positions;

    if (category === 'primary') {
      let spacing = 0;
      if (nodeCount > 1) {
        spacing = Math.min(LAYOUT_CONSTANTS.MIN_SPACING, LAYOUT_CONSTANTS.MAX_TOTAL_SPREAD / (nodeCount - 1));
      }
      const centerOffset = (counters.primary - (nodeCount - 1) / 2) * spacing;
      const angle = arcCenter + centerOffset;

      return {
        x: centerX + Math.cos(angle) * radiusX,
        y: centerY + Math.sin(angle) * radiusY
      };
    } else if (category === 'aggregator') {
      const aggregatorNodes = positions.aggregatorNodes || [];
      let aggregatorSpacing = 0;

      if (aggregatorNodes.length > 1 && primarySpread > 0) {
        const calculatedSpacing = (primarySpread * LAYOUT_CONSTANTS.AGGREGATOR_SPACING_FRACTION) / (aggregatorNodes.length - 1);
        aggregatorSpacing = Math.max(calculatedSpacing, LAYOUT_CONSTANTS.MIN_AGGREGATOR_SPACING);
      }

      const centerOffset = (counters.aggregator - (aggregatorNodes.length - 1) / 2) * aggregatorSpacing;

      return {
        x: centerX + radiusX - LAYOUT_CONSTANTS.AGGREGATOR_X_OFFSET,
        y: centerY + centerOffset
      };
    } else if (category === 'unified') {
      const unifiedNodes = positions.unifiedNodes || [];
      const centerOffset = (counters.unified - (unifiedNodes.length - 1) / 2) * LAYOUT_CONSTANTS.UNIFIED_SPACING;

      return {
        x: centerX + radiusX - LAYOUT_CONSTANTS.UNIFIED_X_OFFSET,
        y: centerY + centerOffset
      };
    }

    return { x: 0, y: 0 };
  }

  function calculateNodeSize(value, category) {
    const safeValue = value || 1;

    switch (category) {
      case 'primary':
        return Math.max(
          NODE_SIZE_CONSTANTS.PRIMARY_MIN_SIZE,
          Math.min(
            NODE_SIZE_CONSTANTS.PRIMARY_MAX_SIZE,
            Math.sqrt(safeValue / NODE_SIZE_CONSTANTS.PRIMARY_SCALE_DIVISOR) * NODE_SIZE_CONSTANTS.PRIMARY_SCALE_MULTIPLIER + NODE_SIZE_CONSTANTS.PRIMARY_BASE_SIZE
          )
        );
      case 'unified':
        return Math.max(
          NODE_SIZE_CONSTANTS.UNIFIED_MIN_SIZE,
          Math.min(
            NODE_SIZE_CONSTANTS.UNIFIED_MAX_SIZE,
            safeValue / NODE_SIZE_CONSTANTS.UNIFIED_SCALE_DIVISOR * NODE_SIZE_CONSTANTS.UNIFIED_SCALE_MULTIPLIER + NODE_SIZE_CONSTANTS.UNIFIED_BASE_SIZE
          )
        );
      case 'aggregator':
      default:
        return Math.max(
          NODE_SIZE_CONSTANTS.AGGREGATOR_MIN_SIZE,
          Math.min(
            NODE_SIZE_CONSTANTS.AGGREGATOR_MAX_SIZE,
            safeValue / NODE_SIZE_CONSTANTS.AGGREGATOR_SCALE_DIVISOR * NODE_SIZE_CONSTANTS.AGGREGATOR_SCALE_MULTIPLIER + NODE_SIZE_CONSTANTS.AGGREGATOR_BASE_SIZE
          )
        );
    }
  }

  function createNodeFromData(nodeData, counters, layout, positions) {
    const position = calculateNodePosition(nodeData, nodeData.category, counters, layout, positions);

    let nodeColor;
    if (nodeData.category === 'primary') {
      nodeColor = COLORS.PRIMARY_NODE;
    } else if (nodeData.category === 'aggregator' || nodeData.category === 'unified') {
      const sourceKey = nodeData.node_id.replace('infores:', '');
      nodeColor = getSourceColor(sourceKey);
    }

    return {
      id: nodeData.node_id,
      name: nodeData.node_name || nodeData.node_id,
      nodeCategory: nodeData.category,
      value: nodeData.value || 0,
      total_all_sources: nodeData.total_all_sources,
      x: position.x,
      y: position.y,
      symbol: 'circle',
      symbolSize: calculateNodeSize(nodeData.value, nodeData.category),
      itemStyle: nodeColor ? { color: nodeColor } : undefined,
      label: {
        show: true,
        position: nodeData.category === 'primary' ? 'left' : 'inside',
        fontSize: nodeData.category === 'primary' ? 12 : 11,
        color: nodeData.category === 'primary' ? COLORS.PRIMARY_LABEL : COLORS.NON_PRIMARY_LABEL,
        fontWeight: nodeData.category === 'primary' ? 'normal' : 'bold',
        distance: nodeData.category === 'primary' ? 8 : 0,
        formatter: function(params) {
          if (nodeData.category === 'primary' && params.name && params.name.length > 20) {
            return params.name.substring(0, 17) + '...';
          }
          return params.name;
        }
      },
      fixed: true
    };
  }

  function formatTooltip(params, links) {
    if (params.dataType === 'node' && params.data.nodeCategory === 'primary') {
      // Primary node tooltip with connections breakdown
      const sourceId = params.data.id;
      const connections = links.filter(link => link.source === sourceId);

      let tooltipContent = `<strong>${params.data.name}</strong><br/>`;
      tooltipContent += `${params.data.value.toLocaleString()} total connections`;

      if (connections.length > 1) {
        tooltipContent += `<br/><br/>`;
        connections.forEach(conn => {
          const targetName = conn.target.replace('infores:', '');
          tooltipContent += `${targetName}: ${conn.value.toLocaleString()}<br/>`;
        });
      }

      return tooltipContent;
    } else if (params.dataType === 'node') {
      // Aggregator and unified node tooltips with dual counts
      if (params.data.total_all_sources && params.data.total_all_sources !== params.data.value) {
        return `<strong>${params.data.id.replace('infores:', '')}</strong><br/>${params.data.value.toLocaleString()} from selected sources<br/>(${params.data.total_all_sources.toLocaleString()} from all sources)`;
      } else {
        return `<strong>${params.data.id.replace('infores:', '')}</strong><br/>${params.data.value.toLocaleString()} total connections`;
      }
    } else if (params.dataType === 'edge') {
      // Edge tooltip
      return `${params.data.source.replace('infores:', '')} → ${params.data.target.replace('infores:', '')}<br/>Connections: ${params.data.value.toLocaleString()}`;
    }

    return '';
  }

  // === DATA PROCESSING ===
  $: processedData = (() => {
    if (!networkData || !Array.isArray(networkData) || networkData.length === 0) {
      return {
        nodeData: [],
        linkData: [],
        primaryNodes: [],
        aggregatorNodes: [],
        unifiedNodes: [],
        limitedNodes: []
      };
    }

    const nodeData = networkData.filter(d => d.type === 'node');
    const linkData = networkData.filter(d => d.type === 'link');

    const sortedNodes = nodeData.sort((a, b) => {
      if (a.category !== b.category) {
        const categoryOrder = { 'primary': 0, 'aggregator': 1, 'unified': 2 };
        return categoryOrder[a.category] - categoryOrder[b.category];
      }
      return (b.value || 0) - (a.value || 0);
    });

    const primaryNodes = sortedNodes.filter(n => n.category === 'primary').slice(0, topNPrimarySources).reverse();
    const aggregatorNodes = sortedNodes.filter(n => n.category === 'aggregator');
    const unifiedNodes = sortedNodes.filter(n => n.category === 'unified');
    const limitedNodes = [...primaryNodes, ...aggregatorNodes, ...unifiedNodes];

    return {
      nodeData,
      linkData,
      primaryNodes,
      aggregatorNodes,
      unifiedNodes,
      limitedNodes
    };
  })();

  $: primaryNodeCount = processedData.primaryNodes.length;

  // === LAYOUT CALCULATIONS ===
  $: layout = (() => {
    const { primaryNodes, aggregatorNodes, unifiedNodes } = processedData;
    const totalNodes = primaryNodes.length + aggregatorNodes.length + unifiedNodes.length;
    const nodeCount = primaryNodes.length;

    const radiusX = LAYOUT_CONSTANTS.OVAL_RADIUS_X;
    const radiusY = Math.max(
      LAYOUT_CONSTANTS.MIN_RADIUS_Y,
      Math.min(LAYOUT_CONSTANTS.MAX_RADIUS_Y, totalNodes * LAYOUT_CONSTANTS.RADIUS_Y_SCALE_FACTOR)
    );
    const centerX = LAYOUT_CONSTANTS.CENTER_X;
    const centerY = Math.max(
      LAYOUT_CONSTANTS.MIN_CENTER_Y,
      Math.min(LAYOUT_CONSTANTS.MAX_CENTER_Y, LAYOUT_CONSTANTS.CENTER_Y_BASE + totalNodes * LAYOUT_CONSTANTS.CENTER_Y_SCALE_FACTOR)
    );

    const { dynamicAngleRange, positioning } = calculateAngleRangeForNodeCount(nodeCount);

    return {
      radiusX,
      radiusY,
      centerX,
      centerY,
      nodeCount,
      dynamicAngleRange,
      positioning,
      arcCenter: LAYOUT_CONSTANTS.ARC_CENTER
    };
  })();
  // === POSITION CALCULATIONS ===
  $: positions = (() => {
    const { primaryNodes, aggregatorNodes, unifiedNodes } = processedData;
    const { centerX, centerY, radiusX, radiusY, nodeCount, arcCenter } = layout;

    const primaryYPositions = [];
    const primaryXPositions = [];
    const primaryAngles = [];

    for (let index = 0; index < primaryNodes.length; index++) {
      let spacing = 0;
      if (nodeCount > 1) {
        spacing = Math.min(LAYOUT_CONSTANTS.MIN_SPACING, LAYOUT_CONSTANTS.MAX_TOTAL_SPREAD / (nodeCount - 1));
      }
      const centerOffset = (index - (nodeCount - 1) / 2) * spacing;
      const angle = arcCenter + centerOffset;
      const yPos = centerY + Math.sin(angle) * radiusY;
      const xPos = centerX + Math.cos(angle) * radiusX;
      primaryYPositions.push(yPos);
      primaryXPositions.push(xPos);
      primaryAngles.push(angle);
    }

    const primarySpread = primaryYPositions.length > 1 ?
      Math.max(...primaryYPositions) - Math.min(...primaryYPositions) : 0;
    const primaryXSpread = primaryXPositions.length > 1 ?
      Math.max(...primaryXPositions) - Math.min(...primaryXPositions) : 0;
    const minMaxPrimaryX = primaryXPositions.length > 0 ? {
      min: Math.min(...primaryXPositions),
      max: Math.max(...primaryXPositions)
    } : { min: 0, max: 0 };

    return {
      primaryYPositions,
      primaryXPositions,
      primaryAngles,
      primarySpread,
      primaryXSpread,
      minMaxPrimaryX,
      aggregatorNodes,
      unifiedNodes
    };
  })();

  // === DEBUG INFO UPDATES ===
  $: {
    const { aggregatorNodes, unifiedNodes } = processedData;
    const { nodeCount, centerX, centerY, radiusY, positioning, dynamicAngleRange } = layout;
    const { primarySpread, primaryXSpread, primaryYPositions, primaryXPositions, primaryAngles, minMaxPrimaryX } = positions;

    debugInfo.nodeCount = nodeCount;
    debugInfo.centerX = centerX;
    debugInfo.centerY = centerY;
    debugInfo.radiusY = radiusY;
    debugInfo.positioning = positioning;
    debugInfo.arcDegrees = Math.round(dynamicAngleRange * 180 / Math.PI);
    debugInfo.primarySpread = Math.round(primarySpread);
    debugInfo.primaryXSpread = Math.round(primaryXSpread);
    debugInfo.primaryYPositions = primaryYPositions.map(y => Math.round(y));
    debugInfo.primaryXPositions = primaryXPositions.map(x => Math.round(x));
    debugInfo.primaryAngles = primaryAngles.map(a => Math.round(a * 180 / Math.PI));
    debugInfo.minMaxPrimaryX = {
      min: Math.round(minMaxPrimaryX.min),
      max: Math.round(minMaxPrimaryX.max)
    };
    debugInfo.aggregatorYPositions = [];
    debugInfo.aggregatorXPositions = [];
    debugInfo.unifiedYPositions = [];
    debugInfo.unifiedXPositions = [];
    debugInfo.aggregatorCount = aggregatorNodes.length;
    debugInfo.unifiedCount = unifiedNodes.length;
  }

  // === NODE CREATION ===
  $: nodes = (() => {
    const { limitedNodes, aggregatorNodes, primaryNodes } = processedData;
    const { centerX, centerY, radiusX, radiusY, nodeCount, arcCenter } = layout;
    const { primarySpread } = positions;

    const counters = { primary: 0, aggregator: 0, unified: 0 };
    const nodeMap = new Map();

    limitedNodes.forEach(nodeData => {
      if (nodeData.node_id && !nodeMap.has(nodeData.node_id)) {
        const node = createNodeFromData(nodeData, counters, layout, positions);

        // Update debug info for positioning
        if (nodeData.category === 'aggregator' && counters.aggregator === 0) {
          debugInfo.aggregatorSpacing = Math.round(
            positions.aggregatorNodes.length > 1 && positions.primarySpread > 0
              ? Math.max(
                  (positions.primarySpread * LAYOUT_CONSTANTS.AGGREGATOR_SPACING_FRACTION) / (positions.aggregatorNodes.length - 1),
                  LAYOUT_CONSTANTS.MIN_AGGREGATOR_SPACING
                )
              : 0
          );
        }

        // Capture positions for debug
        if (nodeData.category === 'aggregator') {
          debugInfo.aggregatorYPositions.push(Math.round(node.y));
          debugInfo.aggregatorXPositions.push(Math.round(node.x));
        } else if (nodeData.category === 'unified') {
          debugInfo.unifiedYPositions.push(Math.round(node.y));
          debugInfo.unifiedXPositions.push(Math.round(node.x));
        }

        counters[nodeData.category]++;
        nodeMap.set(nodeData.node_id, node);
      }
    });

    return Array.from(nodeMap.values());
  })();

  // === DYNAMIC HEIGHT CALCULATION ===
  $: {
    const { primaryNodes } = processedData;

    if (primaryNodes.length > 0) {
      const scalingFactor = Math.min(primaryNodes.length, HEIGHT_CONSTANTS.MAX_SOURCES) / HEIGHT_CONSTANTS.MAX_SOURCES;
      const calculatedHeight = HEIGHT_CONSTANTS.BASE_HEIGHT + (scalingFactor * (HEIGHT_CONSTANTS.MAX_HEIGHT - HEIGHT_CONSTANTS.BASE_HEIGHT));

      dynamicHeight = Math.round(calculatedHeight) + 'px';

      // Update debug info
      if (nodes.length > 0) {
        const allYPositions = nodes.map(n => n.y);
        const minY = Math.min(...allYPositions);
        const maxY = Math.max(...allYPositions);

        debugInfo.contentBounds = { minY: Math.round(minY), maxY: Math.round(maxY) };
        debugInfo.dynamicHeight = parseInt(dynamicHeight);
        debugInfo.heightCalculation = `${HEIGHT_CONSTANTS.BASE_HEIGHT} + (${primaryNodes.length}/${HEIGHT_CONSTANTS.MAX_SOURCES}) * ${HEIGHT_CONSTANTS.MAX_HEIGHT - HEIGHT_CONSTANTS.BASE_HEIGHT} = ${Math.round(calculatedHeight)}px`;
      }
    } else {
      dynamicHeight = height;
    }
  }

  // === LINKS CREATION ===
  $: links = (() => {
    const { linkData } = processedData;

    return linkData
      .filter(d => d.source && d.target && d.value)
      .map(d => {
        let edgeColor = 'rgba(157, 121, 214, 0.6)';

        const sourceNode = nodes.find(n => n.id === d.source);
        const targetNode = nodes.find(n => n.id === d.target);

        let aggregatorNode = null;

        if (sourceNode && (sourceNode.id.includes('robokop') || sourceNode.id.includes('rtxkg2')) &&
            !sourceNode.id.includes(',')) {
          aggregatorNode = sourceNode;
        }
        else if (targetNode && (targetNode.id.includes('robokop') || targetNode.id.includes('rtxkg2')) &&
                 !targetNode.id.includes(',')) {
          aggregatorNode = targetNode;
        }

        if (aggregatorNode && aggregatorNode.itemStyle && aggregatorNode.itemStyle.color) {
          edgeColor = aggregatorNode.itemStyle.color + '80';
        }

        return {
          source: d.source,
          target: d.target,
          value: d.value,
          lineStyle: {
            width: Math.max(1, Math.min(10, Math.sqrt(d.value / 10000) * 5 + 1)),
            color: edgeColor
          },
          label: {
            show: false,
            position: 'middle',
            formatter: function(params) {
              return params.value.toLocaleString();
            },
            fontSize: 10,
            color: '#333',
            backgroundColor: 'rgba(255, 255, 255, 0.8)',
            borderRadius: 3,
            padding: [2, 4]
          },
          emphasis: {
            label: {
              show: true
            }
          }
        };
      });
  })();

  // === FINAL DEBUG UPDATES ===
  $: {
    if (nodes.length > 0) {
      debugInfo.sampleNodeData = nodes.slice(0, 3).map(n => ({
        id: n.id,
        category: n.nodeCategory,
        value: n.value,
        x: n.x,
        y: n.y,
        total_all_sources: n.total_all_sources
      }));

      const allX = nodes.map(n => n.x);
      const allY = nodes.map(n => n.y);
      debugInfo.actualBounds = {
        minX: Math.min(...allX),
        maxX: Math.max(...allX),
        minY: Math.min(...allY),
        maxY: Math.max(...allY)
      };
    }
  }

  // === ECHARTS CONFIGURATION ===
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

<ECharts config={networkOption} data={networkData} height={dynamicHeight} width="100%" />

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