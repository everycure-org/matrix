

<script>
  // THIS FILE IS LARGELY CLAUDE GENERATED
  import { getSourceColor } from '../_lib/colors';
  import { ECharts } from '@evidence-dev/core-components';
  
  export let networkData = [];
  export let title = 'Network Graph';
  export let topNPrimarySources = 25;
  export let height = '900px';
  
  let networkOption = {};
  let useForceLayout = true; // Toggle between 'none' and 'force' layout
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
  
  $: {
    // Process the network data
    let nodes = [];
    let links = [];
    
    if (networkData && Array.isArray(networkData) && networkData.length > 0) {
      // Separate nodes and links
      const nodeData = networkData.filter(d => d.type === 'node');
      const linkData = networkData.filter(d => d.type === 'link');
      
      // Sort nodes by category priority, then by value (largest first)
      const sortedNodes = nodeData.sort((a, b) => {
        if (a.category !== b.category) {
          const categoryOrder = { 'primary': 0, 'aggregator': 1, 'unified': 2 };
          return categoryOrder[a.category] - categoryOrder[b.category];
        }
        return (b.value || 0) - (a.value || 0); // Larger values first
      });
      
      // Adjust primary sources for oval layout
      const primaryNodes = sortedNodes.filter(n => n.category === 'primary').slice(0, topNPrimarySources).reverse();
      const aggregatorNodes = sortedNodes.filter(n => n.category === 'aggregator');
      const unifiedNodes = sortedNodes.filter(n => n.category === 'unified');
      const limitedNodes = [...primaryNodes, ...aggregatorNodes, ...unifiedNodes];

      // Store primary node count for tooltip positioning
      primaryNodeCount = primaryNodes.length;
      
      // Calculate dynamic oval layout for primary sources
      const radiusX = 150; // Horizontal radius (narrower)
      
      // Dynamic sizing based on content
      const totalNodes = primaryNodes.length + aggregatorNodes.length + unifiedNodes.length;
      const baseRadiusY = Math.max(100, Math.min(250, totalNodes * 15)); // Scale radius with content
      const baseCenterY = Math.max(200, Math.min(400, 150 + totalNodes * 8)); // Scale center with content
      
      // Use dynamic scaling for all node counts (removes stepped behavior)
      let radiusY = baseRadiusY;
      let centerY = baseCenterY;
      
      const centerX = 300; // Move primary sources further right to use available space
      
      // Count-based arc sizing for better visual distribution
      let dynamicAngleRange;
      const nodeCount = primaryNodes.length;
      
      // Update debug info
      debugInfo.nodeCount = nodeCount;
      debugInfo.centerX = centerX;
      debugInfo.centerY = centerY;
      debugInfo.radiusY = radiusY;
      
      if (nodeCount <= 2) {
        dynamicAngleRange = Math.PI * 0.5; // 90 degrees - quarter circle for breathing room
        debugInfo.positioning = 'small (≤2)';
      } else if (nodeCount <= 5) {
        dynamicAngleRange = Math.PI * (2/3); // 120 degrees - comfortable spacing
        debugInfo.positioning = 'small (3-5)';
      } else if (nodeCount <= 10) {
        dynamicAngleRange = Math.PI * (5/6); // 150 degrees - good distribution
        debugInfo.positioning = 'medium (6-10)';
      } else {
        dynamicAngleRange = Math.PI * 0.65; // 117 degrees - slightly larger than original max to prevent collisions
        debugInfo.positioning = 'large (11+)';
      }
      
      // Convert to degrees for display
      debugInfo.arcDegrees = Math.round(dynamicAngleRange * 180 / Math.PI);
      
      // Center the arc around the left side (Math.PI = 180 degrees = left side)
      const arcCenter = Math.PI; // Left side of oval
      const startAngle = arcCenter - (dynamicAngleRange / 2); // Start above center
      const endAngle = arcCenter + (dynamicAngleRange / 2); // End below center
      const angleRange = dynamicAngleRange;
      
      // Calculate primary Y and X positions using our current algorithm for matching aggregator spacing
      const primaryYPositions = [];
      const primaryXPositions = [];
      const primaryAngles = [];
      for (let index = 0; index < primaryNodes.length; index++) {
        // Use tight clustering with adaptive spacing based on node count
        let spacing = 0;
        if (nodeCount > 1) {
          // Adaptive spacing: smaller angles for more nodes to keep them clustered
          const maxTotalSpread = Math.PI * 0.4; // Maximum 72 degrees total spread
          spacing = Math.min(Math.PI * 0.08, maxTotalSpread / (nodeCount - 1));
        }
        const centerOffset = (index - (nodeCount - 1) / 2) * spacing;
        const angle = arcCenter + centerOffset;
        const yPos = centerY + Math.sin(angle) * radiusY;
        const xPos = centerX + Math.cos(angle) * radiusX;
        primaryYPositions.push(yPos);
        primaryXPositions.push(xPos);
        primaryAngles.push(angle);
      }
      
      // Calculate actual primary Y and X spreads
      const primarySpread = primaryYPositions.length > 1 ?
        Math.max(...primaryYPositions) - Math.min(...primaryYPositions) : 0;
      const primaryXSpread = primaryXPositions.length > 1 ?
        Math.max(...primaryXPositions) - Math.min(...primaryXPositions) : 0;
      const minMaxPrimaryX = primaryXPositions.length > 0 ? {
        min: Math.min(...primaryXPositions),
        max: Math.max(...primaryXPositions)
      } : { min: 0, max: 0 };

      // Update debug info
      debugInfo.primarySpread = Math.round(primarySpread);
      debugInfo.primaryXSpread = Math.round(primaryXSpread);
      debugInfo.primaryYPositions = primaryYPositions.map(y => Math.round(y));
      debugInfo.primaryXPositions = primaryXPositions.map(x => Math.round(x));
      debugInfo.primaryAngles = primaryAngles.map(a => Math.round(a * 180 / Math.PI)); // Convert to degrees
      debugInfo.minMaxPrimaryX = {
        min: Math.round(minMaxPrimaryX.min),
        max: Math.round(minMaxPrimaryX.max)
      };
      debugInfo.aggregatorYPositions = []; // Reset for new calculation
      debugInfo.aggregatorXPositions = []; // Reset for new calculation
      debugInfo.unifiedYPositions = []; // Reset for new calculation
      debugInfo.unifiedXPositions = []; // Reset for new calculation
      debugInfo.aggregatorCount = aggregatorNodes.length;
      debugInfo.unifiedCount = unifiedNodes.length;
      
      // Note: Aggregator and unified positioning now calculated inline with the nodes
      
      // Pre-calculate primary node sizes for dynamic spacing
      const primarySizes = primaryNodes.map(d => {
        const size = Math.max(8, Math.min(35, Math.sqrt((d.value || 1) / 100000) * 3 + 8));
        return { node: d, size: size };
      });
      
      // Calculate cumulative spacing based on node sizes
      const totalSpacing = primarySizes.reduce((total, item, index) => {
        if (index === 0) return item.size;
        return total + item.size + Math.max(20, item.size * 0.3); // Minimum 20px gap, or 30% of node size
      }, 0);
      
      // Add padding at the end to ensure last node has proper spacing
      const totalSpacingWithPadding = totalSpacing + Math.max(20, primarySizes[primarySizes.length - 1]?.size * 0.3 || 20);
      
      let primaryCount = 0, aggregatorCount = 0, unifiedCount = 0;
      let cumulativeSpacing = 0;
      
      const nodeMap = new Map();
      
      limitedNodes.forEach(d => {
        if (d.node_id && !nodeMap.has(d.node_id)) {
          // Assign non-overlapping positions based on category
          let xPosition = 0, yPosition = 0;
          
          if (d.category === 'primary') {
            let angle;
            
            // Use tight clustering with adaptive spacing based on node count
            let spacing = 0;
            if (nodeCount > 1) {
              // Adaptive spacing: smaller angles for more nodes to keep them clustered
              const maxTotalSpread = Math.PI * 0.4; // Maximum 72 degrees total spread
              spacing = Math.min(Math.PI * 0.08, maxTotalSpread / (nodeCount - 1));
            }
            const centerOffset = (primaryCount - (nodeCount - 1) / 2) * spacing;
            angle = arcCenter + centerOffset;
            
            xPosition = centerX + Math.cos(angle) * radiusX;
            yPosition = centerY + Math.sin(angle) * radiusY;
            primaryCount++;
          } else if (d.category === 'aggregator') {
            xPosition = centerX + radiusX - 200;
            // Math that naturally centers with a fraction of primary Y spread for more even spacing
            let aggregatorSpacing;
            if (aggregatorNodes.length > 1 && primarySpread > 0) {
              // Use 50% of the primary spread for more even aggregator spacing
              const fractionOfPrimarySpread = 0.5;
              const calculatedSpacing = (primarySpread * fractionOfPrimarySpread) / (aggregatorNodes.length - 1);

              // Prevent overlap: ensure minimum spacing based on aggregator node size
              const minimumSpacing = 12;
              aggregatorSpacing = Math.max(calculatedSpacing, minimumSpacing);
            } else {
              aggregatorSpacing = 0; // Single aggregator stays at center
            }
            // Update debug info for first aggregator
            if (aggregatorCount === 0) {
              debugInfo.aggregatorSpacing = Math.round(aggregatorSpacing);
            }
            const centerOffset = (aggregatorCount - (aggregatorNodes.length - 1) / 2) * aggregatorSpacing;
            yPosition = centerY + centerOffset;
            
            // Capture aggregator position for debug
            debugInfo.aggregatorYPositions.push(Math.round(yPosition));
            debugInfo.aggregatorXPositions.push(Math.round(xPosition));
            aggregatorCount++;
          } else if (d.category === 'unified') {
            xPosition = centerX + radiusX - 150;
            // Math that naturally centers: when unifiedNodes.length=1, offset=0
            const unifiedSpacing = 80; // doesn't matter much since usually only 1
            const centerOffset = (unifiedCount - (unifiedNodes.length - 1) / 2) * unifiedSpacing;
            yPosition = centerY + centerOffset;

            // Capture unified position for debug
            debugInfo.unifiedYPositions.push(Math.round(yPosition));
            debugInfo.unifiedXPositions.push(Math.round(xPosition));
            unifiedCount++;
          }
          
          // Get color based on node ID and category
          let nodeColor;
          if (d.category === 'primary') {
            nodeColor = '#88C0D0'; // Light blue for primary sources
          } else if (d.category === 'aggregator' || d.category === 'unified') {
            const sourceKey = d.node_id.replace('infores:', '');
            nodeColor = getSourceColor(sourceKey);
          }

          nodeMap.set(d.node_id, {
            id: d.node_id,
            name: d.node_name || d.node_id, // Use display name if available, fallback to ID
            nodeCategory: d.category, // Store as custom property, not ECharts category
            value: d.value || 0,
            total_all_sources: d.total_all_sources, // Include unfiltered totals for tooltip context
            x: xPosition,
            y: yPosition,
            symbol: 'circle', // All circles
            symbolSize: d.category === 'primary' ? 
              Math.max(8, Math.min(35, Math.sqrt((d.value || 1) / 100000) * 3 + 8)) : // Primary: square root scaling for better differentiation
              d.category === 'unified' ? 
                Math.max(60, Math.min(100, (d.value || 0) / 1000000 * 0.8 + 60)) : // Unified: largest circles
                Math.max(45, Math.min(80, (d.value || 0) / 1000000 * 0.6 + 45)), // Aggregators: medium circles
            itemStyle: nodeColor ? { color: nodeColor } : undefined,
            label: {
              show: true,
              position: d.category === 'primary' ? 'left' : 'inside',
              fontSize: d.category === 'primary' ? 12 : 11,
              color: d.category === 'primary' ? '#333' : '#fff',
              fontWeight: d.category === 'primary' ? 'normal' : 'bold',
              distance: d.category === 'primary' ? 8 : 0,
              formatter: function(params) {
                if (d.category === 'primary' && params.name && params.name.length > 20) {
                  return params.name.substring(0, 17) + '...';
                }
                return params.name;
              }
            },
            fixed: true
          });
        }
      });
      
      nodes = Array.from(nodeMap.values());

      // Calculate dynamic height using smooth function based on primary node count
      if (primaryNodes.length > 0) {
        const baseHeight = 300; // Minimum height for 1 source
        const maxHeight = 900;  // Target height for 25 sources
        const maxSources = 25;  // Reference point for maximum scaling

        // Smooth linear interpolation from baseHeight to maxHeight
        const scalingFactor = Math.min(primaryNodes.length, maxSources) / maxSources;
        const calculatedHeight = baseHeight + (scalingFactor * (maxHeight - baseHeight));

        dynamicHeight = Math.round(calculatedHeight) + 'px';

        // Calculate content bounds for debug info
        const allYPositions = nodes.map(n => n.y);
        const minY = Math.min(...allYPositions);
        const maxY = Math.max(...allYPositions);

        // Update debug info
        debugInfo.contentBounds = { minY: Math.round(minY), maxY: Math.round(maxY) };
        debugInfo.dynamicHeight = parseInt(dynamicHeight);
        debugInfo.heightCalculation = `${baseHeight} + (${primaryNodes.length}/${maxSources}) * ${maxHeight - baseHeight} = ${Math.round(calculatedHeight)}px`;
      } else {
        dynamicHeight = height; // Fallback to original height
      }

      // Create links, filtering out invalid ones
      links = linkData
        .filter(d => d.source && d.target && d.value)
        .map(d => {
          // Determine edge color based on aggregator node
          let edgeColor = 'rgba(157, 121, 214, 0.6)'; // default purple
          
          // Find which node is the aggregator by checking if it's robokop or rtxkg2
          const sourceNode = nodes.find(n => n.id === d.source);
          const targetNode = nodes.find(n => n.id === d.target);
          
          let aggregatorNode = null;
          
          // Check if source is an aggregator (robokop or rtxkg2)
          if (sourceNode && (sourceNode.id.includes('robokop') || sourceNode.id.includes('rtxkg2')) && 
              !sourceNode.id.includes(',')) { // Single aggregator, not unified
            aggregatorNode = sourceNode;
          } 
          // Check if target is an aggregator
          else if (targetNode && (targetNode.id.includes('robokop') || targetNode.id.includes('rtxkg2')) && 
                   !targetNode.id.includes(',')) { // Single aggregator, not unified
            aggregatorNode = targetNode;
          }
          
          if (aggregatorNode && aggregatorNode.itemStyle && aggregatorNode.itemStyle.color) {
            edgeColor = aggregatorNode.itemStyle.color + '80'; // Add transparency
          }
          
          return {
            source: d.source,
            target: d.target,
            value: d.value,
            lineStyle: {
              width: Math.max(1, Math.min(10, Math.sqrt(d.value / 10000) * 5 + 1)), // Thickness based on count
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

    // Update debug info with sample node data
    debugInfo.sampleNodeData = nodes.slice(0, 3).map(n => ({
      id: n.id,
      category: n.nodeCategory,
      value: n.value,
      total_all_sources: n.total_all_sources
    }));
    }

    networkOption = {
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
        max: 500  // centerX(300) + radiusX(150) + margin = ~450-500
      },
      yAxis: {
        show: false,
        type: 'value',
        min: 0,
        max: parseInt(dynamicHeight) // Dynamic based on actual content bounds
      },
      tooltip: {
        show: true,
        position: function(point, params, dom, rect, size) {
          if (params.dataType === 'node' && params.data.nodeCategory === 'primary') {
            // Position tooltip based on primary node count
            // Right for ≤5 nodes, left for ≥6 nodes (labels get pushed off screen with fewer nodes)
            if (primaryNodeCount <= 5) {
              return [point[0] + 20, point[1] - size.contentSize[1] / 2]; // Right
            } else {
              return [point[0] - size.contentSize[0] - 20, point[1] - size.contentSize[1] / 2]; // Left
            }
          } else if (params.dataType === 'node' && params.data.nodeCategory === 'unified') {
            // Position tooltip to the left for unified KG
            return [point[0] - size.contentSize[0] - 20, point[1] - size.contentSize[1] / 2];
          }
          // Default positioning for aggregator nodes
          return [point[0] + 20, point[1] - size.contentSize[1] / 2];
        },
        formatter: function(params) {
          if (params.dataType === 'node' && params.data.nodeCategory === 'primary') {
            // Find all connections from this primary source to aggregators
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
            // Enhanced tooltip for aggregator and unified nodes with dual counts
            if (params.data.total_all_sources && params.data.total_all_sources !== params.data.value) {
              return `<strong>${params.data.id.replace('infores:', '')}</strong><br/>${params.data.value.toLocaleString()} from selected sources<br/>(${params.data.total_all_sources.toLocaleString()} from all sources)`;
            } else {
              // Default tooltip for nodes without dual counts
              return `<strong>${params.data.id.replace('infores:', '')}</strong><br/>${params.data.value.toLocaleString()} total connections`;
            }
          } else if (params.dataType === 'edge') {
            // Tooltip for edges
            return `${params.data.source.replace('infores:', '')} → ${params.data.target.replace('infores:', '')}<br/>Connections: ${params.data.value.toLocaleString()}`;
          }
        }
      },
      series: [{
        type: 'graph',
        layout: useForceLayout ? 'force' : 'none',
        coordinateSystem: useForceLayout ? null : 'cartesian2d',
        ...(useForceLayout ? {
          force: {
            repulsion: 50,
            gravity: 0.1,
            edgeLength: 100,
            layoutAnimation: false,
            friction: 0.9
          }
        } : {}),
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
  }
</script>

<ECharts config={networkOption} data={networkData} height={dynamicHeight} width="100%" />

<!-- Debug Information Display -->
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
    <strong>Sample Node Data:</strong> {JSON.stringify(debugInfo.sampleNodeData, null, 2)}
  </div>
</div>