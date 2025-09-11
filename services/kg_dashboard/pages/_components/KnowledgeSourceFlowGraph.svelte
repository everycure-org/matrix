

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
    aggregatorYPositions: [],
    aggregatorCount: 0
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
      
      // Calculate dynamic oval layout for primary sources
      const radiusX = 150; // Horizontal radius (narrower)
      
      // Dynamic sizing based on content
      const totalNodes = primaryNodes.length + aggregatorNodes.length + unifiedNodes.length;
      const baseRadiusY = Math.max(100, Math.min(250, totalNodes * 15)); // Scale radius with content
      const baseCenterY = Math.max(200, Math.min(400, 150 + totalNodes * 8)); // Scale center with content
      
      // Adjust radius and center based on node count for better small-count layouts
      let radiusY, centerY;
      if (primaryNodes.length <= 5) {
        // For small node counts, use fixed reasonable spacing centered in viewport
        radiusY = Math.max(50, primaryNodes.length * 20); // 20px per node, minimum 50px
        centerY = 300; // Fixed center position
      } else {
        // For larger node counts, use dynamic scaling
        radiusY = baseRadiusY;
        centerY = baseCenterY;
      }
      
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
      
      // Calculate primary Y positions using our current algorithm for matching aggregator spacing
      const primaryYPositions = [];
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
        primaryYPositions.push(yPos);
      }
      
      // Calculate actual primary Y spread
      const primarySpread = primaryYPositions.length > 1 ? 
        Math.max(...primaryYPositions) - Math.min(...primaryYPositions) : 0;
      
      // Update debug info
      debugInfo.primarySpread = Math.round(primarySpread);
      debugInfo.primaryYPositions = primaryYPositions.map(y => Math.round(y));
      debugInfo.aggregatorYPositions = []; // Reset for new calculation
      debugInfo.aggregatorCount = aggregatorNodes.length;
      
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
            // Math that naturally centers and matches primary Y spread
            let aggregatorSpacing;
            if (aggregatorNodes.length > 1 && primarySpread > 0) {
              // Use EXACTLY the same Y spread as primary nodes - no minimum override
              aggregatorSpacing = primarySpread / (aggregatorNodes.length - 1);
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
            aggregatorCount++;
          } else if (d.category === 'unified') {
            xPosition = centerX + radiusX - 150;
            // Math that naturally centers: when unifiedNodes.length=1, offset=0
            const unifiedSpacing = 80; // doesn't matter much since usually only 1
            const centerOffset = (unifiedCount - (unifiedNodes.length - 1) / 2) * unifiedSpacing;
            yPosition = centerY + centerOffset;
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
    }
    
    networkOption = {
      legend: {
        show: false
      },
      title: {
        text: `${title} - Top ${topNPrimarySources} Primary Sources`,
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
        max: 600  // centerY(374) + radiusY + margin = ~550-600
      },
      tooltip: {
        show: true,
        position: function(point, params, dom, rect, size) {
          if (params.dataType === 'node' && (params.data.nodeCategory === 'primary' || params.data.nodeCategory === 'unified')) {
            // Position tooltip to the left for primary sources and unified KG
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
            tooltipContent += `Total connections: ${params.data.value.toLocaleString()}`;
            
            if (connections.length > 1) {
              tooltipContent += `<br/><br/>`;
              connections.forEach(conn => {
                const targetName = conn.target.replace('infores:', '');
                tooltipContent += `${targetName}: ${conn.value.toLocaleString()}<br/>`;
              });
            }
            
            return tooltipContent;
          } else if (params.dataType === 'node') {
            // Default tooltip for other nodes
            return `<strong>${params.data.id.replace('infores:', '')}</strong><br/>Total connections: ${params.data.value.toLocaleString()}`;
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

<ECharts config={networkOption} data={networkData} {height} width="100%" />

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
      <strong>Algorithm:</strong> {debugInfo.nodeCount <= 5 ? 'Equal Angular' : 'Cumulative Spacing'}<br/>
      <strong>Y-Scaling:</strong> {debugInfo.nodeCount <= 5 ? 'Fixed' : 'Dynamic'}<br/>
      <strong>Radius X:</strong> 150px (fixed)
    </div>
  </div>
  <div style="margin-top: 10px; font-size: 11px;">
    <strong>Primary Y Spread:</strong> {debugInfo.primarySpread}px<br/>
    <strong>Aggregator Spacing:</strong> {debugInfo.aggregatorSpacing}px<br/>
    <strong>Primary Y Positions:</strong> [{debugInfo.primaryYPositions.join(', ')}]<br/>
    <strong>Aggregator Y Positions:</strong> [{debugInfo.aggregatorYPositions.join(', ')}] (Count: {debugInfo.aggregatorCount})
  </div>
</div>