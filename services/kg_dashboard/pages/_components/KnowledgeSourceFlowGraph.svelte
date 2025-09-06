// THIS FILE IS LARGELY CLAUDE GENERATED

<script>
  import { getSourceColor } from '../_lib/colors';
  import { ECharts } from '@evidence-dev/core-components';
  
  export let networkData = [];
  export let title = 'Network Graph';
  export let topNPrimarySources = 25;
  export let height = '900px';
  
  let networkOption = {};
  
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
      
      // Calculate oval layout for primary sources (left half)
      const radiusX = 150; // Horizontal radius (narrower)
      const radiusY = 250; // Vertical radius (taller for more space)
      const centerX = 400; // Center X position of the oval
      const centerY = 300; // Center Y position of the oval
      const startAngle = Math.PI * 0.7; // Start at ~126 degrees (upper left)
      const endAngle = Math.PI * 1.3; // End at ~234 degrees (lower left) - extends more to the right
      const angleRange = endAngle - startAngle; // Total angle span
      
      // Position aggregators and unified KG relative to the circle
      const aggregatorStartY = centerY - 50; // Position near center of circle
      const aggregatorSpacing = 100;
      const unifiedY = aggregatorNodes.length > 1 ? 
        aggregatorStartY + (aggregatorNodes.length - 1) * aggregatorSpacing / 2 :
        aggregatorStartY;
      
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
            // Find this node's size info
            const sizeInfo = primarySizes.find(item => item.node.node_id === d.node_id);
            const currentNodeSize = sizeInfo ? sizeInfo.size : 15;
            
            // Add spacing before this node (except for the first one)
            if (primaryCount > 0) {
              cumulativeSpacing += Math.max(20, currentNodeSize * 0.3);
            }
            
            // Calculate position based on cumulative spacing (including space before this node)
            const spacingRatio = cumulativeSpacing / totalSpacingWithPadding;
            const angle = startAngle + spacingRatio * angleRange;
            xPosition = centerX + Math.cos(angle) * radiusX;
            yPosition = centerY + Math.sin(angle) * radiusY;
            
            // Add this node's size to cumulative spacing for next calculation
            cumulativeSpacing += currentNodeSize;
            primaryCount++;
          } else if (d.category === 'aggregator') {
            xPosition = 350; // Move aggregators much closer to the oval
            yPosition = aggregatorStartY + aggregatorCount * aggregatorSpacing;
            aggregatorCount++;
          } else if (d.category === 'unified') {
            xPosition = 450; // Move unified KG much closer to aggregators
            yPosition = unifiedY;
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
      series: [{
        type: 'graph',
        layout: 'none', // Try disabling force layout to use our fixed positions
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
        tooltip: {
          show: true,
          position: function(point, params, dom, rect, size) {
            if (params.dataType === 'node' && params.data.nodeCategory === 'primary') {
              // Position tooltip to the left for primary sources
              return [point[0] - size.contentSize[0] - 20, point[1] - size.contentSize[1] / 2];
            }
            // Default positioning for other nodes
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