import { LAYOUT_CONSTANTS } from './constants.js';
import { CATEGORY_COLORS } from '../colors.js';

/**
 * Calculate positions for category nodes arranged in an oval
 * Node size is scaled by distinct_nodes count
 */
export function calculateCategoryPositions(categories) {
  const { CENTER_X, CENTER_Y, OUTER_RADIUS_X, OUTER_RADIUS_Y, CATEGORY_NODE_SIZE } = LAYOUT_CONSTANTS;
  const count = categories.length;

  // Find min/max for scaling
  const distinctNodeCounts = categories.map(category => category.distinct_nodes || 0);
  const minNodes = Math.min(...distinctNodeCounts);
  const maxNodes = Math.max(...distinctNodeCounts);

  // Scale node sizes between 20 and 60 pixels
  // Minimum of 20px ensures nodes are always clickable and visible
  // Maximum of 60px prevents nodes from overlapping on the oval
  const minSize = 20;
  const maxSize = 60;

  return categories.map((category, index) => {
    const angle = (2 * Math.PI * index) / count - Math.PI / 2; // Start from top
    const x = CENTER_X + OUTER_RADIUS_X * Math.cos(angle);
    const y = CENTER_Y + OUTER_RADIUS_Y * Math.sin(angle);

    // Scale node size based on distinct nodes
    const nodeCount = category.distinct_nodes || 0;
    // If all categories have same count (maxNodes === minNodes), use default size
    const scaledSize = maxNodes > minNodes
      ? minSize + (maxSize - minSize) * (nodeCount - minNodes) / (maxNodes - minNodes)
      : CATEGORY_NODE_SIZE;

    return {
      name: category.connected_category,
      value: category.total_edges,
      distinctNodes: category.distinct_nodes,
      x: x,
      y: y,
      symbolSize: scaledSize,
      category: 'category',
      itemStyle: {
        color: CATEGORY_COLORS[category.connected_category] || CATEGORY_COLORS['Other']
      }
    };
  });
}

/**
 * Create the center node representing the key node
 */
export function createCenterNode(keyNodeName) {
  const { CENTER_X, CENTER_Y, CENTER_NODE_SIZE } = LAYOUT_CONSTANTS;

  return {
    name: keyNodeName,
    value: 0,
    x: CENTER_X,
    y: CENTER_Y,
    symbolSize: CENTER_NODE_SIZE,
    category: 'center',
    itemStyle: {
      color: '#1e40af', // Blue color for key node
      borderWidth: 2,
      borderColor: '#fff'
    },
    label: {
      show: true,
      position: 'bottom',
      fontSize: 12,
      fontWeight: 'bold',
      color: '#333',
      distance: 8
    }
  };
}

/**
 * Create links from center node to all category nodes
 * Link width is scaled by total_edges count
 */
export function createLinks(keyNodeName, categoryNodes) {
  const { LINK_CURVENESS } = LAYOUT_CONSTANTS;

  // Find min/max for scaling link widths
  const edgeCounts = categoryNodes.map(node => node.value || 0);
  const minEdges = Math.min(...edgeCounts);
  const maxEdges = Math.max(...edgeCounts);

  // Scale link widths between 2 and 12 pixels
  // Minimum of 2px keeps thin connections visible
  // Maximum of 12px prevents dominant connections from overwhelming the viz
  const minWidth = 2;
  const maxWidth = 12;

  return categoryNodes.map(node => {
    const edgeCount = node.value || 0;
    // If all links have same edge count, use default width of 4px
    const scaledWidth = maxEdges > minEdges
      ? minWidth + (maxWidth - minWidth) * (edgeCount - minEdges) / (maxEdges - minEdges)
      : 4;

    return {
      source: keyNodeName,
      target: node.name,
      value: node.value,
      lineStyle: {
        color: node.itemStyle.color,
        width: scaledWidth,
        curveness: LINK_CURVENESS,
        opacity: 0.6
      }
    };
  });
}

/**
 * Format tooltip content
 */
export function formatTooltip(params) {
  if (params.dataType === 'node') {
    if (params.data.category === 'center') {
      return `<strong>${params.data.name}</strong><br/>Key Node`;
    }
    return `<strong>${params.data.name}</strong><br/>Distinct Nodes: ${params.data.distinctNodes.toLocaleString()}<br/>Total Edges: ${params.data.value.toLocaleString()}`;
  } else if (params.dataType === 'edge') {
    return `<strong>${params.data.source}</strong> → <strong>${params.data.target}</strong><br/>Total Edges: ${params.data.value.toLocaleString()}`;
  }
  return '';
}
