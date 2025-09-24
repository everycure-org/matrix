import { getSourceColor } from '../colors';
import { COLORS } from './constants.js';
import { calculateNodePosition, calculateNodeSize } from './utils.js';

export function createNodeFromData(nodeData, counters, layout, positions) {
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

export function createNodes(processedData, layout, positions) {
  const { limitedNodes } = processedData;
  const counters = { primary: 0, aggregator: 0, unified: 0 };
  const nodeMap = new Map();

  limitedNodes.forEach(nodeData => {
    if (nodeData.node_id && !nodeMap.has(nodeData.node_id)) {
      const node = createNodeFromData(nodeData, counters, layout, positions);
      counters[nodeData.category]++;
      nodeMap.set(nodeData.node_id, node);
    }
  });

  return Array.from(nodeMap.values());
}