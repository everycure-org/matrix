import { getSourceColor } from '../colors';
import { COLORS, DEFAULT_LEVEL_CONFIG } from './constants.js';
import { calculateNodePosition, calculateNodeSize } from './utils.js';

export function createNodeFromData(nodeData, counters, layout, positions, levelConfig = DEFAULT_LEVEL_CONFIG) {
  const position = calculateNodePosition(nodeData, nodeData.category, counters, layout, positions, levelConfig);

  // Find level configuration for this node's category
  const levelInfo = levelConfig.find(level => level.name === nodeData.category);
  const isFirstLevel = levelInfo === levelConfig[0];

  let nodeColor;
  if (isFirstLevel) {
    nodeColor = COLORS.PRIMARY_NODE;
  } else {
    const sourceKey = nodeData.node_id.replace('infores:', '');
    nodeColor = getSourceColor(sourceKey);
  }

  // Use level-specific label configuration or defaults
  const labelConfig = levelInfo?.label || {
    position: isFirstLevel ? 'left' : 'inside',
    fontSize: isFirstLevel ? 12 : 11,
    fontWeight: isFirstLevel ? 'normal' : 'bold',
    distance: isFirstLevel ? 8 : 0
  };

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
      position: labelConfig.position,
      fontSize: labelConfig.fontSize,
      color: isFirstLevel ? COLORS.PRIMARY_LABEL : COLORS.NON_PRIMARY_LABEL,
      fontWeight: labelConfig.fontWeight,
      distance: labelConfig.distance,
      formatter: function(params) {
        if (isFirstLevel && params.name && params.name.length > 20) {
          return params.name.substring(0, 17) + '...';
        }
        return params.name;
      }
    },
    fixed: true
  };
}

export function createNodes(processedData, layout, positions, levelConfig = DEFAULT_LEVEL_CONFIG) {
  const { limitedNodes } = processedData;

  // Initialize counters for each level
  const counters = {};
  levelConfig.forEach(level => {
    counters[level.name] = 0;
  });

  const nodeMap = new Map();

  limitedNodes.forEach(nodeData => {
    if (nodeData.node_id && !nodeMap.has(nodeData.node_id)) {
      const node = createNodeFromData(nodeData, counters, layout, positions, levelConfig);
      counters[nodeData.category]++;
      nodeMap.set(nodeData.node_id, node);
    }
  });

  return Array.from(nodeMap.values());
}