import { DEFAULT_LEVEL_CONFIG } from './constants.js';

export function processNetworkData(networkData, levelConfig = DEFAULT_LEVEL_CONFIG, topNPrimarySources = 25) {
  if (!networkData || !Array.isArray(networkData) || networkData.length === 0) {
    const emptyResult = {
      nodeData: [],
      linkData: [],
      limitedNodes: []
    };

    // Add empty arrays for each level
    levelConfig.forEach(level => {
      emptyResult[`${level.name}Nodes`] = [];
    });

    return emptyResult;
  }

  const nodeData = networkData.filter(d => d.type === 'node');
  const linkData = networkData.filter(d => d.type === 'link');

  // Create category order from levelConfig
  const categoryOrder = {};
  levelConfig.forEach((level, index) => {
    categoryOrder[level.name] = index;
  });

  const sortedNodes = nodeData.sort((a, b) => {
    if (a.category !== b.category) {
      return (categoryOrder[a.category] ?? 999) - (categoryOrder[b.category] ?? 999);
    }
    return (b.value || 0) - (a.value || 0);
  });

  const result = {
    nodeData,
    linkData,
    limitedNodes: []
  };

  // Process nodes for each level
  levelConfig.forEach(level => {
    let levelNodes = sortedNodes.filter(n => n.category === level.name);

    // Apply top N limit for first level (typically primary)
    if (level === levelConfig[0]) {
      levelNodes = levelNodes.slice(0, topNPrimarySources).reverse();
    }

    result[`${level.name}Nodes`] = levelNodes;
    result.limitedNodes.push(...levelNodes);
  });

  // Add backward-compatible semantic properties for readability
  // This hybrid approach allows flexible level configuration while maintaining readable code
  result.primaryNodes = result[`${levelConfig[0]?.name}Nodes`] || [];
  result.aggregatorNodes = result[`${levelConfig[1]?.name}Nodes`] || [];
  result.unifiedNodes = result[`${levelConfig[2]?.name}Nodes`] || [];

  return result;
}