import { DEFAULT_LEVEL_CONFIG } from './constants.js';

export function processNetworkData(nodeData, linkData, levelConfig = DEFAULT_LEVEL_CONFIG, topNPrimarySources) {

  // return empty result if no data is provided
  if (!nodeData || !Array.isArray(nodeData) || nodeData.length === 0) {
    const emptyResult = {
      nodeData: [],
      linkData: linkData || [],
      limitedNodes: []
    };

    // Add empty arrays for each level
    levelConfig.forEach(level => {
      emptyResult[`${level.name}Nodes`] = [];
    });

    return emptyResult;
  }

  // Create category order from levelConfig
  const categoryOrder = {};
  levelConfig.forEach((level, index) => {
    categoryOrder[level.name] = index;
  });

  // Validate all node categories exist in levelConfig
  const invalidCategories = nodeData
    .map(node => node.category)
    .filter(category => categoryOrder[category] === undefined);

  if (invalidCategories.length > 0) {
    const uniqueInvalid = [...new Set(invalidCategories)];
    const expectedCategories = Object.keys(categoryOrder);
    throw new Error(`Unknown node categories: ${uniqueInvalid.join(', ')}. Expected one of: ${expectedCategories.join(', ')}`);
  }

  const sortedNodes = nodeData.sort((a, b) => {
    if (a.category !== b.category) {
      return categoryOrder[a.category] - categoryOrder[b.category];
    }
    return (b.value || 0) - (a.value || 0);
  });

  const result = {
    nodeData,
    linkData: linkData || [],
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

  // Dynamic properties like result.primaryNodes, result.aggregatorNodes, etc. are created above
  // based on the level configuration names, providing both flexibility and semantic access

  return result;
}