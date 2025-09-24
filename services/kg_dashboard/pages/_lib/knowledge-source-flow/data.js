export function processNetworkData(networkData, topNPrimarySources) {
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
}