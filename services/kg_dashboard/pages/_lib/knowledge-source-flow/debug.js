import { LAYOUT_CONSTANTS } from './constants.js';

export function createDebugInfo() {
  return {
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
    sampleNodeData: [],
    actualBounds: { minX: 0, maxX: 0, minY: 0, maxY: 0 }
  };
}

export function updateDebugInfo(debugInfo, processedData, layout, positions, nodes, dynamicHeight) {
  const { aggregatorNodes, unifiedNodes } = processedData;
  const { nodeCount, centerX, centerY, radiusY, positioning, dynamicAngleRange } = layout;
  const { primarySpread, primaryXSpread, primaryYPositions, primaryXPositions, primaryAngles, minMaxPrimaryX } = positions;

  // Basic layout info
  debugInfo.nodeCount = nodeCount;
  debugInfo.centerX = centerX;
  debugInfo.centerY = centerY;
  debugInfo.radiusY = radiusY;
  debugInfo.positioning = positioning;
  debugInfo.arcDegrees = Math.round(dynamicAngleRange * 180 / Math.PI);
  debugInfo.primarySpread = Math.round(primarySpread);
  debugInfo.primaryXSpread = Math.round(primaryXSpread);
  debugInfo.aggregatorCount = aggregatorNodes.length;
  debugInfo.unifiedCount = unifiedNodes.length;

  // Position arrays
  debugInfo.primaryYPositions = primaryYPositions.map(y => Math.round(y));
  debugInfo.primaryXPositions = primaryXPositions.map(x => Math.round(x));
  debugInfo.primaryAngles = primaryAngles.map(a => Math.round(a * 180 / Math.PI));
  debugInfo.minMaxPrimaryX = {
    min: Math.round(minMaxPrimaryX.min),
    max: Math.round(minMaxPrimaryX.max)
  };

  // Calculate aggregator spacing
  if (aggregatorNodes.length > 1 && primarySpread > 0) {
    debugInfo.aggregatorSpacing = Math.round(
      Math.max(
        (primarySpread * LAYOUT_CONSTANTS.AGGREGATOR_SPACING_FRACTION) / (aggregatorNodes.length - 1),
        LAYOUT_CONSTANTS.MIN_AGGREGATOR_SPACING
      )
    );
  }

  // Node position tracking (reset arrays first)
  debugInfo.aggregatorYPositions = [];
  debugInfo.aggregatorXPositions = [];
  debugInfo.unifiedYPositions = [];
  debugInfo.unifiedXPositions = [];

  // Sample node data and bounds
  if (nodes && nodes.length > 0) {
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
    const minY = Math.min(...allY);
    const maxY = Math.max(...allY);

    debugInfo.actualBounds = {
      minX: Math.min(...allX),
      maxX: Math.max(...allX),
      minY: Math.min(...allY),
      maxY: Math.max(...allY)
    };
    debugInfo.contentBounds = { minY: Math.round(minY), maxY: Math.round(maxY) };
  }

  // Height calculation
  debugInfo.dynamicHeight = parseInt(dynamicHeight);

  return debugInfo;
}