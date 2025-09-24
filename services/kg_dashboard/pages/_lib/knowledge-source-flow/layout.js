import { LAYOUT_CONSTANTS, DEFAULT_LEVEL_CONFIG } from './constants.js';
import { calculateAngleRangeForNodeCount, calculateLevelXPositions, calculateAdaptiveRadius, calculateAdaptiveSpread } from './utils.js';

export function calculateLayout(processedData, levelConfig = DEFAULT_LEVEL_CONFIG) {
  // Use semantic access for readability, with dynamic fallback
  const { primaryNodes, aggregatorNodes, unifiedNodes } = processedData;
  const totalNodes = primaryNodes.length + aggregatorNodes.length + unifiedNodes.length;
  const nodeCount = primaryNodes.length;

  const radiusX = LAYOUT_CONSTANTS.OVAL_RADIUS_X;

  // For arc layouts, use adaptive radius based on primary node count
  // For non-arc layouts, fall back to total-nodes-based calculation
  const primaryLevel = levelConfig[0];
  let radiusY;
  if (primaryLevel?.layout === 'arc') {
    radiusY = calculateAdaptiveRadius(primaryNodes.length, primaryLevel.arc);
  } else {
    radiusY = Math.max(
      LAYOUT_CONSTANTS.MIN_RADIUS_Y,
      Math.min(LAYOUT_CONSTANTS.MAX_RADIUS_Y, totalNodes * LAYOUT_CONSTANTS.RADIUS_Y_SCALE_FACTOR)
    );
  }
  const centerX = LAYOUT_CONSTANTS.CENTER_X;
  const centerY = Math.max(
    LAYOUT_CONSTANTS.MIN_CENTER_Y,
    Math.min(LAYOUT_CONSTANTS.MAX_CENTER_Y, LAYOUT_CONSTANTS.CENTER_Y_BASE + totalNodes * LAYOUT_CONSTANTS.CENTER_Y_SCALE_FACTOR)
  );

  const { dynamicAngleRange, positioning } = calculateAngleRangeForNodeCount(nodeCount);

  return {
    radiusX,
    radiusY,
    centerX,
    centerY,
    nodeCount,
    dynamicAngleRange,
    positioning,
    arcCenter: LAYOUT_CONSTANTS.ARC_CENTER,
    levelConfig
  };
}

export function calculatePositions(processedData, layout, levelConfig = DEFAULT_LEVEL_CONFIG) {
  const { centerX, centerY, radiusX, radiusY, nodeCount, arcCenter } = layout;
  const { primaryNodes, aggregatorNodes, unifiedNodes } = processedData;

  // Calculate X positions for each level
  const levelXPositions = calculateLevelXPositions(levelConfig);

  // Calculate primary spread for reference (used by other layout strategies)
  const primaryLevel = levelConfig[0];
  const primaryYPositions = [];
  const primaryXPositions = [];
  const primaryAngles = [];

  // Calculate positions for the first level (typically primary with arc layout)
  if (primaryLevel.layout === 'arc') {
    // Use adaptive arc configuration based on node count
    const adaptiveRadiusY = calculateAdaptiveRadius(nodeCount, primaryLevel.arc);
    const adaptiveSpread = calculateAdaptiveSpread(nodeCount, primaryLevel.arc);
    const arcCenter = primaryLevel.arc?.center || LAYOUT_CONSTANTS.ARC_CENTER;

    for (let index = 0; index < primaryNodes.length; index++) {
      let spacing = 0;
      if (nodeCount > 1) {
        spacing = adaptiveSpread / (nodeCount - 1);
      }
      const centerOffset = (index - (nodeCount - 1) / 2) * spacing;
      const angle = arcCenter + centerOffset;
      const yPos = centerY + Math.sin(angle) * adaptiveRadiusY;
      const xPos = levelXPositions[primaryLevel.name] + Math.cos(angle) * radiusX;
      primaryYPositions.push(yPos);
      primaryXPositions.push(xPos);
      primaryAngles.push(angle);
    }
  }

  const primarySpread = primaryYPositions.length > 1 ?
    Math.max(...primaryYPositions) - Math.min(...primaryYPositions) : 0;
  const primaryXSpread = primaryXPositions.length > 1 ?
    Math.max(...primaryXPositions) - Math.min(...primaryXPositions) : 0;
  const minMaxPrimaryX = primaryXPositions.length > 0 ? {
    min: Math.min(...primaryXPositions),
    max: Math.max(...primaryXPositions)
  } : { min: 0, max: 0 };

  // Add data for each level to the positions object
  const levelData = {};
  levelConfig.forEach(level => {
    levelData[`${level.name}Nodes`] = processedData[`${level.name}Nodes`] || [];
  });

  return {
    primaryYPositions,
    primaryXPositions,
    primaryAngles,
    primarySpread,
    primaryXSpread,
    minMaxPrimaryX,
    levelXPositions,
    ...levelData
  };
}