import { LAYOUT_CONSTANTS } from './constants.js';
import { calculateAngleRangeForNodeCount } from './utils.js';

export function calculateLayout(processedData) {
  const { primaryNodes, aggregatorNodes, unifiedNodes } = processedData;
  const totalNodes = primaryNodes.length + aggregatorNodes.length + unifiedNodes.length;
  const nodeCount = primaryNodes.length;

  const radiusX = LAYOUT_CONSTANTS.OVAL_RADIUS_X;
  const radiusY = Math.max(
    LAYOUT_CONSTANTS.MIN_RADIUS_Y,
    Math.min(LAYOUT_CONSTANTS.MAX_RADIUS_Y, totalNodes * LAYOUT_CONSTANTS.RADIUS_Y_SCALE_FACTOR)
  );
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
    arcCenter: LAYOUT_CONSTANTS.ARC_CENTER
  };
}

export function calculatePositions(processedData, layout) {
  const { primaryNodes, aggregatorNodes, unifiedNodes } = processedData;
  const { centerX, centerY, radiusX, radiusY, nodeCount, arcCenter } = layout;

  const primaryYPositions = [];
  const primaryXPositions = [];
  const primaryAngles = [];

  for (let index = 0; index < primaryNodes.length; index++) {
    let spacing = 0;
    if (nodeCount > 1) {
      spacing = Math.min(LAYOUT_CONSTANTS.MIN_SPACING, LAYOUT_CONSTANTS.MAX_TOTAL_SPREAD / (nodeCount - 1));
    }
    const centerOffset = (index - (nodeCount - 1) / 2) * spacing;
    const angle = arcCenter + centerOffset;
    const yPos = centerY + Math.sin(angle) * radiusY;
    const xPos = centerX + Math.cos(angle) * radiusX;
    primaryYPositions.push(yPos);
    primaryXPositions.push(xPos);
    primaryAngles.push(angle);
  }

  const primarySpread = primaryYPositions.length > 1 ?
    Math.max(...primaryYPositions) - Math.min(...primaryYPositions) : 0;
  const primaryXSpread = primaryXPositions.length > 1 ?
    Math.max(...primaryXPositions) - Math.min(...primaryXPositions) : 0;
  const minMaxPrimaryX = primaryXPositions.length > 0 ? {
    min: Math.min(...primaryXPositions),
    max: Math.max(...primaryXPositions)
  } : { min: 0, max: 0 };

  return {
    primaryYPositions,
    primaryXPositions,
    primaryAngles,
    primarySpread,
    primaryXSpread,
    minMaxPrimaryX,
    aggregatorNodes,
    unifiedNodes
  };
}