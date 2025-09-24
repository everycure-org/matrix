import { HEIGHT_CONSTANTS } from './constants.js';

export function calculateDynamicHeight(primaryNodes, fallbackHeight) {
  if (!primaryNodes || primaryNodes.length === 0) {
    return fallbackHeight;
  }

  const scalingFactor = Math.min(primaryNodes.length, HEIGHT_CONSTANTS.MAX_SOURCES) / HEIGHT_CONSTANTS.MAX_SOURCES;
  const calculatedHeight = HEIGHT_CONSTANTS.BASE_HEIGHT + (scalingFactor * (HEIGHT_CONSTANTS.MAX_HEIGHT - HEIGHT_CONSTANTS.BASE_HEIGHT));

  return Math.round(calculatedHeight) + 'px';
}

