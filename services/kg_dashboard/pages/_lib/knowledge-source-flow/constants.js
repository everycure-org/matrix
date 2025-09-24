export const LAYOUT_CONSTANTS = {
  OVAL_RADIUS_X: 150,
  MIN_RADIUS_Y: 300,
  MAX_RADIUS_Y: 600,
  RADIUS_Y_SCALE_FACTOR: 25,
  CENTER_X: 300,
  MIN_CENTER_Y: 200,
  MAX_CENTER_Y: 400,
  CENTER_Y_BASE: 150,
  CENTER_Y_SCALE_FACTOR: 8,
  MAX_TOTAL_SPREAD: Math.PI * 0.4, // 72 degrees - narrower left side arc
  MIN_SPACING: Math.PI * 0.05, // Smaller minimum spacing to allow more spread
  ARC_CENTER: Math.PI, // 180 degrees - pointing straight left for left-side arc
  TOTAL_WIDTH: 450,
  VERTICAL_SPACING_FRACTION: 0.5,
  MIN_VERTICAL_SPACING: 12,
  DEFAULT_VERTICAL_SPACING: 80
};

export const DEFAULT_LEVEL_CONFIG = [
  {
    name: 'primary',
    layout: 'arc',
    label: { position: 'left', fontSize: 12, fontWeight: 'normal', distance: 8 },
    arc: {
      spread: Math.PI * 0.6, // 108 degrees - configurable arc spread
      center: Math.PI, // 180 degrees - arc center position
      radiusScaling: {
        method: 'logarithmic',  // use logarithmic scaling for more aggressive scaling
        reference: 10,          // lower reference for more aggressive scaling
        minScale: 0.5,          // minimum scale factor (50% of max radius)
        maxScale: 1.0           // maximum scale factor (100% of max radius)
      },
      spreadScaling: {
        method: 'logarithmic',  // spread scales logarithmically with node count
        reference: 25,          // reference node count
        minScale: 0.4,          // minimum spread scale (40% of configured spread)
        maxScale: 1.0           // maximum spread scale (100% of configured spread)
      }
    }
  },
  {
    name: 'aggregator',
    layout: 'vertical',
    label: { position: 'inside', fontSize: 11, fontWeight: 'bold', distance: 0 }
  },
  {
    name: 'unified',
    layout: 'vertical',
    label: { position: 'inside', fontSize: 11, fontWeight: 'bold', distance: 0 }
  }
];

export const NODE_SIZE_CONSTANTS = {
  PRIMARY_MIN_SIZE: 8,
  PRIMARY_MAX_SIZE: 35,
  PRIMARY_SCALE_DIVISOR: 100000,
  PRIMARY_SCALE_MULTIPLIER: 3,
  PRIMARY_BASE_SIZE: 8,
  UNIFIED_MIN_SIZE: 60,
  UNIFIED_MAX_SIZE: 100,
  UNIFIED_SCALE_DIVISOR: 1000000,
  UNIFIED_SCALE_MULTIPLIER: 0.8,
  UNIFIED_BASE_SIZE: 60,
  AGGREGATOR_MIN_SIZE: 45,
  AGGREGATOR_MAX_SIZE: 80,
  AGGREGATOR_SCALE_DIVISOR: 1000000,
  AGGREGATOR_SCALE_MULTIPLIER: 0.6,
  AGGREGATOR_BASE_SIZE: 45
};

export const HEIGHT_CONSTANTS = {
  BASE_HEIGHT: 300,      // Minimum height for 1 source
  MAX_HEIGHT: 900,       // Target height for 25 sources
  MAX_SOURCES: 25        // Reference point for maximum scaling
};

export const COLORS = {
  PRIMARY_NODE: '#88C0D0',
  PRIMARY_LABEL: '#333',
  NON_PRIMARY_LABEL: '#fff',
  DEFAULT_EDGE: 'rgba(157, 121, 214, 0.6)' // Default purple edge color
};

export const EDGE_CONSTANTS = {
  AGGREGATOR_ALPHA: 0.5, // 50% transparency for aggregator-colored edges
  DEFAULT_ALPHA: 0.6     // Default edge transparency
};