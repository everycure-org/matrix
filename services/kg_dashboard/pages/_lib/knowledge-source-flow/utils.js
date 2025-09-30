import { COLORS, LAYOUT_CONSTANTS, NODE_SIZE_CONSTANTS, DEFAULT_LEVEL_CONFIG } from './constants.js';

export function addTransparencyToColor(color, alpha = 0.5) {
  if (!color) {
    return COLORS.DEFAULT_EDGE;
  }

  // Handle hex colors (e.g., '#FF5733')
  if (color.startsWith('#')) {
    const hex = color.slice(1);

    // Validate hex format
    if (!/^[0-9A-Fa-f]{6}$/.test(hex)) {
      console.warn(`Invalid hex color format: ${color}, using default`);
      return COLORS.DEFAULT_EDGE;
    }

    // Convert hex to RGB
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);

    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  // Handle rgb/rgba colors
  if (color.startsWith('rgb')) {
    // Extract RGB values
    const rgbMatch = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if (rgbMatch) {
      const [, r, g, b] = rgbMatch;
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    // Already rgba, replace alpha
    const rgbaMatch = color.match(/rgba\((\d+),\s*(\d+),\s*(\d+),\s*[\d.]+\)/);
    if (rgbaMatch) {
      const [, r, g, b] = rgbaMatch;
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
  }

  // Fallback for unsupported formats
  console.warn(`Unsupported color format: ${color}, using default`);
  return COLORS.DEFAULT_EDGE;
}

export function calculateAngleRangeForNodeCount(nodeCount) {
  if (nodeCount <= 2) {
    return {
      dynamicAngleRange: Math.PI * 0.5,
      positioning: 'small (≤2)'
    };
  } else if (nodeCount <= 5) {
    return {
      dynamicAngleRange: Math.PI * (2/3),
      positioning: 'small (3-5)'
    };
  } else if (nodeCount <= 10) {
    return {
      dynamicAngleRange: Math.PI * (5/6),
      positioning: 'medium (6-10)'
    };
  } else {
    return {
      dynamicAngleRange: Math.PI * 0.65,
      positioning: 'large (11+)'
    };
  }
}

export function calculateNodeSize(value, category) {
  const safeValue = value || 1;

  switch (category) {
    case 'primary':
      return Math.max(
        NODE_SIZE_CONSTANTS.PRIMARY_MIN_SIZE,
        Math.min(
          NODE_SIZE_CONSTANTS.PRIMARY_MAX_SIZE,
          Math.sqrt(safeValue / NODE_SIZE_CONSTANTS.PRIMARY_SCALE_DIVISOR) * NODE_SIZE_CONSTANTS.PRIMARY_SCALE_MULTIPLIER + NODE_SIZE_CONSTANTS.PRIMARY_BASE_SIZE
        )
      );
    case 'unified':
      return Math.max(
        NODE_SIZE_CONSTANTS.UNIFIED_MIN_SIZE,
        Math.min(
          NODE_SIZE_CONSTANTS.UNIFIED_MAX_SIZE,
          safeValue / NODE_SIZE_CONSTANTS.UNIFIED_SCALE_DIVISOR * NODE_SIZE_CONSTANTS.UNIFIED_SCALE_MULTIPLIER + NODE_SIZE_CONSTANTS.UNIFIED_BASE_SIZE
        )
      );
    case 'aggregator':
    default:
      return Math.max(
        NODE_SIZE_CONSTANTS.AGGREGATOR_MIN_SIZE,
        Math.min(
          NODE_SIZE_CONSTANTS.AGGREGATOR_MAX_SIZE,
          safeValue / NODE_SIZE_CONSTANTS.AGGREGATOR_SCALE_DIVISOR * NODE_SIZE_CONSTANTS.AGGREGATOR_SCALE_MULTIPLIER + NODE_SIZE_CONSTANTS.AGGREGATOR_BASE_SIZE
        )
      );
  }
}

const LAYOUT_STRATEGIES = {
  arc: (nodeIndex, nodeCount, layout, levelX, referenceSpread, levelConfig) => {
    const { centerY, radiusX } = layout;

    // Calculate adaptive values based on node count
    const adaptiveRadiusY = calculateAdaptiveRadius(nodeCount, levelConfig?.arc);
    const adaptiveSpread = calculateAdaptiveSpread(nodeCount, levelConfig?.arc);
    const arcCenter = levelConfig?.arc?.center || LAYOUT_CONSTANTS.ARC_CENTER;

    let spacing = 0;
    if (nodeCount > 1) {
      spacing = adaptiveSpread / (nodeCount - 1);
    }
    const centerOffset = (nodeIndex - (nodeCount - 1) / 2) * spacing;
    const angle = arcCenter + centerOffset;

    return {
      x: levelX + Math.cos(angle) * radiusX,
      y: centerY + Math.sin(angle) * adaptiveRadiusY
    };
  },

  vertical: (nodeIndex, nodeCount, layout, levelX, referenceSpread) => {
    const { centerY } = layout;

    let spacing = LAYOUT_CONSTANTS.DEFAULT_VERTICAL_SPACING;
    if (referenceSpread && nodeCount > 1) {
      const calculatedSpacing = (referenceSpread * LAYOUT_CONSTANTS.VERTICAL_SPACING_FRACTION) / (nodeCount - 1);
      spacing = Math.max(calculatedSpacing, LAYOUT_CONSTANTS.MIN_VERTICAL_SPACING);
    }

    const centerOffset = (nodeIndex - (nodeCount - 1) / 2) * spacing;

    return {
      x: levelX,
      y: centerY + centerOffset
    };
  },

  horizontal: (nodeIndex, nodeCount, layout, levelX) => {
    const { centerY } = layout;
    const spacing = 50; // Default horizontal spacing
    const centerOffset = (nodeIndex - (nodeCount - 1) / 2) * spacing;

    return {
      x: levelX + centerOffset,
      y: centerY
    };
  }
};

export function calculateNodePosition(nodeData, category, counters, layout, positions, levelConfig) {
  const levelInfo = levelConfig.find(level => level.name === category);
  if (!levelInfo) {
    return { x: 0, y: 0 };
  }

  const levelXPositions = positions.levelXPositions || {};
  const levelX = levelXPositions[category] || 0;

  const levelNodes = positions[`${category}Nodes`] || [];
  const nodeCount = levelNodes.length;
  const nodeIndex = counters[category];

  const strategy = LAYOUT_STRATEGIES[levelInfo.layout] || LAYOUT_STRATEGIES.vertical;

  // Use primary spread as reference for vertical layouts
  const referenceSpread = positions.primarySpread || 0;

  return strategy(nodeIndex, nodeCount, layout, levelX, referenceSpread, levelInfo);
}

export function calculateScaleFactor(nodeCount, scaling) {
  if (!scaling) return 1.0;

  const { method, reference, minScale, maxScale } = scaling;
  const ratio = nodeCount / reference;

  let scaleFactor;
  switch (method) {
    case 'logarithmic':
      // Logarithmic scaling: smooth transition, less aggressive than linear
      scaleFactor = Math.log(nodeCount + 1) / Math.log(reference + 1);
      break;
    case 'linear':
      // Linear scaling: direct proportional relationship
      scaleFactor = ratio;
      break;
    case 'proportional':
    default:
      // Proportional with square root smoothing for middle values
      scaleFactor = Math.sqrt(ratio);
      break;
  }

  // Clamp to min/max bounds
  return Math.max(minScale, Math.min(maxScale, scaleFactor));
}

export function calculateAdaptiveRadius(nodeCount, arcConfig, maxCanvasHeight = 400) {
  if (!arcConfig?.radiusScaling) {
    // Fallback to original calculation
    return Math.max(
      LAYOUT_CONSTANTS.MIN_RADIUS_Y,
      Math.min(LAYOUT_CONSTANTS.MAX_RADIUS_Y, nodeCount * LAYOUT_CONSTANTS.RADIUS_Y_SCALE_FACTOR)
    );
  }

  const scaleFactor = calculateScaleFactor(nodeCount, arcConfig.radiusScaling);

  // Simple approach: just make radius much smaller for few nodes
  if (nodeCount <= 3) {
    return Math.max(80, LAYOUT_CONSTANTS.MIN_RADIUS_Y * scaleFactor);
  }

  const baseRadius = LAYOUT_CONSTANTS.MIN_RADIUS_Y;
  const maxRadius = LAYOUT_CONSTANTS.MAX_RADIUS_Y;

  return baseRadius + (maxRadius - baseRadius) * scaleFactor;
}

export function calculateAdaptiveSpread(nodeCount, arcConfig) {
  if (!arcConfig?.spreadScaling) {
    return arcConfig?.spread || LAYOUT_CONSTANTS.MAX_TOTAL_SPREAD;
  }

  const scaleFactor = calculateScaleFactor(nodeCount, arcConfig.spreadScaling);
  const baseSpread = arcConfig.spread * arcConfig.spreadScaling.minScale;
  const maxSpread = arcConfig.spread;

  return baseSpread + (maxSpread - baseSpread) * scaleFactor;
}

export function calculateLevelXPositions(levelConfig, totalWidth = LAYOUT_CONSTANTS.TOTAL_WIDTH) {
  const numLevels = levelConfig.length;
  const positions = {};

  if (numLevels === 1) {
    positions[levelConfig[0].name] = totalWidth / 2;
  } else {
    levelConfig.forEach((level, index) => {
      positions[level.name] = (index / (numLevels - 1)) * totalWidth;
    });
  }

  return positions;
}

export function formatTooltip(params, links, nodes = []) {
  if (params.dataType === 'node' && params.data.nodeCategory === 'primary') {
    // Primary node tooltip with connections breakdown
    const sourceId = params.data.id;
    const connections = links.filter(link => link.source === sourceId);

    let tooltipContent = `<strong>${params.data.name}</strong><br/>`;
    tooltipContent += `${params.data.value.toLocaleString()} total connections`;

    if (connections.length > 1) {
      tooltipContent += `<br/><br/>`;
      connections.forEach(conn => {
        const targetName = conn.target.replace('infores:', '');
        tooltipContent += `${targetName}: ${conn.value.toLocaleString()}<br/>`;
      });
    }

    return tooltipContent;
  } else if (params.dataType === 'node') {
    // Aggregator and unified node tooltips with dual counts
    if (params.data.total_all_sources && params.data.total_all_sources !== params.data.value) {
      return `<strong>${params.data.id.replace('infores:', '')}</strong><br/>${params.data.value.toLocaleString()} from selected sources<br/>(${params.data.total_all_sources.toLocaleString()} from all sources)`;
    } else {
      return `<strong>${params.data.id.replace('infores:', '')}</strong><br/>${params.data.value.toLocaleString()} total connections`;
    }
  } else if (params.dataType === 'edge') {
    // Edge tooltip - use node display names instead of IDs
    const sourceNode = nodes.find(node => node.id === params.data.source);
    const targetNode = nodes.find(node => node.id === params.data.target);

    const sourceName = sourceNode ? sourceNode.name : params.data.source.replace('infores:', '');
    const targetName = targetNode ? targetNode.name : params.data.target.replace('infores:', '');

    return `${sourceName} → ${targetName}<br/>Connections: ${params.data.value.toLocaleString()}`;
  }

  return '';
}