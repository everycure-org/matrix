import { COLORS, LAYOUT_CONSTANTS, NODE_SIZE_CONSTANTS } from './constants.js';

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

export function calculateNodePosition(nodeData, category, counters, layout, positions) {
  const { centerX, centerY, radiusX, radiusY, nodeCount, arcCenter } = layout;
  const { primarySpread } = positions;

  if (category === 'primary') {
    let spacing = 0;
    if (nodeCount > 1) {
      spacing = Math.min(LAYOUT_CONSTANTS.MIN_SPACING, LAYOUT_CONSTANTS.MAX_TOTAL_SPREAD / (nodeCount - 1));
    }
    const centerOffset = (counters.primary - (nodeCount - 1) / 2) * spacing;
    const angle = arcCenter + centerOffset;

    return {
      x: centerX + Math.cos(angle) * radiusX,
      y: centerY + Math.sin(angle) * radiusY
    };
  } else if (category === 'aggregator') {
    const aggregatorNodes = positions.aggregatorNodes || [];
    let aggregatorSpacing = 0;

    if (aggregatorNodes.length > 1 && primarySpread > 0) {
      const calculatedSpacing = (primarySpread * LAYOUT_CONSTANTS.AGGREGATOR_SPACING_FRACTION) / (aggregatorNodes.length - 1);
      aggregatorSpacing = Math.max(calculatedSpacing, LAYOUT_CONSTANTS.MIN_AGGREGATOR_SPACING);
    }

    const centerOffset = (counters.aggregator - (aggregatorNodes.length - 1) / 2) * aggregatorSpacing;

    return {
      x: centerX + radiusX - LAYOUT_CONSTANTS.AGGREGATOR_X_OFFSET,
      y: centerY + centerOffset
    };
  } else if (category === 'unified') {
    const unifiedNodes = positions.unifiedNodes || [];
    const centerOffset = (counters.unified - (unifiedNodes.length - 1) / 2) * LAYOUT_CONSTANTS.UNIFIED_SPACING;

    return {
      x: centerX + radiusX - LAYOUT_CONSTANTS.UNIFIED_X_OFFSET,
      y: centerY + centerOffset
    };
  }

  return { x: 0, y: 0 };
}

export function formatTooltip(params, links) {
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
    // Edge tooltip
    return `${params.data.source.replace('infores:', '')} → ${params.data.target.replace('infores:', '')}<br/>Connections: ${params.data.value.toLocaleString()}`;
  }

  return '';
}