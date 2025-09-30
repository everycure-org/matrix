// Base color palette for known sources
export const sourceColorMap = {
  "robokop": "#1CEAD0",                // cyan
  "rtxkg2": "#4C73F0",                 // lighter purple
  "primekg": "#EF4C9E",                // hot pink
  "robokop,rtxkg2": "#10D461",         // neon green
  "primekg,robokop": "#ED953C",        // coral orange
  "primekg,rtxkg2": "#604CEF",         // deep purple
  "primekg,robokop,rtxkg2": "#E0BC05", // golden yellow
  "disease_list": "#C56492",
  "drug_list": "#51AFA6",
};

// Fallback color palette for unknown sources
export const fallbackColors = [
  "#EF604C", // vibrant coral red
  "#B4D843", // lime yellow-green
  "#B912D1", // electric purple
  "#A74CEF", // violet purple
  "#3AC7B6", // teal
  "#6287D3", // blue
  "#D8AB47", // orange-yellow
  "#6065C1", // darker blue
  "#4645A9", // navy
  "#C74EDE", // magenta
  "#E5F758", // bright yellow-green
  "#4DCFB5", // mint
  "#0E0F65", // dark blue
  "#C66D3C", // burnt orange
  "#644DFF", // bright purple
];

// Define a consistent order for known data sources
export const sourceOrder = [
  "primekg",
  "robokop",
  "rtxkg2",
  "disease_list",
  "drug_list",
  "primekg,robokop",
  "robokop,rtxkg2",
  "rtxkg2,primekg",
  "primekg,robokop,rtxkg2"
];

// Generate a deterministic color for unknown sources
function generateColorForSource(source) {
  // Simple hash function to get consistent colors for the same source
  let hash = 0;
  for (let i = 0; i < source.length; i++) {
    const char = source.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }

  // Use absolute value and modulo to get a fallback color index
  const colorIndex = Math.abs(hash) % fallbackColors.length;
  return fallbackColors[colorIndex];
}

// Centralized sorting logic for sources
export function sortSourcesByOrder(sources) {
  return sources.sort((a, b) => {
    const aIndex = sourceOrder.indexOf(a);
    const bIndex = sourceOrder.indexOf(b);

    // Both are known sources
    if (aIndex !== -1 && bIndex !== -1) {
      return aIndex - bIndex;
    }

    // a is known, b is unknown - a comes first
    if (aIndex !== -1 && bIndex === -1) {
      return -1;
    }

    // a is unknown, b is known - b comes first
    if (aIndex === -1 && bIndex !== -1) {
      return 1;
    }

    // Both are unknown - sort alphabetically
    return a.localeCompare(b);
  });
}

// Sort data array by a series column using the source order
export function sortDataBySeriesOrder(data, seriesColumn) {
  return [...data].sort((a, b) => {
    const aSource = a[seriesColumn];
    const bSource = b[seriesColumn];

    const aIndex = sourceOrder.indexOf(aSource);
    const bIndex = sourceOrder.indexOf(bSource);

    if (aIndex !== -1 && bIndex !== -1) return aIndex - bIndex;
    if (aIndex !== -1) return -1;
    if (bIndex !== -1) return 1;

    // Both unknown → sort alphabetically
    return aSource.localeCompare(bSource);
  });
}

// Enhanced function to get colors in a consistent order
export function getOrderedColors(uniqueSources) {
  // Sort sources using the centralized logic
  const orderedSources = sortSourcesByOrder([...uniqueSources]);

  return orderedSources.map(source =>
    sourceColorMap[source] || generateColorForSource(source)
  );
}

export function getSeriesColors(data, seriesColumn) {
  const uniqueSources = [...new Set(data.map(row => row[seriesColumn]))];

  // Create an object mapping each series name to its color
  const seriesColors = {};
  const orderedColors = getOrderedColors(uniqueSources);

  // Sort sources using the centralized logic
  const orderedSources = sortSourcesByOrder([...uniqueSources]);

  // Map each source to its color
  orderedSources.forEach((source, index) => {
    seriesColors[source] = orderedColors[index];
  });

  console.log('Series color mapping:', seriesColors);
  return seriesColors;
}

// Utility function to add new known sources dynamically
export function addKnownSource(sourceName, color, insertIndex = null) {
  sourceColorMap[sourceName] = color;

  if (insertIndex !== null && insertIndex >= 0 && insertIndex <= sourceOrder.length) {
    sourceOrder.splice(insertIndex, 0, sourceName);
  } else {
    sourceOrder.push(sourceName);
  }
}

// Utility function to get color for a single source
export function getSourceColor(source) {
  return sourceColorMap[source] || generateColorForSource(source);
}

// Sort data by source column using consistent source ordering
export function sortDataBySource(data, sourceColumn) {
  return [...data].sort((a, b) => {
    const aSource = a[sourceColumn];
    const bSource = b[sourceColumn];

    const aIndex = sourceOrder.indexOf(aSource);
    const bIndex = sourceOrder.indexOf(bSource);

    // Both are known sources - use sourceOrder priority
    if (aIndex !== -1 && bIndex !== -1) {
      return aIndex - bIndex;
    }

    // Known source comes before unknown source
    if (aIndex !== -1 && bIndex === -1) {
      return -1;
    }

    // Unknown source comes after known source
    if (aIndex === -1 && bIndex !== -1) {
      return 1;
    }

    // Both unknown - sort alphabetically
    return aSource.localeCompare(bSource);
  });
}

// Function to preview color assignments for a list of sources
export function previewColorAssignments(sources) {
  const colors = getOrderedColors(sources);
  const preview = {};

  sources.forEach((source, index) => {
    preview[source] = colors[index];
  });

  return preview;
}