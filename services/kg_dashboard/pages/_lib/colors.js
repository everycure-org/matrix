export const sourceColorMap = {
  // Base sources - your preferred colors
  "robokop": "#8be9fd",                  // cyan (keeping)
  "rtxkg2": "#bd93f9",                   // purple (keeping)
  "ec_medical": "#ff79c6",               // pink (keeping)

  // Two-way combinations - complementary colors with good contrast
  "robokop,rtxkg2": "#00d4aa",           // teal (cyan + purple blend)
  "ec_medical,robokop": "#ff6b35",       // coral orange (warm contrast to cool cyan/pink)
  "rtxkg2,ec_medical": "#a78bfa",        // lighter purple (purple + pink blend)

  // Three-way combination - distinctive accent
  "ec_medical,robokop,rtxkg2": "#fbbf24" // golden yellow (warm, highly visible)
};


// Define a consistent order for data sources
export const sourceOrder = [
  "ec_medical",
  "robokop",
  "rtxkg2",
  "ec_medical,robokop",
  "robokop,rtxkg2",
  "rtxkg2,ec_medical",
  "ec_medical,robokop,rtxkg2"
];

// Function to get colors in a consistent order
export function getOrderedColors(uniqueSources) {
  // Sort sources based on our predefined order
  const orderedSources = uniqueSources.sort((a, b) => {
    const aIndex = sourceOrder.indexOf(a);
    const bIndex = sourceOrder.indexOf(b);
    // If not found in order, put at end
    if (aIndex === -1) return 1;
    if (bIndex === -1) return -1;
    return aIndex - bIndex;
  });

  return orderedSources.map(source => sourceColorMap[source] || "#6272a4");
}