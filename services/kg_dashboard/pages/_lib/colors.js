export const sourceColorMap = {
  "robokop": "#1CEAD0",                   // cyan
  "rtxkg2": "#4C73F0",                    // lighter purple
  "ec_medical": "#EF4C9E",                // hot pink
  "robokop,rtxkg2": "#10D461",            // neon green
  "ec_medical,robokop": "#ED953C",        // coral orange
  "rtxkg2,ec_medical": "#604CEF",         // deep purple
  "ec_medical,robokop,rtxkg2": "#E0BC05", // golden yellow
  "disease_list": "#C56492" ,
  "drug_list": "#51AFA6",
  // Expansion palette for future use:
  // "#EF604C", // vibrant coral red
  // "#B4D843", // lime yellow-green
  // "#B912D1", // electric purple
  // "#A74CEF", // violet purple
  // "#3AC7B6", // teal
};

// '#6287D3', '#6065C1', '#4645A9', '#C74EDE',
// '#C56492', '#D8AB47', '#E5F758', '#51AFA6',
// '#4DCFB5', '#0E0F65', '#C66D3C', '#644DFF'


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