// Constants for the key node chord diagram visualization

export const LAYOUT_CONSTANTS = {
  CENTER_X: 350,
  CENTER_Y: 200,
  CENTER_NODE_SIZE: 50,
  // Oval shape (280x150) instead of circle reduces vertical space
  // while maintaining good node separation and readability
  OUTER_RADIUS_X: 280,  // Wider radius for oval shape
  OUTER_RADIUS_Y: 150,  // Shorter radius for oval shape
  CATEGORY_NODE_SIZE: 40,
  LINK_CURVENESS: 0,    // Straight lines for cleaner look
  CANVAS_WIDTH: 700,
  CANVAS_HEIGHT: 400
};

// Color palette for different biolink categories
export const CATEGORY_COLORS = {
  'ChemicalEntity': '#51AFA6',      // teal - drugs/chemicals
  'Gene': '#4C73F0',                // blue - genes
  'Protein': '#604CEF',             // purple - proteins
  'Disease': '#EF604C',             // red - diseases
  'PhenotypicFeature': '#EF4C9E',   // pink - phenotypes
  'AnatomicalEntity': '#B4D843',    // lime green - anatomy
  'BiologicalProcess': '#3AC7B6',   // cyan - processes
  'MolecularActivity': '#D8AB47',   // gold - molecular activity
  'GenomicEntity': '#6287D3',       // light blue - genomic
  'OrganismTaxon': '#B912D1',       // magenta - organisms
  'Procedure': '#C66D3C',           // burnt orange - procedures
  'Activity': '#E5F758',            // yellow-green - activities
  'Population': '#C56492',          // mauve - populations
  'Other': '#999999'                // gray - other
};

export const LABEL_CONFIG = {
  fontSize: 12,
  fontWeight: 'normal',
  color: '#333',
  distance: 8
};

export const SELECTION_STYLE = {
  selectedBorderWidth: 4,
  selectedBorderColor: '#000',
  normalBorderWidth: 0,
  selectedOpacity: 1.0,
  unselectedOpacity: 0.3
};
