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

export const LABEL_CONFIG = {
  fontSize: 12,
  fontWeight: 'normal',
  color: 'inherit', // Inherits from parent to support dark mode
  distance: 8
};

export const SELECTION_STYLE = {
  selectedBorderWidth: 4,
  selectedBorderColor: '#000',
  normalBorderWidth: 0,
  selectedOpacity: 1.0,
  unselectedOpacity: 0.3
};
