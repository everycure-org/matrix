import { COLORS, EDGE_CONSTANTS } from './constants.js';
import { addTransparencyToColor } from './utils.js';

export function createLinks(processedData, nodes) {
  const { linkData } = processedData;

  return linkData
    .filter(d => d.source && d.target && d.value)
    .map(d => {
      let edgeColor = COLORS.DEFAULT_EDGE;

      const sourceNode = nodes.find(n => n.id === d.source);
      const targetNode = nodes.find(n => n.id === d.target);

      // Find aggregator node by category to use its color for the edge
      let aggregatorNode = null;
      if (sourceNode && sourceNode.nodeCategory === 'aggregator') {
        aggregatorNode = sourceNode;
      }
      else if (targetNode && targetNode.nodeCategory === 'aggregator') {
        aggregatorNode = targetNode;
      }

      // Apply aggregator color with proper transparency
      if (aggregatorNode && aggregatorNode.itemStyle && aggregatorNode.itemStyle.color) {
        edgeColor = addTransparencyToColor(aggregatorNode.itemStyle.color, EDGE_CONSTANTS.AGGREGATOR_ALPHA);
      }

      return {
        source: d.source,
        target: d.target,
        value: d.value,
        lineStyle: {
          width: Math.max(1, Math.min(10, Math.sqrt(d.value / 10000) * 5 + 1)),
          color: edgeColor
        },
        label: {
          show: false,
          position: 'middle',
          formatter: function(params) {
            return params.value.toLocaleString();
          },
          fontSize: 10,
          color: '#333',
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          borderRadius: 3,
          padding: [2, 4]
        },
        emphasis: {
          label: {
            show: true
          }
        }
      };
    });
}