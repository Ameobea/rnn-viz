export interface NodeVizEdge {
  tx: string;
  rx: string;
  weight: number;
  controlPoints: Point[];
  labelPosition: Point;
}

export interface Point {
  x: number;
  y: number;
}

export interface NodeVizLayout {
  positionByNodeID: Map<string, { pos: Point; width: number; height: number; label: string }>;
  edges: NodeVizEdge[];
}

export const parseGraphvizPlainExt = (
  input: string,
  worldWidth: number,
  worldHeight: number
): NodeVizLayout => {
  const positionByNodeID = new Map<
    string,
    { pos: Point; width: number; height: number; label: string }
  >();
  const edges: NodeVizEdge[] = [];

  const lines = input.split('\n');

  let graphWidth = 0;
  let graphHeight = 0;
  let scale = 1;

  for (const line of lines) {
    const parts = line.split(' ');
    if (parts[0] === 'graph') {
      graphWidth = parseFloat(parts[2]);
      graphHeight = parseFloat(parts[3]);
      scale = Math.min(worldWidth / graphWidth, worldHeight / graphHeight);
    } else if (parts[0] === 'node') {
      const width = parseFloat(parts[4]) * scale;
      const height = parseFloat(parts[5]) * scale;
      const label = parts[6].replace(/\\n/g, '\n');

      // Pos of node is center, but we want top left
      const centerY = graphHeight - parseFloat(parts[3]);
      const pos = {
        x: parseFloat(parts[2]) * scale - width / 2,
        y: centerY * scale - height / 2,
      };

      positionByNodeID.set(parts[1], { pos, width, height, label });
    } else if (parts[0] === 'edge') {
      const controlPoints: Point[] = [];
      const controlPointCount = parseInt(parts[3], 10);
      for (let i = 4; i < 4 + controlPointCount * 2; i += 2) {
        controlPoints.push({
          x: parseFloat(parts[i]) * scale,
          y: (graphHeight - parseFloat(parts[i + 1])) * scale,
        });
      }

      const weight = parseFloat(parts[4 + controlPointCount * 2]);
      const labelPosition = {
        x: parseFloat(parts[4 + controlPointCount * 2 + 1]) * scale,
        y: (graphHeight - parseFloat(parts[4 + controlPointCount * 2 + 2])) * scale,
      };

      const edge: NodeVizEdge = {
        tx: parts[1],
        rx: parts[2],
        weight,
        controlPoints: controlPoints,
        labelPosition,
      };

      edges.push(edge);
    }
  }

  return { positionByNodeID, edges };
};
