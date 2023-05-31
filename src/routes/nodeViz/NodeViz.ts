import { PIXI, Viewport } from '../../viz/RNNViz/pixi';
import type { RNNGraph, SparseNeuron } from '../rnn/graph';
import { getColor } from './ColorScale';
import { parseGraphvizPlainExt, type Point } from './plainExtParsing';

const Conf = {
  LabelColor: 0xffffff,
  NodeLabelFontSize: 30,
  EdgeLabelFontSize: 20,
  WorldWidth: 2000,
  WorldHeight: 2000,
  EdgeWidth: 3,
  ArrowheadSize: 24,
  ArrowheadHeightRatio: 0.6,
  TextResolution: 4,
};

class Edge {
  public graphics: PIXI.Graphics;
  private labelText: PIXI.Text;
  private controlPoints: Point[];
  private start: Point;
  private end: Point;
  private txNode: VizNode;
  private color: number;

  constructor(
    controlPoints: Point[],
    start: Point,
    end: Point,
    label: { text: string; position: Point; fontSize: number; weight: number },
    lineWidth: number,
    txNode: VizNode
  ) {
    this.graphics = new PIXI.Graphics();
    this.controlPoints = controlPoints;
    this.start = start;
    this.end = end;
    this.txNode = txNode;
    this.color = getColor(txNode.inner.getOutput() * label.weight);

    this.labelText = new PIXI.Text(
      label.text,
      new PIXI.TextStyle({ fontSize: label.fontSize, fill: Conf.LabelColor })
    );
    this.labelText.resolution = Conf.TextResolution;
    this.labelText.texture.baseTexture.mipmap = PIXI.MIPMAP_MODES.ON;
    // Label position is center, but we want top left
    const labelX = label.position.x - this.labelText.width / 2;
    const labelY = label.position.y - label.fontSize / 2;
    this.labelText.position.set(labelX, labelY);
    this.graphics.addChild(this.labelText);

    this.drawSpline(this.color, lineWidth);
  }

  private drawSpline(color: number, lineWidth: number): void {
    this.graphics.clear();
    this.graphics.lineStyle(lineWidth, color);

    this.graphics.moveTo(this.start.x, this.start.y);
    for (let i = 0; i < this.controlPoints.length - 3; i += 3) {
      this.graphics.bezierCurveTo(
        this.controlPoints[i].x,
        this.controlPoints[i].y,
        this.controlPoints[i + 1].x,
        this.controlPoints[i + 1].y,
        this.controlPoints[i + 2].x,
        this.controlPoints[i + 2].y
      );
    }
    this.graphics.lineTo(this.end.x, this.end.y);

    this.drawArrowHead(this.end, this.controlPoints[this.controlPoints.length - 1], color);
  }

  private drawArrowHead(end: Point, prev: Point, color: number): void {
    const angle = Math.atan2(end.y - prev.y, end.x - prev.x);

    const point1 = {
      x: end.x - Conf.ArrowheadSize * Math.cos(angle - (Math.PI / 6) * Conf.ArrowheadHeightRatio),
      y: end.y - Conf.ArrowheadSize * Math.sin(angle - (Math.PI / 6) * Conf.ArrowheadHeightRatio),
    };
    const point2 = {
      x: end.x - Conf.ArrowheadSize * Math.cos(angle + (Math.PI / 6) * Conf.ArrowheadHeightRatio),
      y: end.y - Conf.ArrowheadSize * Math.sin(angle + (Math.PI / 6) * Conf.ArrowheadHeightRatio),
    };

    this.graphics.lineStyle(1, color);
    this.graphics.beginFill(color);
    this.graphics.drawPolygon([end.x, end.y, point1.x, point1.y, point2.x, point2.y]);
    this.graphics.endFill();
  }

  public update(): void {
    this.color = this.txNode.getColor();
    this.drawSpline(this.color, 1);
  }
}

const getNodeLabelFromNodeID = (nodeID: string): string => {
  // TODO
  return nodeID;
};

class VizNode {
  public name: string;
  public x: number;
  public y: number;
  public width: number;
  public height: number;
  public inner: SparseNeuron;
  public graphics: PIXI.Graphics = new PIXI.Graphics();

  constructor(
    name: string,
    x: number,
    y: number,
    width: number,
    height: number,
    inner: SparseNeuron,
    onSelect: (node: VizNode) => void
  ) {
    this.name = name;
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
    this.inner = inner;

    const label = new PIXI.Text(
      getNodeLabelFromNodeID(name),
      new PIXI.TextStyle({ fontSize: Conf.NodeLabelFontSize, fill: Conf.LabelColor })
    );
    label.resolution = Conf.TextResolution;
    label.texture.baseTexture.mipmap = PIXI.MIPMAP_MODES.ON;
    label.position.set(x + width / 2 - label.width / 2, y + height / 2 - label.height / 2);
    this.graphics.addChild(label);
    this.graphics.interactive = true;
    this.graphics.cursor = 'pointer';
    this.graphics.on('pointerdown', () => onSelect(this));
    this.render();
  }

  private render() {
    this.graphics.clear();
    const value = this.inner.getOutput();
    const color = getColor(value);

    this.graphics.lineStyle(2, 0xffffff);
    this.graphics.beginFill(color);
    this.graphics.drawRect(this.x, this.y, this.width, this.height);
    this.graphics.endFill();
  }

  public getColor(): number {
    return getColor(this.inner.getOutput());
  }
}

export class NodeViz {
  private app: PIXI.Application;
  private graph: RNNGraph;
  private nodes: VizNode[] = [];
  private edges: Edge[] = [];
  private destroyed = false;

  constructor(canvas: HTMLCanvasElement, graphvizLayoutData: string, graph: RNNGraph) {
    this.graph = graph;
    this.app = new PIXI.Application({
      antialias: true,
      resolution: window.devicePixelRatio,
      autoDensity: true,
      view: canvas,
      height: canvas.height,
      width: canvas.width,
      backgroundColor: 0,
    });
    // TODO: Handle resize
    const viewport = new Viewport({
      screenWidth: canvas.width,
      screenHeight: canvas.height,
      worldWidth: Conf.WorldWidth,
      worldHeight: Conf.WorldHeight,
      events: this.app.renderer.events,
    });
    viewport.drag().pinch().wheel();
    this.app.stage.addChild(viewport);

    const { positionByNodeID, edges } = parseGraphvizPlainExt(
      graphvizLayoutData,
      Conf.WorldWidth,
      Conf.WorldHeight
    );
    const nodesByID = new Map<string, VizNode>();

    for (const [nodeID, { pos, width, height }] of positionByNodeID) {
      const node = graph.allConnectedNeuronsByID.get(nodeID);
      if (!node) {
        throw new Error(`Node ${nodeID} not found in graph`);
      }

      const vizNode = new VizNode(nodeID, pos.x, pos.y, width, height, node, node =>
        this.handleNodeSelect(node)
      );
      this.nodes.push(vizNode);
      nodesByID.set(nodeID, vizNode);
    }

    for (const edge of edges) {
      const labelText = edge.weight.toFixed(Math.round(edge.weight) === edge.weight ? 0 : 2);
      const txNode = nodesByID.get(edge.tx);
      if (!txNode) {
        throw new Error(`Node ${edge.tx} not found in graph`);
      }

      const vizEdge = new Edge(
        edge.controlPoints,
        edge.start,
        edge.end,
        {
          fontSize: Conf.EdgeLabelFontSize,
          position: edge.labelPosition,
          text: labelText,
          weight: edge.weight,
        },
        Conf.EdgeWidth,
        txNode
      );
      this.edges.push(vizEdge);
    }

    // Add edges first so that they are behind nodes
    for (const edge of this.edges) {
      viewport.addChild(edge.graphics);
    }
    for (const node of this.nodes) {
      viewport.addChild(node.graphics);
    }
  }

  private handleNodeSelect(node: VizNode) {
    // TODO
    console.log(node);
  }

  public destroy() {
    if (this.destroyed) {
      console.warn('NodeViz already destroyed');
      return;
    }
    this.destroyed = true;
    this.app.destroy(false, { children: true, texture: true, baseTexture: true });
  }
}
