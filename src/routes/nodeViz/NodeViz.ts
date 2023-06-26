import { writable, type Writable } from 'svelte/store';
import { PIXI, Viewport, GlowFilter } from '../../viz/pixi';
import {
  StateNeuron,
  type RNNGraph,
  type SparseNeuron,
  InputNeuron,
  OutputNeuron,
} from '../rnn/graph';
import { getColor } from './ColorScale';
import { parseGraphvizPlainExt, type Point } from './plainExtParsing';

const Conf = {
  LabelColor: 0xffffff,
  NodeLabelFontSize: 30,
  EdgeLabelFontSize: 20,
  WorldWidth: 2000,
  WorldHeight: 2000,
  EdgeWidth: 3,
  ArrowheadSize: 20,
  ArrowheadHeightRatio: 0.65,
  TextResolution: 4,
};

class VizEdge {
  public graphics: PIXI.Graphics;
  private labelText: PIXI.Text;
  private controlPoints: Point[];
  private txNode: VizNode;
  private color: number;
  private weight: number;

  constructor(
    controlPoints: Point[],
    label: { text: string; position: Point; fontSize: number; weight: number },
    txNode: VizNode,
    onSelect: (edge: VizEdge) => void
  ) {
    this.graphics = new PIXI.Graphics();
    this.controlPoints = controlPoints;
    this.txNode = txNode;
    this.weight = label.weight;
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

    this.graphics.interactive = true;
    this.graphics.cursor = 'pointer';
    this.graphics.on('pointerdown', evt => {
      onSelect(this);
      evt.stopPropagation();
    });

    this.drawSpline(this.color);

    // Adapted from excellent solution by @SignDawn / @oushu1liangqi1:
    // https://github.com/pixijs/pixijs/issues/7058#issuecomment-1385224212
    let linePoly: PIXI.Polygon | undefined;
    let arrowheadPoly: PIXI.Polygon | undefined;
    this.graphics.hitArea = {
      contains: (x: number, y: number) => {
        if (!this.graphics.geometry.points.length) {
          return false;
        }

        if (!linePoly || !arrowheadPoly) {
          const points = this.graphics.geometry.points;
          const odd: { x: number; y: number; z: number }[] = [];
          const even: { x: number; y: number; z: number }[] = [];

          for (let index = 0; index * 2 < points.length - 6; index++) {
            const x = points[index * 2];
            const y = points[index * 2 + 1];
            const z = points[index * 2 + 2];
            if (index % 2 === 0) {
              odd.push({ x, y, z });
            } else {
              even.push({ x, y, z });
            }
          }
          linePoly = new PIXI.Polygon([...odd, ...even.reverse()]);

          const arrowheadPoints = points.slice(-6);
          arrowheadPoly = new PIXI.Polygon(arrowheadPoints);
        }

        return linePoly.contains(x, y) || arrowheadPoly.contains(x, y);
      },
    };
  }

  private drawSpline(color: number): void {
    if (this.controlPoints.length < 4) {
      throw new Error('At least 4 control points are required.');
    }

    if (this.controlPoints.length % 3 !== 1) {
      throw new Error('Invalid number of control points; expected 3n + 1');
    }

    this.graphics.clear();
    this.graphics.lineStyle(Conf.EdgeWidth, color);

    const start = this.controlPoints[0];
    this.graphics.moveTo(start.x, start.y);

    const startPoint = this.controlPoints[0];
    this.graphics.moveTo(startPoint.x, startPoint.y);

    let p1: Point, p2: Point, p3: Point;
    for (let i = 1; i < this.controlPoints.length; i += 3) {
      p1 = this.controlPoints[i];
      p2 = this.controlPoints[i + 1];
      p3 = this.controlPoints[i + 2];
      this.graphics.bezierCurveTo(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
    }

    this.drawArrowHead(p3!, p2!, p1!, this.color);
  }

  private drawArrowHead(end: Point, prev1: Point, prev2: Point, color: number): void {
    // Computes the direction of the last bezier curve in the spline to determine
    // the arrowhead direction
    const dx = end.x - prev1.x;
    const dy = end.y - prev1.y;
    const angle = Math.atan2(dy, dx);

    const point1 = {
      x: end.x - Conf.ArrowheadSize * Math.cos(angle - (Math.PI / 6) * Conf.ArrowheadHeightRatio),
      y: end.y - Conf.ArrowheadSize * Math.sin(angle - (Math.PI / 6) * Conf.ArrowheadHeightRatio),
    };
    const point2 = {
      x: end.x - Conf.ArrowheadSize * Math.cos(angle + (Math.PI / 6) * Conf.ArrowheadHeightRatio),
      y: end.y - Conf.ArrowheadSize * Math.sin(angle + (Math.PI / 6) * Conf.ArrowheadHeightRatio),
    };

    this.graphics.lineStyle(0, color);
    this.graphics.beginFill(color);
    this.graphics.drawPolygon([end.x, end.y, point1.x, point1.y, point2.x, point2.y]);
    const poly = new PIXI.Polygon([end.x, end.y, point1.x, point1.y, point2.x, point2.y]);
    this.graphics.drawPolygon(poly);
    this.graphics.endFill();
  }

  public update(): void {
    this.color = getColor(this.txNode.inner.getOutput() * this.weight);
    this.drawSpline(this.color);
  }

  public onSelect() {
    // Add a glow effect
    this.graphics.filters = [
      new GlowFilter({
        alpha: 0.7,
        color: 0xffffff,
        distance: 40,
        outerStrength: 4,
      }),
    ];
  }

  public onDeselect() {
    this.graphics.filters = [];
  }
}

export class VizNode {
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
    labelText: string,
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
      labelText,
      new PIXI.TextStyle({ fontSize: Conf.NodeLabelFontSize, fill: Conf.LabelColor })
    );
    label.resolution = Conf.TextResolution;
    label.texture.baseTexture.mipmap = PIXI.MIPMAP_MODES.ON;
    label.position.set(x + width / 2 - label.width / 2, y + height / 2 - label.height / 2);
    this.graphics.addChild(label);
    this.graphics.interactive = true;
    this.graphics.cursor = 'pointer';
    this.graphics.on('pointerdown', evt => {
      onSelect(this);
      evt.stopPropagation();
    });
    this.update();
  }

  public update() {
    this.graphics.clear();
    const value = this.inner.getOutput();
    const color = getColor(value);

    const shape: 'circle' | 'square' = (() => {
      if (
        this.inner instanceof StateNeuron ||
        this.inner instanceof InputNeuron ||
        this.inner instanceof OutputNeuron
      ) {
        return 'circle';
      }
      return 'square';
    })();

    this.graphics.lineStyle(2, 0xffffff);
    if (shape === 'square') {
      this.graphics.beginFill(color);
      this.graphics.drawRect(this.x, this.y, this.width, this.height);
      this.graphics.endFill();
    } else if (shape === 'circle') {
      this.graphics.beginFill(color);
      this.graphics.drawCircle(this.x + this.width / 2, this.y + this.height / 2, this.width / 2);
      this.graphics.endFill();
    } else {
      throw new Error('Invalid shape: ' + shape);
    }
  }

  public getColor(): number {
    return getColor(this.inner.getOutput());
  }

  public onSelect() {
    // Add a glow effect
    this.graphics.filters = [
      new GlowFilter({
        alpha: 0.8,
        color: 0xffffff,
        distance: this.width,
        outerStrength: 1.8,
      }),
    ];
  }

  public onDeselect() {
    this.graphics.filters = [];
  }
}

export class NodeViz {
  private app: PIXI.Application;
  private viewport: Viewport;
  private didSetInitialZoom = false;
  private graph: RNNGraph;
  public nodes: VizNode[] = [];
  private edges: VizEdge[] = [];
  private destroyed = false;
  public selected: Writable<VizNode | VizEdge | null> = writable(null);

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
    this.viewport = new Viewport({
      screenWidth: canvas.width,
      screenHeight: canvas.height,
      worldWidth: Conf.WorldWidth,
      worldHeight: Conf.WorldHeight,
      events: this.app.renderer.events,
    });
    this.viewport.drag().pinch().wheel();
    this.app.stage.addChild(this.viewport);

    // Handle background clicks
    this.viewport.interactive = true;
    this.viewport.cursor = 'default';
    this.viewport.on('pointerdown', () => void this.handleBackgroundClick());

    const { nodes, edges } = parseGraphvizPlainExt(
      graphvizLayoutData,
      Conf.WorldWidth,
      Conf.WorldHeight,
      graph
    );
    const nodesByID = new Map<string, VizNode>();

    for (const [nodeID, { pos, width, height, label }] of nodes) {
      const node = graph.allConnectedNeuronsByID.get(nodeID);
      if (!node) {
        throw new Error(`Node ${nodeID} not found in graph`);
      }

      const vizNode = new VizNode(nodeID, pos.x, pos.y, width, height, label, node, node =>
        this.handleNodeSelect(node)
      );
      this.nodes.push(vizNode);
      nodesByID.set(nodeID, vizNode);
    }

    for (const edge of edges) {
      let labelText = edge.weight.toFixed(Math.round(edge.weight) === edge.weight ? 0 : 2);
      // trim trailing zeros
      labelText = labelText.replace(/\.?0+$/, '');
      const txNode = nodesByID.get(edge.tx);
      if (!txNode) {
        throw new Error(`Node ${edge.tx} not found in graph`);
      }

      const vizEdge = new VizEdge(
        edge.controlPoints,
        {
          fontSize: Conf.EdgeLabelFontSize,
          position: edge.labelPosition,
          text: labelText,
          weight: edge.weight,
        },
        txNode,
        edge => this.handleEdgeSelect(edge)
      );
      this.edges.push(vizEdge);
    }

    const container = new PIXI.Container();
    this.viewport.addChild(container);

    // Add edges first so that they are behind nodes
    for (const edge of this.edges) {
      container.addChild(edge.graphics);
    }
    for (const node of this.nodes) {
      container.addChild(node.graphics);
    }
  }

  private handleNodeSelect(node: VizNode) {
    this.selected.update(selected => {
      selected?.onDeselect();
      node.onSelect();
      return node;
    });
  }

  private handleEdgeSelect(edge: VizEdge) {
    this.selected.update(selected => {
      selected?.onDeselect();
      edge.onSelect();
      return edge;
    });
  }

  private handleBackgroundClick() {
    this.selected.update(selected => {
      selected?.onDeselect();
      return null;
    });
  }

  private buildOneInputSeq(): Float32Array[] {
    const input = new Float32Array(this.graph.inputDim);
    for (let i = 0; i < input.length; i++) {
      const val: 1 | -1 = Math.random() > 0.5 ? 1 : -1;
      input[i] = val;
    }
    return [input];
  }

  public reset() {
    this.graph.reset(this.buildOneInputSeq());
    this.update();
  }

  private update() {
    for (const node of this.nodes) {
      node.update();
    }
    for (const edge of this.edges) {
      edge.update();
    }
  }

  public progressTimestep() {
    const nextInput = this.buildOneInputSeq();
    this.graph.advanceSequence(nextInput);
    this.graph.evaluateOneTimestep();
    this.update();
  }

  public toggleSelecteNodeID(nodeID: string) {
    const node = this.nodes.find(node => node.name === nodeID);
    if (!node) {
      throw new Error(`Node ${nodeID} not found`);
    }
    this.selected.update(selected => {
      if (selected === node) {
        node.onDeselect();
        return null;
      } else {
        selected?.onDeselect();
        node.onSelect();
        return node;
      }
    });
  }

  public handleResize(newWidth: number, newHeight: number) {
    this.app.renderer.resize(newWidth, newHeight);
    this.viewport.resize(newWidth, newHeight, Conf.WorldWidth, Conf.WorldHeight);

    // Need to do this here since apparently resizing right away clobbers the zoom
    if (!this.didSetInitialZoom) {
      this.didSetInitialZoom = true;
      const container = this.viewport.children[0];
      const bbox = container.getLocalBounds();
      const center = new PIXI.Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
      const zoom =
        Math.min(this.viewport.worldWidth / bbox.width, this.viewport.worldHeight / bbox.height) *
        0.7;
      console.log({ container, bbox, zoom, center });
      this.viewport.setZoom(zoom, true);
      this.viewport.moveCenter(center);
    }
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
