import { writable, type Writable } from 'svelte/store';
import {
  StateNeuron,
  type RNNGraph,
  type SparseNeuron,
  InputNeuron,
  OutputNeuron,
} from '../rnn/graph';
import { getColor } from './ColorScale';
import { parseGraphvizPlainExt, type Point } from './plainExtParsing';

import * as d3 from './d3';

const hexColorToCSS = (color: number): string => {
  const r = (color >> 16) & 0xff;
  const g = (color >> 8) & 0xff;
  const b = color & 0xff;
  return `rgb(${r}, ${g}, ${b})`;
};

const Conf = {
  LabelColor: '#ffffff',
  NodeLabelFontSize: 46,
  EdgeLabelFontSize: 20,
  WorldWidth: 2000,
  WorldHeight: 2000,
  EdgeWidth: 5,
  ArrowheadSize: 20,
  ArrowheadHeightRatio: 0.65,
  TextResolution: 4,
  PaddingPercent: 0.02,
  NodeBorderWidth: 4,
};

class VizEdge {
  public group: d3.Selection<SVGGElement, undefined, null, undefined>;
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
    this.group = d3.create('svg:g');
    this.controlPoints = controlPoints;
    this.txNode = txNode;
    this.weight = label.weight;
    this.color = getColor(txNode.inner.getOutput() * label.weight);

    if (label.text && !Number.isNaN(label.position.x) && !Number.isNaN(label.position.y)) {
      this.group
        .append('text')
        .text(label.text)
        .attr('x', label.position.x)
        .attr('y', label.position.y)
        .attr('font-size', label.fontSize)
        .attr('fill', Conf.LabelColor)
        .attr('text-anchor', 'middle');
    }

    this.group.on('click', () => onSelect(this));

    this.drawSpline(this.color);
  }

  private drawSpline(color: number): void {
    if (this.controlPoints.length < 4) {
      throw new Error('At least 4 control points are required.');
    }

    if (this.controlPoints.length % 3 !== 1) {
      throw new Error('Invalid number of control points; expected 3n + 1');
    }

    const start = this.controlPoints[0];

    const path = d3.path();
    path.moveTo(start.x, start.y);

    let p1: Point, p2: Point, p3: Point;
    for (let i = 1; i < this.controlPoints.length; i += 3) {
      p1 = this.controlPoints[i];
      p2 = this.controlPoints[i + 1];
      p3 = this.controlPoints[i + 2];
      path.bezierCurveTo(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
    }

    this.group
      .append('path')
      .attr('stroke', hexColorToCSS(color))
      .attr('stroke-width', Conf.EdgeWidth)
      .attr('fill', 'none')
      .attr('d', path.toString())
      .attr('stroke-linecap', 'round')
      .attr('stroke-linejoin', 'round')
      .attr('shape-rendering', 'geometricPrecision')
      .attr('cursor', 'pointer');

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

    this.group
      .append('path')
      .attr('d', `M ${end.x} ${end.y} L ${point1.x} ${point1.y} L ${point2.x} ${point2.y} Z`)
      .attr('fill', hexColorToCSS(color))
      .attr('stroke', hexColorToCSS(color))
      .attr('stroke-width', 3)
      .attr('cursor', 'pointer')
      .attr('shape-rendering', 'geometricPrecision')
      .attr('class', 'arrowhead');
  }

  public update(): void {
    this.color = getColor(this.txNode.inner.getOutput() * this.weight);
    this.group.selectAll('path').attr('stroke', hexColorToCSS(this.color));
    this.group.selectAll('.arrowhead').attr('fill', hexColorToCSS(this.color));
  }

  public onSelect() {
    this.group
      .select('path')
      .attr('stroke-width', Conf.EdgeWidth * 1.5)
      .attr('filter', 'url(#sofGlow)');
  }

  public onDeselect() {
    this.group.select('path').attr('stroke-width', Conf.EdgeWidth).attr('filter', null);
  }
}

export class VizNode {
  public group: d3.Selection<SVGGElement, undefined, null, undefined>;
  public name: string;
  public x: number;
  public y: number;
  public width: number;
  public height: number;
  public inner: SparseNeuron;

  constructor(
    name: string,
    x: number,
    y: number,
    width: number,
    height: number,
    labelText: string,
    inner: SparseNeuron,
    onSelect: (node: VizNode) => void,
    labelFontSizeOverride?: number
  ) {
    this.name = name;
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
    this.inner = inner;

    this.group = d3.create('svg:g');

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

    let elem:
      | d3.Selection<SVGRectElement, undefined, null, undefined>
      | d3.Selection<SVGCircleElement, undefined, null, undefined>;
    if (shape === 'square') {
      elem = this.group
        .append('rect')
        .attr('x', this.x)
        .attr('y', this.y)
        .attr('width', this.width)
        .attr('height', this.height);
    } else if (shape === 'circle') {
      elem = this.group
        .append('circle')
        .attr('cx', this.x + this.width / 2)
        .attr('cy', this.y + this.height / 2)
        .attr('r', this.width / 2);
    } else {
      throw new Error('Invalid shape: ' + shape);
    }

    elem.attr('stroke', 'white');
    elem.attr('stroke-width', Conf.NodeBorderWidth);
    elem.attr('cursor', 'pointer');
    elem.attr('shape-rendering', 'geometricPrecision');
    elem.on('click', () => onSelect(this));

    const labelFontSize = labelFontSizeOverride ?? Conf.NodeLabelFontSize;
    this.group
      .append('text')
      .attr('pointer-events', 'none')
      .text(labelText)
      .attr('x', x + width / 2)
      .attr('y', y + height / 2 + 2)
      .attr('font-size', labelText.length <= 1 ? labelFontSize : labelFontSize * 0.8)
      .attr('font-weight', '500')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-family', 'sans-serif');

    this.update();
  }

  private computeLabelColor(nodeColor: number): number {
    const r = (nodeColor >> 16) & 0xff;
    const g = (nodeColor >> 8) & 0xff;
    const b = nodeColor & 0xff;

    // Gamma correction
    const rLinear = (r / 255) ** 2.2;
    const gLinear = (g / 255) ** 2.2;
    const bLinear = (b / 255) ** 2.2;

    const luminance = 0.2126 * rLinear + 0.7152 * gLinear + 0.0722 * bLinear;
    return luminance > 0.5 ? 0x000000 : 0xffffff;
  }

  public update() {
    const value = this.inner.getOutput();
    const color = getColor(value);

    this.group.selectAll('rect').attr('fill', hexColorToCSS(color));
    this.group.selectAll('circle').attr('fill', hexColorToCSS(color));

    const labelColor = this.computeLabelColor(color);
    this.group.selectAll('text').attr('fill', hexColorToCSS(labelColor));
  }

  public getColor(): number {
    return getColor(this.inner.getOutput());
  }

  public onSelect() {
    this.group.select('rect').attr('filter', 'url(#sofGlow)');
    this.group.select('circle').attr('filter', 'url(#sofGlow)');
  }

  public onDeselect() {
    this.group.select('rect').attr('filter', null);
    this.group.select('circle').attr('filter', null);
  }
}

export class NodeViz {
  private svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private graph: RNNGraph;
  public nodes: VizNode[] = [];
  private edges: VizEdge[] = [];
  public selected: Writable<VizNode | VizEdge | null> = writable(null);

  constructor(
    svg: SVGSVGElement,
    graphvizLayoutData: string,
    graph: RNNGraph,
    labelFontSizeOverride?: number
  ) {
    this.graph = graph;
    this.svg = d3.select(svg).on('click', evt => {
      if (evt.target === svg) {
        this.handleBackgroundClick();
      }
    });
    const container = this.svg.append('g');

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

      const vizNode = new VizNode(
        nodeID,
        pos.x,
        pos.y,
        width,
        height,
        label,
        node,
        node => this.handleNodeSelect(node),
        labelFontSizeOverride
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

    // Add edges first so that they are behind nodes
    for (const edge of this.edges) {
      container.node()!.appendChild(edge.group.node()!);
    }
    for (const node of this.nodes) {
      container.node()!.appendChild(node.group.node()!);
    }

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .on('zoom', evt => void container.attr('transform', evt.transform));
    this.svg.call(zoom);

    // Set initial zoom + pan so the entire viz is visible and centered
    const bbox = container.node()!.getBBox();
    const bboxCenterX = bbox.x + bbox.width / 2;
    const bboxCenterY = bbox.y + bbox.height / 2;
    const paddingX = Conf.PaddingPercent * svg.clientWidth;
    const paddingY = Conf.PaddingPercent * svg.clientHeight;
    const paddedWidth = bbox.width + 2 * paddingX;
    const paddedHeight = bbox.height + 2 * paddingY;
    const scale = Math.min(
      (svg.clientWidth - 2 * paddingX) / paddedWidth,
      (svg.clientHeight - 2 * paddingY) / paddedHeight
    );
    const translateX = svg.clientWidth / 2 - scale * (bboxCenterX + paddingX);
    const translateY = svg.clientHeight / 2 - scale * (bboxCenterY + paddingY);
    zoom.transform(this.svg, d3.zoomIdentity.translate(translateX, translateY).scale(scale));
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
    // input[0] = -1;
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
}
