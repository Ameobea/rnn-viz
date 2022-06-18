/**
 * A group of nodes that represent scalar values.  The nodes will be rendered next to each other
 * and are expected to represent a vector of some kind.
 */

import * as PIXI from 'pixi.js';

import { Node } from './Node';

export class NodeGroup extends PIXI.Container {
  private nodes: Node[];

  public get size(): number {
    return this.nodes.length;
  }

  constructor(size: number) {
    super();
    this.nodes = new Array(size).fill(null).map(() => new Node());
  }

  public draw(values: number[] | Float32Array) {
    if (values.length !== this.size) {
      throw new Error(`Expected ${this.size} values, got ${values.length}`);
    }

    this.nodes.forEach((node, i) => node.draw(values[i]));
  }
}
