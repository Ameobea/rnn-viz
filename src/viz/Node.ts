import { PIXI } from './pixi';
import { NODE_SIZE } from './conf';

export class Node extends PIXI.Graphics {
  constructor() {
    super();
  }

  public draw(value: number) {
    this.clear();
    this.drawRect(0, 0, NODE_SIZE, NODE_SIZE);
    // TODO
  }
}
