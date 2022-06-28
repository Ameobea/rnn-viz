import { RNN, type RNNDefinition } from 'src/nn/RNN';
import { WORLD_SIZE } from './conf';
import { NodeGroup } from './NodeGroup';
import { PIXI, Viewport } from './pixi';

class RNNVizLayer extends PIXI.Container {
  private nodeGroups: NodeGroup[];

  constructor(nodeGroupSizes: number[]) {
    super();
    this.nodeGroups = nodeGroupSizes.map(groupSize => new NodeGroup(groupSize));
  }

  public draw(values: number[][] | Float32Array[]) {
    // TODO
  }
}

export class RNNViz {
  private model: RNN;
  private app: PIXI.Application;
  private container: Viewport;
  private layers: RNNVizLayer[];
  private resizeObserver: ResizeObserver;

  constructor(canvas: HTMLCanvasElement, def: RNNDefinition) {
    this.model = new RNN(def);

    this.app = new PIXI.Application({
      antialias: true,
      resolution: window.devicePixelRatio,
      autoDensity: true,
      view: canvas,
      height: canvas.height,
      width: canvas.width,
      backgroundColor: 0,
    });

    this.container = new Viewport({
      screenWidth: window.innerWidth,
      screenHeight: window.innerHeight,
      worldHeight: WORLD_SIZE,
      worldWidth: WORLD_SIZE,
      interaction: this.app.renderer.plugins.interaction,
    });
    this.container.drag({ mouseButtons: 'middle-left' }).pinch().wheel();

    this.resizeObserver = new ResizeObserver(() => this.handleResize(canvas));
    this.resizeObserver.observe(canvas);

    this.draw();
  }

  private draw() {
    // TODO
    // this.layers.forEach(layer => layer.draw())
  }

  private handleResize = (canvas: HTMLCanvasElement) => {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    console.log('Resizing: ', { width, height });
    this.app.renderer.resize(width, height);
    this.container.resize(width, height);
  };

  public dispose() {
    this.resizeObserver.disconnect();
    this.app.destroy(false, { children: true, texture: true, baseTexture: true });
  }
}
