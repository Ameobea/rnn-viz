import * as THREE from 'three';
import { TrackballControls } from 'three/examples/jsm/controls/TrackballControls';
import SpriteText from 'three-spritetext';

import { Axis } from './Axis';

import { MAX_VOXEL_COUNT, RESOLUTION, VOXEL_SIZE } from './conf';

export class VoxelPlot3D {
  private canvasResizeObserver: ResizeObserver;
  private scene: THREE.Scene;
  private renderer: THREE.WebGLRenderer;
  private controls: TrackballControls;
  private camera: THREE.PerspectiveCamera;
  private engine: typeof import('../../engineComp/engine') | null = null;
  private params: { xWeight: number; yWeight: number; zWeight: number; bias: number } = {
    xWeight: -1 / 3,
    yWeight: 1,
    zWeight: 2 / 3,
    bias: -1 / 3,
  };

  private voxelContainer = new THREE.InstancedMesh(
    new THREE.BoxBufferGeometry(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE),
    new THREE.MeshNormalMaterial(),
    MAX_VOXEL_COUNT
  );

  constructor(canvas: HTMLCanvasElement) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);

    this.camera = new THREE.PerspectiveCamera(
      75,
      canvas.clientWidth / canvas.clientHeight,
      0.1,
      1000
    );
    this.camera.position.set(2, 1.8, 1.6);
    this.camera.lookAt(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);

    this.controls = new TrackballControls(this.camera, canvas);
    this.controls.rotateSpeed = 3.5;
    this.controls.update();

    this.scene.add(this.voxelContainer);

    this.canvasResizeObserver = new ResizeObserver(_entries => {
      this.handleResize(canvas);
    });

    import('../../engineComp/engine').then(async engineMod => {
      await engineMod.default();
      this.engine = engineMod;
      this.plotFunction();
      this.animate();
    });

    const array = this.voxelContainer.instanceMatrix.array as Float32Array;
    // Initialize all instances to be scale 1, position 0,0,0, and rotation 0,0,0
    for (let i = 0; i < MAX_VOXEL_COUNT; i++) {
      array[i * 16 + 0] = 1;
      array[i * 16 + 5] = 1;
      array[i * 16 + 10] = 1;
      array[i * 16 + 15] = 1;
    }

    // Draw axis between all edges of the bounds
    this.scene.add(
      new Axis(new THREE.Vector3(-1.03, -1.03, -1.03), new THREE.Vector3(-1.03, -1.03, 1.03))
    );
    this.scene.add(
      new Axis(new THREE.Vector3(-1.03, -1.03, -1.03), new THREE.Vector3(1.03, -1.03, -1.03))
    );
    this.scene.add(
      new Axis(new THREE.Vector3(-1.03, -1.03, -1.03), new THREE.Vector3(-1.03, 1.03, -1.03))
    );
    this.scene.add(
      new Axis(new THREE.Vector3(1.03, 1.03, 1.03), new THREE.Vector3(-1.03, 1.03, 1.03))
    );
    this.scene.add(
      new Axis(new THREE.Vector3(1.03, 1.03, 1.03), new THREE.Vector3(1.03, -1.03, 1.03))
    );
    this.scene.add(
      new Axis(new THREE.Vector3(1.03, 1.03, 1.03), new THREE.Vector3(1.03, 1.03, -1.03))
    );
    this.scene.add(
      new Axis(new THREE.Vector3(-1.03, 1.03, -1.03), new THREE.Vector3(1.03, 1.03, -1.03))
    );
    this.scene.add(
      new Axis(new THREE.Vector3(-1.03, 1.03, -1.03), new THREE.Vector3(-1.03, 1.03, 1.03))
    );
    this.scene.add(
      new Axis(new THREE.Vector3(1.03, -1.03, 1.03), new THREE.Vector3(1.03, -1.03, -1.03))
    );
    this.scene.add(
      new Axis(new THREE.Vector3(-1.03, -1.03, 1.03), new THREE.Vector3(1.03, -1.03, 1.03))
    );
    this.scene.add(
      new Axis(new THREE.Vector3(-1.03, 1.03, 1.03), new THREE.Vector3(-1.03, -1.03, 1.03))
    );
    this.scene.add(
      new Axis(new THREE.Vector3(1.03, 1.03, -1.03), new THREE.Vector3(1.03, -1.03, -1.03))
    );

    // Create labels at each vertex
    [-1, 1].forEach(x => {
      [-1, 1].forEach(y => {
        [-1, 1].forEach(z => {
          const label = new SpriteText(
            `${x === 1 ? 'T' : 'F'},${y === 1 ? 'T' : 'F'},${z === 1 ? 'T' : 'F'}`,
            0.1,
            'white'
          );
          label.position.set(x * 1.08, y * 1.08, z * 1.08);
          this.scene.add(label);
        });
      });
    });
  }

  private plotFunction() {
    if (!this.engine) {
      return;
    }

    const voxelPositions = this.engine.compute_voxel_positions(
      this.params.xWeight,
      this.params.yWeight,
      this.params.zWeight,
      this.params.bias,
      RESOLUTION
    );
    const count = voxelPositions.length / 3;

    this.voxelContainer.count = count;
    const array = this.voxelContainer.instanceMatrix.array as Float32Array;
    for (let i = 0; i < count; i++) {
      array[i * 16 + 12] = voxelPositions[i * 3 + 0];
      array[i * 16 + 13] = voxelPositions[i * 3 + 1];
      array[i * 16 + 14] = voxelPositions[i * 3 + 2];
    }
    this.voxelContainer.instanceMatrix.needsUpdate = true;
  }

  private handleResize = (canvas: HTMLCanvasElement) => {
    // TODO
  };

  public animate = () => {
    this.renderer.render(this.scene, this.camera);
    this.controls.update();
    requestAnimationFrame(this.animate);
  };

  public setParams = (xWeight: number, yWeight: number, zWeight: number, bias: number) => {
    this.params = { xWeight, yWeight, zWeight, bias };
    this.plotFunction();
  };

  public dispose() {
    this.canvasResizeObserver.disconnect();
    this.renderer.dispose();
  }
}
