import * as THREE from 'three';

export class Axis extends THREE.Group {
  constructor(from: THREE.Vector3, to: THREE.Vector3) {
    super();
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array([from.x, from.y, from.z, to.x, to.y, to.z]);
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.getAttribute('position').needsUpdate = true;
    const material = new THREE.LineBasicMaterial({ color: 0xffffff });
    const line = new THREE.Line(geometry, material);
    this.add(line);
  }
}
