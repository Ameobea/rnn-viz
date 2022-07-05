<script lang="ts">
  import { onMount } from 'svelte';
  import type { VoxelPlot3D } from '../viz/VoxelPlot3D/VoxelPlot3D';
  let canvas: HTMLCanvasElement | null = null;

  let xWeight = 0.7067233920097351;
  let yWeight = 0.7426995038986206;
  let zWeight = -1.2776010036468506;
  let bias = 0.2680431008338928;

  let viz: VoxelPlot3D | null = null;
  $: if (viz) {
    viz.setParams(xWeight, yWeight, zWeight, bias);
  }

  onMount(async () => {
    const { VoxelPlot3D } = await import('../viz/VoxelPlot3D/VoxelPlot3D');
    if (!canvas) {
      throw new Error('Expected canvas to be mounted by now');
    }

    viz = new VoxelPlot3D(canvas);

    return () => viz?.dispose();
  });
</script>

<div class="root">
  <canvas bind:this={canvas} />

  <div class="label-container">
    <label for="x-weight-input">x weight: </label>{xWeight.toFixed(2)}
  </div>
  <input id="x-weight-input" type="range" min={-3} max={3} step={0.01} bind:value={xWeight} />
  <div class="label-container">
    <label for="y-weight-input">y weight: </label>{yWeight.toFixed(2)}
  </div>
  <input id="y-weight-input" type="range" min={-3} max={3} step={0.01} bind:value={yWeight} />
  <div class="label-container">
    <label for="z-weight-input">z weight: </label>{zWeight.toFixed(2)}
  </div>
  <input id="z-weight-input" type="range" min={-3} max={3} step={0.01} bind:value={zWeight} />
  <div class="label-container">
    <label for="bias-input">bias: </label>{bias.toFixed(2)}
  </div>
  <input id="bias-input" type="range" min={-3} max={3} step={0.01} bind:value={bias} />
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  canvas {
    height: 500px;
    width: 500px;
  }
</style>
