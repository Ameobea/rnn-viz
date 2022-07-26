<script lang="ts">
  import { onMount } from 'svelte';
  import type { VoxelPlot3D } from '../viz/VoxelPlot3D/VoxelPlot3D';
  let canvas: HTMLCanvasElement | null = null;

  let xWeight = -1;
  let yWeight = 3;
  let zWeight = 2;
  let bias = -1;

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

  <div class="controls">
    <div class="input-container">
      <div class="label-container">
        <label for="x-weight-input">x weight: </label>{xWeight.toFixed(2)}
      </div>
      <input id="x-weight-input" type="range" min={-3} max={3} step={0.01} bind:value={xWeight} />
    </div>

    <div class="input-container">
      <div class="label-container">
        <label for="y-weight-input">y weight: </label>{yWeight.toFixed(2)}
      </div>
      <input id="y-weight-input" type="range" min={-3} max={3} step={0.01} bind:value={yWeight} />
    </div>

    <div class="input-container">
      <div class="label-container">
        <label for="z-weight-input">z weight: </label>{zWeight.toFixed(2)}
      </div>
      <input id="z-weight-input" type="range" min={-3} max={3} step={0.01} bind:value={zWeight} />
    </div>

    <div class="input-container">
      <div class="label-container">
        <label for="bias-input">bias: </label>{bias.toFixed(2)}
      </div>
      <input id="bias-input" type="range" min={-3} max={3} step={0.01} bind:value={bias} />
    </div>
  </div>
</div>

<style lang="css">
  :global(html) {
    overflow: hidden;
    max-width: 500px;
    max-height: 600px;
    font-family: 'Open Sans', 'PT Sans', 'Roboto', 'Helvetica Neue', Helvetica, Arial, sans-serif;
  }

  .root {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  canvas {
    height: 500px;
    width: 500px;
    cursor: grab;
  }

  .controls {
    padding-top: 6px;
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
  }

  .controls .input-container {
    display: flex;
    flex-direction: column;
    gap: 1px;
    flex-basis: 50%;
    padding-left: 8px;
    padding-right: 8px;
    padding-bottom: 3px;
    box-sizing: border-box;
  }
</style>
