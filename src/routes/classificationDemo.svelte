<script lang="ts">
  import { onMount } from 'svelte';

  import 'uplot/dist/uPlot.min.css';

  const functions = {
    linear: 0,
    relu: 1,
    sigmoid: 2,
    gcu: 3,
    tanh: 4,
    ameo: 5,
    softLeakyAmeo: 6,
  } as const;
  const functionName = 'softLeakyAmeo' as const; // TODO: query param
  const fnID = functions[functionName];

  let canvasRef: HTMLCanvasElement | null = null;

  let engine: typeof import('../engineComp/engine') | null = null;
  let xWeight = 0.3;
  let yWeight = 0.3;
  let bias = 0;

  $: if (canvasRef && engine) {
    const pixels: Uint8Array = engine.plot_classification(
      canvasRef.width,
      fnID,
      xWeight,
      yWeight,
      bias
    );
    const ctx = canvasRef.getContext('2d')!;
    ctx.putImageData(
      new ImageData(new Uint8ClampedArray(pixels.buffer), canvasRef.width, canvasRef.height),
      0,
      0
    );
  }

  onMount(async () => {
    engine = await import('../engineComp/engine').then(async engine => {
      await engine.default();
      return engine;
    });
  });
</script>

<div class="root">
  <canvas bind:this={canvasRef} width={400} height={400} />

  <div class="sliders-container">
    <label for="x-weight">X Weight: {xWeight.toFixed(1)}</label>
    <input id="x-weight" type="range" min={-1.5} max={1.5} step={0.1} bind:value={xWeight} />
    <label for="y-weight">Y Weight: {yWeight.toFixed(1)}</label>
    <input id="y-weight" type="range" min={-1.5} max={1.5} step={0.1} bind:value={yWeight} />
    <label for="bias">Bias: {bias.toFixed(1)}</label>
    <input id="bias" type="range" min={-1.5} max={1.5} step={0.1} bind:value={bias} />
  </div>
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
    margin: 8px;
  }

  canvas {
    width: 300px;
    height: 300px;
  }

  label {
    font-family: sans-serif;
  }

  .sliders-container {
    display: flex;
    flex-direction: column;
    margin-top: 20px;

    max-width: 400px;
  }
</style>
