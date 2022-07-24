<script lang="ts" context="module">
  const drawLabels = (
    ctx: CanvasRenderingContext2D,
    canvasSize: number,
    negTargetValue: number,
    posTargetValue: number
  ) => {
    // Draw label in each corner of the canvas
    ctx.font = '14px sans-serif';
    ctx.fillStyle = 'white';
    // Top left
    const topLeftVal = `${negTargetValue}, ${negTargetValue}`;
    ctx.fillText(topLeftVal, 5, 15);
    // Top right
    const topRightVal = `${posTargetValue}, ${negTargetValue}`;
    ctx.fillText(topRightVal, canvasSize - ctx.measureText(topRightVal).width - 5, 15);
    // Bottom left
    const bottomLeftVal = `${negTargetValue}, ${posTargetValue}`;
    ctx.fillText(bottomLeftVal, 5, canvasSize - 6);
    // Bottom right
    const bottomRightVal = `${posTargetValue}, ${posTargetValue}`;
    ctx.fillText(
      bottomRightVal,
      canvasSize - ctx.measureText(bottomRightVal).width - 5,
      canvasSize - 6
    );
  };
</script>

<script lang="ts">
  import { onMount } from 'svelte';

  import 'uplot/dist/uPlot.min.css';

  const functions = {
    linear: [0, 0, 1],
    relu: [1, 0, 1],
    sigmoid: [2, 0, 1],
    gcu: [3, -1, 1],
    tanh: [4, -1, 1],
    ameo: [5, -1, 1],
    softLeakyAmeo: [6, -1, 1],
    interpolatedAmeo: [7, -1, 1],
    gaussian: [8, 0, 1],
  } as const;
  let functionName: keyof typeof functions = 'linear' as const;
  $: [fnID, negTargetValue, posTargetValue] = functions[functionName];

  let showFnPicker = false;
  let enableAmeo = false;
  let canvasSize = 310;
  let canvasRef: HTMLCanvasElement | null = null;
  $: ctx = canvasRef?.getContext('2d');

  let engine: typeof import('../engineComp/engine') | null = null;
  let xWeight = 0.3;
  let yWeight = 0.3;
  let bias = 0;

  $: if (ctx && engine) {
    ctx.clearRect(0, 0, canvasSize, canvasSize);
    const pixels: Uint8Array = engine.plot_classification(
      canvasSize,
      fnID,
      xWeight,
      yWeight,
      bias,
      posTargetValue,
      negTargetValue
    );
    ctx.putImageData(
      new ImageData(new Uint8ClampedArray(pixels.buffer), canvasSize, canvasSize),
      // This needs to be set to 1 instead of 0 because otherwise there is some bug that causes things not to render
      1,
      0
    );

    drawLabels(ctx, canvasSize, negTargetValue, posTargetValue);
  }

  onMount(async () => {
    const searchParams = new URLSearchParams(window.location.search);
    showFnPicker = searchParams.has('fnPicker');

    if (searchParams.has('ameo')) {
      functionName = 'interpolatedAmeo' as const;
    } else if (searchParams.has('gaussian')) {
      functionName = 'gaussian' as const;
      xWeight = 2.5;
      yWeight = 2.5;
      bias = -2.5;
    } else if (searchParams.has('gcu')) {
      functionName = 'gcu' as const;
      xWeight = 1;
      yWeight = 1;
      bias = 0.6;
    } else if (searchParams.has('softLeakyAmeo')) {
      functionName = 'softLeakyAmeo' as const;
      xWeight = 1;
      yWeight = -1;
      bias = 1;
    } else if (searchParams.has('interpolatedAmeo')) {
      functionName = 'interpolatedAmeo' as const;
      xWeight = 1;
      yWeight = -1;
      bias = 1;
    }

    enableAmeo = functionName.toLowerCase().includes('ameo');

    engine = await import('../engineComp/engine').then(async engine => {
      await engine.default();
      return engine;
    });
  });
</script>

<div class="root">
  <canvas
    bind:this={canvasRef}
    width={canvasSize}
    height={canvasSize}
    style="width: {canvasSize}px; height: {canvasSize}px; margin-bottom: 20px;"
  />

  {#if showFnPicker}
    <label for="fn-picker">Activation Function</label>

    <select id="fn-picker" bind:value={functionName} style="width: 200px">
      {#if enableAmeo}
        <option value="ameo">Ameo</option>
        <option value="softLeakyAmeo">Soft+Leaky Ameo</option>
        <option value="interpolatedAmeo">Interpolated Ameo (factor 0.1)</option>
      {:else}
        <option value="linear">Linear</option>
        <option value="relu">ReLU</option>
        <option value="sigmoid">Sigmoid</option>
        <option value="tanh">Tanh</option>
        <option value="gaussian">Gaussian</option>
        <option value="gcu">GCU</option>
      {/if}
    </select>
  {/if}

  <div
    class="sliders-container"
    style="max-width: {canvasSize}px; margin-top: {showFnPicker ? 20 : -4}px;"
  >
    <label for="x-weight">X Weight: {xWeight.toFixed(1)}</label>
    <input id="x-weight" type="range" min={-3} max={3} step={0.1} bind:value={xWeight} />
    <label for="y-weight">Y Weight: {yWeight.toFixed(1)}</label>
    <input id="y-weight" type="range" min={-3} max={3} step={0.1} bind:value={yWeight} />
    <label for="bias">Bias: {bias.toFixed(1)}</label>
    <input id="bias" type="range" min={-3} max={3} step={0.1} bind:value={bias} />
  </div>
</div>

<style lang="css">
  :global(html) {
    overflow: hidden;
  }

  .root {
    display: flex;
    flex-direction: column;
    margin: 8px;
  }

  label {
    font-family: sans-serif;
  }

  .sliders-container {
    display: flex;
    flex-direction: column;
    margin-top: 20px;
  }
</style>
