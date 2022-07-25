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
  let useTruthTableControls = false;
  let truthTable: [boolean, boolean, boolean, boolean] = [true, false, false, true];
  let enableAmeo = false;
  let canvasSize = 310;
  let canvasRef: HTMLCanvasElement | null = null;
  $: ctx = canvasRef?.getContext('2d');

  $: if (useTruthTableControls) {
    const vals = truthTable.map(b => (b ? 't' : 'f')).join('');

    let params: [number, number, number];
    switch (vals) {
      case 'ffff':
        params = [0, 0, 1];
        break;
      case 'ffft':
        params = [1, 1, -3];
        break;
      case 'fftf':
        params = [1, -1, -3];
        break;
      case 'fftt':
        params = [0, -1, 0];
        break;
      case 'ftff':
        params = [-1, 1, -3];
        break;
      case 'ftft':
        params = [-1, 0, 0];
        break;
      case 'fttf':
        params = [1, -1, 1];
        break;
      case 'fttt':
        params = [1, 1, 3];
        break;
      case 'tfff':
        params = [-1, -1, -3];
        break;
      case 'tfft':
        params = [-1, 1, -1];
        break;
      case 'tftf':
        params = [0, 1, 0];
        break;
      case 'tftt':
        params = [1, -1, 3];
        break;
      case 'ttff':
        params = [1, 0, 0];
        break;
      case 'ttft':
        params = [-1, 1, 3];
        break;
      case 'tttf':
        params = [-1, -1, 3];
        break;
      case 'tttt':
        params = [0, 0, -1];
        break;
      default:
        throw new Error('Invalid truth table');
    }

    xWeight = params[0];
    yWeight = params[1];
    bias = params[2];
  }

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
    useTruthTableControls = searchParams.has('useTruthTable');

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

<div class="root" style="width: {canvasSize}px;">
  <canvas
    bind:this={canvasRef}
    width={canvasSize}
    height={canvasSize}
    style="width: {canvasSize}px; height: {canvasSize}px; margin-bottom: 20px;"
  />

  {#if showFnPicker}
    <label for="fn-picker" style={useTruthTableControls ? 'margin-top: -14px' : undefined}>
      Activation Function
    </label>

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

  {#if useTruthTableControls}
    <div class="truth-controls-container">
      <div class="params">
        <div>
          <div>X Weight</div>
          <div>{xWeight.toFixed(0)}</div>
        </div>
        <div>
          <div>Y Weight</div>
          <div>{yWeight.toFixed(0)}</div>
        </div>
        <div>
          <div>Bias</div>
          <div>{bias.toFixed(0)}</div>
        </div>
      </div>
      <table class="truth-table-controls">
        <thead>
          <tr>
            <th>Y</th>
            <th>X</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td class="truth-table-input" data-val="f">F</td>
            <td class="truth-table-input" data-val="f">F</td>
            <td>
              <input type="checkbox" bind:checked={truthTable[0]} />
            </td>
          </tr>
          <tr>
            <td class="truth-table-input" data-val="t">T</td>
            <td class="truth-table-input" data-val="f">F</td>
            <td>
              <input type="checkbox" bind:checked={truthTable[1]} />
            </td>
          </tr>
          <tr>
            <td class="truth-table-input" data-val="f">F</td>
            <td class="truth-table-input" data-val="t">T</td>
            <td>
              <input type="checkbox" bind:checked={truthTable[2]} />
            </td>
          </tr>
          <tr>
            <td class="truth-table-input" data-val="t">T</td>
            <td class="truth-table-input" data-val="t">T</td>
            <td>
              <input type="checkbox" bind:checked={truthTable[3]} />
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  {:else}
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
  {/if}
</div>

<style lang="css">
  :global(html) {
    overflow: hidden;
    font-family: 'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif;
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

  .truth-table-controls {
    width: 125px;
    margin-left: auto;
    margin-right: auto;
    border-collapse: collapse;
  }

  .truth-table-controls td,
  .truth-table-controls th {
    border: 1px solid #ccc;
    text-align: center;
    font-family: 'Open Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    width: 6px;
    padding: 3px 0;
    user-select: none;
  }

  .truth-table-controls td input {
    width: 16px;
    height: 16px;
    margin-top: 3px;
    margin-bottom: -2px;
    margin-right: 0;
    margin-left: 0;
    display: inline;
    cursor: pointer;
  }

  .truth-table-controls td[data-val='f'] {
    background-color: #811;
  }

  .truth-table-controls td[data-val='t'] {
    background-color: #181;
  }

  .truth-controls-container {
    margin-top: 8px;
    display: flex;
    flex-direction: row;
  }

  .params {
    display: flex;
    flex-direction: column;
    justify-content: center;
    font-size: 26px;
    gap: 4px;
    flex: 1;
  }

  .params > div {
    display: flex;
    flex-direction: row;
  }

  .params > div > div:first-of-type {
    flex: 0.7;
  }

  .params > div > div:last-of-type {
    flex: 0.3;
  }
</style>
