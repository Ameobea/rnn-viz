<script context="module" lang="ts">
  import type { EChartsOption } from 'echarts';

  type VariantParams =
    | { type: 'single'; smooth: boolean; leaky: boolean }
    | { type: 'interpolated'; factor: number; leaky: boolean }
    | { type: 'gaussian' }
    | { type: 'gcu' };

  const buildDefaultVariant = (vType: VariantParams['type']): VariantParams => {
    switch (vType) {
      case 'interpolated':
        return { type: vType, factor: 0.5, leaky: false };
      case 'single':
        return { type: vType, smooth: false, leaky: false };
      default:
        return { type: vType };
    }
  };

  const gaussian = (x: number) => Math.pow(Math.E, -x * x);
  const gaussianGrad = (x: number) => -2 * x * gaussian(x);

  const gcu = (x: number) => x * Math.cos(x);
  const gcuGrad = (x: number) => Math.cos(x) - x * Math.sin(x);

  const buildActivationSeries = (
    params: VariantParams,
    engine: typeof import('../engineComp/engine'),
    ameoActivationMod: typeof import('../nn/ameoActivation') | null
  ): EChartsOption['series'] => {
    const pointCount = 250;
    const range = [-4, 4];
    const outputColor = '#13f2ef';
    const derivativeColor = '#6125d9';

    const xs = new Float32Array(
      new Array(pointCount)
        .fill(0)
        .map((_, i) => ((range[1] - range[0]) * i) / (pointCount - 1) + range[0])
    );

    if (params.type === 'gaussian' || params.type === 'gcu') {
      const y = params.type === 'gaussian' ? gaussian : gcu;
      const yGrad = params.type === 'gaussian' ? gaussianGrad : gcuGrad;

      const ys = xs.map(y);
      const dys = xs.map(yGrad);

      return [
        {
          data: [...xs].map((x, i) => [x, ys[i]]),
          type: 'line',
          symbol: 'none',
          name: 'Output',
          color: outputColor,
        },
        {
          data: [...xs].map((x, i) => [x, dys[i]]),
          type: 'line',
          symbol: 'none',
          name: 'Derivative',
          color: derivativeColor,
        },
      ];
    }

    if (!ameoActivationMod) {
      let factor: number;
      const leakyness = params.leaky ? 0.1 : 0;

      switch (params.type) {
        case 'interpolated': {
          factor = params.factor;
          break;
        }
        case 'single': {
          factor = params.smooth ? 0 : 1;
          break;
        }
        default: {
          throw new Error(`Invalid variant type: ${(params as any).type}`);
        }
      }

      const ys = engine.apply_batch_fused_interpolated_ameo(factor, leakyness, xs);
      const dys = engine.apply_batch_fused_interpolated_ameo_grad(
        factor,
        leakyness,
        xs,
        new Float32Array(xs.length).fill(1)
      );

      return [
        {
          data: [...xs].map((x, i) => [x, ys[i]]),
          type: 'line',
          symbol: 'none',
          name: 'Output',
          color: outputColor,
        },
        {
          data: [...xs].map((x, i) => [x, dys[i]]),
          type: 'line',
          symbol: 'none',
          name: 'Derivative',
          color: derivativeColor,
        },
      ];
    }

    const [activationFn, activationGradFn] =
      params.type === 'interpolated'
        ? (() => {
            if (!ameoActivationMod) {
              throw new Error('unreachable');
            }

            const activation = new ameoActivationMod.InterpolatedAmeo(params.factor, 0.05);
            return [
              (x: number) => activation.apply(ameoActivationMod.tfc.tensor1d([x])).dataSync()[0],
              (x: number) =>
                ameoActivationMod.tfc
                  .grad((x: Tensor<Rank>) => activation.apply(x))(
                    ameoActivationMod.tfc.tensor1d([x])
                  )
                  .dataSync()[0],
            ];
          })()
        : (() => {
            if (!ameoActivationMod) {
              throw new Error('unreachable');
            }

            if (!params.smooth && !params.leaky) {
              const activation = new ameoActivationMod.Ameo();
              return [
                (x: number) => activation.apply(ameoActivationMod.tfc.tensor1d([x])).dataSync()[0],
                (x: number) =>
                  ameoActivationMod.tfc
                    .grad(activation.apply)(ameoActivationMod.tfc.tensor1d([x]))
                    .dataSync()[0],
              ];
            }

            if (params.leaky && !params.smooth) {
              const activation = new ameoActivationMod.LeakyAmeo(0.05);
              return [
                (x: number) => activation.apply(ameoActivationMod.tfc.tensor1d([x])).dataSync()[0],
                (x: number) =>
                  ameoActivationMod.tfc
                    .grad((x: Tensor<Rank>) => activation.apply(x))(
                      ameoActivationMod.tfc.tensor1d([x])
                    )
                    .dataSync()[0],
              ];
            }

            const activation = new ameoActivationMod.SoftLeakyAmeo(params.leaky ? 0.05 : null);
            return [
              (x: number) => activation.apply(ameoActivationMod.tfc.tensor1d([x])).dataSync()[0],
              (x: number) =>
                ameoActivationMod.tfc
                  .grad((x: Tensor<Rank>) => activation.apply(x))(
                    ameoActivationMod.tfc.tensor1d([x])
                  )
                  .dataSync()[0],
            ];
          })();

    const data = [...xs].map((x, i: number) => {
      const y = activationFn(x);
      return [x, y];
    });
    const gradData = activationGradFn
      ? [...xs].map(x => {
          const y = activationGradFn(x);
          return [x, y];
        })
      : null;
    return [
      { data, type: 'line', symbol: 'none', name: 'Output', color: outputColor },
      gradData
        ? {
            data: gradData,
            type: 'line',
            symbol: 'none',
            name: 'Derivative',
            color: derivativeColor,
          }
        : { data: [], type: 'line' },
    ];
  };
</script>

<script lang="ts">
  import * as echarts from 'echarts/core.js';
  import { LineChart } from 'echarts/charts.js';
  import { GridComponent } from 'echarts/components.js';
  import { SVGRenderer } from 'echarts/renderers.js';
  // import { CanvasRenderer } from 'echarts/renderers.js';
  import { LegendComponent } from 'echarts/components';
  import { onMount } from 'svelte';

  import type { Rank, Tensor } from '@tensorflow/tfjs';

  echarts.use([LineChart, GridComponent, SVGRenderer, LegendComponent]);

  function rnn(initialState: number[], inputs: number[][]): number[][] {
    function rnnStep(state: number[], input: number[]): [number[], number[]] {
      const combined = [...state, ...input];

      const newState = applyRecurrentTree(combined);
      const output = applyOutputTree(combined);
      return [newState, output];
    }

    type Acc = [number[], number[][]];

    const [_finalState, outputs] = inputs.reduce(
      ([state, outputs]: Acc, input) => {
        const [newState, output] = rnnStep(state, input);
        return [newState, [...outputs, output]];
      },
      [initialState, []] as Acc
    );
    return outputs;
  }

  let variant: VariantParams = { type: 'interpolated', factor: 1, leaky: true };
  let ameoActivationMod: typeof import('../nn/ameoActivation') | null = null;
  let engine: typeof import('../engineComp/engine') | null = null;
  let chartContainer: HTMLDivElement | null = null;
  let chartInst: echarts.ECharts | null = null;

  $: if (chartInst && engine) {
    chartInst.setOption(
      {
        series: buildActivationSeries(variant, engine, ameoActivationMod),
      },
      false,
      true
    );
  }

  onMount(async () => {
    const searchParams = new URLSearchParams(window.location.search);
    if (searchParams.has('gaussian')) {
      variant = { type: 'gaussian' };
    } else if (searchParams.has('gcu')) {
      variant = { type: 'gcu' };
    } else if (searchParams.has('interpolatedAmeo')) {
      variant = { type: 'interpolated', factor: 0.1, leaky: true };
    }

    if (!chartContainer) {
      throw new Error('Chart container ref not set');
    }

    engine = await import('../engineComp/engine').then(async engine => {
      await engine.default();
      return engine;
    });

    // import('../nn/ameoActivation').then(mod => {
    //   // mod.setWasmEngine(engine);
    //   mod.tfc.setBackend('cpu');
    //   ameoActivationMod = mod;
    // });

    const option: EChartsOption = {
      backgroundColor: '#040404',
      xAxis: {
        splitNumber: 6,
        type: 'value',
        splitLine: {
          lineStyle: {
            opacity: 0.1,
          },
        },
        min: -4,
        max: 4,
        axisLabel: {
          color: '#eee',
        },
      },
      yAxis: {
        splitNumber: 8,
        type: 'value',
        splitLine: {
          lineStyle: {
            opacity: 0.1,
          },
        },
        min: -5.5,
        max: 5.5,
        axisLabel: {
          showMinLabel: false,
          showMaxLabel: false,
          color: '#eee',
        },
      },
      series: buildActivationSeries(variant, engine!, ameoActivationMod),
      legend: {
        show: true,
        bottom: 22,
        textStyle: {
          color: '#eee',
        },
      },
      grid: {
        bottom: 70,
        top: 14,
        left: 26,
        right: 45,
      },
    };

    chartInst = echarts.init(chartContainer);
    chartInst.setOption(option);
  });

  const handleChange = (partial: Partial<VariantParams>) => {
    if (!partial.type) {
      throw new Error('Must provide type');
    }

    if (partial.type === variant.type) {
      variant = { ...variant, ...partial } as any;
    } else {
      variant = { ...buildDefaultVariant(partial.type), ...partial } as any;
    }
  };
</script>

<div class="root">
  <div class="activation-plot-chart" bind:this={chartContainer} />

  {#if variant.type !== 'gaussian' && variant.type !== 'gcu'}
    <div class="controls">
      <!-- <div>
        <label for="smooth-checkbox">Smooth</label>
        <input
          type="checkbox"
          id="smooth-checkbox"
          checked={variant.type === 'single' && variant.smooth}
          on:change={evt => handleChange({ type: 'single', smooth: evt.currentTarget.checked })}
        />
      </div> -->
      <div>
        <label for="leaky-checkbox">Leaky</label>
        <input
          type="checkbox"
          id="leaky-checkbox"
          checked={variant.type === 'single' && variant.leaky}
          on:change={evt =>
            handleChange({ type: 'interpolated', leaky: evt.currentTarget.checked })}
        />
      </div>
      <!-- <div>
        <label for="leaky-checkbox">Interpolated</label>
        <input
          type="checkbox"
          id="leaky-checkbox"
          checked={variant.type === 'interpolated'}
          on:change={evt =>
            handleChange({ type: evt.currentTarget.checked ? 'interpolated' : 'single' })}
        />
      </div> -->
      {#if variant.type === 'interpolated'}
        <div>
          <label for="interpolation-factor-slider">Interpolation Factor</label>
          <input
            style="flex: 1; margin-left: 4px; margin-right: 10px"
            type="range"
            min={0}
            max={1}
            step={0.01}
            id="interpolation-factor-slider"
            value={variant.factor}
            on:change={evt =>
              handleChange({ type: 'interpolated', factor: +evt.currentTarget.value })}
          />
        </div>
      {/if}
    </div>
  {/if}
</div>

<style lang="css">
  :global(html) {
    overflow: hidden;
  }

  .root {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    font-family: 'Open Sans', 'PT Sans', 'Roboto', 'Helvetica Neue', Helvetica, Arial, sans-serif;
  }

  .activation-plot-chart {
    width: 460px;
    height: 340px;
  }

  .controls {
    margin-left: 4px;
    margin-right: 4px;
    z-index: 2;
    display: flex;
    flex-direction: column;
    margin-top: -20px;
    width: 100%;
  }

  .controls > div {
    display: flex;
    gap: 6px;
    flex-direction: row;
    margin-bottom: 4px;
  }
</style>
