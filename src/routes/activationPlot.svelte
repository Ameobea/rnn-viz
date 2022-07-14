<script context="module" lang="ts">
  import type { EChartsOption } from 'echarts';

  type VariantParams =
    | { type: 'single'; smooth: boolean; leaky: boolean }
    | { type: 'interpolated'; factor: number };

  const buildDefaultVariant = (vType: VariantParams['type']): VariantParams => {
    switch (vType) {
      case 'interpolated':
        return { type: vType, factor: 0.5 };
      case 'single':
        return { type: vType, smooth: false, leaky: false };
      default:
        throw new Error(`Invalid variant type: ${vType as any}`);
    }
  };

  const mkAmeo = (leakyness: number | null) => (x: number) => {
    if (x <= -2) {
      return leakyness === null ? 0 : leakyness * (x + 2);
    } else if (x <= -1) {
      return x + 2;
    } else if (x <= 0) {
      return -x;
    } else if (x <= 1) {
      return x;
    } else {
      return leakyness === null ? 1 : leakyness * (x - 1) + 1;
    }
  };

  const mkSmoothAmeo = (leakyness: number | null) => (x: number) => {
    if (x <= -2) return leakyness === null ? 0 : leakyness * (x + 2);
    else if (x <= -1.5) return 8 * Math.pow(x + 2, 4);
    else if (x <= -0.5)
      return -8 * Math.pow(x, 4) + -32 * Math.pow(x, 3) + -48 * Math.pow(x, 2) + -32 * x - 7;
    else if (x <= 0.5) return 8 * Math.pow(x, 4);
    else if (x <= 1)
      return -8 * Math.pow(x, 4) + 32 * Math.pow(x, 3) + -48 * Math.pow(x, 2) + 32 * x - 7;
    else return leakyness === null ? 1 : leakyness * (x - 1) + 1;
  };

  const scaleAndShiftActivationFunction =
    (fn: (x: number) => number): ((x: number) => number) =>
    (x: number) => {
      const y = fn(0.5 * x - 0.5);
      return (y - 0.5) * 2;
    };

  const buildActivationSeries = (
    params: VariantParams,
    ameoActivationMod: typeof import('../nn/ameoActivation') | null
  ): EChartsOption['series'] => {
    const [activationFn, activationGradFn] =
      params.type === 'interpolated'
        ? (() => {
            if (!ameoActivationMod) {
              throw new Error('ameo activation mod not loaded');
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
            if (ameoActivationMod) {
              if (!params.smooth && !params.leaky) {
                const activation = new ameoActivationMod.Ameo();
                return [
                  (x: number) =>
                    activation.apply(ameoActivationMod.tfc.tensor1d([x])).dataSync()[0],
                  (x: number) =>
                    ameoActivationMod.tfc
                      .grad(activation.apply)(ameoActivationMod.tfc.tensor1d([x]))
                      .dataSync()[0],
                ];
              }

              if (params.leaky && !params.smooth) {
                const activation = new ameoActivationMod.LeakyAmeo(0.05);
                return [
                  (x: number) =>
                    activation.apply(ameoActivationMod.tfc.tensor1d([x])).dataSync()[0],
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
            }

            return [
              scaleAndShiftActivationFunction(
                params.smooth
                  ? mkSmoothAmeo(params.leaky ? 0.05 : null)
                  : mkAmeo(params.leaky ? 0.05 : null)
              ),
              undefined,
            ];
          })();

    const pointCount = 250;
    const range = [-4, 4];

    const data = new Array(pointCount).fill(null).map((_, i: number) => {
      const x = range[0] + ((range[1] - range[0]) * i) / (pointCount - 1);
      const y = activationFn(x);
      return [x, y];
    });
    const gradData = activationGradFn
      ? new Array(pointCount).fill(null).map((_, i: number) => {
          const x = range[0] + ((range[1] - range[0]) * i) / (pointCount - 1);
          const y = activationGradFn(x);
          return [x, y];
        })
      : null;
    return [
      { data, type: 'line', symbol: 'none' },
      gradData ? { data: gradData, type: 'line', symbol: 'none' } : { data: [], type: 'line' },
    ];
  };
</script>

<script lang="ts">
  import * as echarts from 'echarts/core.js';
  import { LineChart } from 'echarts/charts.js';
  import { GridComponent } from 'echarts/components.js';
  import { SVGRenderer } from 'echarts/renderers.js';
  import { onMount } from 'svelte';

  import type { Rank, Tensor } from '@tensorflow/tfjs';

  echarts.use([LineChart, GridComponent, SVGRenderer]);

  let variant: VariantParams = { type: 'single', smooth: false, leaky: false };
  let ameoActivationMod: typeof import('../nn/ameoActivation') | null = null;
  let chartContainer: HTMLDivElement | null = null;
  let chartInst: echarts.ECharts | null = null;

  $: if (chartInst) {
    chartInst.setOption(
      {
        series: buildActivationSeries(variant, ameoActivationMod),
      },
      false,
      true
    );
  }

  onMount(async () => {
    if (!chartContainer) {
      throw new Error('Chart container ref not set');
    }

    // TODO: Gate to be only for some dev mode or something, probably
    const engine = await import('../engineComp/engine').then(async engine => {
      await engine.default();
      return engine;
    });
    import('../nn/ameoActivation').then(mod => {
      // mod.setWasmEngine(engine);
      mod.tfc.setBackend('cpu');
      ameoActivationMod = mod;
    });

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
        },
      },
      series: buildActivationSeries(variant, ameoActivationMod),
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

  <label for="smooth-checkbox">Smooth</label>
  <input
    type="checkbox"
    id="smooth-checkbox"
    checked={variant.type === 'single' && variant.smooth}
    on:change={evt => handleChange({ type: 'single', smooth: evt.currentTarget.checked })}
  />
  <label for="leaky-checkbox">Leaky</label>
  <input
    type="checkbox"
    id="leaky-checkbox"
    checked={variant.type === 'single' && variant.leaky}
    on:change={evt => handleChange({ type: 'single', leaky: evt.currentTarget.checked })}
  />
  {#if ameoActivationMod}
    <label for="leaky-checkbox">Interpolated</label>
    <input
      type="checkbox"
      id="leaky-checkbox"
      checked={variant.type === 'interpolated'}
      on:change={evt =>
        handleChange({ type: evt.currentTarget.checked ? 'interpolated' : 'single' })}
    />
    {#if variant.type === 'interpolated'}
      <label for="interpolation-factor-slider">Interpolation Factor</label>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        id="interpolation-factor-slider"
        value={variant.factor}
        on:change={evt => handleChange({ type: 'interpolated', factor: +evt.currentTarget.value })}
      />
    {/if}
  {/if}
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }

  .activation-plot-chart {
    width: 500px;
    height: 400px;
  }
</style>
