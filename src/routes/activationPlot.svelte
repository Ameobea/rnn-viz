<script context="module" lang="ts">
  import type { EChartOption } from 'echarts';

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
      const y = fn(1.5 * x - 0.5);
      return (y - 0.5) * 2;
    };

  const buildActivationSeries = (smooth: boolean, leaky: boolean): EChartOption['series'] => {
    const activationFn = scaleAndShiftActivationFunction(
      smooth ? mkSmoothAmeo(leaky ? 0.05 : null) : mkAmeo(leaky ? 0.05 : null)
    );
    const pointCount = 250;
    const range = [-1.5, 1.5];

    const data = new Array(pointCount).fill(null).map((_, i: number) => {
      const x = range[0] + ((range[1] - range[0]) * i) / (pointCount - 1);
      const y = activationFn(x);
      return [x, y];
    });
    return [{ data, type: 'line', symbol: 'none' }];
  };
</script>

<script lang="ts">
  import * as echarts from 'echarts/core.js';
  import { LineChart } from 'echarts/charts.js';
  import { GridComponent } from 'echarts/components.js';
  import { SVGRenderer } from 'echarts/renderers.js';
  echarts.use([LineChart, GridComponent, SVGRenderer]);

  import { onMount } from 'svelte';
  import type { ECBasicOption } from 'echarts/types/dist/shared';

  let smooth = false;
  let leaky = false;

  let chartContainer: HTMLDivElement | null = null;
  let chartInst: echarts.ECharts | null = null;

  $: if (chartInst) {
    chartInst.setOption(
      {
        series: buildActivationSeries(smooth, leaky),
      },
      false,
      true
    );
  }

  onMount(() => {
    if (!chartContainer) {
      throw new Error('Chart container ref not set');
    }

    const option: ECBasicOption = {
      backgroundColor: '#040404',
      xAxis: {
        type: 'value',
        splitLine: {
          lineStyle: {
            opacity: 0.1,
          },
        },
        min: -1.5,
        max: 1.5,
        axisLabel: {
          showMinLabel: false,
          showMaxLabel: false,
          interval: 1 / 3,
        },
      },
      yAxis: {
        type: 'value',
        splitLine: {
          lineStyle: {
            opacity: 0.1,
          },
        },
        min: -1.5,
        max: 1.5,
        axisLabel: {
          showMinLabel: false,
          showMaxLabel: false,
        },
      },
      series: buildActivationSeries(smooth, leaky),
    };

    chartInst = echarts.init(chartContainer);
    chartInst.setOption(option);
  });
</script>

<div class="root">
  <div class="activation-plot-chart" bind:this={chartContainer} />

  <label for="smooth-checkbox">Smooth</label>
  <input type="checkbox" id="smooth-checkbox" bind:checked={smooth} />
  <label for="leaky-checkbox">Leaky</label>
  <input type="checkbox" id="leaky-checkbox" bind:checked={leaky} />
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }

  .activation-plot-chart {
    width: 500px;
    height: 300px;
  }
</style>
