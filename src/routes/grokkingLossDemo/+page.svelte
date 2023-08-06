<script lang="ts">
  import { browser } from '$app/environment';
  import UPlot from 'uplot';
  import 'uplot/dist/uPlot.min.css';
  import type { PageData } from './$types';

  export let data: PageData;
  const xs = new Array(data.baseLosses.length).fill(null).map((_, i) => i * 10);

  const uPlotInst: UPlot | null = browser
    ? new UPlot({
        width: Math.min(window.innerWidth, 800),
        height: 400,
        series: [
          {},
          {
            label: 'Base Loss',
            stroke: 'red',
            scale: 'loss',
          },
          {
            label: 'Regularization Loss',
            stroke: 'orange',
          },
        ],
        axes: [
          { show: true, label: 'Batch', stroke: '#eee' },
          {
            show: true,
            label: 'Loss',
            stroke: '#eee',
            ticks: { show: true, stroke: '#ccc', width: 1 },
            grid: { show: true, stroke: '#cccccc22', width: 1 },
            scale: 'loss',
          },
        ],
        legend: { show: true },
        scales: {
          x: { time: false },
          loss: { log: 10 },
        },
      })
    : null;
  uPlotInst?.setData([xs, data.baseLosses, data.regLosses]);

  const usePlot = (node: HTMLDivElement) => {
    if (uPlotInst) {
      node.appendChild(uPlotInst.root);
    }
  };
</script>

<div class="root" use:usePlot />

<style lang="css">
  :global(html, body) {
    overflow: hidden;
  }

  .root {
    overflow: hidden;
  }
</style>
