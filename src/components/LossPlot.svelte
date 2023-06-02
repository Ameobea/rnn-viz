<script lang="ts">
  import UPlot from 'uplot';
  import 'uplot/dist/uPlot.min.css';

  export let totalLosses: number[];
  export let lossesWithoutRegularization: number[];
  export let iters: number;

  $: xs = new Array(iters).fill(null).map((_, i) => i);

  let uPlotInst: UPlot | null = null;

  $: if (uPlotInst) {
    uPlotInst.setData([xs, totalLosses, lossesWithoutRegularization]);
  }

  const renderChart = (containerNode: HTMLElement) => {
    uPlotInst = new UPlot({
      width: window.innerWidth / 2 - 20,
      height: 600,
      series: [
        {},
        {
          label: 'Total Loss',
          stroke: 'red',
          scale: 'loss',
        },
        {
          label: 'Loss w/o Regularization',
          stroke: 'yellow',
          scale: 'loss',
        },
      ],
      axes: [
        {},
        {
          show: true,
          label: 'MSE',
          stroke: '#ccc',
          ticks: { show: true, stroke: '#ccc', width: 1 },
          grid: { show: true, stroke: '#cccccc88', width: 1 },
          scale: 'loss',
        },
      ],
      scales: {
        x: { time: false },
        loss: {
          /*range: [0, 2]*/
        },
        acc: { range: [0, 100] },
      },
      id: 'loss-plot',
    });

    containerNode.appendChild(uPlotInst.root);
  };
</script>

<div class="root" use:renderChart />
