<script lang="ts">
  import UPlot from 'uplot';
  import 'uplot/dist/uPlot.min.css';

  export let totalLosses: number[];
  export let finalLosses: number[];
  export let accuracies: number[];
  export let iters: number;

  let uPlotInst: UPlot | null = null;

  $: if (uPlotInst) {
    uPlotInst.setData([
      new Array<number>(iters).fill(0).map((_, i) => i),
      totalLosses,
      accuracies.map(x => x * 100),
    ]);
  }

  const renderChart = (containerNode: HTMLElement) => {
    uPlotInst = new UPlot({
      width: window.innerWidth / 2 - 20,
      height: 600,
      series: [
        {},
        {
          label: 'Test Loss',
          stroke: 'red',
          scale: 'loss',
        },
        {
          label: 'Traing Loss',
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
        // {
        //   show: true,
        //   side: 1,
        //   label: 'Validation Accuracy (%)',
        //   stroke: '#ccc',
        //   scale: 'acc',
        // },
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

    uPlotInst.setData([finalLosses.map((_, i) => i), finalLosses, totalLosses]);
    containerNode.appendChild(uPlotInst.root);
  };
</script>

<div class="root" use:renderChart />
