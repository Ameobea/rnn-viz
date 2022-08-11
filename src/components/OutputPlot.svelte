<script context="module" lang="ts">
  // 16 unique categorical colors
  const colors = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
  ];
</script>

<script lang="ts">
  import UPlot from 'uplot';
  import 'uplot/dist/uPlot.min.css';

  export let outputs: { values: number[]; name: string }[];
  export let xIncrement = 1;

  let uPlotInst: UPlot | null = null;

  $: {
    if (uPlotInst && outputs.length) {
      uPlotInst.setData([
        new Array<number>(outputs[0].values.length).fill(0).map((_, i) => i * xIncrement),
        ...outputs.map(o => o.values),
      ]);
    }
  }

  const renderChart = (containerNode: HTMLElement) => {
    uPlotInst = new UPlot({
      width: window.innerWidth - 20,
      height: 480,
      series: [
        { label: 'input sum' },
        ...outputs.map((o, i) => ({
          label: o.name,
          stroke: colors[i],
          width: o.name === 'combined' ? 3 : 1,
        })),
      ],
      legend: {
        live: false,
      },
      axes: [
        {
          show: true,
          label: 'Summed Input (decimal)',
          stroke: '#ccc',
          ticks: { size: 14, stroke: '#cccccc55', width: 1 },
        },
        {
          show: true,
          label: 'Neuron Output',
          stroke: '#ccc',
          ticks: { show: true, stroke: '#ccc', width: 1 },
          grid: { show: true, stroke: '#cccccc55', width: 1, dash: [1, 3] },
        },
      ],
      scales: {
        x: { time: false },
      },
      id: 'loss-plot',
    });
    containerNode.appendChild(uPlotInst.root);
  };
</script>

<div class="root" use:renderChart />
