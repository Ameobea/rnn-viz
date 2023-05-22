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

  export let data: {
    number: number;
    complexity: number;
    sample_count: number;
    area_fraction: number;
    truth_table: boolean[];
    formula: string;
  }[];

  let uPlotInst: UPlot | null = null;

  $: {
    if (uPlotInst && data) {
      // Sort data by area fraction descending
      data.sort((a, b) => b.area_fraction - a.area_fraction);

      uPlotInst.setData([
        new Array(data.length).fill(null).map((_, i) => i),
        data.map(d => d.complexity),
        // data.map(d => {
        //   const trueCount = d.truth_table.filter(t => t).length;
        //   const falseCount = d.truth_table.length - trueCount;
        //   return Math.min(trueCount, falseCount);
        // }),
        data.map(d => d.sample_count),
      ]);
    }
  }

  let innerWidth = 800;
  $: width = Math.min(innerWidth, 1400);

  $: if (uPlotInst) {
    uPlotInst.setSize({ width, height: 600 });
  }

  const renderChart = (containerNode: HTMLElement) => {
    // generate bar builder with 60% bar (40% gap) & 100px max bar width
    const _bars60_100 = UPlot.paths.bars!({ size: [0.6, 100] });

    uPlotInst = new UPlot({
      width,
      height: 600,
      series: [
        {
          label: 'Formula',
          scale: 'x',
          value: (self: uPlot, rawValue: number, seriesIdx: number, idx: number | null) =>
            idx !== null ? data[idx]?.formula ?? '' : '',
        },
        {
          label: 'Boolean Complexity',
          stroke: colors[0],
          fill: colors[0],
          width: 1,
          paths: _bars60_100,
          points: { show: false },
          scale: 'complexity',
        },
        { label: 'Sample Count', stroke: colors[2], width: 2, scale: 'sample_count' },
      ],
      axes: [
        {},
        {
          show: true,
          label: 'Boolean Complexity',
          stroke: '#ccc',
          ticks: { show: true, stroke: '#ccc', width: 1 },
          grid: { show: true, stroke: '#cccccc88', width: 1 },
          scale: 'complexity',
        },
        {
          show: true,
          label: 'Sample Count',
          stroke: '#ccc',
          ticks: {
            show: true,
            stroke: '#ccc',
            width: 1,
          },
          grid: { show: false },
          scale: 'sample_count',
          side: 1,
        },
      ],
      legend: {
        // live: false,
      },
      scales: {
        x: { time: false },
        complexity: {},
        sample_count: { distr: 3 },
      },
      id: 'function-complexity-plot',
    });
    containerNode.appendChild(uPlotInst.root);
  };
</script>

<svelte:window bind:innerWidth />
<div class="root" use:renderChart />

<style lang="css">
  :global(html) {
    overflow: hidden;
  }

  .root {
    display: flex;
    margin-right: auto;
    margin-left: auto;
  }
</style>
