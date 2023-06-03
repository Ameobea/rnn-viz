<script lang="ts" context="module">
  const clamp = (min: number, max: number, val: number) => Math.min(max, Math.max(min, val));

  const buildInst = (nodeID: string) => {
    let stroke = '#03a9f4';
    switch (nodeID.split('_')[0]) {
      case 'input':
        stroke = '#12ff12';
        break;
      case 'output':
        stroke = '#ee12ee';
        break;
    }

    return new UPlot({
      height: 60,
      width: Math.min((window?.innerWidth ?? 10000) - 100, 600),
      pxAlign: true,
      series: [
        {},
        {
          stroke,
          paths: UPlot.paths.stepped!({ align: 1 }),
          points: {
            show: false,
          },
          scale: 'out',
        },
      ],
      cursor: {
        show: false,
      },
      axes: [
        {
          show: false,
        },
        {
          show: false,
        },
      ],
      legend: {
        show: false,
      },
      scales: {
        x: { time: false },
        out: {
          range: [-1.2, 1.2],
        },
      },
    });
  };
</script>

<script lang="ts">
  import UPlot from 'uplot';
  import 'uplot/dist/uPlot.min.css';

  export let currentTimestep: number;
  export let neuronOutputHistory: Map<string, number[]>;
  export let selectedNodeID: string | null;
  export let visibleNodeIDs: string[];

  let expanded = false;

  $: nodeIDsToRender = [...neuronOutputHistory.keys()]
    .filter(
      nodeID =>
        nodeID.startsWith('input') || nodeID.startsWith('output') || visibleNodeIDs.includes(nodeID)
    )
    .sort((a, b) => {
      // first input_n
      // then layer_n_state_n
      // then layer_n_recurrent_n
      // then layer_n_output_n
      // then post_layer_output_n
      // then output_n
      //
      // All sorted secondarily by n
      const aParts = a.split('_');
      const bParts = b.split('_');
      const aType = aParts[0];
      const bType = bParts[0];

      const partWeight = (part: string) => {
        if (part.startsWith('input')) {
          return 0;
        } else if (part.startsWith('layer')) {
          return 1;
        } else if (part.startsWith('post')) {
          return 2;
        } else if (part.startsWith('output')) {
          return 3;
        } else {
          return 4;
        }
      };

      const aPartWeight = partWeight(aType);
      const bPartWeight = partWeight(bType);

      if (aPartWeight !== bPartWeight) {
        return aPartWeight - bPartWeight;
      }

      const aNum = parseInt(aParts[aParts.length - 1]);
      const bNum = parseInt(bParts[bParts.length - 1]);

      return aNum - bNum;
    });

  let uPlotInsts: { nodeID: string; inst: UPlot }[] = [];

  $: xs = new Array<number>(currentTimestep + 2).fill(0).map((_, i) => i);
  $: for (const { inst, nodeID } of uPlotInsts) {
    const xsSlice = xs.slice(-21);
    const ys: (number | null)[] = (neuronOutputHistory.get(nodeID) ?? [])
      .slice(-20)
      .map(y => clamp(-1, 1, y));
    ys.push(null);
    inst.setData([xsSlice, ys]);
  }

  const renderChart = (container: HTMLDivElement, nodeID: string) => {
    const inst = buildInst(nodeID);
    uPlotInsts.push({ nodeID, inst });
    container.appendChild(inst.root);
  };
</script>

{#if expanded}
  <div class="root">
    {#each nodeIDsToRender as nodeID}
      <div
        class={`node-output-chart-container${
          selectedNodeID === nodeID ? ' node-output-graph-selected' : ''
        }`}
        id={nodeID}
      >
        <div class="node-output-chart-label">{nodeID}</div>
        <div class="chart-container" use:renderChart={nodeID} />
        <div class="last-output-value">
          <button
            class="remove-chart-button"
            on:click={() => {
              // TODO
            }}
          >
            ✕
          </button>
          {neuronOutputHistory.get(nodeID)?.[currentTimestep].toFixed(2)}
        </div>
      </div>
    {/each}
  </div>
  <button
    class="collapse"
    on:click={() => {
      expanded = false;
    }}
  >
    Hide
  </button>
{:else}
  <button
    class="collapsed"
    on:click={() => {
      expanded = true;
    }}
  >
    Show Outputs Timeline
  </button>
{/if}

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
    position: absolute;
    bottom: 0;
    left: 0;
    background-color: #111;
    height: 300px;
    overflow-y: auto;
    border: 1px solid #444;
    box-sizing: border-box;
  }

  button {
    font-family: monospace;
    font-size: 14px;
  }

  .collapsed {
    width: 220px;
    position: absolute;
    bottom: 0;
    left: 0;
  }

  .collapse {
    width: 100px;
    position: absolute;
    bottom: 300px;
    height: 20px;
    line-height: 0;
  }

  .node-output-chart-container {
    border-bottom: 1px solid #444;
    display: flex;
    position: relative;
    padding-right: 4px;
  }

  .node-output-chart-label {
    font-size: 10px;
    font-family: monospace;
    position: absolute;
    top: 0;
    left: 0;
    background-color: rgba(0, 0, 0, 0.2);
    z-index: 1;
    padding-left: 2px;
    padding-top: 1px;
  }

  .node-output-graph-selected {
    background-color: #03777daa;
  }

  .last-output-value {
    font-size: 10px;
    font-family: monospace;
    width: 28px;
    text-align: right;
    display: flex;
    align-items: flex-end;
    justify-content: flex-end;
    padding-left: 3px;
    padding-right: 3px;
    padding-bottom: 1px;
    border-left: 1px solid #444;
  }

  .last-output-value .remove-chart-button {
    font-size: 15px;
    font-weight: bold;
    font-family: monospace;
    line-height: 0;
    padding: 0;
    border: none;
    background-color: transparent;
    color: #a10707;
    cursor: pointer;
    position: absolute;
    top: 6px;
    right: 0px;
  }
</style>
