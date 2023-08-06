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
  import type { Writable } from 'svelte/store';

  import UPlot from 'uplot';
  import 'uplot/dist/uPlot.min.css';

  export let currentTimestep: number;
  export let neuronOutputHistory: Map<string, number[]>;
  export let selectedNodeID: string | null;
  export let toggleSelecteNodeID: (nodeID: string) => void;
  export let visibleNodeIDs: Writable<string[]>;
  export let expanded = false;

  $: nodeIDsToRender = [...neuronOutputHistory.keys()]
    .filter(nodeID => $visibleNodeIDs.includes(nodeID))
    .sort((a, b) => {
      // first input_n
      // then layer_n_state_n
      // then layer_n_recurrent_n
      // then layer_n_output_n
      // then post_layer_output_n
      // then output_n
      //
      // All sorted secondarily by n

      const [aType, aNum] = a.split('_');
      const [bType, bNum] = b.split('_');
      if (aType === 'input' && bType !== 'input') {
        return -1;
      }
      if (aType !== 'input' && bType === 'input') {
        return 1;
      }
      if (aType === 'output' && bType !== 'output') {
        return 1;
      }
      if (aType !== 'output' && bType === 'output') {
        return -1;
      }

      const [aLayer, aSubType, aSubNum] = aNum.split('_');
      const [bLayer, bSubType, bSubNum] = bNum.split('_');

      if (aLayer !== bLayer) {
        return parseInt(aLayer) - parseInt(bLayer);
      }

      if (aSubType !== bSubType) {
        return aSubType.localeCompare(bSubType);
      }

      return parseInt(aSubNum) - parseInt(bSubNum);
    });

  const uPlotInstsByNodeID: Map<string, UPlot> = new Map();

  const getDataForInst = (
    nodeID: string,
    currentTimestep: number,
    neuronOutputHistory: Map<string, number[]>
  ) => {
    const xs = new Array<number>(20).fill(currentTimestep).map((_, i) => i);
    const ys: (number | null)[] = (neuronOutputHistory.get(nodeID) ?? [])
      .slice(-19)
      .map(y => clamp(-1, 1, y));
    ys.push(null);
    return { xs, ys };
  };

  $: for (const [nodeID, inst] of uPlotInstsByNodeID.entries()) {
    const { xs, ys } = getDataForInst(nodeID, currentTimestep, neuronOutputHistory);
    inst.setData([xs, ys]);
  }

  const renderChart = (container: HTMLDivElement, nodeID: string) => {
    const inst =
      uPlotInstsByNodeID.get(nodeID) ??
      (() => {
        const inst = buildInst(nodeID);
        uPlotInstsByNodeID.set(nodeID, inst);
        return inst;
      })();
    container.appendChild(inst.root);

    const { xs, ys } = getDataForInst(nodeID, currentTimestep, neuronOutputHistory);
    inst.setData([xs, ys]);

    return {
      destroy: () => {
        container.removeChild(inst.root);
        inst.destroy();
        uPlotInstsByNodeID.delete(nodeID);
      },
      update: (newNodeID: string) => {
        if (newNodeID !== nodeID) {
          throw new Error('should never happen in keyed each');
        }
      },
    };
  };
</script>

{#if expanded}
  <div class="root">
    {#each nodeIDsToRender as nodeID (nodeID)}
      <div
        class={`node-output-chart-container${
          selectedNodeID === nodeID ? ' node-output-graph-selected' : ''
        }`}
        id={nodeID}
      >
        <div class="node-output-chart-label">{nodeID}</div>
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        <div
          class="chart-container"
          use:renderChart={nodeID}
          on:click={() => toggleSelecteNodeID(nodeID)}
        />
        <div class="last-output-value">
          <button
            class="remove-chart-button"
            on:click={() => visibleNodeIDs.update(nodeIDs => nodeIDs.filter(id => id !== nodeID))}
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
    Open Logic Analyzer
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
    height: 368px;
    overflow-y: auto;
    border: 1px solid #444;
    box-sizing: border-box;
  }

  button {
    font-family: monospace;
    font-size: 15px;
    padding: 4px 0px;
    border-radius: 0;
    outline: none;
    border: 1px solid #444;
    cursor: pointer;
  }

  .collapsed {
    width: calc(min(220px, 50vw));
    position: absolute;
    bottom: 0;
    left: 0;
  }

  .collapse {
    width: 100px;
    position: absolute;
    bottom: 368px;
    height: 20px;
    line-height: 0;
  }

  .node-output-chart-container {
    border-bottom: 1px solid #444;
    display: flex;
    position: relative;
    padding-right: 4px;
  }

  .chart-container {
    cursor: pointer;
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
    background-color: #03777d55;
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
