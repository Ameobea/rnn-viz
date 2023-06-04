<script lang="ts" context="module">
  import { onDestroy, onMount } from 'svelte';
  import { RNNGraph } from '../rnn/graph';
  import { browser } from '$app/environment';
  import type { NodeViz } from './NodeViz';
  import { ColorScaleLegend, getColor } from './ColorScale';

  type FetchLayoutState =
    | { type: 'notFetched' }
    | { type: 'loading' }
    | { type: 'error'; error: string }
    | { type: 'loaded'; layoutData: string };

  const buildPlainExtLayout = async (graphDotviz: string): Promise<string> => {
    const res = await fetch('https://dot-server-mi7imxlw6a-uw.a.run.app/dot_to_plainext', {
      method: 'POST',
      headers: {
        'Content-Type': 'text/plain',
      },
      body: graphDotviz,
    });
    if (!res.ok) {
      throw await res.text();
    }
    const data = await res.text();
    return data;
  };
</script>

<script lang="ts">
  import NodeInfo from './NodeInfo.svelte';
  import LogicAnalyzer from './LogicAnalyzer.svelte';
  import NodeVizControls from './NodeVizControls.svelte';
  import { writable, type Writable } from 'svelte/store';

  export let serializedRNNGraph: string;

  const { graph, graphDotviz }: { graph: RNNGraph; graphDotviz: string } = (() => {
    const graph = RNNGraph.deserialize(JSON.parse(serializedRNNGraph));
    const graphDotviz = browser
      ? graph.buildGraphviz({
          arrowhead: false,
          cluster: false,
          // edgeLabels: false,
          aspectRatio:
            window.innerWidth && window.innerHeight
              ? Math.min(Math.max(window.innerHeight / window.innerWidth, 0.65), 1.5)
              : undefined,
        })
      : '';
    return { graph, graphDotviz };
  })();
  const currentTimestep = graph.currentTimestep;

  let windowWidth = browser ? window.innerWidth : 0;
  let windowHeight = browser ? window.innerHeight : 0;
  $: isMobile = windowWidth < 600;

  let layoutDataState: FetchLayoutState = { type: 'notFetched' };

  $: if (layoutDataState.type === 'notFetched' && browser) {
    layoutDataState = { type: 'loading' };
    buildPlainExtLayout(graphDotviz)
      .then(layoutData => {
        layoutDataState = { type: 'loaded', layoutData };
      })
      .catch(error => {
        layoutDataState = { type: 'error', error };
      });
  }

  let NodeVizMod: typeof import('./NodeViz') | null = null;
  let viz: NodeViz | null = null;
  let logicAnalyzerOpen = false;
  $: selected = viz?.selected;
  const logicAnalyzerVisibleNodeIDs: Writable<string[]> = writable([]);
  $: selectedNode =
    NodeVizMod && selected && $selected instanceof NodeVizMod.VizNode ? $selected : null;
  $: selectedNodeID = selectedNode?.name ?? null;

  $: viz?.handleResize(windowWidth, windowHeight);

  onMount(() => {
    if (!browser) {
      return;
    }

    import('./NodeViz').then(mod => {
      NodeVizMod = mod;
    });
  });

  const useNodeViz = (canvas: HTMLCanvasElement) => {
    if (!browser) {
      return;
    }

    if (layoutDataState.type !== 'loaded') {
      throw new Error('Layout data not loaded');
    } else if (!NodeVizMod) {
      throw new Error('NodeViz module not loaded');
    }

    if (viz) {
      viz.destroy();
    }
    viz = new NodeVizMod.NodeViz(canvas, layoutDataState.layoutData, graph);
  };

  const useColorScaleLegend = (node: HTMLDivElement) => {
    if (!browser) {
      return;
    }

    const legend: SVGSVGElement = ColorScaleLegend(getColor, {
      height: isMobile ? 12 : 24,
      width: isMobile ? 200 : 300,
    });
    legend.style.zIndex = '2';
    legend.style.position = 'absolute';
    legend.style.top = '0';
    legend.style.right = '10px';
    node.appendChild(legend);
  };

  onDestroy(() => void viz?.destroy());
</script>

<svelte:window bind:innerWidth={windowWidth} bind:innerHeight={windowHeight} />
<div class="root" use:useColorScaleLegend>
  {#if !NodeVizMod || layoutDataState.type === 'loading'}
    <p>Loading...</p>
  {:else if layoutDataState.type === 'error'}
    <p>Error: {layoutDataState.error}</p>
  {:else if layoutDataState.type === 'loaded'}
    <canvas use:useNodeViz width={windowWidth} height={windowHeight} />
    {#if viz}
      <NodeVizControls {viz} />
      {#if !isMobile}
        <LogicAnalyzer
          currentTimestep={$currentTimestep}
          neuronOutputHistory={graph.neuronOutputHistory}
          {selectedNodeID}
          visibleNodeIDs={$logicAnalyzerVisibleNodeIDs ?? []}
          bind:expanded={logicAnalyzerOpen}
        />
      {/if}
      {#if selectedNode}
        <NodeInfo
          node={selectedNode}
          addToLogicAnalyzer={() => {
            logicAnalyzerVisibleNodeIDs.update(ids => {
              if (!selectedNode || ids.includes(selectedNode.name)) {
                return ids;
              }
              return [...ids, selectedNode.name];
            });
          }}
          {isMobile}
          logicAnalyzerVisibleNodeIDs={$logicAnalyzerVisibleNodeIDs}
          {logicAnalyzerOpen}
          currentTimestep={$currentTimestep}
        />
      {/if}
    {/if}
  {/if}
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }
</style>
