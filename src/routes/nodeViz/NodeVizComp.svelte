<script lang="ts" context="module">
  import { onDestroy, onMount } from 'svelte';
  import type { RNNGraph } from '../rnn/graph';
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
  export let graph: RNNGraph;

  $: graphDotviz = browser ? graph.buildGraphviz({ arrowhead: false, cluster: false }) : '';

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

  let windowWidth = browser ? window.innerWidth : 0;
  let windowHeight = browser ? window.innerHeight : 0;

  let viz: NodeViz | null = null;
  let NodeVizMod: typeof import('./NodeViz') | null = null;

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

    const legend: SVGSVGElement = ColorScaleLegend(getColor, { height: 24, width: 300 });
    legend.style.zIndex = '2';
    legend.style.position = 'absolute';
    legend.style.top = '0';
    legend.style.right = '14px';
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
  {/if}
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }
</style>
