<script lang="ts" context="module">
  import { onMount } from 'svelte';
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
    // check if we have a cached version
    if (
      localStorage.getItem('lastGraphDotviz') === graphDotviz &&
      localStorage.getItem('lastGraphPlainExt')
    ) {
      return localStorage.getItem('lastGraphPlainExt') ?? '';
    }

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

    // cache the result
    localStorage.setItem('lastGraphDotviz', graphDotviz);
    localStorage.setItem('lastGraphPlainExt', data);

    return data;
  };
</script>

<script lang="ts">
  import NodeInfo from './NodeInfo.svelte';
  import LogicAnalyzer from './LogicAnalyzer.svelte';
  import NodeVizControls from './NodeVizControls.svelte';
  import { writable, type Writable } from 'svelte/store';

  export let serializedRNNGraph: string | RNNGraph;
  export let excludedNodeIDs: string[] = [];
  export let aspectRatio: number | undefined = undefined;

  const { graph, graphDotviz }: { graph: RNNGraph; graphDotviz: string } = (() => {
    const graph =
      typeof serializedRNNGraph === 'string'
        ? RNNGraph.deserialize(JSON.parse(serializedRNNGraph))
        : serializedRNNGraph;
    const graphDotviz = browser
      ? graph.buildGraphviz({
          arrowhead: false,
          cluster: false,
          edgeLabels: false,
          clusterInputs: false,
          aspectRatio:
            aspectRatio ??
            (window.innerWidth && window.innerHeight
              ? Math.min(Math.max(window.innerHeight / window.innerWidth, 0.65), 1.5)
              : undefined),
          excludedNodeIDs,
        })
      : '';
    return { graph, graphDotviz };
  })();
  const currentTimestep = graph.currentTimestep;

  let windowWidth = browser ? window.innerWidth : 0;
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
  const logicAnalyzerVisibleNodeIDs: Writable<string[]> = writable(
    [...graph.neuronOutputHistory.keys()].filter(
      nodeID => nodeID.startsWith('input') || nodeID.startsWith('output')
    )
  );
  $: selectedNode =
    NodeVizMod && selected && $selected instanceof NodeVizMod.VizNode ? $selected : null;
  $: selectedNodeID = selectedNode?.name ?? null;
  const toggleSelecteNodeID = (nodeID: string) => viz?.toggleSelecteNodeID(nodeID);

  onMount(() => {
    if (!browser) {
      return;
    }

    import('./NodeViz').then(mod => {
      NodeVizMod = mod;
    });
  });

  const useNodeViz = (svg: SVGSVGElement) => {
    if (!browser) {
      return;
    }

    if (layoutDataState.type !== 'loaded') {
      throw new Error('Layout data not loaded');
    } else if (!NodeVizMod) {
      throw new Error('NodeViz module not loaded');
    }

    viz = new NodeVizMod.NodeViz(svg, layoutDataState.layoutData, graph);
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
</script>

<svelte:window bind:innerWidth={windowWidth} />
<div class="root" use:useColorScaleLegend>
  {#if !NodeVizMod || layoutDataState.type === 'loading'}
    <p>Loading...</p>
  {:else if layoutDataState.type === 'error'}
    <p>Error: {layoutDataState.error}</p>
  {:else if layoutDataState.type === 'loaded'}
    <svg use:useNodeViz>
      <!-- FROM: https://codepen.io/dipscom/pen/mVYjPw -->
      <defs>
        <filter id="sofGlow" height="300%" width="300%" x="-75%" y="-75%">
          <!-- Thicken out the original shape -->
          <feMorphology operator="dilate" radius="10" in="SourceAlpha" result="thicken" />

          <!-- Use a gaussian blur to create the soft blurriness of the glow -->
          <feGaussianBlur in="thicken" stdDeviation="10" result="blurred" />

          <!-- Change the colour -->
          <feFlood flood-color="rgb(160,160,160)" result="glowColor" />

          <!-- Color in the glows -->
          <feComposite in="glowColor" in2="blurred" operator="in" result="softGlow_colored" />

          <!--	Layer the effects together -->
          <feMerge>
            <feMergeNode in="softGlow_colored" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
    </svg>
    {#if viz}
      <NodeVizControls {viz} />
      {#if !isMobile}
        <LogicAnalyzer
          currentTimestep={$currentTimestep}
          neuronOutputHistory={graph.neuronOutputHistory}
          {selectedNodeID}
          visibleNodeIDs={logicAnalyzerVisibleNodeIDs}
          bind:expanded={logicAnalyzerOpen}
          {toggleSelecteNodeID}
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
    width: 100vw;
    height: 100vh;
  }

  svg {
    width: 100%;
    height: 100%;
  }
</style>
