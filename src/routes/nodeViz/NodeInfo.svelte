<script lang="ts" context="module">
  const colorToCSS = (color: number): string => {
    const r = (color >> 16) & 0xff;
    const g = (color >> 8) & 0xff;
    const b = color & 0xff;
    return `rgb(${r}, ${g}, ${b})`;
  };
</script>

<script lang="ts">
  import { StateNeuron } from '../rnn/graph';
  import { getColor } from './ColorScale';
  import type { VizNode } from './NodeViz';

  export let isMobile: boolean;
  export let logicAnalyzerOpen: boolean;
  export let logicAnalyzerVisibleNodeIDs: string[];
  export let node: VizNode;
  export let addToLogicAnalyzer: () => void;
  export let currentTimestep: number;

  $: stateNode = node.inner instanceof StateNeuron ? node.inner : null;
  // Fake dependency on `currentTimestamp` to force update every time it changes
  let curOutput = node.inner.getOutput();
  let curOutputColor = getColor(curOutput);
  $: if (currentTimestep > 0) {
    curOutput = node.inner.getOutput();
    curOutputColor = getColor(curOutput);
  }

  $: left = (() => {
    if (isMobile) {
      return 0;
    }
    if (logicAnalyzerOpen) {
      return '640px';
    }
    return 'calc(min(220px, 50vw))';
  })();

  // logic analyzer is not enabled on mobile
  $: isInLogicAnalyzer = isMobile || logicAnalyzerVisibleNodeIDs.includes(node.name);
</script>

<div class="root" style="left: {left}">
  <h3>{node.name}</h3>
  {#if stateNode}
    <div class="info-item">
      Initial state:
      <span class="output-display" style={`color: ${colorToCSS(getColor(stateNode.initialState))}`}>
        {stateNode.initialState}
      </span>
    </div>
  {/if}
  <div class="info-item">
    Current {stateNode ? 'state' : 'output'}:
    <span class="output-display" style={`color: ${colorToCSS(curOutputColor)}`}>{curOutput}</span>
  </div>
  <div class="info-item">
    Bias:
    <span class="output-display" style={`color: ${colorToCSS(getColor(node.inner.bias))}`}
      >{node.inner.bias}</span
    >
  </div>
  {#if !isInLogicAnalyzer}
    <div class="add-to-logic-analyzer">
      <button on:click={addToLogicAnalyzer}>Add to logic analyzer</button>
    </div>
  {/if}
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
    width: calc(min(400px, 50vw));
    position: absolute;
    bottom: 0;
    background-color: #111;
    font-family: monospace;
    border: 1px solid #444;
    box-sizing: border-box;
    min-height: 200px;
  }

  h3 {
    margin: 0;
    padding: 10px;
    color: #fff;
    font-size: 14px;
  }

  .info-item {
    padding-left: 10px;
    padding-right: 10px;
  }

  .add-to-logic-analyzer {
    display: flex;
    flex-direction: row;
    align-items: space-between;
    padding: 10px;
    margin-top: auto;
  }

  .add-to-logic-analyzer button {
    flex: 1;
    font-family: monospace;
    font-size: 12px;
    border-radius: 0;
    border: none;
    cursor: pointer;
  }

  .output-display {
    background-color: #000;
    padding-left: 2px;
    padding-right: 2px;
  }
</style>
