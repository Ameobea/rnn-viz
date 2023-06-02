<script lang="ts">
  import type { NodeViz } from './NodeViz';

  export let viz: NodeViz;
</script>

<div class="root">
  <button on:click={() => viz.reset()}> Reset </button>
  <button on:click={() => viz.progressTimestep()}> +1 timestep </button>
</div>

<style lang="css">
  /* Minimal, monospaced, dark theme UI */

  .root {
    display: flex;
    flex-direction: column;
    position: absolute;
    bottom: 0;
    right: 0;
    width: 300px;
    height: 90px;
    background-color: #222;
  }

  button {
    font-family: monospace;
    font-size: 14px;
    background: #444;
    color: #fff;
    border: none;
    padding: 8px;
    margin: 4px;
    cursor: pointer;
  }
</style>
