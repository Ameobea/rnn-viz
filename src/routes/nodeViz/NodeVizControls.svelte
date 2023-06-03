<script lang="ts">
  import type { NodeViz } from './NodeViz';

  const PlayInterval = 500;

  export let viz: NodeViz;

  let playState: { isPlaying: true; intervalHandle: number } | { isPlaying: false } = {
    isPlaying: false,
  };

  const startPlaying = () => {
    if (playState.isPlaying) {
      return;
    }

    playState = {
      isPlaying: true,
      intervalHandle: setInterval(() => {
        viz.progressTimestep();
      }, PlayInterval) as any,
    };
  };

  const stopPlaying = () => {
    if (!playState.isPlaying) {
      return;
    }

    clearInterval(playState.intervalHandle);
    playState = { isPlaying: false };
  };

  const togglePlaying = () => {
    if (playState.isPlaying) {
      stopPlaying();
    } else {
      startPlaying();
    }
  };
</script>

<div class="root">
  <button
    on:click={() => {
      stopPlaying();
      viz.reset();
    }}>Reset</button
  >
  <button disabled={playState.isPlaying} on:click={() => viz.progressTimestep()}>
    +1 timestep
  </button>
  <button
    on:click={() => {
      togglePlaying();
    }}
  >
    {playState.isPlaying ? 'Pause' : 'Play'}
  </button>
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
    position: absolute;
    bottom: 0;
    right: 0;
    width: 200px;
    background-color: #222;
  }

  button {
    font-family: monospace;
    font-size: 11px;
    background: #444;
    color: #fff;
    border: none;
    padding: 8px;
    margin: 4px;
    cursor: pointer;
  }

  button:disabled {
    opacity: 0.5;
    cursor: default;
  }
</style>
