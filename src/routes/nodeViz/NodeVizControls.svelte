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
    }}
  >
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960" height="24" width="24">
      <path
        fill="#fff"
        d="M451-122q-123-10-207-101t-84-216q0-77 35.5-145T295-695l43 43q-56 33-87 90.5T220-439q0 100 66 173t165 84v60Zm60 0v-60q100-12 165-84.5T741-439q0-109-75.5-184.5T481-699h-20l60 60-43 43-133-133 133-133 43 43-60 60h20q134 0 227 93.5T801-439q0 125-83.5 216T511-122Z"
      />
    </svg>
  </button>
  <button disabled={playState.isPlaying} on:click={() => viz.progressTimestep()}>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960" height="24" width="24">
      <path
        fill="#fff"
        d="M680-240v-480h60v480h-60Zm-460 0v-480l346 240-346 240Zm60-240Zm0 125 181-125-181-125v250Z"
      />
    </svg>
  </button>
  <button on:click={() => void togglePlaying()}>
    {#if playState.isPlaying}
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960" height="24" width="24">
        <path
          fill="#fff"
          d="M525-200v-560h235v560H525Zm-325 0v-560h235v560H200Zm385-60h115v-440H585v440Zm-325 0h115v-440H260v440Zm0-440v440-440Zm325 0v440-440Z"
        />
      </svg>
    {:else}
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 1000 960" height="24" width="24">
        <path fill="#fff" d="M320-203v-560l440 280-440 280Zm60-280Zm0 171 269-171-269-171v342Z" />
      </svg>
    {/if}
  </button>
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: row;
    position: absolute;
    top: 0;
    left: 0;
    width: 110px;
    background-color: #222;
  }

  button {
    background: #444;
    color: #fff;
    border: none;
    padding: 2px;
    margin: 4px;
    cursor: pointer;
    height: 30px;
    width: 30px;
  }

  button:active {
    background: #555;
  }

  button:disabled {
    opacity: 0.5;
    cursor: default;
  }
</style>
