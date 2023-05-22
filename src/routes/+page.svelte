<script lang="ts">
  import type { RNNDefinition } from '../nn/RNN';

  import { onMount } from 'svelte';

  let canvas: HTMLCanvasElement | null = null;

  onMount(async () => {
    const maxSequenceLength = 8;
    const def: RNNDefinition = [
      {
        inputShape: [maxSequenceLength, 1],
        cell: [
          {
            stateSize: 2,
            outputDim: 1,
            recurrentActivation: 'linear',
            outputActivation: 'linear',
            useOutputBias: false,
            useRecurrentBias: false,
            recurrentInitializer: 'glorotNormal',
          },
        ],
        returnSequences: true,
        trainableInitialState: true,
      },
    ];
    const { RNNViz } = await import('../viz/RNNViz');
    if (!canvas) {
      throw new Error('Expected canvas to be mounted by now');
    }
    const viz = new RNNViz(canvas, def);
  });
</script>

<div class="root">
  <canvas bind:this={canvas} />
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }
</style>
