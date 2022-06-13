<script lang="ts">
  import type { RNNDefinition } from 'src/nn/RNN';

  import { onMount } from 'svelte';

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
    const { RNNViz } = await import('src/viz/RNNViz');
    const viz = new RNNViz(def);
    console.log(viz);
  });
</script>

<div class="root">TODO</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }
</style>
