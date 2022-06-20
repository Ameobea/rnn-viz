<script lang="ts">
  import { onMount } from 'svelte';

  onMount(async () => {
    const { AmeoTestbed } = await import('../nn/ameoTestbed');

    const targetFunction = ([cond, ifTrue, ifFalse]: number[]) => (cond ? ifTrue : ifFalse);

    const testbed = new AmeoTestbed({
      inputSize: 3,
      targetFunction,
      learningRate: 0.15,
      initialization: { type: 'random', scale: 0.1 },
      iterations: 600,
    });
    const res = testbed.run(1);
  });
</script>
