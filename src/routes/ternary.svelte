<script lang="ts">
  import { onMount } from 'svelte';

  onMount(async () => {
    const { AmeoTestbed } = await import('../nn/ameoTestbed');

    // const targetFunction = ([cond, ifTrue, ifFalse]: boolean[]) => (cond ? ifTrue : ifFalse);
    // const targetFunction = ([a, b, c]: number[]) => (a && b ? c : 0);
    const xor = ([a, b]: boolean[]) => (!a && b) || (!b && a);
    const or = ([a, b]: boolean[]) => a || b;
    const and = ([a, b]: boolean[]) => a && b;
    const xnor = ([a, b]: boolean[]) => (a && b) || (!a && !b);
    const nand = ([a, b]: boolean[]) => !(a && b);
    const not = ([a]: boolean[]) => !a;
    const allTrueButOne = (vals: boolean[]) => vals.filter(val => val).length === vals.length - 1;
    const allTrueButOneOrNotAll = (vals: boolean[]) =>
      allTrueButOne(vals) || vals.every(val => !val);
    const tern = ([a, b, c]: boolean[]) => (a ? b : c);
    const targetFunction = (vals: boolean[]) => allTrueButOneOrNotAll(vals) || tern(vals);

    const testbed = new AmeoTestbed({
      inputSize: 3,
      targetFunction,
      learningRate: 0.1,
      initialization: { type: 'random', scale: 0.3 },
      iterations: 200,
      batchSize: 4,
      variant: 'softLeakyAmeo',
    });
    const res = testbed.run(1);
  });
</script>
