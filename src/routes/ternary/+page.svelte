<script lang="ts">
  import { onMount } from 'svelte';

  onMount(async () => {
    const { AmeoTestbed, formatTestbedRunResult, tf } = await import('../../nn/ameoTestbed');
    tf.setBackend('cpu');

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
    // const targetFunction = (vals: boolean[]) => allTrueButOneOrNotAll(vals) || tern(vals);

    const repro = ([a, b, c]: boolean[]) => {
      // [
      //   ((false, false, false), false),
      //   ((false, false, true), false),
      //   ((false, true, false), false),
      //   ((false, true, true), true),
      //   ((true, false, false), true),
      //   ((true, false, true), false),
      //   ((true, true, false), false),
      //   ((true, true, true), false)
      // ]
      switch (`${a}-${b}-${c}`) {
        case 'false-false-false':
          return false;
        case 'false-false-true':
          return false;
        case 'false-true-false':
          return false;
        case 'false-true-true':
          return true;
        case 'true-false-false':
          return true;
        case 'true-false-true':
          return false;
        case 'true-true-false':
          return false;
        case 'true-true-true':
          return false;
        default:
          throw new Error(`Unhandled: ${a}-${b}-${c}`);
      }
    };

    const attempts = 50;
    const testbed = new AmeoTestbed({
      inputSize: 3,
      targetFunction: repro,
      learningRate: 0.8,
      initialization: { type: 'normalDistribution', stdDev: 1.5 },
      iterations: 2000,
      batchSize: 2,
      variant: 'interpolatedAmeo',
      perfectCostThreshold: 0.15 * 0.15,
      optimizer: 'adam',
    });
    const res = await testbed.run(attempts);
    console.log(formatTestbedRunResult(res));
  });
</script>
