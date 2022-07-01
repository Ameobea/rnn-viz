<script lang="ts">
  import { onMount } from 'svelte';

  onMount(async () => {
    const { AmeoTestbed, formatTestbedRunResult } = await import('../nn/ameoTestbed');

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

    const attempts = 100;
    const testbed = new AmeoTestbed({
      inputSize: 2,
      targetFunction: tern,
      learningRate: 0.35,
      initialization: { type: 'random', scale: 0.6 },
      iterations: 1000,
      batchSize: 2,
      variant: 'ameo',
      perfectCostThreshold: 0.0025,
      optimizer: 'adam',
    });
    const res = testbed.run(attempts);
    console.log(formatTestbedRunResult(res));
  });
</script>
