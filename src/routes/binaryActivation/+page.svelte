<script lang="ts" context="module">
  const countBinary = (bits: number, increment: number, max: number): number[][] => {
    let acc = 0;
    const outs: number[][] = [];

    for (;;) {
      if (acc >= max) {
        break;
      }

      const cur = Math.trunc(acc).toString(2);
      const digits = cur
        .split('')
        .reverse()
        .map(d => (d === '0' ? -1 : 1));
      while (digits.length < bits) {
        digits.push(-1);
      }

      // We shove the remainder into the first bit, which works out for our use case
      digits[0] += (acc - Math.trunc(acc)) * 2;
      outs.push(digits);

      acc += increment;
    }

    return outs;
  };

  const weights0 = [
    0,
    0,
    -1 / 32,
    -1 / 16,
    -1 / 8,
    -1 / 4,
    -1 / 2,
    1,
    0,
    0,
    -1 / 32,
    -1 / 16,
    -1 / 8,
    -1 / 4,
    -1 / 2,
    -1,
  ];
  const bias0 = 0;

  const weights2 = [1 / 6, 2 / 6, 4 / 6, 0, 0, 0, 0, 0, 1 / 6, 2 / 6, 4 / 6, 0, 0, 0, 0, 0];
  const bias2 = -1 / 6;

  const weights3 = [1 / 2, 1, 2, 0, 0, 0, 0, 0, 1 / 2, 1, 2, 0, 0, 0, 0, 0];
  const bias3 = -1 / 2;

  const weights4 = [
    -1 / 32,
    -1 / 16,
    -1 / 8,
    -1 / 4,
    -1 / 2,
    -1,
    0,
    0,
    -1 / 32,
    -1 / 16,
    -1 / 8,
    -1 / 4,
    -1 / 2,
    -1,
    0,
    0,
  ];
  const bias4 = 1 / 32;

  const weights5 = [1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0];
  const bias5 = 1;

  const weights6 = [0.25, 0.5, 1, 0, 0, 0, 0, 0, 0.25, 0.5, 1, 0, 0, 0, 0, 0];
  const bias6 = -1 / 4;

  const weights8 = [1 / 8, 1 / 4, 1 / 2, -1, 0, 0, 0, 0, 1 / 8, 1 / 4, 1 / 2, 1, 0, 0, 0, 0, 0];
  const bias8 = -1 / 8;

  const weights11 = [
    1 / 16,
    1 / 8,
    1 / 4,
    1 / 2,
    -1,
    0,
    0,
    0,
    1 / 16,
    1 / 8,
    1 / 4,
    1 / 2,
    1,
    0,
    0,
    0,
    0,
  ];
  const bias11 = -1 / 16;

  const layer1Neuron7 = (out2: number, out3: number, out6: number) =>
    Math.tanh(-2.25 * out2 + -2.06 * out3 + 1.989 * out6);

  const layer1Neuron9 = (out11: number, out6: number) => Math.tanh(5.38 * out11 + 0.726 * out6);

  const activate = (vals: number[], engine: typeof import('../../engineComp/engine')) => {
    const sum0 = vals.map((val, ix) => val * weights0[ix]).reduce((a, b) => a + b, 0) + bias0;
    const sum2 = vals.map((val, ix) => val * weights2[ix]).reduce((a, b) => a + b, 0) + bias2;
    const sum3 = vals.map((val, ix) => val * weights3[ix]).reduce((a, b) => a + b, 0) + bias3;
    const sum4 = vals.map((val, ix) => val * weights4[ix]).reduce((a, b) => a + b, 0) + bias4;
    const sum5 = vals.map((val, ix) => val * weights5[ix]).reduce((a, b) => a + b, 0) + bias5;
    const sum6 = vals.map((val, ix) => val * weights6[ix]).reduce((a, b) => a + b, 0) + bias6;
    const sum11 = vals.map((val, ix) => val * weights11[ix]).reduce((a, b) => a + b, 0) + bias11;

    const outs = engine.apply_batch_fused_interpolated_ameo(
      0.1,
      0.005,
      new Float32Array([sum0, sum2, sum3, sum4, sum5, sum6, sum11])
    );
    const [out0, out2, out3, out4, out5, out6, out11] = Array.from(outs);

    return { 0: out0, 2: out2, 3: out3, 4: out4, 5: out5, 6: out6, 11: out11 };
  };
</script>

<script lang="ts">
  import { onMount } from 'svelte';

  import OutputPlot from '../../components/OutputPlot.svelte';

  let engine: typeof import('../../engineComp/engine') | null = null;
  onMount(async () => {
    import('../../engineComp/engine').then(async engineMod => {
      await engineMod.default();
      engine = engineMod;
    });
  });

  let fullOutputs: { values: number[]; name: string }[] | null = null;
  $: {
    if (!engine) fullOutputs = null;
    else {
      const fullInputs = countBinary(8, 0.1, 32).map(x => [x, []]);

      const vals = fullInputs
        .map(inputs => {
          const inputs0: number[] = [...inputs[0], ...new Array(8 - inputs[0].length).fill(-1)];
          const inputs1: number[] = [...inputs[1], ...new Array(8 - inputs[0].length).fill(-1)];
          const allInputs = [...inputs0, ...inputs1];
          return activate(allInputs, engine!);
        })
        .reduce(
          (acc, outs) => {
            Object.entries(outs).forEach(([ix, value]) => {
              const entries: number[] = (acc as any)[ix];
              entries.push(value);
            });

            const layer1Neuron7Out = layer1Neuron7(outs[2], outs[3], outs[6]);
            acc['layer1Neuron7'].push(layer1Neuron7Out);

            const layer1Neuron9Out = layer1Neuron9(outs[11], outs[6]);
            acc['layer1Neuron9'].push(layer1Neuron9Out);

            return acc;
          },
          {
            0: [] as number[],
            2: [] as number[],
            3: [] as number[],
            4: [] as number[],
            5: [] as number[],
            6: [] as number[],
            11: [] as number[],
            layer1Neuron7: [] as number[],
            layer1Neuron9: [] as number[],
          }
        );
      fullOutputs = Object.entries(vals)
        .filter(([key]) => ['2', '3', '11'].includes(key))
        .map(([_name, values], i) => ({ name: i.toString(), values }));
    }
  }
</script>

<div class="root">
  <div class="output">
    {#if fullOutputs}
      <OutputPlot outputs={fullOutputs} xIncrement={0.1} />
    {/if}
  </div>
</div>

<style lang="css">
  .root {
    margin-top: -40px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .output {
    font-size: 30px;
    margin-top: 50px;
    font-family: 'Hack', 'IBM Plex Mono', 'Oxygen Mono', 'Input', 'Ubuntu Mono', 'Liberation Mono',
      'Courier New', monospace;
  }
</style>
