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

  // console.log(countBinary(4, 1 / 2, 20));

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
</script>

<script lang="ts">
  import OutputPlot from 'src/components/OutputPlot.svelte';
  import { onMount } from 'svelte';

  let activate:
    | ((
        vals: number[]
      ) => Promise<{ 2: number; 3: number; 4: number; 5: number; 6: number; 11: number }>)
    | null = null;
  let tfc: typeof import('../nn/ameoActivation').tfc | null = null;
  onMount(async () => {
    const activationMod = await import('../nn/ameoActivation');
    activationMod.tfc.setBackend('cpu');
    const activation = new activationMod.InterpolatedAmeo(0.0, 0);

    activate = async (vals: number[]) => {
      const sum2 = vals.map((val, ix) => val * weights2[ix]).reduce((a, b) => a + b, 0) + bias2;
      const sum3 = vals.map((val, ix) => val * weights3[ix]).reduce((a, b) => a + b, 0) + bias3;
      const sum4 = vals.map((val, ix) => val * weights4[ix]).reduce((a, b) => a + b, 0) + bias4;
      const sum5 = vals.map((val, ix) => val * weights5[ix]).reduce((a, b) => a + b, 0) + bias5;
      const sum6 = vals.map((val, ix) => val * weights6[ix]).reduce((a, b) => a + b, 0) + bias6;
      const sum11 = vals.map((val, ix) => val * weights11[ix]).reduce((a, b) => a + b, 0) + bias11;

      const out2 = (
        (await activation
          .apply(activationMod.tfc.tensor1d(new Float32Array([sum2])))
          .data()) as Float32Array
      )[0];
      const out3 = (
        (await activation
          .apply(activationMod.tfc.tensor1d(new Float32Array([sum3])))
          .data()) as Float32Array
      )[0];
      const out4 = (
        (await activation
          .apply(activationMod.tfc.tensor1d(new Float32Array([sum4])))
          .data()) as Float32Array
      )[0];
      const out5 = (
        (await activation
          .apply(activationMod.tfc.tensor1d(new Float32Array([sum5])))
          .data()) as Float32Array
      )[0];
      const out6 = (
        (await activation
          .apply(activationMod.tfc.tensor1d(new Float32Array([sum6])))
          .data()) as Float32Array
      )[0];
      const out11 = (
        (await activation
          .apply(activationMod.tfc.tensor1d(new Float32Array([sum11])))
          .data()) as Float32Array
      )[0];
      return { 2: out2, 3: out3, 4: out4, 5: out5, 6: out6, 11: out11 };
    };
    tfc = activationMod.tfc;
  });

  let output: { 2: number; 3: number; 4: number; 5: number; 6: number; 11: number } | null = null;
  const inputs: [boolean[], boolean[]] = [new Array(8).fill(false), new Array(8).fill(false)];

  $: {
    if (!activate || !tfc) output = null;
    else {
      output = null;
      const inputs0 = inputs[0].map(x => (x ? 1 : -1));
      const inputs1 = inputs[1].map(x => (x ? 1 : -1));

      const allInputs = [...inputs0, ...inputs1];
      console.log(allInputs);
      activate(allInputs).then(out => {
        output = out;
      });
    }
  }

  let fullOutputs: { values: number[]; name: string }[] | null = null;
  $: {
    if (!activate) fullOutputs = null;
    else {
      const fullInputs = countBinary(8, 0.1, 32).map(x => [x, []]);
      const localActivate = activate;
      Promise.all(
        fullInputs.map(async inputs => {
          const inputs0 = [...inputs[0], ...new Array(8 - inputs[0].length).fill(-1)];
          const inputs1 = [...inputs[1], ...new Array(8 - inputs[0].length).fill(-1)];
          const allInputs = [...inputs0, ...inputs1];
          return localActivate(allInputs);
        })
      )
        .then(outs =>
          outs.reduce(
            (acc, outs) => {
              Object.entries(outs).forEach(([ix, value]) => {
                const entries: number[] = (acc as any)[ix];
                entries.push(value);
              });

              // const layer1Neuron7Out = layer1Neuron7(outs[2], outs[3], outs[6]);
              // acc['layer1Neuron7'].push(layer1Neuron7Out);

              // const layer1Neuron9Out = layer1Neuron9(outs[11], outs[6]);
              // acc['layer1Neuron9'].push(layer1Neuron9Out);

              return acc;
            },
            {
              2: [] as number[],
              3: [] as number[],
              4: [] as number[],
              5: [] as number[],
              6: [] as number[],
              11: [] as number[],
              // layer1Neuron7: [] as number[],
              // layer1Neuron9: [] as number[],
            }
          )
        )
        .then(vals => {
          fullOutputs = Object.entries(vals).map(([name, values]) => ({ name, values }));
        });
    }
  }
</script>

<div class="root">
  <!-- <div class="checkboxes-container">
    {#each Array(3) as _, i}
      <input type="checkbox" bind:checked={inputs[0][i]} />
    {/each}
  </div>
  <div class="checkboxes-container">
    {#each Array(3) as _, i}
      <input type="checkbox" bind:checked={inputs[1][i]} />
    {/each}
  </div> -->

  <div class="output">
    <!-- {#if output !== null && output !== undefined}
      2: {output['2']}
      <br />
      3: {output['3']}
      <br />
      4: {output['4']}
      <br />
      5: {output['5']}
      <br />
      6: {output['6']}
      <br />
      11: {output['11']}
      <br />
      combined: {layer1Neuron7(output['2'], output['3'], output['6'])}
    {:else}
      <span>Loading...</span>
    {/if} -->

    {#if fullOutputs}
      <OutputPlot
        outputs={fullOutputs.filter(o => o.name !== '4' && o.name !== '5')}
        xIncrement={0.1}
      />
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

  .checkboxes-container {
    display: flex;
    flex-direction: row;
    margin-bottom: 10px;
  }

  .output {
    font-size: 30px;
    margin-top: 50px;
    font-family: 'Hack', 'IBM Plex Mono', 'Oxygen Mono', 'Input', 'Ubuntu Mono', 'Liberation Mono',
      'Courier New', monospace;
  }
</style>
