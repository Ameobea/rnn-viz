<script lang="ts">
  import { onMount } from 'svelte';

  onMount(async () => {
    const { Ameo, SoftLeakyAmeo, tfc } = await import('../nn/ameoActivation');
    const ameo = new Ameo();
    const ameoGrad = tfc.grad(x => ameo.apply(x));
    const min = -1.2;
    const max = 1.2;
    const input = tfc.tensor1d(
      new Array(200).fill(0).map((_, i) => {
        const x = min + ((max - min) * i) / 199;
        return x;
      })
    );
    console.log('input:', Array.from(input.dataSync()));
    const output = ameo.apply(input, undefined);
    console.log('output:', Array.from(output.dataSync()));
    const grad = ameoGrad(input, undefined);
    console.log('gradient:', Array.from(grad.dataSync()));

    const buildAmeoNeuron =
      ([condWeight, ifTrueWeight, ifFalseWeight]: [number, number, number], bias: number) =>
      ([cond, ifTrue, ifFalse]: [number, number, number]) => {
        const beforeActivation =
          condWeight * cond + ifTrueWeight * ifTrue + ifFalseWeight * ifFalse + bias;
        const activation = ameo.apply(tfc.tensor1d([beforeActivation]));
        return activation;
      };

    const ameoNeuron = buildAmeoNeuron([-1, 1, 4], -3);
    (window as any).ameoNeuron = ameoNeuron;
    console.log(ameoNeuron);
  });
</script>
