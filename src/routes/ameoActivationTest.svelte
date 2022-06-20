<script lang="ts">
  import { onMount } from 'svelte';

  onMount(async () => {
    const { Ameo, tfc } = await import('../nn/ameoActivation');
    const ameo = new Ameo();
    const ameoGrad = tfc.grad(x => ameo.apply(x));
    const input = tfc.tensor1d([-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 2, 3]);
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
