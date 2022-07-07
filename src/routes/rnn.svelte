<script context="module" lang="ts">
  const randomBoolInput = () => (Math.random() > 0.5 ? 1 : -1);

  const xor = (a: -1 | 1, b: -1 | 1): -1 | 1 =>
    (a === -1 && b === 1) || (a === 1 && b === -1) ? 1 : -1;
</script>

<script lang="ts">
  import type { Rank, Tensor } from '@tensorflow/tfjs';

  import { onMount } from 'svelte';

  onMount(async () => {
    const { tf, ...mod } = await import('../nn/customRNN');
    tf.setBackend('cpu');
    const rnn = new mod.MyRNN({
      cell: [
        new mod.MySimpleRNNCell({
          stateSize: 1,
          outputDim: 1,
          outputActivation: { type: 'softLeakyAmeo' },
          recurrentActivation: { type: 'softLeakyAmeo' },
          // recurrentActivation: 'linear',
          useOutputBias: true,
          useRecurrentBias: true,
          biasInitializer: 'glorotNormal',
          recurrentInitializer: 'glorotNormal',
          kernelInitializer: 'glorotNormal',
        }),
        // new mod.MySimpleRNNCell({
        //   stateSize: 2,
        //   outputDim: 1,
        //   outputActivation: null,
        //   // recurrentActivation: { type: 'leakyAmeo' },
        //   recurrentActivation: 'linear',
        //   useOutputBias: false,
        //   useRecurrentBias: false,
        //   biasInitializer: 'zeros',
        //   recurrentInitializer: 'glorotNormal',
        //   kernelInitializer: 'glorotNormal',
        // }),
      ],
      inputShape: [8, 1],
      trainableInitialState: true,
      initialStateActivation: null,
      returnSequences: true,
      returnState: false,
      initialStateInitializer: 'zeros',
      batchSize: 1,
    });

    const model = new tf.Sequential();
    model.add(rnn);
    model.summary();

    const optimizer = tf.train.adam(0.1);
    const f = () => {
      const inputs = new Array(8).fill(null).map(randomBoolInput);
      const expected = inputs.map((v, i) => {
        if (i === 0) {
          return 0;
        }

        return xor(inputs[i - 1], v);
      });

      const inputsTensor = tf.tensor(inputs, [1, inputs.length, 1]);
      // inputsTensor.print();
      const expectedTensor = tf.tensor(expected, [1, expected.length, 1]);
      // expectedTensor.print();

      const actual = model.apply(inputsTensor) as Tensor<Rank>;
      // actual.print();
      // const loss = tf.abs(actual.sub(expectedTensor)).mean() as Scalar;
      // return loss;
      const loss = tf.losses.meanSquaredError(expectedTensor, actual);
      return loss as Tensor<Rank.R0>;
    };
    // f();

    for (let i = 0; i < 200; i++) {
      const loss = optimizer.minimize(f, true)! as Tensor<Rank>;
      loss.print();
    }

    model.weights.forEach(w => {
      console.log(w.name);
      w.read().print();
    });
  });
</script>

<div class="root">TODO</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }
</style>
