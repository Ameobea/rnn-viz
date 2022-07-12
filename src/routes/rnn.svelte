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

    const seqLen = 64;
    const rnn = new mod.MyRNN({
      cell: [
        new mod.MySimpleRNNCell({
          stateSize: 2,
          outputDim: 4,
          // outputActivation: { type: 'softLeakyAmeo' },
          // recurrentActivation: { type: 'softLeakyAmeo' },
          // outputActivation: { type: 'leakyAmeo' },
          // recurrentActivation: { type: 'leakyAmeo' },
          outputActivation: 'tanh',
          recurrentActivation: 'tanh',
          // recurrentActivation: 'linear',
          useOutputBias: true,
          useRecurrentBias: true,
          biasInitializer: 'glorotNormal',
          recurrentInitializer: 'glorotNormal',
          kernelInitializer: 'glorotNormal',
        }),
        // new mod.MySimpleRNNCell({
        //   stateSize: 1,
        //   outputDim: 2,
        //   outputActivation: { type: 'leakyAmeo' },
        //   recurrentActivation: { type: 'leakyAmeo' },
        //   useOutputBias: false,
        //   useRecurrentBias: false,
        //   biasInitializer: 'zeros',
        //   recurrentInitializer: 'glorotNormal',
        //   kernelInitializer: 'glorotNormal',
        // }),
      ],
      inputShape: [seqLen, 1],
      trainableInitialState: true,
      initialStateActivation: null,
      returnSequences: true,
      returnState: false,
      initialStateInitializer: 'glorotNormal',
      // batchSize: 1,
    });

    const model = new tf.Sequential();
    model.add(rnn);
    // model.add(tf.layers.dense({ units: 8, activation: 'tanh' }));
    model.add(
      tf.layers.dense({
        units: 1,
        activation: 'linear',
        kernelInitializer: 'glorotNormal',
        useBias: false,
      })
    );
    model.summary();

    let optimizer = tf.train.adam(0.0015);
    // const optimizer = tf.train.sgd(0.1);

    const oneBatchExamples = () => {
      // const inputs = new Array(seqLen).fill(null).map(randomBoolInput);
      const inputs = new Array(seqLen).fill(0);
      const expected = inputs.map((v, i) => {
        // if (i === 0) {
        //   return 0;
        // }

        // return xor(inputs[i - 1], v);
        // return i % 2 === 0 ? 1 : -1;
        return Math.sin((i / 8) * Math.PI);
      });

      const inputsTensor = tf.tensor(inputs, [inputs.length, 1]);
      // inputsTensor.print();
      const expectedTensor = tf.tensor(expected, [expected.length, 1]);
      // expectedTensor.print();

      return { inputsTensor, expectedTensor };
    };

    const f = (batchSize = 8, print = false) => {
      const batches = [];
      for (let i = 0; i < batchSize; i++) {
        batches.push(oneBatchExamples());
      }

      const inputsTensor = tf.stack(batches.map(b => b.inputsTensor));
      const expectedTensor = tf.stack(batches.map(b => b.expectedTensor));

      const actual = model.apply(inputsTensor) as Tensor<Rank>;
      if (print) {
        actual.print();
      }
      // const loss = tf.abs(actual.sub(expectedTensor)).mean() as Scalar;
      // return loss;
      const loss = tf.losses.meanSquaredError(expectedTensor, actual);
      return loss as Tensor<Rank.R0>;
    };
    // f();

    for (let i = 0; i < 8000; i++) {
      // if (i < 100) {
      //   // optimizer = tf.train.adam(0.08);
      // } else if (i === 500) {
      //   const weights = await optimizer.getWeights();
      //   optimizer = tf.train.adam(0.02);
      //   optimizer.setWeights(weights);
      // }

      const loss = (optimizer.minimize(() => f(1), true)! as Tensor<Rank>).dataSync()[0];
      console.log(loss);
      if (loss < 0.002 || isNaN(loss)) {
        break;
      }

      // if (i > 1000 && loss > 0.43) {
      //   break;
      // }
    }

    model.weights.forEach(w => {
      console.log(w.name);
      w.read().print();
    });

    model.save('downloads://rnn');

    f(1, true);
  });
</script>

<div class="root">TODO</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }
</style>
