<script context="module" lang="ts">
  const randomBoolInput = () => (Math.random() > 0.5 ? 1 : -1);

  const xor = (a: -1 | 1, b: -1 | 1): -1 | 1 =>
    (a === -1 && b === 1) || (a === 1 && b === -1) ? 1 : -1;
</script>

<script lang="ts">
  import type { Rank, Tensor } from '@tensorflow/tfjs';

  import { onMount } from 'svelte';

  const quantizationInterval = 1 / 8;
  const quantizationIntensity = 0.05;

  onMount(async () => {
    const engine = await import('../engineComp/engine').then(async engine => {
      await engine.default();
      return engine;
    });
    const { tf, ...mod } = await import('../nn/customRNN');
    const { QuantizationRegularizer } = await import('src/nn/QuantizationRegularizer');
    tf.setBackend('cpu');

    // const initializer = tf.initializers.randomUniform({ minval: -1.5, maxval: 1.5 });
    // const initializer = tf.initializers.leCunUniform({});
    const initializer = tf.initializers.randomNormal({ mean: 0, stddev: 0.1 });
    const activation = { type: 'interpolatedAmeo' as const, factor: 0.25 };
    // const activation = 'gcu';

    const seqLen = 64;
    const inputDim = 16;
    const outputDim = 8;
    const rnn = new mod.MyRNN({
      cell: [
        new mod.MySimpleRNNCell({
          stateSize: 1,
          outputDim: 16,
          outputActivation: activation,
          recurrentActivation: 'linear',
          // outputActivation: { type: 'leakyAmeo' },
          // recurrentActivation: { type: 'leakyAmeo' },
          // outputActivation: 'tanh',
          // recurrentActivation: 'tanh',
          // recurrentActivation: 'linear',
          useOutputBias: true,
          useRecurrentBias: true,
          biasInitializer: initializer,
          recurrentInitializer: initializer,
          kernelInitializer: initializer,
          // kernelRegularizer: new QuantizationRegularizer(
          //   quantizationInterval,
          //   quantizationIntensity
          // ),
          // recurrentRegularizer: new QuantizationRegularizer(
          //   quantizationInterval,
          //   quantizationIntensity
          // ),
          // biasRegularizer: new QuantizationRegularizer(quantizationInterval, quantizationIntensity),
        }),
        new mod.MySimpleRNNCell({
          stateSize: 1,
          outputDim: 8,
          outputActivation: activation,
          recurrentActivation: 'linear',
          useOutputBias: true,
          useRecurrentBias: true,
          biasInitializer: initializer,
          recurrentInitializer: initializer,
          kernelInitializer: initializer,
          // kernelRegularizer: new QuantizationRegularizer(
          //   quantizationInterval,
          //   quantizationIntensity
          // ),
          // recurrentRegularizer: new QuantizationRegularizer(
          //   quantizationInterval,
          //   quantizationIntensity
          // ),
          // biasRegularizer: new QuantizationRegularizer(quantizationInterval, quantizationIntensity),
        }),
      ],
      inputShape: [seqLen, inputDim],
      trainableInitialState: false,
      initialStateActivation: null,
      returnSequences: true,
      returnState: false,
      initialStateInitializer: 'glorotNormal',
      // batchSize: 1,
    });
    const cell0 = rnn.cell.cells[0];

    const model = new tf.Sequential();
    model.add(rnn);
    console.log(cell0.losses);
    // model.add(tf.layers.dense({ units: 8, activation: 'tanh' }));
    // model.add(
    //   tf.layers.dense({
    //     units: outputDim,
    //     activation: 'linear',
    //     kernelInitializer: 'glorotNormal',
    //     useBias: false,
    //     // kernelRegularizer: new QuantizationRegularizer(quantizationInterval, quantizationIntensity),
    //   })
    // );
    model.summary();
    model.compile({
      loss: tf.losses.absoluteDifference,
      optimizer: tf.train.adam(0.0015),
    });

    let optimizer = tf.train.adam(0.0015);
    // const optimizer = tf.train.sgd(0.001);

    // const reg = new QuantizationRegularizer(quantizationInterval, quantizationIntensity);
    // const x = tf.tensor1d([0, 1 / 3, -1 / 3, 3, 6, -2, 0]);
    // reg.apply(x).print();
    // const x2 = tf.tensor1d([1 / 6]);
    // reg.apply(x2).print();
    // const x3 = tf.tensor1d([1 / 6]);
    // reg.apply(x3).print();
    // throw new Error();

    const oneBatchExamples = () => {
      // const inputs = new Array(seqLen).fill(null).map(randomBoolInput);
      // const inputs = new Array(seqLen * inputDim).fill(0);
      // const expected = inputs.map((v, i) => {
      //   // if (i === 0) {
      //   //   return 0;
      //   // }

      //   // return xor(inputs[i - 1], v);
      //   // return i % 2 === 0 ? 1 : -1;
      //   return Math.sin((i / 8) * Math.PI);
      // });

      // const expected = engine.eight_bit_unsigned_binary_count(seqLen);

      const vals = engine.wrapping_unsigned_8_bit_add(seqLen);
      const inputs = new Float32Array((vals.length / 3) * 2);
      const expected = new Float32Array(vals.length / 3);

      for (let seqIx = 0; seqIx < seqLen; seqIx += 1) {
        let offset = seqIx * (8 * 3);
        for (let j = 0; j < 8 * 2; j++) {
          inputs[seqIx * 16 + j] = vals[offset + j];
        }
        offset += 8 * 2;
        for (let j = 0; j < 8; j++) {
          expected[seqIx * 8 + j] = vals[offset + j];
        }
      }

      // console.log(inputs, expected);

      const inputsTensor = tf.tensor(inputs, [1, seqLen, inputDim]);
      // inputsTensor.print();
      const expectedTensor = tf.tensor(expected, [1, seqLen, outputDim]);
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

      // const loss = tf.losses.meanSquaredError(expectedTensor, actual);
      const loss = tf.losses.absoluteDifference(
        expectedTensor,
        actual,
        undefined,
        tf.Reduction.MEAN
      );
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

      // const loss = (optimizer.minimize(() => f(1), true)! as Tensor<Rank>).dataSync()[0];
      // console.log(loss);
      // if (loss < 0.002 || isNaN(loss)) {
      //   break;
      // }
      const { inputsTensor, expectedTensor } = oneBatchExamples();
      await model.fit(inputsTensor, expectedTensor, {
        batchSize: 1,
        epochs: 1,
        callbacks: {
          onBatchEnd: async (batch, logs) => {
            if (logs) console.log(logs.loss);
          },
        },
      });

      // if (i > 1000 && loss > 0.43) {
      //   break;
      // }
    }

    model.weights.forEach(w => {
      console.log(w.name);
      w.read().print();
    });

    // model.save('downloads://rnn');

    // f(1, true);
  });
</script>

<div class="root">TODO</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }
</style>
