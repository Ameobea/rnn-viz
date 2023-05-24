<script context="module" lang="ts">
  const randomBoolInput = () => (Math.random() > 0.5 ? 1 : -1);

  const xor = (a: -1 | 1, b: -1 | 1): -1 | 1 =>
    (a === -1 && b === 1) || (a === 1 && b === -1) ? 1 : -1;
</script>

<script lang="ts">
  import LossPlot from '../../components/LossPlot.svelte';

  import { onMount } from 'svelte';
  import { get, writable } from 'svelte/store';

  let totalLosses = writable([] as number[]);
  let finalLosses = writable([] as number[]);
  let accuracies = writable([] as number[]);

  const seqLen = 4;
  const inputDim = 1;
  const outputDim = 1;
  const batchSize = 4;
  const epochs = 50000;

  onMount(async () => {
    const engine = await import('../../engineComp/engine').then(async engine => {
      await engine.default();
      return engine;
    });
    const { tf, ...mod } = await import('../../nn/customRNN');
    const { QuantizationRegularizer } = await import('../../nn/QuantizationRegularizer');
    const { ComposedRegularizer } = await import('../../nn/ComposedRegularizer');
    const { SparseRegularizer } = await import('../../nn/SparseRegularizer');
    tf.setBackend('cpu');

    // const initializer = tf.initializers.randomUniform({ minval: -1.5, maxval: 1.5 });
    // const initializer = tf.initializers.leCunUniform({});
    const initializer = tf.initializers.randomNormal({ mean: 0, stddev: 0.5 });
    // const activation = { type: 'interpolatedAmeo' as const, factor: 0.9, leakyness: 1 };
    const activation = { type: 'leakyAmeo' as const, leakyness: 0.2 };
    // const activation = 'linear';

    const quantIntensity = 0.08;
    const sparseIntensity = 0.45;

    const rnn = new mod.MyRNN({
      cell: [
        new mod.MySimpleRNNCell({
          stateSize: 4,
          outputDim: 16,
          outputActivation: activation,
          recurrentActivation: activation,
          useOutputBias: true,
          useRecurrentBias: true,
          biasInitializer: initializer,
          recurrentInitializer: initializer,
          kernelInitializer: initializer,
          kernelRegularizer: new ComposedRegularizer(
            new QuantizationRegularizer(1, quantIntensity),
            new SparseRegularizer(sparseIntensity, 0.1, 15)
          ),
          recurrentRegularizer: new ComposedRegularizer(
            new QuantizationRegularizer(1, quantIntensity),
            new SparseRegularizer(sparseIntensity, 0.1, 15)
          ),
          biasRegularizer: new QuantizationRegularizer(1, 0.2),
        }),
      ],
      inputShape: [seqLen, inputDim],
      trainableInitialState: true,
      initialStateRegularizer: new QuantizationRegularizer(1, 0.2),
      initialStateActivation: null,
      returnSequences: true,
      returnState: false,
      initialStateInitializer: 'glorotNormal',
    });

    const model = new tf.Sequential();
    model.add(rnn);
    // model.add(tf.layers.dense({ units: 8, activation: 'tanh' }));
    model.add(
      tf.layers.dense({
        units: outputDim,
        activation: 'linear',
        kernelInitializer: 'glorotNormal',
        useBias: false,
        kernelRegularizer: new ComposedRegularizer(
          new QuantizationRegularizer(1, quantIntensity),
          new SparseRegularizer(sparseIntensity, 0.1, 20)
        ),
        biasRegularizer: new QuantizationRegularizer(1, 0.2),
      })
    );

    // The loss of the whole network including the regularizers
    let lastLoss = Infinity;
    // The loss of the actual output of the network less the regularizers
    let lastFinalLoss = Infinity;

    model.summary();
    model.compile({
      // loss: tf.losses.meanSquaredError,
      loss: (yTrue, yPred) =>
        tf.tidy(() => {
          const loss = tf.losses.meanSquaredError(yTrue, yPred);
          lastFinalLoss = loss.dataSync()[0];
          return loss;
        }),
      optimizer: tf.train.adam(0.02),
    });

    const oneBatchExamples = () => {
      const inputs: (1 | -1)[] = [];
      const outputs: (1 | -1)[] = [];
      for (let i = 0; i < seqLen; i++) {
        const input = randomBoolInput();
        inputs.push(input);

        if (i === 0 || i === 1) {
          if (input === 1) {
            outputs.push(1);
          } else {
            outputs.push(-1);
          }
          continue;
        }

        const prevInput = inputs[i - 2];
        const output = xor(input, prevInput);
        outputs.push(output);
      }

      return { inputs, outputs };
    };

    for (let i = 0; i < epochs; i++) {
      const inputBatches: number[][] = [];
      const outputBatches: number[][] = [];
      for (let i = 0; i < batchSize; i++) {
        const { inputs, outputs } = oneBatchExamples();
        inputBatches.push(inputs);
        outputBatches.push(outputs);
      }

      const inputsTensor = tf.tensor(inputBatches, [batchSize, seqLen, inputDim]);
      const expectedsTensor = tf.tensor(outputBatches, [batchSize, seqLen, outputDim]);

      await model.fit(inputsTensor, expectedsTensor, {
        batchSize,
        epochs: 1,
        callbacks: {
          onBatchEnd: async (batch, logs) => {
            if (logs) {
              console.log({ trainLoss: logs.loss, finalLoss: lastFinalLoss });
              lastLoss = logs.loss;
              totalLosses.update(l => [...l, logs.loss]);
              finalLosses.update(l => [...l, lastFinalLoss]);
            }
          },
        },
      });

      // decay small weights towards zero
      // if (i % 10 === 25) {
      //   model.weights.forEach(w => {
      //     const data = w.read().dataSync();
      //     for (let i = 0; i < data.length; i++) {
      //       if (Math.abs(data[i]) < 0.02 && Math.random() < 0.4) {
      //         data[i] = 0;
      //       }
      //     }
      //     w.write(tf.tensor(data, w.shape as any));
      //   });
      // }

      await new Promise(resolve => setTimeout(resolve, 0));
      if (lastLoss < 0.01) {
        break;
      }
    }

    model.weights.forEach(w => {
      console.log(w.name);
      w.read().print();
    });

    // model.save('downloads://rnn');
  });
</script>

<LossPlot
  iters={epochs}
  totalLosses={$totalLosses}
  finalLosses={$finalLosses}
  accuracies={$accuracies}
/>
