<script context="module" lang="ts">
  const randomBoolInput = () => (Math.random() > 0.5 ? 1 : -1);

  const xor = (a: -1 | 1, b: -1 | 1): -1 | 1 =>
    (a === -1 && b === 1) || (a === 1 && b === -1) ? 1 : -1;
</script>

<script lang="ts">
  import LossPlot from 'src/components/LossPlot.svelte';

  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';

  let losses = writable([] as number[]);
  let accuracies = writable([] as number[]);

  const seqLen = 64;
  const inputDim = 1;
  const outputDim = 1;
  const batchSize = 128;
  const epochs = 1000;

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
    const initializer = tf.initializers.randomNormal({ mean: 0, stddev: 0.4 });
    const activation = { type: 'interpolatedAmeo' as const, factor: 0.1, leakyness: 1 };
    // const activation = 'linear';

    const rnn = new mod.MyRNN({
      cell: [
        new mod.MySimpleRNNCell({
          stateSize: 4,
          outputDim: 1,
          outputActivation: activation,
          recurrentActivation: activation,
          useOutputBias: true,
          useRecurrentBias: true,
          biasInitializer: initializer,
          recurrentInitializer: initializer,
          kernelInitializer: initializer,
        }),
        // new mod.MySimpleRNNCell({
        //   stateSize: 1,
        //   outputDim: 8,
        //   outputActivation: activation,
        //   recurrentActivation: 'linear',
        //   useOutputBias: true,
        //   useRecurrentBias: true,
        //   biasInitializer: initializer,
        //   recurrentInitializer: initializer,
        //   kernelInitializer: initializer,
        // }),
      ],
      inputShape: [seqLen, inputDim],
      trainableInitialState: true,
      initialStateActivation: null,
      returnSequences: true,
      returnState: false,
      initialStateInitializer: 'glorotNormal',
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
    //   })
    // );
    model.summary();
    model.compile({
      loss: tf.losses.meanSquaredError,
      optimizer: tf.train.adam(0.01),
    });

    const oneBatchExamples = () => {
      let count = 1;
      const inputs = [1];
      const outputs = [-1];
      for (let i = 1; i < seqLen; i++) {
        if (count === 0) {
          inputs.push(1);
          outputs.push(-1);
          count += 1;
          continue;
        } else if (count > 6) {
          inputs.push(-1);
          outputs.push(-1);
          count -= 1;
          continue;
        }

        if (Math.random() > 0.7) {
          inputs.push(1);
          outputs.push(-1);
          count += 1;
        } else {
          inputs.push(-1);
          count -= 1;
          outputs.push(count === 0 ? 1 : -1);
        }
      }

      return { inputs, outputs };
    };

    for (let i = 0; i < epochs; i++) {
      const inputBatches: number[][] = [];
      const outputBatches: number[][] = [];
      for (let i = 0; i < batchSize; i++) {
        const { inputs, outputs } = oneBatchExamples();
        // console.log(inputs, outputs);
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
              console.log(logs.loss);
              losses.update(l => [...l, logs.loss]);
            }
          },
        },
      });

      await new Promise(resolve => setTimeout(resolve, 0));
    }

    model.weights.forEach(w => {
      console.log(w.name);
      w.read().print();
    });

    // model.save('downloads://rnn');
  });
</script>

<LossPlot iters={epochs} losses={$losses} accuracies={$accuracies} />
