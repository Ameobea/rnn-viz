<script context="module" lang="ts">
  const randomBoolInput = () => (Math.random() > 0.5 ? 1 : -1);

  const xor = (a: -1 | 1, b: -1 | 1): -1 | 1 =>
    (a === -1 && b === 1) || (a === 1 && b === -1) ? 1 : -1;
</script>

<script lang="ts">
  import LossPlot from '../../components/LossPlot.svelte';

  import { onMount } from 'svelte';
  import { get, writable } from 'svelte/store';
  import { runGraphTest } from './graphTest';
  import { RNNGraph, type RNNCellWeights, type RNNGraphParams } from './graph';

  let totalLosses = writable([] as number[]);
  let finalLosses = writable([] as number[]);
  let accuracies = writable([] as number[]);

  const seqLen = 6;
  const inputDim = 1;
  const outputDim = 1;
  const batchSize = 32;
  const epochs = 2000;
  const quantIntensity = 0.4;
  const sparseIntensity = 0.65;
  const learningRate = 0.01;

  onMount(async () => {
    // runGraphTest().then(console.log);
    // return;

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
    const initializer = tf.initializers.glorotNormal({});
    // const activation = { type: 'interpolatedAmeo' as const, factor: 0.9, leakyness: 1 };
    const activation = { type: 'leakyAmeo' as const, leakyness: 0.2 };
    // const activation = 'linear';

    const cellParams = [
      {
        stateSize: 8,
        outputDim: 2,
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
      },
    ];
    const cells = cellParams.map(params => new mod.MySimpleRNNCell(params));

    const rnn = new mod.MyRNN({
      cell: cells,
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
    const denseLayerArgs = {
      units: outputDim,
      activation: 'linear' as const,
      kernelInitializer: 'glorotNormal' as const,
      useBias: false,
      kernelRegularizer: new ComposedRegularizer(
        new QuantizationRegularizer(1, quantIntensity),
        new SparseRegularizer(sparseIntensity, 0.1, 20)
      ),
      biasRegularizer: new QuantizationRegularizer(1, 0.2),
    };
    const denseLayer = tf.layers.dense(denseLayerArgs);
    model.add(denseLayer);

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
      optimizer: tf.train.adam(learningRate),
    });

    // const oneBatchExamples = () => {
    //   const inputs: (1 | -1)[] = [];
    //   const outputs: (1 | -1)[] = [];
    //   for (let i = 0; i < seqLen; i++) {
    //     const input = randomBoolInput();
    //     inputs.push(input);

    //     if (i === 0 || i === 1 || i === 2) {
    //       if (input === 1) {
    //         outputs.push(1);
    //       } else {
    //         outputs.push(-1);
    //       }
    //       continue;
    //     }
    //     if (i === 0) {
    //       outputs.push(input);
    //       continue;
    //     }

    //     const prevInput = inputs[i - 3];
    //     // const prevInput = inputs[i - 1];
    //     const output = xor(input, prevInput);
    //     outputs.push(output);
    //   }

    //   return { inputs, outputs };
    // };

    const oneBatchExamples = () => {
      const inputs: number[] = [];
      const outputs: number[] = [];

      const oneVal = () => (Math.random() > 0.5 ? 1 : -1);

      for (let i = 0; i < seqLen; i += 1) {
        const input = oneVal();
        inputs.push(input);

        if (i === 0) {
          outputs.push(-1);
          continue;
        }

        if (input === -1 && inputs[i - 1] === -1 && inputs[i - 2] === -1) {
          outputs.push(1);
          continue;
        }

        // 1 if 2 of the last 3 inputs were 1, else -1
        if (i === 1) {
          const output = inputs[i - 1] === 1 && input ? 1 : -1;
          outputs.push(output);
          continue;
        }

        const output = inputs[i - 1] === 1 && inputs[i - 2] === 1 ? 1 : -1;
        outputs.push(output);
      }

      return { inputs, outputs };
    };

    const totalLossesLocal: number[] = [];
    const finalLossesLocal: number[] = [];

    for (let i = 0; i < epochs; i++) {
      const inputBatches: any[] = [];
      const outputBatches: any[] = [];
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
              lastLoss = logs.loss;
              totalLossesLocal.push(logs.loss);
              finalLossesLocal.push(lastFinalLoss);
              if (i % 100 === 0) {
                totalLosses.set(totalLossesLocal);
                finalLosses.set(finalLossesLocal);
                console.log({ trainLoss: logs.loss, finalLoss: lastFinalLoss });
              }
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

      if (i % 100 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      if (lastLoss < 0.01) {
        break;
      }
    }

    model.weights.forEach(w => {
      console.log(w.name, w.shape);
      w.read().print();
    });

    const params: Partial<RNNGraphParams> = { clipThreshold: 0.01 };
    const cellWeights: RNNCellWeights[] = cells.map((cell, cellIx) => {
      const params = cellParams[cellIx];

      return {
        // TODO
        // initialState: cell.weights.find(w => w.name.includes('initial_state_weights'))!.read(),
        initialState: tf.tensor(new Float32Array(params.stateSize).fill(0), [params.stateSize]),
        outputActivation: params.outputActivation,
        recurrentActivation: params.recurrentActivation,
        outputSize: params.outputDim,
        outputTreeWeights: cell.weights.find(w => w.name.includes('output_tree'))!.read(),
        recurrentTreeWeights: cell.weights.find(w => w.name.includes('recurrent_tree'))!.read(),
        stateSize: params.stateSize,
        outputTreeBias: cell.weights.find(w => w.name.includes('output_bias'))?.read(),
        recurrentTreeBias: cell.weights.find(w => w.name.includes('recurrent_bias'))?.read(),
      };
    });
    const graph = new RNNGraph(
      inputDim,
      outputDim,
      cellWeights,
      [
        {
          weights: denseLayer.weights.find(w => w.name.includes('kernel'))!.read(),
          bias: denseLayer.weights.find(w => w.name.includes('bias'))?.read(),
          activation: denseLayerArgs.activation,
        },
      ],
      params
    );

    console.log(graph.buildGraphviz());
  });
</script>

<LossPlot
  iters={epochs}
  totalLosses={$totalLosses}
  finalLosses={$finalLosses}
  accuracies={$accuracies}
/>
