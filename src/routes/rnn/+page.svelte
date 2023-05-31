<script lang="ts">
  import LossPlot from '../../components/LossPlot.svelte';

  import { onMount } from 'svelte';
  import { get, writable } from 'svelte/store';
  import { runGraphTest } from './graphTest';
  import { RNNGraph, type RNNCellWeights, type RNNGraphParams } from './graph';
  import { oneSeqExamples } from './objective';

  let totalLosses = writable([] as number[]);
  let finalLosses = writable([] as number[]);
  let accuracies = writable([] as number[]);

  const seqLen = oneSeqExamples().inputs.length;
  const inputDim = oneSeqExamples().inputs[0].length;
  const outputDim = oneSeqExamples().outputs[0].length;
  const batchSize = 128;
  const epochs = 40000;
  const quantIntensity = 0.02;
  const sparseIntensity = 0.15;
  const sparseSteepness = 25;
  const learningRate = 0.01;
  const l1 = 0.0;

  onMount(async () => {
    const engine = await import('../../engineComp/engine').then(async engine => {
      await engine.default();
      return engine;
    });
    const { tf, ...mod } = await import('../../nn/customRNN');
    // const { QuantizationRegularizer } = await import('../../nn/QuantizationRegularizer');
    const { ComposedRegularizer } = await import('../../nn/ComposedRegularizer');
    const { SparseRegularizer } = await import('../../nn/SparseRegularizer');
    tf.setBackend('cpu');

    const initializer = tf.initializers.randomNormal({ mean: 0, stddev: 0.2 });
    // const initializer = tf.initializers.leCunUniform({});
    // const initializer = tf.initializers.glorotNormal({});
    const activation = { type: 'interpolatedAmeo' as const, factor: 0.6, leakyness: 1 };
    // const activation = { type: 'leakyAmeo' as const, leakyness: 0.2 };
    // const activation = 'linear';

    const cellParams = [
      {
        stateSize: 4,
        outputDim: 8,
        outputActivation: activation,
        recurrentActivation: activation,
        useOutputBias: true,
        useRecurrentBias: true,
        biasInitializer: initializer,
        recurrentInitializer: initializer,
        kernelInitializer: initializer,
        kernelRegularizer: new ComposedRegularizer(
          // new QuantizationRegularizer(1, quantIntensity),
          new SparseRegularizer(sparseIntensity, 0.025, sparseSteepness, l1)
          // tf.regularizers.l1({ l1 })
        ),
        recurrentRegularizer: new ComposedRegularizer(
          // new QuantizationRegularizer(1, quantIntensity),
          new SparseRegularizer(sparseIntensity, 0.025, sparseSteepness, l1)
          // tf.regularizers.l1({ l1 })
        ),
        // biasRegularizer: new QuantizationRegularizer(1, 0.2),
      },
    ];
    const cells = cellParams.map(params => new mod.MySimpleRNNCell(params));

    const rnn = new mod.MyRNN({
      cell: cells,
      inputShape: [seqLen, inputDim],
      trainableInitialState: true,
      initialStateRegularizer: new ComposedRegularizer(
        new SparseRegularizer(sparseIntensity, 0.025, sparseSteepness, l1)
        // tf.regularizers.l1({ l1 })
      ),
      initialStateActivation: null,
      returnSequences: true,
      returnState: false,
      initialStateInitializer: 'glorotNormal' as const,
    });

    const model = new tf.Sequential();
    model.add(rnn);
    const denseLayerArgs = {
      units: outputDim,
      activation: 'linear' as const,
      kernelInitializer: 'glorotNormal' as const,
      useBias: false,
      kernelRegularizer: new ComposedRegularizer(
        // new QuantizationRegularizer(1, quantIntensity),
        new SparseRegularizer(sparseIntensity, 0.025, sparseSteepness)
      ),
      // biasRegularizer: new QuantizationRegularizer(1, 0.2),
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

    const totalLossesLocal: number[] = [];
    const finalLossesLocal: number[] = [];

    for (let i = 0; i < epochs; i++) {
      const inputBatches: any[] = [];
      const outputBatches: any[] = [];
      for (let i = 0; i < batchSize; i++) {
        const { inputs, outputs } = oneSeqExamples();
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
      if (i % 20 === 0) {
        model.weights.forEach(w => {
          if (!w.trainable) {
            return;
          }

          const data = w.read().dataSync();
          for (let i = 0; i < data.length; i++) {
            if (Math.abs(data[i]) < 0.1) {
              data[i] = data[i] * 0.95;
            }
          }
          w.write(tf.tensor(data, w.shape as any));
        });
      }

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

    const params: Partial<RNNGraphParams> = { clipThreshold: 0.01, quantizationInterval: 0 };
    const cellWeights: RNNCellWeights[] = cells.map((cell, cellIx) => {
      const params = cellParams[cellIx];

      return {
        initialState: rnn.weights
          // initial states are in reverse order for some reason
          .find(w => w.name.includes(`/initial_state_weights_${cells.length - cellIx - 1}`))!
          .read(),
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

    // Try very aggressive clipping to start with and scale down until we get a valid graph
    const clipThresholds = [
      0.7, 0.5, 0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0,
    ];
    let graph: RNNGraph | undefined;
    let isValid = false;

    const buildGraph = () =>
      RNNGraph.fromWeights(
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

    for (const clipThreshold of clipThresholds) {
      params.clipThreshold = clipThreshold;
      graph = buildGraph();

      isValid = graph.validate(oneSeqExamples);
      if (!isValid) {
        console.log(`Invalid graph with clipThreshold ${clipThreshold}`);
        continue;
      }

      console.log(`Valid graph with clipThreshold ${clipThreshold}`);

      // Try progressively finer quantization until we get a valid graph
      const quantizationIntervals = [1, 0.5, 0.5, 0.25, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005];
      for (const quantizationInterval of quantizationIntervals) {
        params.quantizationInterval = quantizationInterval;
        graph = buildGraph();

        isValid = graph.validate(oneSeqExamples);
        if (!isValid) {
          console.log(`Invalid graph with quantizationInterval ${quantizationInterval}`);
          continue;
        }

        console.log(`Valid graph with quantizationInterval ${quantizationInterval}`);
        break;
      }
      break;
    }

    if (!isValid) {
      console.error('No valid graph found');
    }

    console.log(graph!.buildGraphviz());

    console.log(graph);

    console.log(graph!.serialize());

    console.log(RNNGraph.deserialize(graph!.serialize()));
  });
</script>

<LossPlot
  iters={epochs}
  totalLosses={$totalLosses}
  finalLosses={$finalLosses}
  accuracies={$accuracies}
/>
