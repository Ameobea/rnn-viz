<script lang="ts">
  import type { Variable, Rank, Tensor } from '@tensorflow/tfjs';

  import LossPlot from 'src/components/LossPlot.svelte';
  import { onDestroy, onMount } from 'svelte';
  import { writable } from 'svelte/store';

  const truncateWeights = true;
  const epochs = 8001;
  const inputDim = 16;
  const outputDim = 8;
  const batchSize = 256;
  const learningRate = 0.02;

  let losses = writable([] as number[]);
  let accuracies = writable([] as number[]);
  let stopped = false;

  onMount(async () => {
    const engine = await import('../engineComp/engine').then(async engine => {
      await engine.default();
      return engine;
    });
    const { tf } = await import('../nn/customRNN');
    const { GCUActivation, SineActivation } = await import('src/nn/gcuActivation');
    const { QuantizationRegularizer } = await import('../nn/QuantizationRegularizer');
    const ameoActivationModule = await import('../nn/ameoActivation');
    ameoActivationModule.setWasmEngine(engine);
    tf.setBackend('cpu');

    const seed = 91134302318.33333;
    const seed2 = 0x9352;
    engine.seed_rng(seed, seed2);
    const mkInitializer = (i: number) =>
      tf.initializers.randomNormal({ mean: 0, stddev: 0.1, seed: seed + i });
    const buildActivation = () => new ameoActivationModule.InterpolatedAmeo(0.1);
    // const buildActivation = () => new GCUActivation();
    // const buildActivation = () => new SineActivation();
    // const buildActivation = null as any;

    const model = new tf.Sequential();
    model.add(tf.layers.inputLayer({ inputShape: [inputDim] }));
    const layer1 = tf.layers.dense({
      units: 12,
      useBias: true,
      activation: 'tanh',
      kernelInitializer: mkInitializer(0),
      biasInitializer: mkInitializer(1),
      // kernelRegularizer: new QuantizationRegularizer(1 / 16, 0.002),
      // biasRegularizer: new QuantizationRegularizer(1 / 16, 0.0001),
    });
    if (buildActivation) (layer1 as any).activation = buildActivation();
    model.add(layer1);

    const layer2 = tf.layers.dense({
      units: 10,
      useBias: true,
      activation: 'tanh',
      kernelInitializer: mkInitializer(2),
      biasInitializer: mkInitializer(3),
      // kernelRegularizer: new QuantizationRegularizer(0.1, 0.002),
    });
    // if (buildActivation) (layer2 as any).activation = buildActivation();
    model.add(layer2);

    const layer3 = tf.layers.dense({
      units: outputDim,
      useBias: true,
      activation: 'tanh',
      kernelInitializer: mkInitializer(4),
      biasInitializer: mkInitializer(5),
      // kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 0.08, seed: seed + 10 }),
      // biasInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 0.08, seed: seed + 10 }),
      kernelRegularizer: tf.regularizers.l1l2({ l1: 0.0002, l2: 0.0 }),
      biasRegularizer: tf.regularizers.l1l2({ l1: 0.0002, l2: 0.0 }),
    });
    // if (buildActivation) (layer3 as any).activation = buildActivation();
    model.add(layer3);

    model.summary();
    let lastPreds: Variable<Rank> | null = null;
    let lastLabels: Variable<Rank> | null = null;
    model.compile({
      // loss: tf.losses.absoluteDifference,
      loss: (labels: Tensor<Rank>, preds: Tensor<Rank>) => {
        lastLabels = labels.clone().variable();
        lastPreds = preds.clone().variable();
        return tf.losses.meanSquaredError(labels, preds);
      },
      optimizer: tf.train.adam(learningRate),
    });

    const oneBatchExamples = (batchSize: number, validate = false) => {
      const vals = validate
        ? engine.wrapping_unsigned_8_bit_add_full_validation()
        : engine.wrapping_unsigned_8_bit_add(batchSize);
      batchSize = validate ? 256 * 256 : batchSize;
      const inputs = new Float32Array((vals.length / 3) * 2);
      const expected = new Float32Array(vals.length / 3);

      for (let seqIx = 0; seqIx < batchSize; seqIx += 1) {
        let offset = seqIx * (8 * 3);
        for (let j = 0; j < 8 * 2; j++) {
          inputs[seqIx * 16 + j] = vals[offset + j];
        }
        offset += 8 * 2;
        for (let j = 0; j < 8; j++) {
          expected[seqIx * 8 + j] = vals[offset + j];
        }
      }

      const inputsTensor = tf.tensor(inputs, [batchSize, inputDim]);
      const expectedTensor = tf.tensor(expected, [batchSize, outputDim]);

      return { inputsTensor, expectedTensor };
    };

    let perfect = false;
    for (let epoch = 0; epoch < epochs; epoch++) {
      if (stopped) break;

      if (epoch === 1500) {
        model.optimizer = tf.train.adam(0.0005);
      }

      const { inputsTensor, expectedTensor } = oneBatchExamples(batchSize);
      const history = await model.fit(inputsTensor, expectedTensor, {
        batchSize,
        epochs: 1,
      });

      const loss = history.history.loss[0] as number;
      if (Number.isNaN(loss)) {
        throw new Error('NaN loss');
      }
      losses.update(l => [...l, loss]);

      const doFullValidation = epoch >= 1750 && epoch % 250 === 0;
      const validationCount = doFullValidation ? 256 * 256 : batchSize;
      if (doFullValidation) {
        // truncate almost zero weights
        let truncatedCount = 0;
        let roundedCount = 0;
        const threshold = 0.015;
        for (let layerIx = 1; layerIx < (truncateWeights ? 4 : 0); layerIx += 1) {
          const weightsForLayer = await Promise.all(
            model.layers[layerIx]
              .getWeights(true)
              .map(
                async t => [t.shape as number[], await (t.data() as Promise<Float32Array>)] as const
              )
          );

          for (let varIx = 0; varIx < weightsForLayer.length; varIx += 1)
            for (let weightIx = 0; weightIx < weightsForLayer[varIx][1].length; weightIx++) {
              const weight = weightsForLayer[varIx][1][weightIx];
              if (Math.abs(weight) < 0.025) {
                truncatedCount += 1;
                weightsForLayer[varIx][1][weightIx] = 0;
                continue;
              }

              if (layerIx > 1) {
                continue;
              }

              // If closer than threshold to 0.5, set to 0.5
              if (Math.abs(weight - 0.5) < threshold) {
                roundedCount += 1;
                weightsForLayer[varIx][1][weightIx] = 0.5;
              }

              // If closer than threshold to -0.5, set to -0.5
              if (Math.abs(weight + 0.5) < threshold) {
                roundedCount += 1;
                weightsForLayer[varIx][1][weightIx] = -0.5;
              }

              // If closer than threshold to 0.25, set to 0.25
              if (Math.abs(weight - 0.25) < threshold) {
                roundedCount += 1;
                weightsForLayer[varIx][1][weightIx] = 0.25;
              }

              // If closer than threshold to -0.25, set to -0.25
              if (Math.abs(weight + 0.25) < threshold) {
                roundedCount += 1;
                weightsForLayer[varIx][1][weightIx] = -0.25;
              }

              // If closer than threshold to 0.125, set to 0.125
              if (Math.abs(weight - 0.125) < threshold) {
                roundedCount += 1;
                weightsForLayer[varIx][1][weightIx] = 0.125;
              }

              // If closer than threshold to -0.125, set to -0.125
              if (Math.abs(weight + 0.125) < threshold) {
                roundedCount += 1;
                weightsForLayer[varIx][1][weightIx] = -0.125;
              }

              // If closer than threshold to 1.0, set to 1.0
              if (Math.abs(weight - 1.0) < threshold) {
                roundedCount += 1;
                weightsForLayer[varIx][1][weightIx] = 1.0;
              }

              // If closer than threshold to -1.0, set to -1.0
              if (Math.abs(weight + 1.0) < threshold) {
                roundedCount += 1;
                weightsForLayer[varIx][1][weightIx] = -1.0;
              }
            }

          model.layers[layerIx].setWeights(
            weightsForLayer.map(([shape, data]) => tf.tensor(data, shape))
          );
        }
        console.log(`Truncated ${truncatedCount} weights`);
        console.log(`Rounded ${roundedCount} weights`);

        const validationData = oneBatchExamples(validationCount, doFullValidation);

        lastPreds = ((await model.predict(validationData.inputsTensor)) as Tensor<Rank>).variable();
        lastLabels = validationData.expectedTensor.variable();
      }

      if (!lastPreds || !lastLabels) {
        throw new Error('No last preds or labels');
      }
      const lastPredsData = await lastPreds.data();
      const lastLabelsData = await lastLabels.data();

      let successCount = 0;
      for (let batchIx = 0; batchIx < validationCount; batchIx++) {
        const pred = lastPredsData.slice(batchIx * outputDim, (batchIx + 1) * outputDim);
        const expected = lastLabelsData.slice(batchIx * outputDim, (batchIx + 1) * outputDim);
        // valid if the sign of all elements is the same
        const isValid = pred.every((p, ix) => Math.sign(p) === Math.sign(expected[ix]));
        if (isValid) successCount += 1;
      }
      lastPreds.dispose();
      lastPreds = null;
      lastLabels.dispose();
      lastLabels = null;
      const validationAccuracy = successCount / validationCount;
      accuracies.update(a => [...a, validationAccuracy]);

      if (doFullValidation && successCount === validationCount) {
        console.log('Perfect solution found after ' + epoch + ' epochs');
        perfect = true;
        // break;
      }

      if (epoch % 5 === 0) await new Promise(r => setTimeout(r, 0));
    }

    if (!perfect) {
      throw new Error('Failed to find perfect solution');
    }

    model.weights.forEach(w => {
      console.log(w.name);
      w.read().print(true);
    });
  });

  onDestroy(() => {
    stopped = true;
  });
</script>

<LossPlot iters={epochs} losses={$losses} accuracies={$accuracies} />
