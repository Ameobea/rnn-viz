<script lang="ts">
  import type { Rank, Tensor } from '@tensorflow/tfjs';

  import LossPlot from 'src/components/LossPlot.svelte';
  import { onDestroy, onMount } from 'svelte';
  import { writable } from 'svelte/store';

  const epochs = 4000;
  const inputDim = 16;
  const outputDim = 8;
  const batchSize = 256;

  let losses = writable([] as number[]);
  let accuracies = writable([] as number[]);
  let stopped = false;

  onMount(async () => {
    const engine = await import('../engineComp/engine').then(async engine => {
      await engine.default();
      return engine;
    });
    const { tf } = await import('../nn/customRNN');
    const { GCUActivation } = await import('src/nn/gcuActivation');
    const ameoActivationModule = await import('../nn/ameoActivation');
    tf.setBackend('cpu');

    const initializer = tf.initializers.randomNormal({ mean: 0, stddev: 0.1 });
    const buildActivation = () => new ameoActivationModule.InterpolatedAmeo(0.25);
    // const buildActivation = () => new GCUActivation();
    // const buildActivation = null as any;

    const model = new tf.Sequential();
    model.add(tf.layers.inputLayer({ inputShape: [inputDim] }));
    const layer1 = tf.layers.dense({
      units: 24,
      useBias: true,
      activation: 'tanh',
      kernelInitializer: initializer,
      biasInitializer: initializer,
    });
    if (buildActivation) (layer1 as any).activation = buildActivation();
    model.add(layer1);

    const layer2 = tf.layers.dense({
      units: 12,
      useBias: true,
      activation: 'tanh',
      kernelInitializer: initializer,
      biasInitializer: initializer,
    });
    if (buildActivation) (layer2 as any).activation = buildActivation();
    model.add(layer2);

    const layer3 = tf.layers.dense({
      units: outputDim,
      useBias: true,
      activation: 'tanh',
      kernelInitializer: initializer,
      biasInitializer: initializer,
    });
    if (buildActivation) (layer3 as any).activation = buildActivation();
    model.add(layer3);

    const optimizer = tf.train.adam(0.0015);
    model.summary();
    model.compile({
      // loss: tf.losses.absoluteDifference,
      loss: tf.losses.meanSquaredError,
      optimizer,
    });

    const oneBatchExamples = (batchSize: number) => {
      const vals = engine.wrapping_unsigned_8_bit_add(batchSize);
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

    for (let epoch = 0; epoch < epochs; epoch++) {
      const { inputsTensor, expectedTensor } = oneBatchExamples(batchSize);
      const history = await model.fit(inputsTensor, expectedTensor, {
        batchSize,
        epochs: 1,
      });

      const loss = history.history.loss[0] as number;
      console.log(loss);
      losses.update(l => [...l, loss]);

      const validationData = oneBatchExamples(batchSize);
      const preds = (
        (await model.predict(validationData.inputsTensor)) as Tensor<Rank>
      ).dataSync() as Float32Array;
      const expecteds = validationData.expectedTensor.dataSync() as Float32Array;

      let successCount = 0;
      for (let batchIx = 0; batchIx < batchSize; batchIx++) {
        const pred = preds.slice(batchIx * outputDim, (batchIx + 1) * outputDim);
        const expected = expecteds.slice(batchIx * outputDim, (batchIx + 1) * outputDim);
        // valid if the sign of all elements is the same
        const isValid = pred.every((p, ix) => Math.sign(p) === Math.sign(expected[ix]));
        if (isValid) successCount += 1;
      }
      const validationAccuracy = successCount / batchSize;
      accuracies.update(a => [...a, validationAccuracy]);

      await new Promise(r => setTimeout(r, 0));
    }

    model.weights.forEach(w => {
      console.log(w.name);
      w.read().print();
    });
  });

  onDestroy(() => {
    stopped = true;
  });
</script>

<LossPlot iters={epochs} losses={$losses} accuracies={$accuracies} />
