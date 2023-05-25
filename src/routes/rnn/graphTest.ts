// my_rnn_MyRNN2/RNNCell_0/output_tree (2) [3, 3]
// print.ts:34 Tensor
//     [[-0.9981869, -0.001011 , 0.0009836 ],
//      [0.002684  , -0.9993024, -0.0009874],
//      [0.0039492 , -0.0010899, -0.0053482]]
// +page.svelte:205 my_rnn_MyRNN2/RNNCell_0/output_bias [3]
// print.ts:34 Tensor
//     [0.0017771, 0.003676, -1.0009207]
// +page.svelte:205 my_rnn_MyRNN2/RNNCell_0/recurrent_tree (2) [3, 1]
// print.ts:34 Tensor
//     [[0.0009768 ],
//      [-0.0088487],
//      [0.0054622 ]]
// +page.svelte:205 my_rnn_MyRNN2/RNNCell_0/recurrent_bias [1]
// print.ts:34 Tensor
//     [0.0016593]
// +page.svelte:205 my_rnn_MyRNN2/initial_state_weights_0 [1]
// print.ts:34 Tensor
//     [-0.1309134]

import { RNNGraph, type RNNCellWeights, type PostLayerWeights, type RNNGraphParams } from './graph';

export const runGraphTest = async () => {
  const { tf } = await import('../../nn/customRNN');

  const inputDim = 2;
  const outputDim = 3;
  const cellWeights: RNNCellWeights[] = [
    {
      recurrentActivation: { type: 'leakyAmeo', leakyness: 0.1 },
      outputActivation: { type: 'leakyAmeo', leakyness: 0.1 },
      outputSize: 3,
      outputTreeWeights: tf.tensor(
        [
          -0.9981869, -0.001011, 0.0009836, 0.002684, -0.9993024, -0.0009874, 0.0039492, -0.0010899,
          -0.0053482,
        ],
        [3, 3]
      ),
      recurrentTreeWeights: tf.tensor([0.0009768, -0.0088487, 0.0054622], [3, 1]),
      stateSize: 1,
      initialState: tf.tensor([-0.1309134], [1]),
      outputTreeBias: tf.tensor([0.0017771, 0.003676, -1.0009207], [3]),
      recurrentTreeBias: tf.tensor([0.0016593], [1]),
    },
  ];
  const postLayers: PostLayerWeights[] = [];
  const params: RNNGraphParams = { clipThreshold: 0.1, quantizationInterval: 1 };

  const graph = new RNNGraph(inputDim, outputDim, cellWeights, postLayers, params);
  console.log(graph);

  const inputSeq = [
    new Float32Array([1, -1]),
    new Float32Array([1, -1]),
    new Float32Array([-1, -1]),
    new Float32Array([-1, 1]),
    new Float32Array([1, 1]),
  ];
  const outputSeq = graph.evaluate(inputSeq);
  console.log(outputSeq);

  return graph.buildGraphviz();
};
