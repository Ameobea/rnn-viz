<script lang="ts" context="module">
  import {
    RNNGraph,
    type PostLayerWeights,
    type RNNCellWeights,
    type RNNGraphParams,
  } from '../rnn/graph';
  import type { AmeoActivationIdentifier } from '../../nn/customRNN';

  const flattenToF32Array = (vals: number[][]): Float32Array => {
    const flat = new Float32Array(vals.length * vals[0].length);
    for (let i = 0; i < vals.length; i++) {
      flat.set(vals[i], i * vals[0].length);
    }
    return flat;
  };

  interface NodeVizWeightsDef {
    inputDim: number;
    outputDim: number;
    rnnCells: RNNCellWeights[];
    postLayerWeights: PostLayerWeights[];
  }

  const convertActivation = (activation: any): AmeoActivationIdentifier => {
    if (typeof activation === 'string') {
      return activation as any;
    }

    switch (activation.id) {
      case 'interpolated_ameo': {
        return {
          type: 'interpolatedAmeo',
          factor: activation.factor,
          leakyness: activation.leakyness,
        };
      }
      default: {
        throw new Error(`Unknown activation: ${activation}`);
      }
    }
  };
</script>

<script lang="ts">
  import type { PageData } from './$types';
  import NodeVizComp from '../nodeViz/NodeVizComp.svelte';
  import { oneSeqExamples } from '../rnn/objective';

  export let data: PageData;
  const weights: NodeVizWeightsDef = {
    inputDim: data.weights.input_dim,
    outputDim: data.weights.output_dim,
    rnnCells: data.weights.cells.map(cell => ({
      initialState: new Float32Array(cell.initial_state),
      stateSize: cell.state_size,
      outputSize: cell.output_dim,
      outputActivation: convertActivation(cell.output_activation),
      recurrentActivation: convertActivation(cell.recurrent_activation),
      outputTreeWeights: flattenToF32Array(cell.output_kernel),
      recurrentTreeWeights: flattenToF32Array(cell.recurrent_kernel),
      outputTreeBias: new Float32Array(cell.output_bias),
      recurrentTreeBias: new Float32Array(cell.recurrent_bias),
    })),
    postLayerWeights: data.weights.post_layers.map(l => ({
      inputDim: l.input_dim,
      outputDim: l.output_dim,
      weights: flattenToF32Array(l.weights),
      bias: new Float32Array(l.bias),
      activation: convertActivation(l.activation ?? 'tanh'),
    })),
  };

  const params: Partial<RNNGraphParams> = { clipThreshold: 0.1, quantizationInterval: 0 };
  const clipThresholds = [
    0.7, 0.5, 0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0,
  ];
  let graph: RNNGraph | undefined;
  let isValid = false;

  const buildGraph = () =>
    RNNGraph.fromWeights(
      weights.inputDim,
      weights.outputDim,
      weights.rnnCells,
      weights.postLayerWeights,
      params
    );

  for (const clipThreshold of clipThresholds) {
    params.clipThreshold = clipThreshold;
    graph = buildGraph();

    isValid = graph.validate(oneSeqExamples, 500, true);
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

      isValid = graph.validate(oneSeqExamples, 500, true);
      if (!isValid) {
        console.log(`Invalid graph with quantizationInterval ${quantizationInterval}`);
        continue;
      }

      console.log(`Valid graph with quantizationInterval ${quantizationInterval}`);
      break;
    }
    break;
  }

  if (!graph) {
    console.error('Could not build a valid graph');
  }

  const finalGraph = graph ?? buildGraph();

  const inputSeq = [
    new Float32Array(finalGraph.inputLayer.inputDim).map(() => (Math.random() > 0.5 ? 1 : -1)),
  ];
  finalGraph.reset(inputSeq);
</script>

<NodeVizComp
  serializedRNNGraph={finalGraph}
  excludedNodeIDs={[
    'layer_0_output_12',
    'layer_0_output_0',
    'layer_0_output_7',
    'input_2',
    'input_1',
    'layer_0_output_14',
    'layer_0_output_5',
    'layer_0_output_10',
    'layer_0_recurrent_8',
    'layer_1_output_11',
    'layer_1_output_9',
    'post_layer_output_0',
    'output_0',
  ]}
/>
