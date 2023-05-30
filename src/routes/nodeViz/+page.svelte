<script lang="ts">
  import { RNNGraph } from '../rnn/graph';
  import NodeVizComp from './NodeVizComp.svelte';

  const serializedRNNGraph = JSON.parse(
    '{"inputLayer":{"neurons":[{"name":"input_0","activation":"linear","weights":[],"bias":0}]},"cells":[{"outputNeurons":[{"weights":[{"weight":1,"index":1}],"bias":0,"name":"layer_0_output_0","activation":{"type":"interpolatedAmeo","factor":0.6,"leakyness":1}}],"recurrentNeurons":[{"weights":[{"weight":1,"index":2}],"bias":0,"name":"layer_0_recurrent_0","activation":{"type":"interpolatedAmeo","factor":0.6,"leakyness":1}},{"weights":[{"weight":-1,"index":0}],"bias":0,"name":"layer_0_recurrent_1","activation":{"type":"interpolatedAmeo","factor":0.6,"leakyness":1}}],"stateNeurons":[{"bias":0.7941224575042725,"name":"layer_0_state_0","activation":"linear","weights":[{"weight":1,"index":0}]},{"bias":-0.9996616244316101,"name":"layer_0_state_1","activation":"linear","weights":[{"weight":1,"index":1}]}],"outputDim":2}],"postLayers":[{"neurons":[{"weights":[{"weight":1,"index":0}],"bias":0,"name":"post_layer_output_0","activation":"linear"}],"outputDim":1}],"outputs":{"neurons":[{"name":"output_0","activation":"linear","weights":[{"weight":1,"index":0}],"bias":0}]}}'
  );
  const graph = RNNGraph.deserialize(serializedRNNGraph);
  graph.setInputSequence([new Float32Array(graph.inputDim).fill(-1)]);
</script>

<NodeVizComp {graph} />
