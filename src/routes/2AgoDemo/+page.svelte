<script lang="ts">
  import NodeVizComp from '../nodeViz/NodeVizComp.svelte';

  const serializedGraph =
    '{"inputLayer":{"neurons":[{"name":"input_0","activation":"linear","weights":[],"bias":0}]},"cells":[{"outputNeurons":[null,null,{"weights":[{"weight":-1,"index":1}],"bias":0,"name":"layer_0_output_2","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.1}}],"recurrentNeurons":[{"weights":[{"weight":1,"index":2}],"bias":0,"name":"layer_0_recurrent_0","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.1}},{"weights":[{"weight":-1,"index":0}],"bias":0,"name":"layer_0_recurrent_1","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.1}}],"stateNeurons":[{"bias":1,"name":"layer_0_state_0","activation":"linear","weights":[{"weight":1,"index":0}]},{"bias":-1,"name":"layer_0_state_1","activation":"linear","weights":[{"weight":1,"index":1}]}],"outputDim":4}],"postLayers":[{"neurons":[{"weights":[{"weight":-1,"index":2}],"bias":0,"name":"post_layer_output_0","activation":"linear"}],"outputDim":1}],"outputs":{"neurons":[{"name":"output_0","activation":"linear","weights":[{"weight":1,"index":0}],"bias":0}]}}';
</script>

<NodeVizComp
  serializedRNNGraph={serializedGraph}
  aspectRatio={0.77}
  labelFontSizeOverride={60}
  rankdir="LR"
  dashSize="8 12"
/>
