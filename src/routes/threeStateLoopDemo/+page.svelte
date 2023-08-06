<script lang="ts" context="module">
  import NodeVizComp from '../nodeViz/NodeVizComp.svelte';

  // prettier-ignore
  const serializedGraph = '{"inputLayer":{"neurons":[null,null]},"cells":[{"outputNeurons":[null,null,null,null,null,null,null,{"weights":[{"weight":-1,"index":3}],"bias":0,"name":"layer_0_output_7","activation":{"type":"interpolatedAmeo","factor":0.6,"leakyness":1}}],"recurrentNeurons":[null,{"weights":[{"weight":1,"index":5}],"bias":0,"name":"layer_0_recurrent_1","activation":{"type":"interpolatedAmeo","factor":0.6,"leakyness":1}},null,{"weights":[{"weight":-1,"index":3},{"weight":1,"index":5}],"bias":-1,"name":"layer_0_recurrent_3","activation":{"type":"interpolatedAmeo","factor":0.6,"leakyness":1}}],"stateNeurons":[null,{"bias":-1,"name":"layer_0_state_1","activation":"linear","weights":[{"weight":1,"index":1}]},null,{"bias":1,"name":"layer_0_state_3","activation":"linear","weights":[{"weight":1,"index":3}]}],"outputDim":8}],"postLayers":[{"neurons":[{"weights":[{"weight":1,"index":7}],"bias":0,"name":"post_layer_output_0","activation":"linear"}],"outputDim":1}],"outputs":{"neurons":[{"name":"output_0","activation":"linear","weights":[{"weight":1,"index":0}],"bias":0}]}}';
</script>

<NodeVizComp serializedRNNGraph={serializedGraph} aspectRatio={0.77} labelFontSizeOverride={90} />
