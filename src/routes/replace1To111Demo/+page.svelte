<script lang="ts">
  import type { InputSeqGenerator } from '../nodeViz/NodeViz';
  import NodeVizComp from '../nodeViz/NodeVizComp.svelte';

  const serializedGraph =
    '{"inputLayer":{"neurons":[{"name":"input_0","activation":"linear","weights":[],"bias":0}]},"cells":[{"outputNeurons":[null,{"weights":[{"weight":-0.38815388083457947,"index":0},{"weight":0.5671479105949402,"index":1},{"weight":-0.6452344059944153,"index":2}],"bias":-0.9780579805374146,"name":"layer_0_output_1","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.1}},null],"recurrentNeurons":[{"weights":[{"weight":1.1957452297210693,"index":0}],"bias":0,"name":"layer_0_recurrent_0","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.1}},{"weights":[{"weight":0.5374318361282349,"index":0},{"weight":0.4408518970012665,"index":1}],"bias":0.5113770961761475,"name":"layer_0_recurrent_1","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.1}}],"stateNeurons":[{"bias":0.731544017791748,"name":"layer_0_state_0","activation":"linear","weights":[{"weight":1,"index":0}]},{"bias":-0.525644063949585,"name":"layer_0_state_1","activation":"linear","weights":[{"weight":1,"index":1}]}],"outputDim":4}],"postLayers":[{"neurons":[{"weights":[{"weight":1.764803409576416,"index":1}],"bias":-0.4024794399738312,"name":"post_layer_output_0","activation":"linear"}],"outputDim":1}],"outputs":{"neurons":[{"name":"output_0","activation":"linear","weights":[{"weight":1,"index":0}],"bias":0}]}}';

  class SeqGen implements InputSeqGenerator {
    private i = 0;

    public reset() {
      this.i = 0;
    }

    public next(): Float32Array {
      // Generates [-1, 1, -1, -1, -1, -1] repeating forever
      const i = this.i;
      this.i += 1;
      switch (i % 6) {
        case 0:
          return new Float32Array([-1]);
        case 1:
          return new Float32Array([1]);
        case 2:
          return new Float32Array([-1]);
        case 3:
          return new Float32Array([-1]);
        case 4:
          return new Float32Array([-1]);
        case 5:
          return new Float32Array([-1]);
        default:
          throw new Error('unreachable');
      }
    }
  }

  const seqGen = new SeqGen();
</script>

<NodeVizComp
  serializedRNNGraph={serializedGraph}
  aspectRatio={1}
  labelFontSizeOverride={50}
  inputSeqGenerator={seqGen}
  defaultLogicAnalyzerVisibleNodeIDs={['layer_0_state_0', 'layer_0_state_1']}
/>
