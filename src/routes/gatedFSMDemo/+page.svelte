<script lang="ts" context="module">
  import type { InputSeqGenerator } from '../nodeViz/NodeViz';
  import NodeVizComp from '../nodeViz/NodeVizComp.svelte';

  const serializedGraph =
    '{"inputLayer":{"neurons":[{"name":"input_0","activation":"linear","weights":[],"bias":0},{"name":"input_1","activation":"linear","weights":[],"bias":0},{"name":"input_2","activation":"linear","weights":[],"bias":0}]},"cells":[{"outputNeurons":[{"weights":[{"weight":-0.4814823269844055,"index":1}],"bias":0,"name":"layer_0_output_0","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},null,null,null,null,{"weights":[{"weight":0.270023375749588,"index":1}],"bias":0,"name":"layer_0_output_5","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},null,{"weights":[{"weight":1.0577094554901123,"index":0},{"weight":0.8551025390625,"index":1},{"weight":-1.3045403957366943,"index":2}],"bias":0.5151251554489136,"name":"layer_0_output_7","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},null,null,{"weights":[],"bias":0.2660384476184845,"name":"layer_0_output_10","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},null,{"weights":[{"weight":0.533193826675415,"index":2}],"bias":0,"name":"layer_0_output_12","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},null,{"weights":[{"weight":-0.36277785897254944,"index":1}],"bias":0,"name":"layer_0_output_14","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":-0.6887124180793762,"index":0}],"bias":-0.3121510446071625,"name":"layer_0_output_15","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}}],"recurrentNeurons":[null,null,null,null,null,null,null,null,null,null],"stateNeurons":[],"outputDim":16},{"outputNeurons":[null,null,null,null,null,null,null,null,null,{"weights":[{"weight":-0.40992486476898193,"index":5},{"weight":-0.7490429282188416,"index":7},{"weight":-0.38042396306991577,"index":12},{"weight":0.2802499830722809,"index":14},{"weight":-0.46259093284606934,"index":22},{"weight":-0.43170997500419617,"index":23},{"weight":-0.6122638583183289,"index":24}],"bias":0.5237740874290466,"name":"layer_1_output_9","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},null,{"weights":[{"weight":0.47070661187171936,"index":0},{"weight":0.748959481716156,"index":7},{"weight":-0.6853590607643127,"index":12},{"weight":-0.6263692378997803,"index":15},{"weight":-0.42471301555633545,"index":19},{"weight":-0.3335980772972107,"index":21}],"bias":-0.3092273473739624,"name":"layer_1_output_11","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},null,null,null,null],"recurrentNeurons":[{"weights":[{"weight":-0.49626392126083374,"index":19},{"weight":-0.38963818550109863,"index":21}],"bias":0,"name":"layer_1_recurrent_0","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":-0.332940012216568,"index":15}],"bias":-0.9362188577651978,"name":"layer_1_recurrent_1","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},null,{"weights":[{"weight":-0.30583733320236206,"index":21},{"weight":0.5991976857185364,"index":22},{"weight":0.3954133093357086,"index":23},{"weight":0.476666659116745,"index":25}],"bias":0.2873994708061218,"name":"layer_1_recurrent_3","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":0.317425012588501,"index":19},{"weight":0.2500830888748169,"index":21}],"bias":0,"name":"layer_1_recurrent_4","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":1.2417453527450562,"index":15},{"weight":-0.606984555721283,"index":16},{"weight":0.6414817571640015,"index":20},{"weight":-0.5019652247428894,"index":21},{"weight":-0.6486623883247375,"index":23}],"bias":0.6129547357559204,"name":"layer_1_recurrent_5","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":-0.5857794284820557,"index":16},{"weight":-0.26059022545814514,"index":21},{"weight":-0.33627328276634216,"index":23}],"bias":0,"name":"layer_1_recurrent_6","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":0.8145925402641296,"index":15},{"weight":0.4571588337421417,"index":17},{"weight":0.41865187883377075,"index":19},{"weight":0.3289593756198883,"index":21},{"weight":-0.45543718338012695,"index":22},{"weight":-0.4251612424850464,"index":23}],"bias":0.3430020213127136,"name":"layer_1_recurrent_7","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":-0.24037007987499237,"index":10}],"bias":0,"name":"layer_1_recurrent_8","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}},{"weights":[{"weight":0.6364306807518005,"index":19},{"weight":0.49986082315444946,"index":21},{"weight":-0.26047226786613464,"index":22}],"bias":0,"name":"layer_1_recurrent_9","activation":{"type":"interpolatedAmeo","factor":0.5,"leakyness":0.01}}],"stateNeurons":[{"bias":0.5273516178131104,"name":"layer_1_state_0","activation":"linear","weights":[{"weight":1,"index":0}]},{"bias":0.8671452403068542,"name":"layer_1_state_1","activation":"linear","weights":[{"weight":1,"index":1}]},null,{"bias":0,"name":"layer_1_state_3","activation":"linear","weights":[{"weight":1,"index":3}]},{"bias":-0.4251083731651306,"name":"layer_1_state_4","activation":"linear","weights":[{"weight":1,"index":4}]},{"bias":0,"name":"layer_1_state_5","activation":"linear","weights":[{"weight":1,"index":5}]},{"bias":0.6216710209846497,"name":"layer_1_state_6","activation":"linear","weights":[{"weight":1,"index":6}]},{"bias":0.5517396926879883,"name":"layer_1_state_7","activation":"linear","weights":[{"weight":1,"index":7}]},{"bias":-0.49197861552238464,"name":"layer_1_state_8","activation":"linear","weights":[{"weight":1,"index":8}]},{"bias":-0.3938024640083313,"name":"layer_1_state_9","activation":"linear","weights":[{"weight":1,"index":9}]}],"outputDim":16}],"postLayers":[{"neurons":[{"weights":[{"weight":-1.274767279624939,"index":9},{"weight":1.2751485109329224,"index":11}],"bias":-0.9995079040527344,"name":"post_layer_output_0","activation":"linear"}],"outputDim":1}],"outputs":{"neurons":[{"name":"output_0","activation":"linear","weights":[{"weight":1,"index":0}],"bias":0}]}}';
  const excludedNodeIDs = [
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
    'layer_1_recurrent_8',
    'layer_1_state_8',
  ];
  const defaultLogicAnalyzerVisibleNodeIDs = [
    'layer_1_state_0',
    'layer_1_state_4',
    'layer_1_state_3',
    'layer_1_state_6',
  ];

  class SeqGen implements InputSeqGenerator {
    private i = 0;

    public reset() {
      this.i = 0;
    }

    public next(): Float32Array {
      const i = this.i;
      this.i += 1;

      if (i % 12 < 6) {
        return new Float32Array([1, -1, -1]);
      }
      return new Float32Array([-1, -1, -1]);
    }
  }

  const seqGen = new SeqGen();
</script>

<NodeVizComp
  serializedRNNGraph={serializedGraph}
  {excludedNodeIDs}
  aspectRatio={1.5}
  labelFontSizeOverride={50}
  {defaultLogicAnalyzerVisibleNodeIDs}
  inputSeqGenerator={seqGen}
/>
