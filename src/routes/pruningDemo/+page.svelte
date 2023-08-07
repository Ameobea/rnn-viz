<script lang="ts">
  import NodeVizComp from '../nodeViz/NodeVizComp.svelte';

  const serializedGraph = JSON.stringify({
    inputLayer: { neurons: [{ name: 'input_0', activation: 'linear', weights: [], bias: 0 }] },
    cells: [
      {
        outputNeurons: [
          {
            weights: [],
            bias: 0.5883655548095703,
            name: 'layer_0_output_0',
            activation: { type: 'interpolatedAmeo', factor: 0.5, leakyness: 1 },
          },
          {
            weights: [{ weight: 0.4903032183647156, index: 1 }],
            bias: 0.16279402375221252,
            name: 'layer_0_output_1',
            activation: { type: 'interpolatedAmeo', factor: 0.5, leakyness: 1 },
          },
          {
            weights: [{ weight: 0.579464316368103, index: 0 }],
            bias: 1.1870207786560059,
            name: 'layer_0_output_2',
            activation: { type: 'interpolatedAmeo', factor: 0.5, leakyness: 1 },
          },
          {
            weights: [{ weight: -0.8984912633895874, index: 1 }],
            bias: -0.0956144779920578,
            name: 'layer_0_output_3',
            activation: { type: 'interpolatedAmeo', factor: 0.5, leakyness: 1 },
          },
        ],
        recurrentNeurons: [
          {
            weights: [{ weight: 1.5696280002593994, index: 0 }],
            bias: -0.10378765314817429,
            name: 'layer_0_recurrent_0',
            activation: { type: 'interpolatedAmeo', factor: 0.5, leakyness: 1 },
          },
        ],
        stateNeurons: [
          {
            bias: 0.5493731498718262,
            name: 'layer_0_state_0',
            activation: 'linear',
            weights: [{ weight: 1, index: 0 }],
          },
        ],
        outputDim: 4,
      },
    ],
    postLayers: [
      {
        neurons: [
          {
            weights: [
              { weight: 0.602713406085968, index: 1 },
              { weight: -0.8723659515380859, index: 3 },
            ],
            bias: 0.08960767090320587,
            name: 'post_layer_output_0',
            activation: 'linear',
          },
        ],
        outputDim: 1,
      },
    ],
    outputs: {
      neurons: [
        { name: 'output_0', activation: 'linear', weights: [{ weight: 1, index: 0 }], bias: 0 },
      ],
    },
  });
</script>

<NodeVizComp serializedRNNGraph={serializedGraph} aspectRatio={0.77} />
