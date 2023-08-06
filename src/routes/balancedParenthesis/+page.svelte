<script lang="ts">
  import type { InputSeqGenerator } from '../nodeViz/NodeViz';
  import NodeVizComp from '../nodeViz/NodeVizComp.svelte';

  const serializedGraph =
    '{"inputLayer":{"neurons":[{"name":"input_0","activation":"linear","weights":[],"bias":0}]},"cells":[{"outputNeurons":[{"weights":[{"weight":0.9258173108100891,"index":0},{"weight":-4.172152042388916,"index":1},{"weight":0.2808956503868103,"index":2}],"bias":0.5771756768226624,"name":"layer_0_output_0","activation":{"type":"interpolatedAmeo","factor":1,"leakyness":0.1}},null,{"weights":[{"weight":0.8388127088546753,"index":0},{"weight":3.2579290866851807,"index":1},{"weight":-0.2192409336566925,"index":2}],"bias":-0.6698281764984131,"name":"layer_0_output_2","activation":{"type":"interpolatedAmeo","factor":1,"leakyness":0.1}},null,null,null],"recurrentNeurons":[{"weights":[{"weight":-0.04513341188430786,"index":0},{"weight":-1.000068187713623,"index":1}],"bias":0,"name":"layer_0_recurrent_0","activation":{"type":"interpolatedAmeo","factor":1,"leakyness":0.1}},{"weights":[{"weight":1.0060919523239136,"index":2}],"bias":0,"name":"layer_0_recurrent_1","activation":{"type":"interpolatedAmeo","factor":1,"leakyness":0.1}}],"stateNeurons":[{"bias":0.23326684534549713,"name":"layer_0_state_0","activation":"linear","weights":[{"weight":1,"index":0}]},{"bias":0.31625017523765564,"name":"layer_0_state_1","activation":"linear","weights":[{"weight":1,"index":1}]}],"outputDim":8}],"postLayers":[{"neurons":[{"weights":[{"weight":2.6380178928375244,"index":0},{"weight":-3.3781895637512207,"index":2}],"bias":-0.2598108947277069,"name":"post_layer_output_0","activation":"linear"}],"outputDim":1}],"outputs":{"neurons":[{"name":"output_0","activation":"linear","weights":[{"weight":1,"index":0}],"bias":0}]}}';

  class RandomValidParenthesisGenerator implements InputSeqGenerator {
    private depth = 0;
    private maxDepth = 8;

    public reset() {
      this.depth = 0;
    }

    public next(): Float32Array {
      const input: [1 | -1] = (() => {
        if (this.depth < this.maxDepth && (this.depth === 0 || Math.random() > 0.5)) {
          this.depth += 1;
          return [-1];
        } else {
          this.depth -= 1;
          return [1];
        }
      })();
      return Float32Array.from(input);
    }
  }

  // Generates deterministic parenthesis sequences like (((((((())))))))(((((((())))))))...
  class RepeatedParenthesisGenerator implements InputSeqGenerator {
    private depth = 0;
    private maxDepth = 8;
    private closing = false;

    public reset() {
      this.depth = 0;
      this.closing = false;
    }

    public next(): Float32Array {
      if (this.closing) {
        if (this.depth <= 0) {
          throw new Error('Invalid state');
        }
        this.depth -= 1;
        this.closing = this.depth > 0;
        return Float32Array.from([1]);
      } else {
        if (this.depth >= this.maxDepth) {
          throw new Error('Invalid state');
        }
        this.depth += 1;
        this.closing = this.depth >= this.maxDepth;
        return Float32Array.from([-1]);
      }
    }
  }

  // const inputSeqGenerator = new RandomValidParenthesisGenerator();
  const inputSeqGenerator = new RepeatedParenthesisGenerator();
</script>

<NodeVizComp
  serializedRNNGraph={serializedGraph}
  aspectRatio={0.82}
  labelFontSizeOverride={90}
  {inputSeqGenerator}
  defaultLogicAnalyzerVisibleNodeIDs="ALL"
/>
