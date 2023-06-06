const seqLen = 9;

// const oneSeqExamples = () => {
//   const inputs: (1 | -1)[] = [];
//   const outputs: (1 | -1)[] = [];
//   for (let i = 0; i < seqLen; i++) {
//     const input = randomBoolInput();
//     inputs.push(input);

//     if (i === 0 || i === 1 || i === 2) {
//       if (input === 1) {
//         outputs.push(1);
//       } else {
//         outputs.push(-1);
//       }
//       continue;
//     }
//     if (i === 0) {
//       outputs.push(input);
//       continue;
//     }

//     const prevInput = inputs[i - 3];
//     // const prevInput = inputs[i - 1];
//     const output = xor(input, prevInput);
//     outputs.push(output);
//   }

//   return { inputs, outputs };
// };

// const oneSeqExamples = () => {
//   const inputs: number[] = [];
//   const outputs: number[] = [];

//   const oneVal = () => (Math.random() > 0.5 ? 1 : -1);

//   for (let i = 0; i < seqLen; i += 1) {
//     const input = oneVal();
//     inputs.push(input);

//     if (i === 0) {
//       outputs.push(-1);
//       continue;
//     }

//     // 1 if 2 of the last 3 inputs were 1, else -1
//     if (i === 1) {
//       const output = inputs[i - 1] === 1 && input === 1 ? 1 : -1;
//       outputs.push(output);
//       continue;
//     }

//     const output = inputs[i - 1] === 1 && inputs[i - 2] === 1 ? 1 : -1;
//     outputs.push(output);
//   }

//   return { inputs, outputs };
// };

const xor = (a: -1 | 1, b: -1 | 1): -1 | 1 =>
  (a === -1 && b === 1) || (a === 1 && b === -1) ? 1 : -1;

// export const oneSeqExamples = () => {
//   const inputs: [number, number][] = [];
//   const outputs: [number][] = [];

//   const oneVal = () => (Math.random() > 0.5 ? 1 : -1);

//   for (let i = 0; i < seqLen; i += 1) {
//     const input: [1 | -1, 1 | -1] = [oneVal(), oneVal()];
//     // const output: [number] =
//     //   input[i - 1] === -1
//     //     ? [-input[1]]
//     //     : [xor(input[0] as 1 | -1, (inputs[i - 1]?.[1] as 1 | -1 | undefined) ?? -1)];
//     const output: [number] = (() => {
//       let cond = false;
//       if (i === 0) {
//         return [xor(input[0], -1)];
//       }
//       if (i === 1) {
//         cond = input[0] === -1;
//       } else if (i === 2) {
//         cond = input[1] === -1;
//       }

//       if (cond) {
//         return [-input[1]];
//       }
//       return [xor(input[0], (inputs[i - 1]?.[1] ?? -1) as 1 | -1)];
//     })();
//     inputs.push(input);
//     outputs.push(output);
//   }

//   return { inputs, outputs };
// };

// -1, -1, 1 loop
/*
digraph "RNN" {
  graph [ ratio = "0.75", rankdir =TB, center =true, splines = "spline", overlap = "false", nodesep =0.32 ];
  node [ shape =square ];
  edge [ arrowhead =none ];

  subgraph "cluster_outputs" {
    graph [ rank =sink ];
    node [ fontsize =10 ];
    "output_0" [ label = "OUT0" ];
  }

  subgraph "layer_0" {
  subgraph "state" {
    node [ shape =circle ];
    "layer_0_state_1" [ label = "S" ];
    "layer_0_state_3" [ label = "S" ];
  }

  subgraph "recurrent" {
    "layer_0_recurrent_1" [ label = "N" ];
    "layer_0_recurrent_3" [ label = "N" ];
  }

  subgraph "output" {
    "layer_0_output_7" [ label = "N" ];
  }

  }

  subgraph "post_layer_0" {
    "post_layer_output_0" [ label = "N" ];
  }

  subgraph "cluster_inputs" {
    graph [ rank =source ];
    node [ shape =circle, fontsize =10 ];
  }

  "post_layer_output_0" -> "output_0" [ label = "1" ];
  "layer_0_output_7" -> "post_layer_output_0" [ label = "1" ];
  "layer_0_state_1" -> "layer_0_output_7" [ label = "-1" ];
  "layer_0_recurrent_1" -> "layer_0_state_1" [ label = "1" ];
  "layer_0_state_3" -> "layer_0_recurrent_1" [ label = "1" ];
  "layer_0_recurrent_3" -> "layer_0_state_3" [ label = "1" ];
  "layer_0_state_1" -> "layer_0_recurrent_3" [ label = "-1" ];
  "layer_0_state_3" -> "layer_0_recurrent_3" [ label = "1" ];
}

Serialized graph: {"inputLayer":{"neurons":[null,null]},"cells":[{"outputNeurons":[null,null,null,null,null,null,null,{"weights":[{"weight":-1,"index":3}],"bias":0,"name":"layer_0_output_7","activation":{"type":"interpolatedAmeo","factor":0.6,"leakyness":1}}],"recurrentNeurons":[null,{"weights":[{"weight":1,"index":5}],"bias":0,"name":"layer_0_recurrent_1","activation":{"type":"interpolatedAmeo","factor":0.6,"leakyness":1}},null,{"weights":[{"weight":-1,"index":3},{"weight":1,"index":5}],"bias":-1,"name":"layer_0_recurrent_3","activation":{"type":"interpolatedAmeo","factor":0.6,"leakyness":1}}],"stateNeurons":[null,{"bias":-1,"name":"layer_0_state_1","activation":"linear","weights":[{"weight":1,"index":1}]},null,{"bias":1,"name":"layer_0_state_3","activation":"linear","weights":[{"weight":1,"index":3}]}],"outputDim":8}],"postLayers":[{"neurons":[{"weights":[{"weight":1,"index":7}],"bias":0,"name":"post_layer_output_0","activation":"linear"}],"outputDim":1}],"outputs":{"neurons":[{"name":"output_0","activation":"linear","weights":[{"weight":1,"index":0}],"bias":0}]}}

LEARNED:

s0[0] = T
s1[0] = F

s0[n] = XNOR(s0[n-1], s1[n-1])
s1[n] = NOT(s0[n-1])
out   = s1[n]

EXPECTED:

s0[0] = F
s1[0] = F
s2[0] = T

s0[n] = s2[n-1]
s1[n] = s0[n-1]
s2[n] = s1[n-1]
out   = s1[n]

*/
// export const oneSeqExamples = () => {
//   const inputs: [number, number][] = [];
//   const outputs: [number][] = [];

//   const oneVal = () => (Math.random() > 0.5 ? 1 : -1);

//   for (let i = 0; i < seqLen; i += 1) {
//     const input: [1 | -1, 1 | -1] = [oneVal(), oneVal()];
//     const output: [1 | -1] = (() => {
//       if (i % 3 === 0) {
//         return [-1];
//       } else if (i % 3 === 1) {
//         return [-1];
//       } else {
//         return [1];
//       }
//     })();
//     inputs.push(input);
//     outputs.push(output);
//   }

//   return { inputs, outputs };
// };

// Simple 2-ago
/*
digraph "RNN" {
subgraph "cluster_outputs" {
  "output_0";
}

subgraph "cluster_layer_0" {
subgraph "cluster_state" {
  "layer_0_state_0";
  "layer_0_state_1";
}

subgraph "cluster_recurrent" {
  "layer_0_recurrent_0";
  "layer_0_recurrent_1";
}

subgraph "cluster_output" {
  "layer_0_output_1";
}

}

subgraph "cluster_post_layer_0" {
  "post_layer_output_0";
}

subgraph "cluster_inputs" {
  "input_0";
}

  "post_layer_output_0" -> "output_0" [ label = "1" ];
  "layer_0_output_1" -> "post_layer_output_0" [ label = "-1" ];
  "layer_0_state_0" -> "layer_0_output_1" [ label = "-1" ];
  "layer_0_recurrent_0" -> "layer_0_state_0" [ label = "1" ];
  "layer_0_state_1" -> "layer_0_recurrent_0" [ label = "1" ];
  "layer_0_recurrent_1" -> "layer_0_state_1" [ label = "1" ];
  "input_0" -> "layer_0_recurrent_1" [ label = "-1" ];
}
*/
// export const oneSeqExamples = () => {
//   const inputs: [number][] = [];
//   const outputs: [number][] = [];

//   const oneVal = () => (Math.random() > 0.5 ? 1 : -1);

//   for (let i = 0; i < seqLen; i += 1) {
//     const input: [1 | -1] = [oneVal()];
//     const output: [number] = [i === 0 || i === 1 ? -1 : inputs[i - 2][0]];
//     inputs.push(input);
//     outputs.push(output);
//   }

//   return { inputs, outputs };
// };

// input[0] is control, input[1] is data.  Has an internal "mode" which determines if data is output directly or inverted.
// Starts out outputting input[1] directly.  When control changes from -1 to 1 or 1 to -1, the mode is inverted.
//
// Has a 10% chance of flipping the control value at each step.
//
// Turned out to be easy:
/*
digraph "RNN" {
subgraph "cluster_outputs" {
  "output_0";
}

subgraph "cluster_layer_0" {
subgraph "cluster_state" {
  "layer_0_state_0";
  "layer_0_state_3";
}

subgraph "cluster_recurrent" {
  "layer_0_recurrent_0";
  "layer_0_recurrent_3";
}

subgraph "cluster_output" {
  "layer_0_output_2";
}

}

subgraph "cluster_layer_1" {

subgraph "cluster_output" {
  "layer_1_output_3";
}

}

subgraph "cluster_post_layer_0" {
  "post_layer_output_0";
}

subgraph "cluster_inputs" {
  "input_0";
  "input_1";
}

  "post_layer_output_0" -> "output_0" [ label = "1" ];
  "layer_1_output_3" -> "post_layer_output_0" [ label = "1" ];
  "layer_0_output_2" -> "layer_1_output_3" [ label = "-1" ];
  "input_1" -> "layer_0_output_2" [ label = "-1" ];
  "layer_0_state_0" -> "layer_0_output_2" [ label = "1" ];
  "layer_0_recurrent_0" -> "layer_0_state_0" [ label = "1" ];
  "layer_0_state_3" -> "layer_0_output_2" [ label = "1" ];
  "layer_0_recurrent_3" -> "layer_0_state_3" [ label = "1" ];
  "input_0" -> "layer_0_recurrent_3" [ label = "-1" ];
}
*/
// export const oneSeqExamples = () => {
//   const inputs: [number, number][] = [];
//   const outputs: [number][] = [];

//   const oneVal = () => (Math.random() > 0.5 ? 1 : -1);

//   let inverted = false;
//   let lastControl = -1;

//   for (let i = 0; i < seqLen; i += 1) {
//     const newControl = Math.random() > 0.9 ? -lastControl : lastControl;
//     const input: [1 | -1, 1 | -1] = [newControl as 1 | -1, oneVal()];
//     const output: [1 | -1] = (() => {
//       if (inverted) {
//         return [-input[1] as 1 | -1];
//       } else {
//         return [input[1] as 1 | -1];
//       }
//     })();
//     inputs.push(input);
//     outputs.push(output);
//     if (newControl !== lastControl) {
//       inverted = !inverted;
//     }
//     lastControl = newControl;
//   }

//   return { inputs, outputs };
// };

//
/*
digraph "RNN" {
  subgraph "outputs" {
    "output_0";
  }

  subgraph "layer_0" {
  subgraph "state" {
    "layer_0_state_4";
    "layer_0_state_7";
  }

  subgraph "recurrent" {
    "layer_0_recurrent_4";
    "layer_0_recurrent_7";
  }

  subgraph "output" {
    "layer_0_output_5";
  }

  }

  subgraph "layer_1" {
  subgraph "state" {
  }

  subgraph "output" {
    "layer_1_output_1";
  }

  }

  subgraph "post_layer_0" {
    "post_layer_output_0";
  }

  subgraph "inputs" {
    "input_0";
  }

    "post_layer_output_0" -> "output_0" [ label = "1" ];
    "layer_1_output_1" -> "post_layer_output_0" [ label = "-1" ];
    "layer_0_output_5" -> "layer_1_output_1" [ label = "-1" ];
    "input_0" -> "layer_0_output_5" [ label = "1" ];
    "layer_0_state_4" -> "layer_0_output_5" [ label = "1" ];
    "layer_0_recurrent_4" -> "layer_0_state_4" [ label = "1" ];
    "layer_0_state_7" -> "layer_0_output_5" [ label = "1" ];
    "layer_0_recurrent_7" -> "layer_0_state_7" [ label = "1" ];
    "input_0" -> "layer_0_recurrent_7" [ label = "-1" ];
    "layer_0_state_4" -> "layer_0_recurrent_7" [ label = "-1" ];
    "layer_0_state_7" -> "layer_0_recurrent_7" [ label = "-1" ];
  }
*/
// export const oneSeqExamples = () => {
//   const inputs: [number][] = [];
//   const outputs: [number][] = [];

//   const oneVal = () => (Math.random() > 0.5 ? 1 : -1);
//   let sum = 0;

//   for (let i = 0; i < seqLen; i += 1) {
//     const input: [1 | -1] = [oneVal()];
//     inputs.push(input);
//     sum += input[0] === 1 ? 1 : 0; // add 1 to sum if input was 1
//     outputs.push([sum % 2 === 0 ? -1 : 1]); // mod 2
//   }

//   return { inputs, outputs };
// };

// can't learn
// export const oneSeqExamples = () => {
//   const inputs: [number][] = [];
//   const outputs: [number][] = [];

//   const oneVal = () => (Math.random() > 0.5 ? 1 : -1);
//   let output = 1;
//   let n = 1;
//   let count = 0;

//   for (let i = 0; i < seqLen; i += 1) {
//     const input: [1 | -1] = [oneVal()];
//     inputs.push(input);

//     if (input[0] === -1) {
//       count += 1; // count the number of consecutive -1s
//     } else {
//       n = count > 0 ? count : n; // set n to the count of -1s, if any
//       n = Math.min(n, 3); // limit n to 3
//       count = 0; // reset the count
//     }

//     if (i % n === 0) {
//       output *= -1; // change output every n steps
//     }

//     outputs.push([output]);
//   }

//   return { inputs, outputs };
// };

/**
 * All 3 inputs -1 or all 3 inputs 1
 *
 * {"inputLayer":{"neurons":[{"name":"input_0","activation":"linear","weights":[],"bias":0},{"name":"input_1","activation":"linear","weights":[],"bias":0},{"name":"input_2","activation":"linear","weights":[],"bias":0}]},"cells":[{"outputNeurons":[null,null,null,{"weights":[{"weight":-0.5,"index":0},{"weight":-0.5,"index":1},{"weight":-0.5,"index":2}],"bias":-1,"name":"layer_0_output_3","activation":{"type":"interpolatedAmeo","factor":0.6,"leakyness":1}},null,null,null,null],"recurrentNeurons":[null],"stateNeurons":[null,null],"outputDim":8}],"postLayers":[{"neurons":[{"weights":[{"weight":-1.5,"index":3}],"bias":0,"name":"post_layer_output_0","activation":"linear"}],"outputDim":1}],"outputs":{"neurons":[{"name":"output_0","activation":"linear","weights":[{"weight":1,"index":0}],"bias":0}]}}
 *
 * Apparently it's representable in a single neuron!  All input weights=-0.5, bias=-1
 */
export const oneSeqExamples = () => {
  const inputs: [number, number, number][] = [];
  const outputs: [number][] = [];

  const oneVal = () => (Math.random() > 0.5 ? 1 : -1);

  for (let i = 0; i < seqLen; i += 1) {
    const input: [1 | -1, 1 | -1, 1 | -1] = [oneVal(), oneVal(), oneVal()];
    const output: [1 | -1] = (() => {
      if (input[0] === input[1] && input[1] === input[2]) {
        return [1];
      }
      return [-1];
    })();
    inputs.push(input);
    outputs.push(output);
  }

  return { inputs, outputs };
};
