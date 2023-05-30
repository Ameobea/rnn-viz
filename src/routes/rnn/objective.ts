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

export const oneSeqExamples = () => {
  const inputs: [number, number][] = [];
  const outputs: [number][] = [];

  const oneVal = () => (Math.random() > 0.5 ? 1 : -1);

  for (let i = 0; i < seqLen; i += 1) {
    const input: [1 | -1, 1 | -1] = [oneVal(), oneVal()];
    // const output: [number] =
    //   input[i - 1] === -1
    //     ? [-input[1]]
    //     : [xor(input[0] as 1 | -1, (inputs[i - 1]?.[1] as 1 | -1 | undefined) ?? -1)];
    const output: [number] = (() => {
      let cond = false;
      if (i === 0) {
        return [xor(input[0], -1)];
      }
      if (i === 1) {
        cond = input[0] === -1;
      } else if (i === 2) {
        cond = input[1] === -1;
      }

      if (cond) {
        return [-input[1]];
      }
      return [xor(input[0], (inputs[i - 1]?.[1] ?? -1) as 1 | -1)];
    })();
    inputs.push(input);
    outputs.push(output);
  }

  return { inputs, outputs };
};

// -1, -1, 1 loop
/*
digraph "RNN" {
  subgraph "outputs" {
    "output_0";
  }

  subgraph "layer_0" {
  subgraph "state" {
    "layer_0_state_0";
    "layer_0_state_1";
  }

  subgraph "recurrent" {
    "layer_0_recurrent_0";
    "layer_0_recurrent_1";
  }

  subgraph "output" {
    "layer_0_output_0";
  }

  }

  subgraph "post_layer_0" {
    "post_layer_output_0";
  }

    "post_layer_output_0" -> "output_0" [ label = "1" ];
    "layer_0_output_0" -> "post_layer_output_0" [ label = "1" ];
    "layer_0_state_0" -> "layer_0_output_0" [ label = "1" ];
    "layer_0_recurrent_0" -> "layer_0_state_0" [ label = "1" ];
    "layer_0_state_0" -> "layer_0_recurrent_0" [ label = "0.500" ];
    "layer_0_state_1" -> "layer_0_recurrent_0" [ label = "1" ];
    "layer_0_recurrent_1" -> "layer_0_state_1" [ label = "1" ];
    "layer_0_state_0" -> "layer_0_recurrent_1" [ label = "-1" ];
    "layer_0_state_1" -> "layer_0_recurrent_1" [ label = "0.500" ];
  }
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
