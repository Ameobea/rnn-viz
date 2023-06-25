import * as fs from 'fs/promises';

import type { PageLoad } from './$types';
import type { AmeoActivationIdentifier } from '../../nn/customRNN';

const WEIGHTS_PATH = '/home/casey/Downloads/weights.json';

export const load: PageLoad = async () => {
  const weightsJSON = await fs.readFile(WEIGHTS_PATH, 'utf-8');
  const weights: {
    input_dim: number;
    output_dim: number;
    cells: {
      state_size: number;
      output_dim: number;
      output_kernel: number[][];
      output_bias: number[];
      recurrent_kernel: number[][];
      recurrent_bias: number[];
      initial_state: number[];
      recurrent_activation: AmeoActivationIdentifier;
      output_activation: AmeoActivationIdentifier;
    }[];
    post_layers: {
      input_dim: number;
      output_dim: number;
      weights: number[][];
      bias: number[];
      activation: AmeoActivationIdentifier;
    }[];
  } = JSON.parse(weightsJSON);
  return { weights };
};
