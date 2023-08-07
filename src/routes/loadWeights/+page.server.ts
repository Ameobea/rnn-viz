import * as fs from 'fs/promises';

import type { PageServerLoad } from './$types';
import type { AmeoActivationIdentifier } from '../../nn/customRNN';

export const load: PageServerLoad = async ({ url, fetch }) => {
  const homeDir = process.env.HOME;
  if (!homeDir) {
    throw new Error('No home dir');
  }
  const weightsPath = `${homeDir}/Downloads/weights.json`;
  const rawWeights = await (async () => {
    const ameotrackID = 'b9i';
    // const ameotrackID = url.searchParams.get('ameotrackID');
    if (ameotrackID) {
      console.log(`https://i.ameo.link/${ameotrackID}.json`);
      fetch(`https://i.ameo.link/${ameotrackID}.json`)
        .then(res => res.text())
        .then(console.log);
      return fetch(`https://i.ameo.link/${ameotrackID}.json`).then(res => res.json());
    }
    return JSON.parse(await fs.readFile(weightsPath, 'utf-8'));
  })();
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
  } = rawWeights;
  return { weights };
};
