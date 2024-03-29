import type { Sequential } from '@tensorflow/tfjs';

import * as tf from '@tensorflow/tfjs';
import {
  MyRNN,
  MySimpleRNNCell,
  type MyRNNLayerArgs,
  type MySimpleRNNCellLayerArgs,
} from 'src/nn/customRNN';

export type RNNLayerDef = Omit<MyRNNLayerArgs, 'cell'> & {
  cell: MySimpleRNNCellLayerArgs[];
};

export type RNNDefinition = RNNLayerDef[];

const buildModel = (layers: RNNLayerDef[]): Sequential => {
  const model = tf.sequential();
  for (const layer of layers) {
    model.add(
      new MyRNN({
        ...layer,
        cell: layer.cell.map(cellDef => new MySimpleRNNCell(cellDef)),
      })
    );
  }
  return model;
};

export class RNN {
  private model: Sequential;

  constructor(def: RNNDefinition) {
    this.model = buildModel(def);
  }
}
