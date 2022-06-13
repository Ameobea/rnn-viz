import { RNN, type RNNDefinition } from 'src/nn/RNN';

export class RNNViz {
  private model: RNN;

  constructor(def: RNNDefinition) {
    this.model = new RNN(def);
  }
}
