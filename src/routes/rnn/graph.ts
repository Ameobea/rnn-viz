import type { Rank, Tensor } from '@tensorflow/tfjs';
import type { AmeoActivationIdentifier } from '../../nn/customRNN';
import { nativeFusedInterpolatedAmeoImplInner } from '../../nn/ameoActivation/native';
import * as GVB from 'graphviz-builder';

export interface RNNCellWeights {
  initialState: Tensor<Rank>;
  stateSize: number;
  outputSize: number;
  recurrentTreeWeights: Tensor<Rank>;
  recurrentTreeBias?: Tensor<Rank>;
  outputTreeWeights: Tensor<Rank>;
  outputTreeBias?: Tensor<Rank>;
  outputActivation: AmeoActivationIdentifier;
  recurrentActivation: AmeoActivationIdentifier;
}

export interface PostLayerWeights {
  weights: Tensor<Rank>;
  bias?: Tensor<Rank>;
  activation: AmeoActivationIdentifier | 'linear';
}

export interface RNNGraphParams {
  clipThreshold: number;
  /**
   * Weights are quantized to the nearest multiple of `quantizationInterval`
   */
  quantizationInterval: number;
}

const DefaultRNNGraphParams: { [K in keyof RNNGraphParams]: NonNullable<RNNGraphParams[K]> } = {
  clipThreshold: 0.1,
  quantizationInterval: 1,
};

const mergeNonNullable = <T>(a: T, b: Partial<T>): T => {
  const result = { ...a };
  for (const key in b) {
    if (b[key] !== undefined) {
      result[key] = b[key]!;
    }
  }
  return result;
};

export interface SparseWeight {
  weight: number;
  /**
   * Index of the neuron in the previous layer
   */
  index: number;
  inputNeuron: SparseNeuron;
}

class SparseNeuron {
  public weights: SparseWeight[];
  public bias: number;
  public name: string;

  public activation: (x: number) => number = x => x;

  constructor(
    weights: SparseWeight[],
    bias: number,
    name: string,
    activation?: (x: number) => number
  ) {
    this.weights = weights;
    this.bias = bias;
    this.name = name;
    if (activation) {
      this.activation = activation;
    }
  }

  public getOutput(): number {
    const weightedSum = this.weights.reduce((sum, { weight, inputNeuron }) => {
      return sum + weight * inputNeuron.getOutput();
    }, this.bias);
    return this.activation(weightedSum);
  }

  public advanceSequence() {
    // no-op
  }
}

class InputNeuron extends SparseNeuron {
  public index: number;
  private inputSequence: Float32Array[] = [];

  constructor(index: number) {
    const name = `input_${index}`;
    super([], 0, name);
    this.index = index;
  }

  public setInputSequence(inputSequence: Float32Array[]) {
    this.inputSequence = inputSequence;
  }

  public getOutput(): number {
    const input = this.inputSequence[0];
    if (!input) {
      throw new Error('Input sequence is empty');
    }
    return input[this.index];
  }

  public advanceSequence() {
    this.inputSequence.shift();
  }
}

class OutputNeuron extends SparseNeuron {
  public index: number;

  constructor(index: number, prevLayer: GraphRNNLayer) {
    const inputNeuron = prevLayer.getNeuron(index);
    const weights = [];
    if (!inputNeuron) {
      console.error(`No input neuron for output neuron ${index}`, { prevLayer, index });
    } else {
      weights.push({ index, inputNeuron, weight: 1 });
    }
    const name = `output_${index}`;
    super(weights, 0, name);
    this.index = index;
  }
}

class StateNeuron extends SparseNeuron {
  public layerIx: number;
  public index: number;
  public initialState: number;
  private state: number;

  constructor(layerIx: number, index: number, initialState: number) {
    const name = `layer_${layerIx}_state_${index}`;
    super([], 0, name);
    this.layerIx = layerIx;
    this.index = index;
    this.initialState = initialState;
    this.state = initialState;
  }

  public getOutput(): number {
    return this.state;
  }

  public advanceSequence() {
    const newState = SparseNeuron.prototype.getOutput.call(this);
    this.state = newState;
  }

  /**
   * Adds the recurrent connection to the recurrent layer.  This is called after
   * the layer has been fully populated so we have access to all the neurons.
   */
  public connect(index: number, recurrentNeuron: SparseNeuron) {
    this.weights.push({ index, inputNeuron: recurrentNeuron, weight: 1 });
  }
}

abstract class GraphRNNLayer {
  abstract outputDim: number;

  abstract getNeuron(outputIx: number): SparseNeuron | undefined;

  abstract advanceSequence(): void;
}

class GraphRNNInputLayer implements GraphRNNLayer {
  public outputDim: number;
  private neurons: InputNeuron[] = [];
  private inputSequence: Float32Array[] = [];

  constructor(inputDim: number) {
    this.outputDim = inputDim;
    for (let i = 0; i < inputDim; ++i) {
      this.neurons.push(new InputNeuron(i));
    }
  }

  public setInputSequence(inputSequence: Float32Array[]) {
    this.inputSequence = inputSequence;
    this.neurons.forEach(neuron => neuron.setInputSequence([...inputSequence]));
  }

  getNeuron(outputIx: number): SparseNeuron | undefined {
    return this.neurons[outputIx];
  }

  public get size(): number {
    return this.neurons.length;
  }

  public advanceSequence() {
    this.neurons.forEach(neuron => neuron.advanceSequence());
  }
}

/**
 * Represents the output tensor of a GraphRNN
 */
class GraphRNNOutputs implements GraphRNNLayer {
  public outputDim: number;
  private neurons: OutputNeuron[] = [];

  constructor(outputDim: number, prevLayer: GraphRNNLayer) {
    this.outputDim = outputDim;
    for (let i = 0; i < outputDim; ++i) {
      this.neurons.push(new OutputNeuron(i, prevLayer));
    }
  }

  getNeuron(outputIx: number): SparseNeuron | undefined {
    return this.neurons[outputIx];
  }

  public get size(): number {
    return this.neurons.length;
  }

  public advanceSequence(): void {
    this.neurons.forEach(neuron => neuron.advanceSequence());
  }
}

const buildActivation = (id: AmeoActivationIdentifier | 'linear'): ((x: number) => number) => {
  if (id === 'linear') {
    return x => x;
  }

  if (typeof id === 'string') {
    throw new Error('Unimplemented');
  }
  switch (id.type) {
    case 'leakyAmeo':
      return x => nativeFusedInterpolatedAmeoImplInner(1, x, id.leakyness ?? 0);
    default:
      throw new Error('Unimplemented');
  }
};

class GraphRNNCell implements GraphRNNLayer {
  public outputDim: number;
  public params: RNNGraphParams;
  public recurrentNeurons: SparseNeuron[] = [];
  public outputNeurons: SparseNeuron[] = [];
  public stateNeurons: StateNeuron[] = [];

  constructor(
    layerIx: number,
    weights: RNNCellWeights,
    params: RNNGraphParams,
    prevLayer: GraphRNNLayer
  ) {
    this.params = params;
    this.outputDim = weights.outputSize;

    if (weights.recurrentTreeWeights.shape.length !== 2) {
      throw new Error('Recurrent tree weights must be 2D');
    }

    const recurrentActivation = buildActivation(weights.recurrentActivation);
    const outputActivation = buildActivation(weights.outputActivation);

    const recurrentWeightsData = clipAndQuantizeWeights(
      weights.recurrentTreeWeights.dataSync() as Float32Array,
      this.params
    );
    const recurrentBiasData = weights.recurrentTreeBias
      ? clipAndQuantizeWeights(weights.recurrentTreeBias.dataSync() as Float32Array, this.params)
      : null;

    const outputWeightsData = clipAndQuantizeWeights(
      weights.outputTreeWeights.dataSync() as Float32Array,
      this.params
    );
    const outputBiasData = weights.outputTreeBias
      ? clipAndQuantizeWeights(weights.outputTreeBias.dataSync() as Float32Array, this.params)
      : null;

    const initialStateData = weights.initialState.dataSync();
    const getInputNeuron = (outputIx: number): SparseNeuron | undefined => {
      // Inputs to the output and recurrent tree are created by concatenating the inputs and the state
      if (outputIx < prevLayer.outputDim) {
        return prevLayer.getNeuron(outputIx);
      }
      const stateIx = outputIx - prevLayer.outputDim;
      if (!this.stateNeurons[stateIx]) {
        const stateNeuron = new StateNeuron(layerIx, stateIx, initialStateData[stateIx]);
        this.stateNeurons[stateIx] = stateNeuron;
      }
      return this.stateNeurons[stateIx];
    };

    for (let outputNeuronIx = 0; outputNeuronIx < weights.outputSize; outputNeuronIx += 1) {
      const weightsForNeuron: SparseWeight[] = [];
      for (let weightIx = 0; weightIx < weights.outputTreeWeights.shape[0]; weightIx += 1) {
        const tensorIx = weightIx * weights.outputTreeWeights.shape[1]! + outputNeuronIx;
        const weight = outputWeightsData[tensorIx];
        if (weight === undefined) {
          throw new Error(`Unexpected undefined weight; tensorIx=${tensorIx}`);
        }
        if (weight === 0) {
          continue;
        }

        const inputNeuron = getInputNeuron(weightIx);
        if (!inputNeuron) {
          continue;
        }
        weightsForNeuron.push({ weight, index: weightIx, inputNeuron });
      }

      if (weightsForNeuron.length === 0 && outputBiasData?.[outputNeuronIx] === 0) {
        continue;
      }
      const name = `layer_${layerIx}_output_${outputNeuronIx}`;
      const neuron = new SparseNeuron(
        weightsForNeuron,
        outputBiasData?.[outputNeuronIx] ?? 0,
        name,
        outputActivation
      );
      this.outputNeurons[outputNeuronIx] = neuron;
    }

    for (let recurrentNeuronIx = 0; recurrentNeuronIx < weights.stateSize; recurrentNeuronIx += 1) {
      const weightsForNeuron: SparseWeight[] = [];
      for (let weightIx = 0; weightIx < weights.recurrentTreeWeights.shape[0]; weightIx += 1) {
        const tensorIx = weightIx * weights.recurrentTreeWeights.shape[1]! + recurrentNeuronIx;
        const weight = recurrentWeightsData[tensorIx];
        if (weight === undefined) {
          throw new Error(`Unexpected undefined weight; tensorIx=${tensorIx}`);
        }
        if (weight === 0) {
          continue;
        }

        const inputNeuron = getInputNeuron(weightIx);
        if (!inputNeuron) {
          continue;
        }
        weightsForNeuron.push({ weight, index: weightIx, inputNeuron });
      }

      if (weightsForNeuron.length === 0 && !recurrentBiasData?.[recurrentNeuronIx]) {
        continue;
      }
      const name = `layer_${layerIx}_recurrent_${recurrentNeuronIx}`;
      const neuron = new SparseNeuron(
        weightsForNeuron,
        recurrentBiasData?.[recurrentNeuronIx] ?? 0,
        name,
        recurrentActivation
      );
      this.recurrentNeurons[recurrentNeuronIx] = neuron;
    }

    // Now that we have populated all neurons, we can add in the recurrent connections between the
    // recurrent tree and the state
    this.stateNeurons.forEach((neuron, index) => {
      const recurrentNeuron = this.recurrentNeurons[index];
      if (!recurrentNeuron) {
        return;
      }
      neuron.connect(index, recurrentNeuron);
    });

    // // Prune out state and recurrent neurons that have no connections
    // this.stateNeurons = this.stateNeurons.filter(neuron => neuron.weights.length > 0);
    // this.recurrentNeurons = this.recurrentNeurons.filter(neuron => neuron.weights.length > 0);
  }

  getNeuron(outputIx: number): SparseNeuron | undefined {
    return this.outputNeurons[outputIx];
  }

  public advanceSequence(): void {
    this.stateNeurons.forEach(neuron => neuron.advanceSequence());
    this.recurrentNeurons.forEach(neuron => neuron.advanceSequence());
    this.outputNeurons.forEach(neuron => neuron.advanceSequence());
  }
}

class GraphPostLayer implements GraphRNNLayer {
  public outputDim: number;
  public neurons: SparseNeuron[] = [];

  constructor(
    weights: PostLayerWeights,
    params: RNNGraphParams,
    prevLayer: GraphRNNLayer,
    outputDim: number
  ) {
    this.outputDim = outputDim;

    const weightsData = clipAndQuantizeWeights(weights.weights.dataSync() as Float32Array, params);
    const biasData = weights.bias
      ? clipAndQuantizeWeights(weights.bias?.dataSync() as Float32Array, params)
      : undefined;

    const getInputNeuron = (outputIx: number): SparseNeuron | undefined =>
      prevLayer.getNeuron(outputIx);

    const outputActivation = buildActivation(weights.activation);

    for (let outputNeuronIx = 0; outputNeuronIx < outputDim; outputNeuronIx += 1) {
      const weightsForNeuron: SparseWeight[] = [];
      for (let weightIx = 0; weightIx < weights.weights.shape[0]; weightIx += 1) {
        const tensorIx = weightIx * weights.weights.shape[1]! + outputNeuronIx;
        const weight = weightsData[tensorIx];
        if (weight === undefined) {
          throw new Error(`Unexpected undefined weight; tensorIx=${tensorIx}`);
        }
        if (weight === 0) {
          continue;
        }

        const inputNeuron = getInputNeuron(weightIx);
        if (!inputNeuron) {
          continue;
        }
        weightsForNeuron.push({ weight, index: weightIx, inputNeuron });
      }

      if (weightsForNeuron.length === 0 && biasData?.[outputNeuronIx] === 0) {
        continue;
      }
      const name = `post_layer_output_${outputNeuronIx}`;
      const neuron = new SparseNeuron(
        weightsForNeuron,
        biasData?.[outputNeuronIx] ?? 0,
        name,
        outputActivation
      );
      this.neurons[outputNeuronIx] = neuron;
    }
  }

  advanceSequence(): void {
    // no-op
  }

  getNeuron(outputIx: number): SparseNeuron | undefined {
    return this.neurons[outputIx];
  }
}

/**
 * Sets weights to 0 if their magnitude is less than or equal to `clipThreshold`.
 */
const clipAndQuantizeWeights = (weights: Float32Array, params: RNNGraphParams): Float32Array => {
  const { clipThreshold, quantizationInterval } = params;
  for (let i = 0; i < weights.length; ++i) {
    if (Math.abs(weights[i]) <= clipThreshold) {
      weights[i] = 0;
    }
  }

  if (!quantizationInterval) {
    return weights;
  }

  // Quantize weights to nearest multiple of `quantizationInterval`
  for (let i = 0; i < weights.length; ++i) {
    weights[i] = Math.round(weights[i] / quantizationInterval) * quantizationInterval;
  }

  return weights;
};

export class RNNGraph {
  private params: RNNGraphParams;
  private inputLayer: GraphRNNInputLayer;
  private outputs: GraphRNNOutputs;
  private cells: GraphRNNCell[] = [];
  private postLayers: GraphPostLayer[] = [];

  constructor(
    inputDim: number,
    outputDim: number,
    rnnCells: RNNCellWeights[],
    postLayers: PostLayerWeights[],
    params?: Partial<RNNGraphParams>
  ) {
    this.params = params
      ? mergeNonNullable(DefaultRNNGraphParams, params)
      : { ...DefaultRNNGraphParams };

    this.inputLayer = new GraphRNNInputLayer(inputDim);

    let prevLayer: GraphRNNLayer = this.inputLayer;
    for (let layerIx = 0; layerIx < rnnCells.length; layerIx += 1) {
      const cell = rnnCells[layerIx];
      const cellLayer = new GraphRNNCell(layerIx, cell, this.params, prevLayer);
      this.cells.push(cellLayer);
      prevLayer = cellLayer;
    }

    for (let postLayerIx = 0; postLayerIx < postLayers.length; postLayerIx += 1) {
      const weights = postLayers[postLayerIx];
      const postLayer = new GraphPostLayer(weights, this.params, prevLayer, outputDim);
      this.postLayers.push(postLayer);
      prevLayer = postLayer;
    }

    this.outputs = new GraphRNNOutputs(outputDim, prevLayer);
  }

  public evaluate(inputSeq: Float32Array[]): Float32Array[] {
    const outputs: Float32Array[] = [];
    console.log({ inputSeq });
    this.inputLayer.setInputSequence([...inputSeq]);
    for (let seqIx = 0; seqIx < inputSeq.length; seqIx += 1) {
      console.log(`seqIx: ${seqIx}`);
      const output = new Float32Array(this.outputs.size);

      // Pull from the output node through the rest of the graph to compute each output
      for (let outputIx = 0; outputIx < output.length; outputIx += 1) {
        const outputNeuron = this.outputs.getNeuron(outputIx);
        if (!outputNeuron) {
          continue;
        }
        output[outputIx] = outputNeuron.getOutput();
      }

      outputs.push(output);

      this.inputLayer.advanceSequence();
      this.cells.forEach(cell => cell.advanceSequence());
      // TODO: Implement post layers
      this.outputs.advanceSequence();
    }
    return outputs;
  }

  public buildGraphviz(): string {
    // Work around bug in `graphviz-builder`: https://github.com/prantlf/graphviz-builder/issues/1
    (window as any).l = undefined;

    const g = GVB.digraph('RNN');

    const outputs = g.addCluster('cluster_outputs');
    for (let outputIx = 0; outputIx < this.outputs.size; outputIx += 1) {
      const neuron = this.outputs.getNeuron(outputIx)!;
      outputs.addNode(neuron.name);
    }

    this.cells.forEach((cell, layerIx) => {
      const layer = g.addCluster(`cluster_layer_${layerIx}`);

      const state = layer.addCluster('cluster_state');
      cell.stateNeurons.forEach(neuron => state.addNode(neuron.name));

      const recurrent = layer.addCluster('cluster_recurrent');
      cell.recurrentNeurons.forEach(neuron => recurrent.addNode(neuron.name));

      const output = layer.addCluster('cluster_output');
      cell.outputNeurons.forEach(neuron => output.addNode(neuron.name));
    });

    this.postLayers.forEach((postLayer, layerIx) => {
      const layer = g.addCluster(`cluster_post_layer_${layerIx}`);
      postLayer.neurons.forEach(neuron => layer.addNode(neuron.name));
    });

    const inputs = g.addCluster('cluster_inputs');
    for (let inputIx = 0; inputIx < this.inputLayer.size; inputIx += 1) {
      inputs.addNode(`input_${inputIx}`, {});
    }

    const processedNodes = new Set<string>();
    const addEdges = (neuron: SparseNeuron) => {
      const alreadyProcessed = processedNodes.has(neuron.name);
      if (alreadyProcessed) {
        return;
      }
      processedNodes.add(neuron.name);

      neuron.weights.forEach(({ inputNeuron, weight }) => {
        g.addEdge(inputNeuron.name, neuron.name, { label: weight.toString() });
        addEdges(inputNeuron);
      });
    };

    // Walk the graph backwards to populate edges
    for (let outputIx = 0; outputIx < this.outputs.size; outputIx += 1) {
      const neuron = this.outputs.getNeuron(outputIx)!;
      addEdges(neuron);
    }

    return g.to_dot();
  }
}
