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
  activation: AmeoActivationIdentifier | 'linear' | 'tanh';
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

export class SparseNeuron {
  public weights: SparseWeight[];
  public bias: number;
  public name: string;
  private activationID: AmeoActivationIdentifier | 'linear' | 'tanh';
  public activation: (x: number) => number = x => x;

  constructor(
    weights: SparseWeight[],
    bias: number,
    name: string,
    activationID: AmeoActivationIdentifier | 'linear' | 'tanh' = 'linear'
  ) {
    this.weights = weights;
    this.bias = bias;
    this.name = name;
    this.activationID = activationID;
    const activation = buildActivation(activationID);
    if (activation) {
      this.activation = activation;
    }
  }

  public getOutput(): number {
    const weightedSum = this.weights.reduce((sum, { weight, inputNeuron }) => {
      const output = inputNeuron.getOutput();
      return sum + weight * output;
    }, this.bias);
    return this.activation(weightedSum);
  }

  public advanceSequence() {
    // no-op
  }

  public serialize(): SerializedSparseNeuron {
    return {
      weights: this.weights.map(({ weight, index }) => ({ weight, index })),
      bias: this.bias,
      name: this.name,
      activation: this.activationID,
    };
  }

  public static deserialize(
    serialized: SerializedSparseNeuron,
    prevLayer: GraphRNNLayer,
    _index: number
  ) {
    const { weights, bias, name, activation } = serialized;

    const neuron = new SparseNeuron(
      weights.map(({ weight, index }) => {
        const inputNeuron = prevLayer.getNeuron(index);
        if (!inputNeuron) {
          throw new Error(`No input neuron for neuron ${name} with index ${index}`);
        }
        return { weight, index, inputNeuron };
      }),
      bias,
      name,
      activation
    );
    return neuron;
  }
}

export class InputNeuron extends SparseNeuron {
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
    if (!this.inputSequence.length) {
      throw new Error('Input sequence is empty');
    }

    const input = this.inputSequence[0];
    if (typeof input[this.index] !== 'number') {
      throw new Error(`Input sequence is missing index ${this.index}`);
    }
    return input[this.index];
  }

  public advanceSequence() {
    this.inputSequence.shift();
  }

  public serialize(): SerializedSparseNeuron {
    return {
      name: this.name,
      activation: 'linear',
      weights: [],
      bias: 0,
    };
  }

  public static deserialize(
    serialized: SerializedSparseNeuron,
    _prevLayer: GraphRNNLayer,
    index: number
  ) {
    const { name } = serialized;
    const neuron = new InputNeuron(index);
    neuron.name = name;
    return neuron;
  }
}

export class OutputNeuron extends SparseNeuron {
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

  public serialize(): SerializedSparseNeuron {
    return {
      name: this.name,
      activation: 'linear',
      weights: this.weights.map(({ weight, index }) => ({ weight, index })),
      bias: this.bias,
    };
  }

  public static deserialize(
    serialized: ReturnType<OutputNeuron['serialize']>,
    prevLayer: GraphRNNLayer
  ) {
    const { name, weights, bias } = serialized;
    const neuron = new OutputNeuron(weights[0].index, prevLayer);
    neuron.name = name;
    neuron.bias = bias;
    return neuron;
  }
}

export class StateNeuron extends SparseNeuron {
  public layerIx: number;
  public index: number;
  public initialState: number;
  private state: number;
  /**
   * When advancing the sequence, we compute next states by pulling through the graph from
   * connected neurons to this state.  However, we have to be sure to wait until all the
   * neurons have computed new states before we update the state of this neuron so that the other
   * state neurons computing new states will have the correct previous state to pull through.
   *
   * This variable holds the new state that we will update to after all the neurons have computed
   * their new states.
   */
  private pendingNewState = 0;

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

  public computeNewState() {
    const newState = SparseNeuron.prototype.getOutput.call(this);
    this.pendingNewState = newState;
  }

  public commitNewState() {
    this.state = this.pendingNewState;
  }

  public reset() {
    this.state = this.initialState;
  }

  /**
   * Adds the recurrent connection to the recurrent layer.  This is called after
   * the layer has been fully populated so we have access to all the neurons.
   */
  public connect(index: number, recurrentNeuron: SparseNeuron) {
    this.weights.push({ index, inputNeuron: recurrentNeuron, weight: 1 });
  }

  public serialize(): SerializedSparseNeuron {
    return {
      bias: this.initialState,
      name: this.name,
      activation: 'linear',
      weights: this.weights.map(({ weight, index }) => ({ weight, index })),
    };
  }

  public static deserialize(serialized: SerializedSparseNeuron, prevLayer: GraphRNNLayer) {
    const { name, weights, bias } = serialized;
    const neuron = new StateNeuron(0, 0, bias);
    neuron.name = name;
    neuron.weights = weights.map(({ weight, index }) => {
      const inputNeuron = prevLayer.getNeuron(index);
      if (!inputNeuron) {
        throw new Error(`No input neuron for state neuron ${name} with index ${index}`);
      }
      return { weight, index, inputNeuron };
    });
    return neuron;
  }
}

export abstract class GraphRNNLayer {
  abstract outputDim: number;

  abstract getNeuron(outputIx: number): SparseNeuron | undefined;

  abstract advanceSequence(): void;

  abstract serialize(): Record<string, any>;

  static deserialize: (serialized: Record<string, any>) => GraphRNNLayer;
}

export class GraphRNNInputLayer implements GraphRNNLayer {
  public inputDim: number;
  public outputDim: number;
  public neurons: InputNeuron[] = [];
  private inputSequence: Float32Array[] = [];

  constructor(inputDim: number) {
    this.inputDim = inputDim;
    this.outputDim = inputDim;
    for (let i = 0; i < inputDim; i += 1) {
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

  public serialize(): SerializedGraphRNNInputLayer {
    return { neurons: serializeSparseNeurons(this.neurons) };
  }

  public static deserialize(serialized: SerializedGraphRNNInputLayer): GraphRNNInputLayer {
    const { neurons } = serialized;
    const layer = new GraphRNNInputLayer(neurons.length);
    layer.neurons = new Array(neurons.length);
    for (let i = 0; i < neurons.length; i += 1) {
      const neuron = neurons[i];
      if (neuron) {
        layer.neurons[i] = InputNeuron.deserialize(neuron, layer, i);
      }
    }
    return layer;
  }
}

const serializeSparseNeurons = (neurons: SparseNeuron[]): (SerializedSparseNeuron | null)[] => {
  const serializedNeurons: (SerializedSparseNeuron | null)[] = [];
  for (let i = 0; i < neurons.length; i += 1) {
    serializedNeurons.push(neurons[i]?.serialize() ?? null);
  }
  return serializedNeurons;
};

interface SerializedGraphRNNInputLayer {
  neurons: (SerializedSparseNeuron | null)[];
}

/**
 * Represents the output tensor of a GraphRNN
 */
export class GraphRNNOutputs implements GraphRNNLayer {
  public outputDim: number;
  public neurons: OutputNeuron[] = [];

  constructor(outputDim: number, prevLayer: GraphRNNLayer) {
    this.outputDim = outputDim;
    for (let i = 0; i < outputDim; i += 1) {
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

  public serialize(): SerializedRNNOutputLayer {
    return { neurons: serializeSparseNeurons(this.neurons) };
  }

  public static deserialize(serialized: SerializedRNNOutputLayer, prevLayer: GraphRNNLayer) {
    const { neurons } = serialized;
    const layer = new GraphRNNOutputs(neurons.length, prevLayer);
    layer.neurons = new Array(neurons.length);
    for (let i = 0; i < neurons.length; i += 1) {
      const neuron = neurons[i];
      if (!neuron) {
        continue;
      }
      layer.neurons[i] = OutputNeuron.deserialize(neuron, prevLayer);
    }
    return layer;
  }
}

interface SerializedRNNOutputLayer {
  neurons: (SerializedSparseNeuron | null)[];
}

const buildActivation = (
  id: AmeoActivationIdentifier | 'linear' | 'tanh'
): ((x: number) => number) => {
  if (id === 'linear') {
    return x => x;
  } else if (id === 'tanh') {
    return x => Math.tanh(x);
  }

  if (typeof id === 'string') {
    throw new Error('Unimplemented');
  }
  switch (id.type) {
    case 'leakyAmeo':
      return x => nativeFusedInterpolatedAmeoImplInner(1, x, id.leakyness ?? 0);
    case 'interpolatedAmeo':
      return x => nativeFusedInterpolatedAmeoImplInner(id.factor, x, id.leakyness ?? 0);
    default:
      throw new Error('Unimplemented');
  }
};

export class GraphRNNCell implements GraphRNNLayer {
  public outputDim: number;
  public recurrentNeurons: SparseNeuron[] = [];
  public outputNeurons: SparseNeuron[] = [];
  public stateNeurons: StateNeuron[] = [];

  constructor(outputDim: number) {
    this.outputDim = outputDim;
  }

  public static fromWeights(
    layerIx: number,
    weights: RNNCellWeights,
    params: RNNGraphParams,
    prevLayer: GraphRNNLayer
  ) {
    const layer = new GraphRNNCell(weights.outputSize);
    layer.outputDim = weights.outputSize;

    if (weights.recurrentTreeWeights.shape.length !== 2) {
      throw new Error('Recurrent tree weights must be 2D');
    }

    const recurrentWeightsData = clipAndQuantizeWeights(
      weights.recurrentTreeWeights.dataSync() as Float32Array,
      params
    );
    const recurrentBiasData = weights.recurrentTreeBias
      ? clipAndQuantizeWeights(weights.recurrentTreeBias.dataSync() as Float32Array, params)
      : null;

    const outputWeightsData = clipAndQuantizeWeights(
      weights.outputTreeWeights.dataSync() as Float32Array,
      params
    );
    const outputBiasData = weights.outputTreeBias
      ? clipAndQuantizeWeights(weights.outputTreeBias.dataSync() as Float32Array, params)
      : null;

    const initialStateData = clipAndQuantizeWeights(
      weights.initialState.dataSync() as Float32Array,
      params
    );
    if (initialStateData.length !== weights.stateSize) {
      console.log({ weights, initialStateData });
      throw new Error(
        `Unexpected initial state length; expected=${weights.stateSize} actual=${initialStateData.length}`
      );
    }

    const getInputNeuron = (outputIx: number): SparseNeuron | undefined => {
      // Inputs to the output and recurrent tree are created by concatenating the inputs and the state
      if (outputIx < prevLayer.outputDim) {
        return prevLayer.getNeuron(outputIx);
      }
      const stateIx = outputIx - prevLayer.outputDim;
      if (!layer.stateNeurons[stateIx]) {
        const initialState = initialStateData[stateIx];
        if (initialState === undefined) {
          throw new Error(`Unexpected undefined initial state; stateIx=${stateIx}`);
        }
        const stateNeuron = new StateNeuron(layerIx, stateIx, initialState);
        layer.stateNeurons[stateIx] = stateNeuron;
      }
      return layer.stateNeurons[stateIx];
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
        weights.outputActivation
      );
      layer.outputNeurons[outputNeuronIx] = neuron;
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
        weights.recurrentActivation
      );
      layer.recurrentNeurons[recurrentNeuronIx] = neuron;
    }

    // Now that we have populated all neurons, we can add in the recurrent connections between the
    // recurrent tree and the state
    layer.stateNeurons.forEach((neuron, index) => {
      const recurrentNeuron = layer.recurrentNeurons[index];
      if (!recurrentNeuron) {
        return;
      }
      neuron.connect(index, recurrentNeuron);
    });

    return layer;
  }

  getNeuron(outputIx: number): SparseNeuron | undefined {
    return this.outputNeurons[outputIx];
  }

  public advanceSequence(): void {
    this.stateNeurons.forEach(neuron => {
      neuron.computeNewState();
      neuron.advanceSequence();
    });
    this.recurrentNeurons.forEach(neuron => neuron.advanceSequence());
    this.outputNeurons.forEach(neuron => neuron.advanceSequence());
  }

  /**
   * Resets all state back to initial values
   */
  public reset(): void {
    this.stateNeurons.forEach(neuron => neuron.reset());
  }

  public serialize(): SerializedGraphRNNCell {
    const outputNeurons = serializeSparseNeurons(this.outputNeurons);
    const recurrentNeurons = serializeSparseNeurons(this.recurrentNeurons);
    const stateNeurons = serializeSparseNeurons(this.stateNeurons);
    return { outputNeurons, recurrentNeurons, stateNeurons, outputDim: this.outputDim };
  }

  public static deserialize(
    layerIx: number,
    serialized: SerializedGraphRNNCell,
    prevLayer: GraphRNNLayer
  ): GraphRNNCell {
    const layer = new GraphRNNCell(serialized.outputDim);

    for (let i = 0; i < serialized.outputNeurons.length; i += 1) {
      const serializedNeuron = serialized.outputNeurons[i];
      if (!serializedNeuron) {
        continue;
      }
      // We can't deserialize directly since neurons might depend recursively on other neurons in this
      // layer that haven't been deserialized yet
      const neuron = new SparseNeuron(
        [],
        serializedNeuron.bias,
        serializedNeuron.name,
        serializedNeuron.activation
      );
      layer.outputNeurons[i] = neuron;
    }

    for (let i = 0; i < serialized.recurrentNeurons.length; i += 1) {
      const serializedNeuron = serialized.recurrentNeurons[i];
      if (!serializedNeuron) {
        continue;
      }

      const neuron = new SparseNeuron(
        [],
        serializedNeuron.bias,
        serializedNeuron.name,
        serializedNeuron.activation
      );
      layer.recurrentNeurons[i] = neuron;
    }

    for (let i = 0; i < serialized.stateNeurons.length; i += 1) {
      const serializedNeuron = serialized.stateNeurons[i];
      if (!serializedNeuron) {
        continue;
      }

      const neuron = new StateNeuron(layerIx, i, serializedNeuron.bias);
      layer.stateNeurons[i] = neuron;
    }

    const getInputNeuron = (outputIx: number): SparseNeuron | undefined => {
      // Inputs to the output and recurrent tree are created by concatenating the previous layer's outputs and the state
      if (outputIx < prevLayer.outputDim) {
        return prevLayer.getNeuron(outputIx);
      }
      const stateIx = outputIx - prevLayer.outputDim;
      if (!layer.stateNeurons[stateIx]) {
        throw new Error(
          `Unexpected undefined state neuron; should have been populated already. stateIx=${stateIx}`
        );
      }
      return layer.stateNeurons[stateIx];
    };

    const fillConnections = (
      neurons: SparseNeuron[],
      serializedNeurons: (SerializedSparseNeuron | null)[]
    ) => {
      neurons.forEach((neuron, index) => {
        const serializedNeuron = serializedNeurons[index];
        if (!serializedNeuron) {
          throw new Error(`Unexpected undefined serialized neuron; index=${index}`);
        }
        serializedNeuron.weights.forEach(weight => {
          const inputNeuron = getInputNeuron(weight.index);
          if (!inputNeuron) {
            throw new Error(`Unexpected undefined input neuron; index=${weight.index}`);
          }
          neuron.weights.push({ weight: weight.weight, index: weight.index, inputNeuron });
        });
      });
    };

    // Now that we have populated all neurons, fill in their connections
    fillConnections(layer.outputNeurons, serialized.outputNeurons);
    fillConnections(layer.recurrentNeurons, serialized.recurrentNeurons);

    // State neurons are special in that they always have only one connection to the corresponding neuron
    // in the recurrent tree, if one exists.
    layer.stateNeurons.forEach((neuron, index) => {
      const recurrentNeuron = layer.recurrentNeurons[index];
      if (!recurrentNeuron) {
        return;
      }

      neuron.weights.push({
        weight: 1,
        index,
        inputNeuron: recurrentNeuron,
      });
    });

    return layer;
  }
}

interface SerializedSparseWeight {
  weight: number;
  index: number;
}

interface SerializedSparseNeuron {
  weights: SerializedSparseWeight[];
  bias: number;
  name: string;
  activation: AmeoActivationIdentifier | 'linear' | 'tanh';
}

interface SerializedGraphRNNCell {
  outputNeurons: (SerializedSparseNeuron | null)[];
  recurrentNeurons: (SerializedSparseNeuron | null)[];
  stateNeurons: (SerializedSparseNeuron | null)[];
  outputDim: number;
}

export class GraphRNNPostLayer implements GraphRNNLayer {
  public outputDim: number;
  public neurons: SparseNeuron[] = [];

  constructor(outputDim: number, neurons: SparseNeuron[]) {
    this.outputDim = outputDim;
    this.neurons = neurons;
  }

  public static fromWeights(
    weights: PostLayerWeights,
    params: RNNGraphParams,
    prevLayer: GraphRNNLayer,
    outputDim: number
  ): GraphRNNPostLayer {
    const layer = new GraphRNNPostLayer(outputDim, []);

    const weightsData = clipAndQuantizeWeights(weights.weights.dataSync() as Float32Array, params);
    const biasData = weights.bias
      ? clipAndQuantizeWeights(weights.bias?.dataSync() as Float32Array, params)
      : undefined;

    const getInputNeuron = (outputIx: number): SparseNeuron | undefined =>
      prevLayer.getNeuron(outputIx);

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
        weights.activation
      );
      layer.neurons[outputNeuronIx] = neuron;
    }

    return layer;
  }

  advanceSequence(): void {
    // no-op
  }

  getNeuron(outputIx: number): SparseNeuron | undefined {
    return this.neurons[outputIx];
  }

  public serialize(): SerializedGraphRNNPostLayer {
    return { neurons: serializeSparseNeurons(this.neurons), outputDim: this.outputDim };
  }

  public static deserialize(
    serialized: SerializedGraphRNNPostLayer,
    prevLayer: GraphRNNLayer
  ): GraphRNNPostLayer {
    const { neurons: serializedNeurons, outputDim } = serialized;
    const neurons: SparseNeuron[] = [];
    for (let i = 0; i < outputDim; i += 1) {
      const neuron = serializedNeurons[i];
      if (!neuron) {
        continue;
      }
      neurons[i] = SparseNeuron.deserialize(neuron, prevLayer, i);
    }
    return new GraphRNNPostLayer(outputDim, neurons);
  }
}

interface SerializedGraphRNNPostLayer {
  neurons: (SerializedSparseNeuron | null)[];
  outputDim: number;
}

/**
 * Sets weights to 0 if their magnitude is less than or equal to `clipThreshold`.
 */
const clipAndQuantizeWeights = (weights: Float32Array, params: RNNGraphParams): Float32Array => {
  const { clipThreshold, quantizationInterval } = params;
  const newWeights = new Float32Array(weights.length);
  for (let i = 0; i < weights.length; i += 1) {
    if (Math.abs(weights[i]) <= clipThreshold) {
      newWeights[i] = 0;
    } else {
      newWeights[i] = weights[i];
    }
  }

  if (!quantizationInterval) {
    return newWeights;
  }

  // Quantize weights to nearest multiple of `quantizationInterval`
  for (let i = 0; i < newWeights.length; i += 1) {
    newWeights[i] = Math.round(newWeights[i] / quantizationInterval) * quantizationInterval;
  }

  return newWeights;
};

export class RNNGraph {
  private inputLayer: GraphRNNInputLayer;
  private outputs: GraphRNNOutputs;
  private cells: GraphRNNCell[] = [];
  private postLayers: GraphRNNPostLayer[] = [];
  public allConnectedNeuronsByID: Map<string, SparseNeuron>;

  constructor(
    inputLayer: GraphRNNInputLayer,
    outputs: GraphRNNOutputs,
    cells: GraphRNNCell[],
    postLayers: GraphRNNPostLayer[]
  ) {
    this.inputLayer = inputLayer;
    this.outputs = outputs;
    this.cells = cells;
    this.postLayers = postLayers;
    this.allConnectedNeuronsByID = new Map();

    this.pruneUnconnectedNeurons();
  }

  private pruneUnconnectedNeurons() {
    // Walk the full connected graph and remove any un-connected neurons
    const allConnectedNeuronsByID: Map<string, SparseNeuron> = new Map();

    const addNeuron = (neuron: SparseNeuron): void => {
      if (allConnectedNeuronsByID.has(neuron.name)) {
        return;
      }

      allConnectedNeuronsByID.set(neuron.name, neuron);
      for (const weight of neuron.weights) {
        addNeuron(weight.inputNeuron);
      }
    };

    for (const output of this.outputs.neurons) {
      if (!output?.weights.length) {
        continue;
      }

      addNeuron(output);
    }

    function filterNeurons<T extends { name: string }>(neurons: T[]): T[] {
      for (let i = 0; i < neurons.length; i += 1) {
        const neuron = neurons[i];
        if (!neuron) {
          continue;
        }
        if (!allConnectedNeuronsByID.has(neuron.name)) {
          delete neurons[i];
        }
      }
      return neurons;
    }

    this.outputs.neurons = filterNeurons(this.outputs.neurons);

    for (const cell of this.cells) {
      cell.recurrentNeurons = filterNeurons(cell.recurrentNeurons);
      cell.stateNeurons = filterNeurons(cell.stateNeurons);
      cell.outputNeurons = filterNeurons(cell.outputNeurons);
    }

    for (const postLayer of this.postLayers) {
      postLayer.neurons = filterNeurons(postLayer.neurons);
    }

    this.inputLayer.neurons = filterNeurons(this.inputLayer.neurons);

    this.allConnectedNeuronsByID = allConnectedNeuronsByID;
  }

  public static fromWeights(
    inputDim: number,
    outputDim: number,
    rnnCells: RNNCellWeights[],
    postLayerWeights: PostLayerWeights[],
    rawParams?: Partial<RNNGraphParams>
  ) {
    const params = rawParams
      ? mergeNonNullable(DefaultRNNGraphParams, rawParams)
      : { ...DefaultRNNGraphParams };

    const inputLayer = new GraphRNNInputLayer(inputDim);
    const cells: GraphRNNCell[] = [];
    const postLayers: GraphRNNPostLayer[] = [];

    let prevLayer: GraphRNNLayer = inputLayer;
    for (let layerIx = 0; layerIx < rnnCells.length; layerIx += 1) {
      const cell = rnnCells[layerIx];
      const cellLayer = GraphRNNCell.fromWeights(layerIx, cell, params, prevLayer);
      cells.push(cellLayer);
      prevLayer = cellLayer;
    }

    for (let postLayerIx = 0; postLayerIx < postLayerWeights.length; postLayerIx += 1) {
      const weights = postLayerWeights[postLayerIx];
      const postLayer = GraphRNNPostLayer.fromWeights(weights, params, prevLayer, outputDim);
      postLayers.push(postLayer);
      prevLayer = postLayer;
    }

    const outputs = new GraphRNNOutputs(outputDim, prevLayer);

    return new RNNGraph(inputLayer, outputs, cells, postLayers);
  }

  public get inputDim(): number {
    return this.inputLayer.inputDim;
  }

  public setInputSequence(inputSeq: Float32Array[]): void {
    this.inputLayer.setInputSequence([...inputSeq]);
  }

  public evaluate(inputSeq: Float32Array[]): Float32Array[] {
    const outputs: Float32Array[] = [];
    this.inputLayer.setInputSequence([...inputSeq]);
    this.cells.forEach(cell => cell.reset());
    for (let seqIx = 0; seqIx < inputSeq.length; seqIx += 1) {
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

      // Advance sequences from bottom to top so that state neurons can pull their new inputs from current graph outputs
      this.outputs.advanceSequence();
      this.postLayers.forEach(layer => layer.advanceSequence());
      for (let cellIx = this.cells.length - 1; cellIx >= 0; cellIx -= 1) {
        this.cells[cellIx].advanceSequence();
      }
      this.inputLayer.advanceSequence();

      this.cells.forEach(cell => cell.stateNeurons.forEach(n => n.commitNewState()));
    }
    return outputs;
  }

  public buildGraphviz(params?: BuildGraphvizParams): string {
    // Work around bug in `graphviz-builder`: https://github.com/prantlf/graphviz-builder/issues/1
    (window as any).l = undefined;

    const g = GVB.digraph('RNN');
    const clusterPrefix = params?.cluster === false ? '' : 'cluster_';

    const outputs = g.addCluster('cluster_outputs');
    outputs.set('rank', 'sink');
    for (let outputIx = 0; outputIx < this.outputs.size; outputIx += 1) {
      const neuron = this.outputs.getNeuron(outputIx)!;
      outputs.addNode(neuron.name);
    }

    this.cells.forEach((cell, layerIx) => {
      const layer = g.addCluster(`${clusterPrefix}layer_${layerIx}`);

      const state = layer.addCluster(`${clusterPrefix}state`);
      cell.stateNeurons.forEach(neuron => state.addNode(neuron.name));

      const recurrent = layer.addCluster(`${clusterPrefix}recurrent`);
      cell.recurrentNeurons.forEach(neuron => recurrent.addNode(neuron.name));

      const output = layer.addCluster(`${clusterPrefix}output`);
      cell.outputNeurons.forEach(neuron => output.addNode(neuron.name));
    });

    this.postLayers.forEach((postLayer, layerIx) => {
      const layer = g.addCluster(`${clusterPrefix}post_layer_${layerIx}`);
      postLayer.neurons.forEach(neuron => layer.addNode(neuron.name));
    });

    const inputs = g.addCluster('cluster_inputs');
    inputs.set('rank', 'source');
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
        g.addEdge(inputNeuron.name, neuron.name, {
          label: weight.toFixed(Math.round(weight) === weight ? 0 : 3),
        });
        addEdges(inputNeuron);
      });
    };

    // Walk the graph backwards to populate edges
    for (let outputIx = 0; outputIx < this.outputs.size; outputIx += 1) {
      const neuron = this.outputs.getNeuron(outputIx)!;
      addEdges(neuron);
    }

    if (params?.arrowhead === false) {
      g.setEdgeAttribut('arrowhead', 'none');
    }
    g.set('ratio', params?.aspectRatio ?? 0.75);
    g.set('rankdir', 'TB');
    g.set('center', true);
    g.set('splines', 'spline');
    g.set('overlap', false);
    g.set('nodesep', 0.32);

    return g.to_dot();
  }

  private validateOneSeq({
    inputs,
    outputs: expectedOuts,
  }: {
    inputs: number[][];
    outputs: number[][];
  }): { isValid: true } | { isValid: false; expected: number[]; actual: number[] } {
    const f32Inputs = inputs.map(input => Float32Array.from(input));
    const actualOuts = this.evaluate(f32Inputs);

    for (let seqIx = 0; seqIx < expectedOuts.length; seqIx += 1) {
      for (let i = 0; i < expectedOuts[seqIx].length; i += 1) {
        if (Math.round(actualOuts[seqIx][i]) !== Math.round(expectedOuts[seqIx][i])) {
          return {
            isValid: false,
            expected: expectedOuts[seqIx],
            actual: Array.from(actualOuts[seqIx]),
          };
        }
      }
    }

    return { isValid: true };
  }

  public validate(
    oneSeqExamples: () => { inputs: number[][]; outputs: number[][] },
    iters = 10_000
  ): boolean {
    for (let i = 0; i < iters; i += 1) {
      const res = this.validateOneSeq(oneSeqExamples());
      if (!res.isValid) {
        console.log(`Validation failed on iteration ${i}`);
        console.log(`Expected: ${res.expected}`);
        console.log(`Actual: ${res.actual}`);
        return false;
      }
    }

    return true;
  }

  public serialize(): SerializedRNNGraph {
    return {
      inputLayer: this.inputLayer.serialize(),
      cells: this.cells.map(cell => cell.serialize()),
      postLayers: this.postLayers.map(layer => layer.serialize()),
      outputs: this.outputs.serialize(),
    };
  }

  public static deserialize(serialized: SerializedRNNGraph): RNNGraph {
    const inputLayer = GraphRNNInputLayer.deserialize(serialized.inputLayer);
    let prevLayer: GraphRNNLayer = inputLayer;

    const cells = serialized.cells.map((serializedCell, layerIx) => {
      const cell = GraphRNNCell.deserialize(layerIx, serializedCell, prevLayer);
      prevLayer = cell;
      return cell;
    });
    const postLayers = serialized.postLayers.map(layer => {
      const postLayer = GraphRNNPostLayer.deserialize(layer, prevLayer);
      prevLayer = postLayer;
      return postLayer;
    });
    const outputs = GraphRNNOutputs.deserialize(serialized.outputs, prevLayer);

    return new RNNGraph(inputLayer, outputs, cells, postLayers);
  }
}

interface BuildGraphvizParams {
  arrowhead?: boolean;
  cluster?: boolean;
  aspectRatio?: number;
}

interface SerializedRNNGraph {
  inputLayer: SerializedGraphRNNInputLayer;
  cells: SerializedGraphRNNCell[];
  postLayers: SerializedGraphRNNPostLayer[];
  outputs: SerializedRNNOutputLayer;
}