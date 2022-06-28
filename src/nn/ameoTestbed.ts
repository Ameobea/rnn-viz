import * as tf from '@tensorflow/tfjs';
import type { Activation } from '@tensorflow/tfjs-layers/dist/activations';

import { Ameo, SoftAmeo, SoftLeakyAmeo } from './ameoActivation';

type Initialization = { type: 'zeroes' } | { type: 'random'; scale: number };

interface AmeoTestbedParams {
  inputSize: number;
  targetFunction(input: boolean[]): boolean;
  learningRate: number;
  initialization: Initialization;
  iterations: number;
  batchSize: number;
  variant: 'ameo' | 'softAmeo' | 'softLeakyAmeo';
}

interface AmeoTestbedRunResult {
  successCount: number;
  failureCount: number;
}

const randomBool = () => Math.random() > 0.5;

export class AmeoTestbed {
  private params: AmeoTestbedParams;

  constructor(params: AmeoTestbedParams) {
    (window as any).tf = tf;
    this.params = params;
  }

  private initializeWeights(): number[] {
    switch (this.params.initialization.type) {
      case 'zeroes': {
        return new Array(this.params.inputSize).fill(0);
      }
      case 'random': {
        const scale = this.params.initialization.scale;
        return new Array(this.params.inputSize).fill(0).map(() => (Math.random() * 2 - 1) * scale);
      }
      default: {
        throw new Error(`Unknown initialization type: ${(this.params.initialization as any).type}`);
      }
    }
  }

  private initializeBias(): number {
    // TODO
    return 0;
  }

  private buildActivation(): Activation {
    switch (this.params.variant) {
      case 'ameo': {
        return new Ameo();
      }
      case 'softAmeo': {
        return new SoftAmeo();
      }
      case 'softLeakyAmeo': {
        return new SoftLeakyAmeo();
      }
      default: {
        throw new Error(`Unknown variant: ${(this.params as any).variant}`);
      }
    }
  }

  public run(attempts: number): AmeoTestbedRunResult {
    let successes = 0;
    let failures = 0;

    // const optimizer = tf.train.sgd(this.params.learningRate);
    const optimizer = tf.train.adam(this.params.learningRate);
    const activation = this.buildActivation();

    for (let i = 0; i < attempts; i++) {
      // const weights = tf.tensor1d(this.initializeWeights()).variable(true);
      const weights = tf.tensor1d([0.2, -0.26, -0.27]).variable(true);
      // const bias = tf.scalar(this.initializeWeights()[0]).variable(true);
      const bias = tf.scalar(0.39).variable(true);
      console.log('Initial weights:', Array.from(weights.dataSync()));
      console.log('Initial bias:', bias.dataSync()[0]);

      const f = (x: tf.Tensor1D) => activation.apply(x.mul(weights).sum().add(bias));
      const loss = (x: tf.Tensor, y: tf.Tensor): tf.Scalar => x.sub(y).square().mean();

      for (let j = 0; j < this.params.iterations; j++) {
        const cost = optimizer.minimize(
          () => {
            const oneExampleLoss = () => {
              const input = new Array(this.params.inputSize).fill(0).map(() => randomBool());
              const expected = tf.scalar(this.params.targetFunction(input) ? 1 : -1);
              const actual = f(tf.tensor1d(input.map(b => (b ? 1 : -1))));
              // console.log('Input:', input);
              // console.log('Expected:', Array.from(expected.dataSync()));
              // console.log('Actual:', Array.from(actual.dataSync()));
              return loss(actual, expected);
            };

            let totalLoss: tf.Scalar | null = null;
            for (let exampleIx = 0; exampleIx < this.params.batchSize; exampleIx++) {
              const exampleLoss = oneExampleLoss();
              totalLoss = totalLoss ? (totalLoss.add(exampleLoss) as tf.Scalar) : exampleLoss;
            }
            return totalLoss!.div(this.params.batchSize);
          },
          true,
          [weights, bias]
        );
        console.log('Cost:', +cost!.dataSync()[0].toFixed(3));
      }

      console.log('Values: ', {
        weights: Array.from(weights.dataSync()),
        bias: Array.from(bias.dataSync()),
      });
    }

    return { successCount: successes, failureCount: failures };
  }
}
