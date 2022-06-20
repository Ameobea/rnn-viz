import * as tf from '@tensorflow/tfjs';

import { Ameo } from './ameoActivation';

type Initialization = { type: 'zeroes' } | { type: 'random'; scale: number };

interface AmeoTestbedParams {
  inputSize: number;
  targetFunction(input: number[]): number;
  learningRate: number;
  initialization: Initialization;
  iterations: number;
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
        return new Array(this.params.inputSize).fill(0).map(() => Math.random() * scale);
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

  public run(attempts: number): AmeoTestbedRunResult {
    let successes = 0;
    let failures = 0;

    // const optimizer = tf.train.sgd(this.params.learningRate);
    const optimizer = tf.train.adam(this.params.learningRate);
    const activation = new Ameo();

    for (let i = 0; i < attempts; i++) {
      // const weights = tf.tensor1d(this.initializeWeights()).variable(true);
      const weights = tf.tensor1d([0.4, -0.6, -0.33]).variable(true);
      // const bias = tf.scalar(this.initializeBias()).variable(true);
      const bias = tf.scalar(-0.4).variable(true);
      console.log('Initial weights:', Array.from(weights.dataSync()));
      console.log('Initial bias:', bias.dataSync()[0]);

      const f = (x: tf.Tensor1D) => activation.apply(x.mul(weights).sum().add(bias));
      const loss = (x: tf.Tensor, y: tf.Tensor): tf.Scalar => x.sub(y).square().mean();

      for (let j = 0; j < this.params.iterations; j++) {
        const input = new Array(this.params.inputSize).fill(0).map(() => (randomBool() ? 1 : 0));

        optimizer.minimize(() => {
          const expected = tf.scalar(this.params.targetFunction(input));
          const actual = f(tf.tensor1d(input));
          // console.log('Input:', input);
          // console.log('Expected:', Array.from(expected.dataSync()));
          // console.log('Actual:', Array.from(actual.dataSync()));
          console.log('Cost:', loss(actual, expected).dataSync()[0]);
          return loss(actual, expected);
        });
      }

      console.log('Values: ', {
        weights: Array.from(weights.dataSync()),
        bias: Array.from(bias.dataSync()),
      });
    }

    return { successCount: successes, failureCount: failures };
  }
}
