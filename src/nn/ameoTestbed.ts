import * as tf from '@tensorflow/tfjs';
import type { Activation } from '@tensorflow/tfjs-layers/dist/activations';

import { Ameo, SoftAmeo, SoftLeakyAmeo } from './ameoActivation';
import { GCUActivation } from './gcuActivation';

type Initialization = { type: 'zeroes' } | { type: 'random'; scale: number };

interface AmeoTestbedParams {
  inputSize: number;
  targetFunction(input: boolean[]): boolean;
  learningRate: number;
  initialization: Initialization;
  iterations: number;
  batchSize: number;
  variant: 'ameo' | 'softAmeo' | 'softLeakyAmeo' | 'gcu';
  perfectCostThreshold: number;
  optimizer: 'sgd' | 'adam';
}

interface AmeoTestbedRunResult {
  perfectFitCount: number;
  linearlySeparableCount: number;
  failureCount: number;
}

enum ValidationRes {
  Perfect = 0,
  LinearlySeparable = 1,
  Failure = 2,
}

const randomBool = () => Math.random() > 0.5;

const formatValidationRes = (validationRes: ValidationRes): string => {
  switch (validationRes) {
    case ValidationRes.Failure:
      return 'Failure';
    case ValidationRes.LinearlySeparable:
      return 'Linearly Separable';
    case ValidationRes.Perfect:
      return 'Perfect';
    default:
      throw new Error('Unreachable');
  }
};

export const formatTestbedRunResult = (runRes: AmeoTestbedRunResult): string => {
  const totalRuns = runRes.failureCount + runRes.linearlySeparableCount + runRes.perfectFitCount;
  return `Perfect: ${runRes.perfectFitCount} ${((runRes.perfectFitCount / totalRuns) * 100).toFixed(
    1
  )}%\nLinearly Seperable: ${runRes.linearlySeparableCount} ${(
    (runRes.linearlySeparableCount / totalRuns) *
    100
  ).toFixed(1)}%\nFailures: ${runRes.failureCount} ${(
    (runRes.failureCount / totalRuns) *
    100
  ).toFixed(1)}%`;
};

tf.setBackend('cpu');

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
      case 'gcu': {
        return new GCUActivation();
      }
      default: {
        throw new Error(`Unknown variant: ${(this.params as any).variant}`);
      }
    }
  }

  private validate = (weights: tf.Variable, bias: tf.Variable): ValidationRes => {
    const activation = this.buildActivation();
    const f = (x: tf.Tensor1D) => activation.apply(x.mul(weights).sum().add(bias));

    // FROM: https://stackoverflow.com/a/55001358/3833068
    const cartesianProduct = <T>(arr: T[][]): T[][] =>
      arr.reduce((a, b) => a.map(x => b.map(y => x.concat(y))).reduce((c, d) => c.concat(d), []), [
        [],
      ] as T[][]);

    const inputsToSearch = cartesianProduct(
      new Array(this.params.inputSize).fill(null).map(() => [false, true])
    );
    console.log(inputsToSearch);

    let validationRes = ValidationRes.Perfect;
    const costs: number[] = [];
    for (const rawInput of inputsToSearch) {
      const input = tf.tensor1d(rawInput.map(b => (b ? 1 : -1)));
      const expected = this.params.targetFunction(rawInput) ? 1 : -1;
      const actual = f(input).dataSync()[0];
      const diff = Math.abs(actual - expected);
      const cost = diff * diff;
      costs.push(cost);
      if (cost < this.params.perfectCostThreshold) {
        validationRes = Math.max(validationRes, ValidationRes.Perfect);
      } else if ((expected === 1 && actual > 0) || (expected === -1 && actual < 0)) {
        validationRes = Math.max(validationRes, ValidationRes.LinearlySeparable);
      } else {
        validationRes = ValidationRes.Failure;
        break;
      }
    }
    if (validationRes !== ValidationRes.Perfect) {
      const totalCost = costs.reduce((acc, val) => acc + val, 0);
      const averageCost = totalCost / inputsToSearch.length;
      const maxCost = Math.max(...costs);
      console.log('Average cost: ', averageCost);
      console.log('Max cost: ', maxCost);
    }
    return validationRes;
  };

  public run(attempts: number): AmeoTestbedRunResult {
    let perfectFitCount = 0;
    let linearlySeparableCount = 0;
    let failureCount = 0;

    // const optimizer = tf.train.sgd(this.params.learningRate);
    let optimizer = (this.params.optimizer === 'adam' ? tf.train.adam : tf.train.sgd)(
      this.params.learningRate
    );
    const activation = this.buildActivation();

    const lastCosts = [];
    for (let i = 0; i < attempts; i++) {
      const weights = tf.tensor1d(this.initializeWeights()).variable(true);
      // const weights = tf.tensor1d([0.2, -0.26, -0.27]).variable(true);
      const bias = tf.scalar(this.initializeWeights()[0]).variable(true);
      // const bias = tf.scalar(0.39).variable(true);
      // console.log('Initial weights:', Array.from(weights.dataSync()));
      // console.log('Initial bias:', bias.dataSync()[0]);

      const f = (x: tf.Tensor1D) => activation.apply(x.mul(weights).sum().add(bias));
      const loss = (x: tf.Tensor, y: tf.Tensor): tf.Scalar => x.sub(y).square().mean();

      for (let j = 0; j < this.params.iterations; j++) {
        if (j === 500) {
          optimizer = (this.params.optimizer === 'adam' ? tf.train.adam : tf.train.sgd)(
            this.params.learningRate / 100
          );
        }
        const costVar = optimizer.minimize(
          () => {
            const oneExampleLoss = () => {
              const input = new Array(this.params.inputSize).fill(0).map(randomBool);
              const expected = tf.scalar(this.params.targetFunction(input) ? 1 : -1);
              const actual = f(tf.tensor1d(input.map(b => (b ? 1 : -1))));
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
        const cost = costVar?.dataSync()[0];
        // console.log('Cost:', cost!.toFixed(3));
        lastCosts.push(cost);
        if (lastCosts.length > 100) {
          lastCosts.shift();
          if (
            lastCosts.every(
              cost => typeof cost === 'number' && cost < this.params.perfectCostThreshold
            )
          ) {
            console.log('Happy early exit');
            break;
          }
        }
      }

      // console.log('Values: ', {
      //   weights: Array.from(weights.dataSync()),
      //   bias: Array.from(bias.dataSync()),
      // });
      const validationRes = this.validate(weights, bias);
      console.log(`[${i}]: ${formatValidationRes(validationRes)}`);
      switch (validationRes) {
        case ValidationRes.Failure:
          failureCount += 1;
          break;
        case ValidationRes.LinearlySeparable:
          linearlySeparableCount += 1;
          break;
        case ValidationRes.Perfect:
          perfectFitCount += 1;
          break;
        default:
          throw new Error('unreachable');
      }
    }

    return { perfectFitCount, linearlySeparableCount, failureCount };
  }
}