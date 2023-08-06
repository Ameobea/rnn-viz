import * as fs from 'fs';
import rawLosses from './rawLosses.json';

const sequence1: number[] = [];
const sequence2: number[] = [];

for (const pair of rawLosses) {
  sequence1.push(pair[0]);
  sequence2.push(pair[1]);
}

const buffer1 = Float32Array.from(sequence1);
const buffer2 = Float32Array.from(sequence2);

const arrayBuffer1 = buffer1.buffer;
const arrayBuffer2 = buffer2.buffer;

fs.writeFileSync('sequence1.bin', Buffer.from(arrayBuffer1));
fs.writeFileSync('sequence2.bin', Buffer.from(arrayBuffer2));
