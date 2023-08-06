import rawLosses from './rawLosses.json';

import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
  const baseLosses = rawLosses.map(([l]) => +l.toFixed(6));
  const regLosses = rawLosses.map(([, l]) => +l.toFixed(6));

  // Downsample to 1000 points (from 10,000) by
  const downsampledBaseLosses = [];
  const downsampledRegLosses = [];

  for (let i = 0; i < 1000; i++) {
    const start = Math.floor((i / 1000) * 10000);
    const end = Math.floor(((i + 1) / 1000) * 10000);
    const baseLossesSlice = baseLosses.slice(start, end);
    const regLossesSlice = regLosses.slice(start, end);
    const baseLossesSum = baseLossesSlice.reduce((a, b) => a + b, 0);
    const regLossesSum = regLossesSlice.reduce((a, b) => a + b, 0);
    downsampledBaseLosses.push(baseLossesSum / baseLossesSlice.length);
    downsampledRegLosses.push(regLossesSum / regLossesSlice.length);
  }

  return { baseLosses: downsampledBaseLosses, regLosses: downsampledRegLosses };
};
