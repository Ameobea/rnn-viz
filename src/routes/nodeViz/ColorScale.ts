// ISC License
//
// Copyright (c) 2021 Observable Inc.
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
// REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
// INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
// LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
// OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
// PERFORMANCE OF THIS SOFTWARE.

import * as d3 from './d3';

interface LegendOptions {
  width: number;
  height: number;
}

type ColorScaler = (value: number) => number;

export const getColor: ColorScaler = (val: number): number => {
  const clamped = -Math.max(-1, Math.min(1, val));
  const scaled = (clamped + 1) / 2;
  const hexString = d3.interpolateRdBu(scaled);
  // Convert from `rgb(r, g, b)` to `0xrrggbb`
  const [r, g, b] = hexString
    .substring(4, hexString.length - 1)
    .split(/, ?/)
    .map(x => parseInt(x, 10));
  return (r << 16) + (g << 8) + b;
};

// Heavily modified version of initial code from https://observablehq.com/@d3/color-legend
//
// Copyright 2021, Observable Inc.
// Released under the ISC license.
export function ColorScaleLegend(colorScaler: ColorScaler, options: LegendOptions): SVGSVGElement {
  const { width, height } = options;

  const ramp = (colorScaler: ColorScaler, n = 256): HTMLCanvasElement => {
    const canvas = document.createElement('canvas');
    canvas.width = n;
    canvas.height = 1;
    const context = canvas.getContext('2d')!;
    for (let i = 0; i < n; ++i) {
      context.fillStyle = `#${colorScaler((i / (n - 1)) * 2 - 1)
        .toString(16)
        .padStart(6, '0')}`;
      context.fillRect(i, 0, 1, 1);
    }
    return canvas;
  };

  const svg = d3
    .create('svg')
    .attr('width', width)
    .attr('height', height)
    .attr('viewBox', [0, 0, width, height])
    .style('overflow', 'visible')
    .style('display', 'block');

  const x = d3.scaleLinear().domain([-1, 1]).range([0, width]);

  svg
    .append('image')
    .attr('x', 0)
    .attr('y', 0)
    .attr('width', width)
    .attr('height', height)
    .attr('preserveAspectRatio', 'none')
    .attr('xlink:href', ramp(colorScaler).toDataURL());

  const axis = d3.axisBottom(x).ticks(3).tickSize(6);

  svg
    .append('g')
    .attr('transform', `translate(0,${height})`)
    .call(axis)
    .call(g => g.select('.domain').remove());

  return svg.node()!;
}
