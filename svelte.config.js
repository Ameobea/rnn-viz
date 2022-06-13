import { resolve } from 'path';
import adapter from '@sveltejs/adapter-static';
import preprocess from 'svelte-preprocess';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // Consult https://github.com/sveltejs/svelte-preprocess
  // for more information about preprocessors
  preprocess: preprocess(),

  kit: {
    inlineStyleThreshold: 2048,
    adapter: adapter({ precompress: true }),
    prerender: {
      concurrency: 6,
      default: true,
    },
    vite: {
      build: {
        sourcemap: true,
      },
      resolve: {
        alias: {
          src: resolve('./src'),
        },
      },
    },
    floc: true,
  },
  experimental: {
    inspector: {
      holdMode: true,
    },
    prebundleSvelteLibraries: true,
  },
};

export default config;
