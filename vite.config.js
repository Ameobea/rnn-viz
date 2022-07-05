import { resolve } from 'path';
import { sveltekit } from '@sveltejs/kit/vite';

const config = {
  plugins: [sveltekit()],
  build: {
    sourcemap: true,
  },
  resolve: {
    alias: {
      src: resolve('./src'),
    },
  },
  optimizeDeps: {
    include: ['echarts'],
  },
  ssr: {
    noExternal: ['echarts'].filter(Boolean),
  },
};

export default config;
