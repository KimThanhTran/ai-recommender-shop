import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// In development `/api/*` is proxied to the FastAPI service (the API has no CORS headers).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: process.env.API_TARGET || 'http://localhost:8000',
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.js',
  },
});
