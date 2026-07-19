import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built app is served by the Flask backend from ./dist at http://127.0.0.1:5000.
// During `npm run dev`, /api and /media are proxied to that backend.
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:5000',
      '/media': 'http://127.0.0.1:5000',
    },
  },
})
