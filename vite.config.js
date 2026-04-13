import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
    server: {
        cors: {
            origin: '*'
        },
        headers: {
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp',
        }
    },
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          tensorflow: ['@tensorflow/tfjs'],
          onnx: ['onnxruntime-web']
        }
      }
    }
  }
})
