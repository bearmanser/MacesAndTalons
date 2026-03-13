import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react-swc'

const normalizeBasePath = (basePath: string) => {
  const trimmed = basePath.trim()

  if (!trimmed || trimmed === '/') {
    return '/'
  }

  return `/${trimmed.replace(/^\/+|\/+$/g, '')}/`
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const defaultBasePath = mode === 'production' ? '/maces-and-talons/' : '/'

  return {
    base: normalizeBasePath(env.VITE_APP_BASE_PATH ?? defaultBasePath),
    plugins: [react()],
  }
})
