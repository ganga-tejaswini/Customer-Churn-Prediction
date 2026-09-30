import { z } from 'zod'

// Single place where environment variables are read and validated.
const envSchema = z.object({
  VITE_APP_NAME: z.string().default('Churn Insight'),
  VITE_API_BASE_URL: z.string().url().default('http://localhost:8000/api'),
  VITE_API_TIMEOUT_MS: z.coerce.number().int().positive().default(15000),
  VITE_USE_MOCKS: z.enum(['true', 'false']).default('false'),
})

const parsed = envSchema.safeParse(import.meta.env)

if (!parsed.success) {
  console.error('Invalid environment configuration:', parsed.error.flatten().fieldErrors)
  throw new Error('Invalid environment configuration. Check your .env file.')
}

export const env = {
  appName: parsed.data.VITE_APP_NAME,
  apiBaseUrl: parsed.data.VITE_API_BASE_URL,
  apiTimeoutMs: parsed.data.VITE_API_TIMEOUT_MS,
  useMocks: parsed.data.VITE_USE_MOCKS === 'true',
  mode: import.meta.env.MODE,
} as const
