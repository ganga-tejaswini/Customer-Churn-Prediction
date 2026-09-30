import axios, { AxiosError } from 'axios'
import { env } from '@/config/env'
import type { ApiError } from '@/types'

/** Shared Axios instance. UI components never import this directly; they use feature services. */
export const apiClient = axios.create({
  baseURL: env.apiBaseUrl,
  timeout: env.apiTimeoutMs,
  headers: { 'Content-Type': 'application/json' },
})

function toApiError(error: unknown): ApiError {
  if (error instanceof AxiosError) {
    if (error.response) {
      const data = error.response.data as { message?: string; detail?: string } | undefined
      return {
        status: error.response.status,
        message: data?.message ?? data?.detail ?? `Request failed with status ${error.response.status}.`,
        details: error.response.data,
      }
    }
    if (error.code === 'ECONNABORTED') return { message: 'The request timed out. Please try again.' }
    return { message: 'Cannot reach the server. Check your connection or the API base URL.' }
  }
  return { message: error instanceof Error ? error.message : 'An unexpected error occurred.' }
}

apiClient.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(toApiError(error)),
)
