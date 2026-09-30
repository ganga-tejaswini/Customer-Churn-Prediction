/** Normalised error shape every service call rejects with. */
export interface ApiError {
  message: string
  status?: number
  details?: unknown
}
