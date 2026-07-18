import type { AnalysisJob, Insights, ScanHistory } from './types'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = payload.detail
    const message = typeof detail === 'string'
      ? detail
      : Array.isArray(detail)
        ? detail.map((item) => item?.msg).filter(Boolean).join('. ')
        : ''
    throw new Error(message || 'Something went wrong. Please try again.')
  }
  return payload as T
}

export type AnalysisSettings = {
  maxCrops: number
  maxSkuCrops: number
  extractSku: boolean
}

export async function startAnalysis(file: File, settings: AnalysisSettings): Promise<string> {
  const form = new FormData()
  form.append('image', file)
  form.append('max_crops', String(settings.maxCrops))
  form.append('max_sku_crops', String(settings.maxSkuCrops))
  form.append('extract_sku', String(settings.extractSku))
  const payload = await request<{ job_id: string }>('/api/analyses', {
    method: 'POST',
    body: form,
  })
  return payload.job_id
}

export function getAnalysis(jobId: string) {
  return request<AnalysisJob>(`/api/analyses/${jobId}`)
}

export function getInsights() {
  return request<Insights>('/api/insights')
}

export function getHistory() {
  return request<{ scans: ScanHistory[]; stats: Record<string, number> }>('/api/scans')
}

export function sendDetectionFeedback(payload: {
  scanId: number
  cropId: number
  feedbackType: 'category' | 'sku'
  verdict: 'correct' | 'incorrect'
  correction?: string
}) {
  return request<{ id: number; status: string; message: string }>('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      scan_id: payload.scanId,
      crop_id: payload.cropId,
      feedback_type: payload.feedbackType,
      verdict: payload.verdict,
      correction: payload.correction || '',
    }),
  })
}

export function askInventory(question: string) {
  return request<{ text: string; source: string; table?: Record<string, unknown>[] }>(
    '/api/questions',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    },
  )
}
