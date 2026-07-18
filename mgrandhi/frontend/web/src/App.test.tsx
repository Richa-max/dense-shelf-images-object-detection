import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import * as api from './api'

vi.mock('./api')

const completeJob = {
  job_id: 'job-1',
  status: 'complete' as const,
  stage: 'complete',
  progress: 100,
  message: 'Shelf report ready',
  result: {
    scan_id: 7,
    image_name: 'shelf.jpg',
    annotated_image: 'data:image/jpeg;base64,test',
    summary: {
      num_items: 2,
      distinct_categories: 1,
      empty_pct: 0.3,
      empty_label: 'Moderate',
      shelf_type: 'Category-specific',
      review_count: 1,
    },
    detections: [
      {
        crop_id: 1,
        category: 'Soft drinks',
        subcategory: 'Cola',
        score: 0.12,
        area: 200,
        box: [1, 2, 3, 4],
        brand: 'Acme',
        product_name: 'Cola',
        sku_text: 'ACME COLA',
        sku_needs_review: 0,
      },
      {
        crop_id: 2,
        category: 'unknown',
        subcategory: 'unknown',
        score: 0,
        area: 100,
        box: [5, 6, 7, 8],
        sku_needs_review: 1,
      },
    ],
    timings: { yolo_s: 0.4, classify_s: 1.2 },
  },
}

describe('ShelfSight operator experience', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    window.history.pushState({}, '', '/')
  })

  it('shows the guided shelf upload as the primary action', () => {
    render(<App />)

    expect(screen.getByRole('heading', { name: /turn a shelf photo/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /choose photo/i })).toBeInTheDocument()
    expect(screen.getByText(/keep the full shelf in frame/i)).toBeInTheDocument()
  })

  it('rejects a non-image file with plain-language guidance', () => {
    const { container } = render(<App />)
    const input = container.querySelector('input[type="file"]') as HTMLInputElement

    fireEvent.change(input, { target: { files: [new File(['text'], 'notes.txt', { type: 'text/plain' })] } })

    expect(screen.getByRole('alert')).toHaveTextContent(/choose a jpg, png, bmp, or webp/i)
  })

  it('renders actionable results after a completed analysis', async () => {
    vi.mocked(api.startAnalysis).mockResolvedValue('job-1')
    vi.mocked(api.getAnalysis).mockResolvedValue(completeJob)
    const { container } = render(<App />)
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(input, { target: { files: [new File(['image'], 'shelf.jpg', { type: 'image/jpeg' })] } })

    fireEvent.click(container.querySelector('.analyze-button') as HTMLButtonElement)

    await waitFor(() => expect(screen.getByRole('heading', { name: /shelf report is ready/i })).toBeInTheDocument())
    expect(api.startAnalysis).toHaveBeenCalledWith(
      expect.any(File),
      { maxCrops: 60, maxSkuCrops: 5, extractSku: true },
    )
    expect(screen.getByRole('button', { name: /analyze shelf/i })).toBeEnabled()
    expect(screen.queryByText(/analysis in progress/i)).not.toBeInTheDocument()
    expect(screen.getByText('Soft drinks')).toBeInTheDocument()
    expect(screen.getByText(/category could not be matched/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /download report/i })).toBeInTheDocument()

    fireEvent.click(screen.getAllByRole('button', { name: /review ai/i })[0])
    fireEvent.click(screen.getAllByRole('button', { name: /^correct$/i })[0])
    await waitFor(() => expect(api.sendDetectionFeedback).toHaveBeenCalledWith({
      scanId: 7,
      cropId: 1,
      feedbackType: 'category',
      verdict: 'correct',
      correction: '',
    }))
  })
})
