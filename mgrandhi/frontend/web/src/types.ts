export type Detection = {
  crop_id: number
  category: string
  subcategory: string
  score: number
  area: number
  box: number[]
  brand?: string
  product_name?: string
  sku_text?: string
  visible_text?: string
  package_size?: string
  barcode?: string
  sku_confidence?: number
  sku_needs_review?: number
  sku_error?: string
}

export type ScanResult = {
  scan_id: number
  image_name: string
  annotated_image: string
  summary: {
    num_items: number
    distinct_categories: number
    empty_pct: number
    empty_label: string
    shelf_type: string
    review_count: number
  }
  detections: Detection[]
  timings: Record<string, number>
  warning?: string
}

export type AnalysisJob = {
  job_id: string
  status: 'queued' | 'processing' | 'complete' | 'failed'
  stage: string
  progress: number
  message: string
  result?: ScanResult
  error?: string
}

export type ScanHistory = {
  id: number
  ts: string
  image_name: string
  num_items: number
  distinct_categories: number
  empty_pct: number
  shelf_type: string
  review_count: number
}

export type PlanogramSlot = {
  slot_index: number
  category: string
  subcategory: string
  brand: string
  facings: number
}

export type PlanogramRow = {
  row_index: number
  slots: PlanogramSlot[]
}

export type PlanogramTemplateSummary = {
  id: number
  name: string
  store_id: string
  shelf_id: string
  ts: string
}

export type PlanogramTemplate = {
  template_id: number
  rows: PlanogramRow[]
}

export type PlanogramResultSlot = {
  row_index: number
  position: number
  status: 'compliant' | 'misplaced' | 'missing' | 'extra'
  expected_key: string | null
  actual_key: string | null
  detail: { crop_id?: number; box?: number[] }
}

export type PlanogramResult = {
  scan_id: number
  template_id: number
  compliance_score: number
  total_expected: number
  total_compliant: number
  missing_count: number
  extra_count: number
  row_count_expected: number
  row_count_detected: number
  slots: PlanogramResultSlot[]
}

export type Insights = {
  summary: {
    total_items: number
    distinct_categories: number
    unknown_items: number
    num_scans: number
  }
  categories: Array<{ category: string; count: number }>
  subcategories: Array<{ subcategory: string; count: number }>
  scans: ScanHistory[]
  feedback: {
    total: number
    category_correct: number
    category_incorrect: number
    sku_correct: number
    sku_incorrect: number
  }
}
