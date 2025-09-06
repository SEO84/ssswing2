// 브라우저 경량 포즈 추론(선택): onnxruntime-web + WebGPU
// MVP에서는 서버 분석을 사용하며, 본 파일은 추후 미리보기 고도화에 활용
import type * as ortTypes from 'onnxruntime-web'

let ort: typeof ortTypes | null = null

export async function ensureOrt() {
  if (!ort) {
    ort = await import('onnxruntime-web')
  }
  return ort
}

export async function warmup() {
  await ensureOrt()
}


