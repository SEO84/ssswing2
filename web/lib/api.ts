export const API = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

export async function getPresign(input: { filename: string, contentType: string }) {
  const res = await fetch(`${API}/videos/presign`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input)
  })
  if (!res.ok) throw new Error('presign 실패')
  return res.json()
}

export async function getTemplates() {
  const res = await fetch(`${API}/analysis/templates`)
  if (!res.ok) throw new Error('템플릿 목록 조회 실패')
  return res.json()
}

export async function createAnalysis(input: { userVideoKey: string, proTemplateId: string }) {
  const res = await fetch(`${API}/analysis`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input)
  })
  if (!res.ok) throw new Error('분석 생성 실패')
  return res.json()
}

// 파일 업로드를 통한 프로 비교 분석 생성(통합 경로)
export async function createAnalysisFromFile(file: File, proTemplateId: string) {
  const form = new FormData()
  form.append('userVideo', file)
  form.append('proTemplateId', proTemplateId)
  const res = await fetch(`${API}/analysis`, { method: 'POST', body: form })
  if (!res.ok) throw new Error('분석 생성 실패(파일 업로드)')
  return res.json()
}

export async function getAnalysis(id: string) {
  const res = await fetch(`${API}/analysis/${id}`)
  if (!res.ok) throw new Error('분석 조회 실패')
  return res.json()
}

export async function getAnalysisProgress(id: string) {
  const res = await fetch(`${API}/analysis/${id}/progress`)
  if (!res.ok) throw new Error('진행 상황 조회 실패')
  return res.json()
}

export async function requestExport(input: { analysisId: string }) {
  const res = await fetch(`${API}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input)
  })
  if (!res.ok) throw new Error('export 실패')
  return res.json()
}

// 사용자 영상 비교 분석 API
export async function createUserComparisonAnalysis(
  video1: File,
  video2: File,
  description: string
): Promise<{ status: string; analysisId: string; message: string }> {
  const formData = new FormData();
  formData.append('video1', video1);
  formData.append('video2', video2);
  formData.append('description', description);

  const response = await fetch(`${API}/analysis/user-comparison`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('사용자 비교 분석 요청에 실패했습니다.');
  }

  return response.json();
}

export async function getUserComparisonResult(analysisId: string): Promise<any> {
  const response = await fetch(`${API}/analysis/user-comparison/${analysisId}`);
  
  if (!response.ok) {
    throw new Error('사용자 비교 분석 결과 조회에 실패했습니다.');
  }

  return response.json();
}

export async function listUserComparisons(): Promise<any> {
  const response = await fetch(`${API}/analysis/user-comparison`);
  
  if (!response.ok) {
    throw new Error('사용자 비교 분석 목록 조회에 실패했습니다.');
  }

  return response.json();
}


