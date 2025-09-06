"use client"
import { useEffect, useState } from 'react'
import { UploadArea } from '@/components/UploadArea'
import { getTemplates, getPresign, createAnalysis, createAnalysisFromFile, getAnalysis } from '@/lib/api'
import { useRouter } from 'next/navigation'

interface ProTemplate {
  id: string
  name: string
  s3Key: string
  description: string
}

export default function ProComparisonStartPage() {
  const router = useRouter()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploaded, setUploaded] = useState<{ key: string, url: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [loadingTemplates, setLoadingTemplates] = useState(true)
  const [proTemplates, setProTemplates] = useState<ProTemplate[]>([])
  const [selectedPro, setSelectedPro] = useState<string>('')

  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        setLoadingTemplates(true)
        const resp = await getTemplates()
        if (resp?.templates?.length) {
          setProTemplates(resp.templates)
          const sidePreferred = resp.templates.find((t: ProTemplate) => /side|사이드/i.test(t.name))
          setSelectedPro(sidePreferred?.id || resp.templates[0].id)
        } else {
          setProTemplates([])
        }
      } finally {
        setLoadingTemplates(false)
      }
    }
    fetchTemplates()
  }, [])

  const onStart = async () => {
    setBusy(true)
    try {
      // 1) 파일이 있으면 멀티파트로 즉시 분석 생성(권장)
      let analysisId: string | undefined
      if (selectedFile) {
        const res = await createAnalysisFromFile(selectedFile, selectedPro)
        analysisId = res.analysisId
      } else {
        // 2) 하위호환: 이미 업로드된 키를 이용한 JSON 경로
        let userKey = uploaded?.key
        if (!userKey) return
        const res = await createAnalysis({ userVideoKey: userKey, proTemplateId: selectedPro })
        analysisId = res.analysisId
      }

      if (!analysisId) return

      // 사용자 비교와 동일하게: 이 페이지에서 폴링 후 완료 시에만 결과 페이지로 이동
      const maxTries = 120 // 약 3분(1.5s*120)
      for (let i = 0; i < maxTries; i++) {
        const data = await getAnalysis(analysisId)
        if (data?.status === 'done') {
          router.push(`/result/${analysisId}`)
          return
        }
        if (data?.status === 'error') {
          alert('분석 중 오류가 발생했습니다. 다시 시도해주세요.')
          return
        }
        await new Promise((r) => setTimeout(r, 1500))
      }
      alert('분석 대기 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-blue-50 to-emerald-50">
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">프로 비교 분석</h1>
          <p className="text-sm text-gray-500">내 영상을 업로드하고, 선택한 프로 스윙과 비교 분석합니다</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl border border-gray-200 p-8">
              <UploadArea onFileUpload={setSelectedFile} onUploaded={setUploaded} />
            </div>
          </div>
          <div className="space-y-6">
            <div className="bg-white rounded-2xl border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">프로 템플릿 선택</h3>
              {loadingTemplates ? (
                <div className="text-sm text-gray-500">템플릿 로딩 중...</div>
              ) : (
                <>
                  <select
                    value={selectedPro}
                    onChange={(e) => setSelectedPro(e.target.value)}
                    className="w-full border border-gray-200 rounded-xl p-3 bg-gray-50 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all duration-200"
                  >
                    {proTemplates.map((p) => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </select>
                  {selectedPro && (
                    <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200 text-sm text-blue-800">
                      선택된 템플릿: {proTemplates.find(p => p.id === selectedPro)?.description}
                    </div>
                  )}
                </>
              )}
              <p className="text-sm text-gray-500 mt-2">비교할 프로 스윙을 선택해주세요</p>
            </div>

            <button
              disabled={(!uploaded && !selectedFile) || busy}
              onClick={onStart}
              className="w-full bg-gradient-to-r from-green-500 to-blue-600 hover:from-green-600 hover:to-blue-700 text-white font-semibold py-4 px-6 rounded-2xl shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {busy ? '분석 중...' : 'AI 분석 시작하기'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}


