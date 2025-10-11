"use client"
import { useEffect, useState } from 'react'
import { UploadArea } from '@/components/UploadArea'
import { getTemplates, getPresign, createAnalysis, createAnalysisFromFile, getAnalysis, API } from '@/lib/api'
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
  const [templateError, setTemplateError] = useState<string>('')
  const [lastFetchedAt, setLastFetchedAt] = useState<string>('')

  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        setLoadingTemplates(true)
        setTemplateError('')
        const resp = await getTemplates()
        if (resp?.templates?.length) {
          setProTemplates(resp.templates)
          const sidePreferred = resp.templates.find((t: ProTemplate) => /side|사이드/i.test(t.name))
          setSelectedPro(sidePreferred?.id || resp.templates[0].id)
        } else {
          setProTemplates([])
        }
        setLastFetchedAt(new Date().toISOString())
      } catch (err: any) {
        setTemplateError(String(err?.message || err))
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
    <div className="min-h-screen relative overflow-hidden">
      {/* 인터랙티브 배경 */}
      <div className="fixed inset-0 z-0">
        {/* 다크 배경 그라데이션 */}
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-black to-gray-800"></div>
        
        {/* 다크 텍스처 오버레이 */}
        <div className="absolute inset-0 opacity-30">
          <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-transparent via-gray-800/20 to-gray-900/40"></div>
        </div>
        
        {/* 미니멀한 배경 요소 */}
        <div className="absolute inset-0 overflow-hidden">
          {/* 단순한 그라데이션 원형들 */}
          <div className="absolute top-1/4 right-1/4 w-64 h-64 bg-gradient-to-br from-green-500/5 to-transparent rounded-full blur-3xl"></div>
          <div className="absolute bottom-1/4 left-1/4 w-80 h-80 bg-gradient-to-br from-blue-500/5 to-transparent rounded-full blur-3xl"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-br from-purple-500/3 to-transparent rounded-full blur-3xl"></div>
        </div>
      </div>

      {/* 헤더 섹션 */}
      <div className="relative z-20 bg-black/80 backdrop-blur-sm shadow-sm border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-blue-600 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">프로 비교 분석</h1>
                <p className="text-sm text-gray-400">내 영상을 업로드하고, 선택한 프로 스윙과 비교 분석합니다</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 메인 콘텐츠 */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 md:py-10">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 sm:gap-8">
          <div className="lg:col-span-2">
            <div className="bg-white/10 backdrop-blur-sm rounded-2xl sm:rounded-3xl border border-white/20 p-4 sm:p-6 md:p-8">
              <UploadArea onFileUpload={setSelectedFile} onUploaded={setUploaded} />
            </div>
          </div>
          <div className="space-y-4 sm:space-y-6">
            <div className="bg-white/10 backdrop-blur-sm rounded-2xl sm:rounded-3xl border border-white/20 p-4 sm:p-6">
              <h3 className="text-base sm:text-lg font-semibold text-white mb-3 sm:mb-4">프로 템플릿 선택</h3>
              {loadingTemplates ? (
                <div className="text-xs sm:text-sm text-gray-400">템플릿 로딩 중...</div>
              ) : proTemplates.length === 0 ? (
                <div className="text-xs sm:text-sm text-red-400 space-y-1">
                  <div>템플릿을 불러올 수 없습니다. S3 연결을 확인하세요.</div>
                  <div className="text-[10px] sm:text-xs text-red-300 break-all">에러: {templateError || '없음'}</div>
                  <div className="text-[10px] sm:text-xs text-gray-400 break-all">API: {API}</div>
                  <div className="text-[10px] sm:text-xs text-gray-400">fetch 시각: {lastFetchedAt || '미시도'}</div>
                </div>
              ) : (
                <>
                  <select
                    value={selectedPro}
                    onChange={(e) => setSelectedPro(e.target.value)}
                    className="w-full border border-white/30 rounded-xl p-2 sm:p-3 text-sm sm:text-base bg-white/10 backdrop-blur-sm text-white focus:bg-white/20 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/50 transition-all duration-200"
                  >
                    {proTemplates.map((p) => (
                      <option key={p.id} value={p.id} className="bg-gray-800 text-white">{p.name}</option>
                    ))}
                  </select>
                  {selectedPro && (
                    <div className="mt-2 sm:mt-3 p-2 sm:p-3 bg-blue-500/20 backdrop-blur-sm rounded-lg border border-blue-400/30 text-xs sm:text-sm text-blue-200">
                      선택된 템플릿: {proTemplates.find(p => p.id === selectedPro)?.description}
                    </div>
                  )}
                  <div className="mt-2 space-y-1">
                    <div className="text-[10px] sm:text-xs text-gray-400 break-all">API: {API}</div>
                    <div className="text-[10px] sm:text-xs text-gray-400">템플릿 개수: {proTemplates.length}</div>
                    {lastFetchedAt && (
                      <div className="text-[10px] sm:text-xs text-gray-400">fetch 시각: {lastFetchedAt}</div>
                    )}
                  </div>
                </>
              )}
              <p className="text-xs sm:text-sm text-gray-400 mt-2">비교할 프로 스윙을 선택해주세요</p>
            </div>

            <button
              disabled={(!uploaded && !selectedFile) || busy || !selectedPro}
              onClick={onStart}
              className="w-full bg-gradient-to-r from-green-500 to-blue-600 hover:from-green-600 hover:to-blue-700 text-white font-semibold py-3 sm:py-4 px-4 sm:px-6 text-sm sm:text-base rounded-2xl shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105"
            >
              {busy ? '분석 중...' : 'AI 분석 시작하기'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}


