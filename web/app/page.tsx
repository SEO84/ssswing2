"use client"
import { useState, useEffect } from 'react'
import { UploadArea } from '@/components/UploadArea'
import { AdSlot } from '@/components/AdSlot'
import { createAnalysis, createAnalysisFromFile, getTemplates, getPresign } from '@/lib/api'
import { useRouter } from 'next/navigation'

interface ProTemplate {
  id: string
  name: string
  s3Key: string
  description: string
}

export default function HomePage() {
  const [uploaded, setUploaded] = useState<{ key: string, url: string } | null>(null)
  const [proTemplates, setProTemplates] = useState<ProTemplate[]>([])
  const [selectedPro, setSelectedPro] = useState<string>('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [loadingTemplates, setLoadingTemplates] = useState(true)
  const [showProComparison, setShowProComparison] = useState(false)
  const router = useRouter()

  // 프로 템플릿 목록 가져오기
  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        setLoadingTemplates(true)
        console.log('프로 템플릿 로딩 시작...')
        
        const response = await getTemplates()
        console.log('API 응답:', response)
        
        // 응답 구조 확인 및 안전한 처리
        if (response && response.templates && Array.isArray(response.templates)) {
          console.log('템플릿 개수:', response.templates.length)
          setProTemplates(response.templates)
          if (response.templates.length > 0) {
            // 사이드(사이드뷰) 우선 선택, 없으면 첫 번째 선택
            const sidePreferred = response.templates.find((t: ProTemplate) => /side|사이드/i.test(t.name))
            setSelectedPro((sidePreferred?.id) || response.templates[0].id)
          }
        } else {
          console.warn('API 응답 구조가 예상과 다릅니다:', response)
          // 기본값 설정
          setProTemplates([
            { id: 'pro_iron_side', name: '프로 아이언(사이드뷰)', s3Key: 'pro/iron_side.mp4', description: '프로 아이언 스윙 템플릿' },
            { id: 'pro_driver_side', name: '프로 드라이버(사이드뷰)', s3Key: 'pro/driver_side.mp4', description: '프로 드라이버 스윙 템플릿' }
          ])
          setSelectedPro('pro_iron_side')
        }
      } catch (error) {
        console.error('프로 템플릿 로딩 실패:', error)
        // 기본값 설정
        setProTemplates([
          { id: 'pro_iron_side', name: '프로 아이언(사이드뷰)', s3Key: 'pro/iron_side.mp4', description: '프로 아이언 스윙 템플릿' },
          { id: 'pro_driver_side', name: '프로 드라이버(사이드뷰)', s3Key: 'pro/driver_side.mp4', description: '프로 드라이버 스윙 템플릿' }
        ])
        setSelectedPro('pro_iron_side')
      } finally {
        setLoadingTemplates(false)
      }
    }

    fetchTemplates()
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-blue-50 to-emerald-50">
      {/* 헤더 섹션 */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-blue-600 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
                  SSSwing
                </h1>
                <p className="text-sm text-gray-500">AI 골프 스윙 분석</p>
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-6 text-sm text-gray-600">
              <a href="/" className="hover:text-green-600 transition-colors duration-200">🏠 홈</a>
              <a href="/user-comparison" className="hover:text-blue-600 transition-colors duration-200">🔄 사용자 비교</a>
              <span>🎯 정확한 분석</span>
              <span>⚡ 빠른 처리</span>
              <span>🔒 안전한 업로드</span>
            </div>
          </div>
        </div>
      </div>

      {/* 메인 콘텐츠 */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* 히어로 섹션 */}
        <div className="text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            프로와 비교하는
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-green-600 to-blue-600">
              AI 골프 스윙 분석
            </span>
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            로그인 없이 영상을 업로드하고, AI가 프로 스윙과 비교 분석하여 
            <span className="font-semibold text-green-600">스윙 속도</span>와 
            <span className="font-semibold text-blue-600">궤도 각도</span>를 점수로 제공합니다.
          </p>
        </div>

        {/* 분석 단계 표시 */}
        <div className="flex justify-center mb-12">
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-green-500 text-white rounded-full flex items-center justify-center font-bold">
                1
              </div>
              <span className="text-gray-700 font-medium">영상 업로드</span>
            </div>
            <div className="w-8 h-0.5 bg-gray-300"></div>
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">
                2
              </div>
              <span className="text-gray-700 font-medium">프로 템플릿 선택</span>
            </div>
            <div className="w-8 h-0.5 bg-gray-300"></div>
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">
                3
              </div>
              <span className="text-gray-700 font-medium">AI 분석</span>
            </div>
            <div className="w-8 h-0.5 bg-gray-300"></div>
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-emerald-500 text-white rounded-full flex items-center justify-center font-bold">
                4
              </div>
              <span className="text-gray-700 font-medium">결과 확인</span>
            </div>
          </div>
        </div>

        {/* 분석 옵션 안내 */}
        <div className="text-center mb-8">
          <p className="text-lg text-gray-600">
            아래에서 원하는 분석 방법을 선택하세요
          </p>
        </div>

        {/* 서비스 소개 */}
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-800 mb-4">
            🎯 SSSwing으로 무엇을 할 수 있나요?
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
            <div className="bg-white p-6 rounded-xl shadow-lg">
              <div className="text-4xl mb-4">🏆</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">프로와 비교 분석</h3>
              <p className="text-gray-600">
                업로드한 스윙을 프로 골퍼의 스윙과 비교하여 정확한 점수와 개선점을 확인하세요
              </p>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-lg">
              <div className="text-4xl mb-4">🔄</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">사용자 영상 비교</h3>
              <p className="text-gray-600">
                두 개의 영상을 업로드하여 직접 비교 분석하고 개선 과정을 추적하세요
              </p>
            </div>
          </div>
        </div>

        {/* 분석 옵션 선택 */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
          <h2 className="text-2xl font-semibold text-gray-800 mb-6 text-center">
            📊 어떤 분석을 원하시나요?
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* 프로와 비교 분석 */}
            <div className="border-2 border-gray-200 rounded-xl p-6 hover:border-green-500 transition-colors duration-200">
              <div className="text-center">
                <div className="text-5xl mb-4">🏆</div>
                <h3 className="text-xl font-semibold text-gray-800 mb-3">프로와 비교 분석</h3>
                <p className="text-gray-600 mb-4">
                  업로드한 스윙을 전문가 스윙과 비교하여 정확한 점수와 개선점을 확인합니다
                </p>
                <a
                  href="/pro-comparison"
                  className="inline-block w-full px-6 py-3 bg-gradient-to-r from-green-500 to-blue-600 text-white font-semibold rounded-lg hover:from-green-600 hover:to-blue-700 transition-all duration-200 transform hover:scale-105"
                >
                  🚀 프로 비교 분석 시작
                </a>
              </div>
            </div>

            {/* 사용자 영상 비교 분석 */}
            <div className="border-2 border-gray-200 rounded-xl p-6 hover:border-blue-500 transition-colors duration-200">
              <div className="text-center">
                <div className="text-5xl mb-4">🔄</div>
                <h3 className="text-xl font-semibold text-gray-800 mb-3">사용자 영상 비교</h3>
                <p className="text-gray-600 mb-4">
                  두 개의 영상을 업로드하여 직접 비교 분석하고 개선 과정을 추적합니다
                </p>
                <a
                  href="/user-comparison"
                  className="inline-block w-full px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-200 transform hover:scale-105"
                >
                  🔄 사용자 비교 분석 시작
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* 기존 프로 비교 분석 폼 */}
        {showProComparison && (
          <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-semibold text-gray-800">
                🏆 프로와 비교 분석
              </h2>
              <button
                onClick={() => setShowProComparison(false)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ✕
              </button>
            </div>
            
            {/* 기존 업로드 및 분석 영역 */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* 업로드 영역 */}
              <div className="lg:col-span-2">
                <div className="bg-gray-50 rounded-2xl border border-gray-200 p-8">
                  <UploadArea onUploaded={setUploaded} onFileUpload={setSelectedFile} />
                </div>
              </div>

              {/* 사이드바 */}
              <div className="space-y-6">
                {/* 프로 템플릿 선택 */}
                <div className="bg-gray-50 rounded-2xl border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    <svg className="w-5 h-5 text-blue-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    프로 템플릿 선택
                  </h3>
                  
                  {loadingTemplates ? (
                    <div className="w-full border border-gray-200 rounded-xl p-3 bg-gray-50 flex items-center justify-center">
                      <svg className="animate-spin h-5 w-5 text-blue-600 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <span className="text-gray-500">프로 템플릿 로딩 중...</span>
                    </div>
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
                        <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
                          <p className="text-sm text-blue-800">
                            <span className="font-medium">선택된 템플릿:</span> {proTemplates.find(p => p.id === selectedPro)?.description}
                          </p>
                        </div>
                      )}
                    </>
                  )}
                  
                  <p className="text-sm text-gray-500 mt-2">
                    비교할 프로 스윙을 선택해주세요
                  </p>
                </div>

                {/* 분석 요청 버튼 */}
                <button
                  disabled={(!uploaded && !selectedFile) || busy}
                  onClick={async () => {
                    // 업로드가 아직 안 된 경우, 자동으로 S3 업로드 후 진행
                    setBusy(true)
                    try {
                      // 파일이 선택된 경우: 멀티파트로 즉시 분석 생성(권장)
                      if (selectedFile) {
                        const res = await createAnalysisFromFile(selectedFile, selectedPro)
                        router.push(`/result/${res.analysisId}`)
                        return
                      }

                      // 파일이 없고 이미 업로드된 키가 있는 경우: 하위호환 경로 유지
                      const userKey = uploaded?.key
                      if (!userKey) return
                      const res = await createAnalysis({ userVideoKey: userKey, proTemplateId: selectedPro })
                      router.push(`/result/${res.analysisId}`)
                    } finally {
                      setBusy(false)
                    }
                  }}
                  className="w-full bg-gradient-to-r from-green-500 to-blue-600 hover:from-green-600 hover:to-blue-700 text-white font-semibold py-4 px-6 rounded-2xl shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                  {busy ? (
                    <div className="flex items-center justify-center">
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      분석 준비 중...
                    </div>
                  ) : (
                    <div className="flex items-center justify-center">
                      <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      AI 분석 시작하기
                    </div>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 특징 섹션 */}
        <div className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="w-16 h-16 bg-green-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">정확한 분석</h3>
            <p className="text-gray-600">AI 기술로 스윙의 속도와 각도를 정밀하게 분석합니다</p>
          </div>
          <div className="text-center">
            <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">빠른 처리</h3>
            <p className="text-gray-600">업로드 후 2-3분 내에 분석 결과를 확인할 수 있습니다</p>
          </div>
          <div className="text-center">
            <div className="w-16 h-16 bg-purple-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">안전한 업로드</h3>
            <p className="text-gray-600">개인정보 없이 안전하게 영상을 업로드하고 분석합니다</p>
          </div>
        </div>
      </div>
    </div>
  )
}


