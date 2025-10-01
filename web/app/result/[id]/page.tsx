"use client"
import useSWR from 'swr'
import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import { getAnalysis, getAnalysisProgress } from '@/lib/api'
import SideBySidePlayer from '@/components/SideBySidePlayer'
import { MergeDownload } from '@/components/MergeDownload'
import { AdSlot } from '@/components/AdSlot'

export default function ResultPage() {
  const params = useParams<{ id: string }>()
  const id = params.id
  const [elapsedTime, setElapsedTime] = useState(0)
  const { data, isLoading, mutate, error } = useSWR(id ? ["analysis", id] : null, () => getAnalysis(id), { refreshInterval: 2000 })
  
  // 진행 상황 조회 (분석 중일 때만)
  const { data: progressData } = useSWR(
    id && data?.status === 'processing' ? ["progress", id] : null, 
    () => getAnalysisProgress(id), 
    { refreshInterval: 1000 }
  )
  
  // 경과 시간 업데이트
  useEffect(() => {
    if (data?.status === 'processing' && progressData?.start_time) {
      const interval = setInterval(() => {
        const startTime = new Date(progressData.start_time)
        const elapsed = Math.floor((Date.now() - startTime.getTime()) / 1000)
        setElapsedTime(elapsed)
      }, 1000)
      
      return () => clearInterval(interval)
    }
  }, [data?.status, progressData?.start_time])

  useEffect(() => {
    if (!data) return
    if (data.status !== 'done') {
      const t = setTimeout(() => mutate(), 2000)
      return () => clearTimeout(t)
    }
  }, [data, mutate])

  // 디버깅: 전체 데이터 로그
  useEffect(() => {
    console.log('=== 디버깅 로그 ===')
    console.log('분석 ID:', id)
    console.log('전체 데이터:', data)
    console.log('로딩 상태:', isLoading)
    console.log('오류:', error)
    
    if (data?.status === 'done') {
      console.log('분석 완료!')
      console.log('스코어:', data.scores)
      console.log('자산:', data.assets)
      console.log('프로 영상 URL:', data.assets?.proUrl)
      console.log('사용자 영상 URL:', data.assets?.userUrl)
      console.log('프로 포즈 데이터:', data.proPoses)
      console.log('사용자 포즈 데이터:', data.userPoses)
    }
  }, [data, isLoading, error, id])

  if (isLoading || !data) {
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

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <div className="w-24 h-24 bg-gradient-to-r from-green-500 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-8 animate-pulse">
              <svg className="w-12 h-12 text-white animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h2 className="text-3xl font-bold text-white mb-4">AI 분석 진행 중...</h2>
            <p className="text-xl text-gray-300 mb-8">서버에서 포즈 분석/정렬/스코어 계산 중입니다. 잠시만 기다려주세요.</p>
            
            {/* 경과 시간 표시 */}
            {elapsedTime > 0 && (
              <div className="mb-8">
                <div className="inline-flex items-center px-6 py-3 bg-blue-500/20 backdrop-blur-sm rounded-full border border-blue-400/30">
                  <svg className="w-5 h-5 text-blue-300 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-lg font-semibold text-blue-200">
                    경과 시간: {Math.floor(elapsedTime / 60)}분 {elapsedTime % 60}초
                  </span>
                </div>
              </div>
            )}
            
            {/* 진행률 표시 */}
            <div className="max-w-md mx-auto">
              <div className="bg-gray-600 rounded-full h-3 mb-4">
                <div 
                  className="bg-gradient-to-r from-green-500 to-blue-600 h-3 rounded-full transition-all duration-500" 
                  style={{ width: `${progressData?.progress || 0}%` }}
                ></div>
              </div>
              <p className="text-sm text-gray-400">
                분석 진행률: {progressData?.progress || 0}%
              </p>
              {progressData?.current_step && (
                <p className="text-sm text-blue-300 mt-2 font-medium">
                  현재 단계: {progressData.current_step}
                </p>
              )}
            </div>

            {/* 분석 단계 표시 */}
            <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
              <div className="bg-white/10 backdrop-blur-sm rounded-3xl p-6 shadow-lg border border-white/20">
                <div className="w-12 h-12 bg-green-500/20 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <svg className="w-6 h-6 text-green-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-white mb-2">영상 업로드 완료</h3>
                <p className="text-sm text-gray-300">사용자 영상이 성공적으로 업로드되었습니다</p>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-3xl p-6 shadow-lg border border-white/20">
                <div className="w-12 h-12 bg-blue-500/20 rounded-xl flex items-center justify-center mx-auto mb-4 animate-pulse">
                  <svg className="w-6 h-6 text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-white mb-2">AI 분석 중</h3>
                <p className="text-sm text-gray-300">포즈 추출 및 스윙 분석을 진행하고 있습니다</p>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-3xl p-6 shadow-lg border border-white/20 opacity-50">
                <div className="w-12 h-12 bg-gray-500/20 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-white mb-2">결과 생성</h3>
                <p className="text-sm text-gray-300">분석 결과를 정리하여 표시합니다</p>
              </div>
            </div>

            {/* 광고 슬롯 */}
            <div className="mt-12">
              <AdSlot />
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (data.status !== 'done') {
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

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <div className="w-24 h-24 bg-gradient-to-r from-yellow-500 to-orange-600 rounded-full flex items-center justify-center mx-auto mb-8">
              <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h2 className="text-3xl font-bold text-white mb-4">
              {data.status === 'error' ? '분석 오류' : '분석 큐 대기/진행 중'}
            </h2>
            <p className="text-xl text-gray-300 mb-8">잠시 후 자동으로 갱신됩니다...</p>
            
            {/* 상태별 메시지 */}
            <div className="bg-white/10 backdrop-blur-sm rounded-3xl p-8 shadow-lg border border-white/20 max-w-2xl mx-auto">
              <div className="flex items-center justify-center space-x-4 mb-4">
                <div className="w-4 h-4 bg-blue-400 rounded-full animate-pulse"></div>
                <span className="text-lg font-medium text-white">
                  {data.status === 'queued' && '분석 대기 중...'}
                  {data.status === 'running' && '분석 진행 중...'}
                  {data.status === 'error' && '분석 중 오류 발생'}
                </span>
              </div>
              <p className="text-gray-300">
                {data.status === 'queued' && '다른 분석이 진행 중입니다. 순서대로 처리됩니다.'}
                {data.status === 'running' && 'AI가 스윙을 분석하고 있습니다. 잠시만 기다려주세요.'}
                {data.status === 'error' && '분석 중 오류가 발생했습니다. 다시 시도해주세요.'}
              </p>
            </div>

            {/* 광고 슬롯 */}
            <div className="mt-12">
              <AdSlot />
            </div>
          </div>
        </div>
      </div>
    )
  }

  const s = data.scores
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
                <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">비교 분석 결과</h1>
                <p className="text-sm text-gray-400">프로와 비교한 상세 분석 결과</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 메인 콘텐츠 */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">

        {/* 점수 결과 (사용자 비교 스타일) */}
        <div className="bg-white/10 backdrop-blur-sm rounded-3xl border border-white/20 shadow-xl p-8 mb-8">
          <h2 className="text-2xl font-semibold text-white mb-6">📊 분석 결과</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-6 bg-gradient-to-br from-blue-500/20 to-blue-600/30 backdrop-blur-sm rounded-xl border border-blue-400/30">
              <div className="text-3xl font-bold text-blue-300 mb-2">{s.angles?.toFixed(1)}%</div>
              <div className="text-sm text-blue-200">관절 각도 점수</div>
            </div>
            <div className="text-center p-6 bg-gradient-to-br from-green-500/20 to-green-600/30 backdrop-blur-sm rounded-xl border border-green-400/30">
              <div className="text-3xl font-bold text-green-300 mb-2">{s.speed?.toFixed(1)}%</div>
              <div className="text-sm text-green-200">속도 타이밍 점수</div>
            </div>
            <div className="text-center p-6 bg-gradient-to-br from-purple-500/20 to-purple-600/30 backdrop-blur-sm rounded-xl border border-purple-400/30">
              <div className="text-3xl font-bold text-purple-300 mb-2">{s.final?.toFixed(1)}%</div>
              <div className="text-sm text-purple-200">최종 종합 점수</div>
            </div>
          </div>
        </div>

        {/* 서버 합성 영상 자동 재생 미리보기 (사용자 비교 스타일) */}
        {data.comparisonVideoUrl && (
          <div className="bg-white/10 backdrop-blur-sm rounded-3xl border border-white/20 shadow-xl p-8 mb-8">
            <h3 className="text-xl font-semibold text-white mb-4">🎬 합성 영상 미리보기</h3>
            <div className="w-full">
              <video className="w-full rounded-lg border border-gray-600" autoPlay loop muted playsInline controls preload="metadata">
                <source src={data.comparisonVideoUrl} />
              </video>
              <p className="mt-2 text-xs text-gray-400">자동 재생 중입니다. 소리가 필요하면 음소거를 해제하세요.</p>
            </div>
          </div>
        )}

        {/* 스켈레톤 비교 */}
        {(data.userPoses || data.proPoses) && (
          <div className="bg-white/10 backdrop-blur-sm rounded-3xl border border-white/20 shadow-xl p-8 mb-8">
            <h3 className="text-xl font-semibold text-white mb-4">🦴 포즈 스켈레톤 비교</h3>
            <SideBySidePlayer
              leftVideoUrl={data.assets?.proUrl}
              rightVideoUrl={data.assets?.userUrl}
              leftPoses={data.proPoses}
              rightPoses={data.userPoses}
            />
          </div>
        )}

        {/* 비교 영상 다운로드 */}
        {data.comparisonVideoUrl && (
          <div className="bg-white/10 backdrop-blur-sm rounded-3xl border border-white/20 shadow-xl p-8 mb-8">
            <h3 className="text-xl font-semibold text-white mb-4">🎬 비교 영상 다운로드</h3>
            <MergeDownload leftSrc={data.assets?.proUrl} rightSrc={data.assets?.userUrl} />
          </div>
        )}

        {/* 네비게이션 버튼들 */}
        <div className="bg-white/10 backdrop-blur-sm rounded-3xl border border-white/20 shadow-xl p-8">
          <h3 className="text-xl font-semibold text-white mb-6 text-center">🚀 다른 분석을 시도해보세요</h3>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <a
              href="/"
              className="px-8 py-4 bg-gradient-to-r from-green-500 to-blue-600 text-white font-semibold rounded-xl hover:from-green-600 hover:to-blue-700 transition-all duration-300 transform hover:scale-105 shadow-lg"
            >
              🏠 홈으로 돌아가기
            </a>
            <a
              href="/pro-comparison"
              className="px-8 py-4 bg-gradient-to-r from-green-500 to-blue-600 text-white font-semibold rounded-xl hover:from-green-600 hover:to-blue-700 transition-all duration-300 transform hover:scale-105 shadow-lg"
            >
              🏆 프로 비교 분석
            </a>
            <a
              href="/user-comparison"
              className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-600 hover:to-purple-700 transition-all duration-300 transform hover:scale-105 shadow-lg"
            >
              🔄 사용자 비교 분석
            </a>
          </div>
          <p className="text-center text-gray-400 mt-4 text-sm">
            새로운 영상으로 다른 분석을 시작하거나 홈으로 돌아가세요
          </p>
        </div>
      </div>
    </div>
  )
}


