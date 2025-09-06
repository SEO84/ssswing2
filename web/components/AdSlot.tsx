"use client"
import { useEffect, useState } from 'react'

export function AdSlot() {
  const [isAdLoaded, setIsAdLoaded] = useState(false)

  useEffect(() => {
    try {
      // @ts-ignore
      ;(window.adsbygoogle = window.adsbygoogle || []).push({})
      // 광고 로드 시뮬레이션
      setTimeout(() => setIsAdLoaded(true), 1000)
    } catch {}
  }, [])

  return (
    <div className="relative overflow-hidden rounded-xl border border-gray-200 bg-gradient-to-br from-blue-50 to-purple-50 p-4 transition-all duration-300 hover:shadow-md">
      {/* 광고 로딩 상태 */}
      {!isAdLoaded && (
        <div className="flex items-center justify-center space-x-2 text-sm text-gray-500">
          <div className="w-4 h-4 border-2 border-blue-300 border-t-blue-600 rounded-full animate-spin"></div>
          <span>광고 로딩 중...</span>
        </div>
      )}

      {/* AdSense 실제 적용 시 아래 주석 해제 및 client/data-ad-slot 교체 */}
      {/* <ins className="adsbygoogle" style={{ display: 'block' }} data-ad-client={process.env.NEXT_PUBLIC_ADSENSE_CLIENT} data-ad-slot="0000000000" data-ad-format="auto" data-full-width-responsive="true"></ins> */}
      
      {/* 임시 광고 플레이스홀더 */}
      <div className={`text-center transition-opacity duration-300 ${isAdLoaded ? 'opacity-100' : 'opacity-0'}`}>
        <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100">
          <div className="flex items-center justify-center space-x-2 mb-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
            <span className="text-xs font-medium text-gray-700">스폰서 광고</span>
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
          </div>
          <div className="bg-gray-100 rounded h-20 flex items-center justify-center">
            <div className="text-center">
              <svg className="w-8 h-8 text-gray-400 mx-auto mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <p className="text-xs text-gray-500">골프 관련 광고</p>
            </div>
          </div>
          <div className="mt-2 text-xs text-gray-600">
            <p className="font-medium">골프 클럽 할인 이벤트</p>
            <p className="text-gray-500">최대 50% 할인</p>
          </div>
        </div>
      </div>

      {/* 광고 배지 */}
      <div className="absolute top-2 right-2">
        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
          광고
        </span>
      </div>
    </div>
  )
}


