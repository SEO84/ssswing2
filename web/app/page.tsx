'use client'


export default function Home() {

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
                <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
                  SSSwing
                </h1>
                <p className="text-sm text-gray-400">AI 골프 스윙 분석</p>
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-6 text-sm text-gray-400">
              <a href="/" className="hover:text-green-400 transition-colors duration-200">🏠 홈</a>
              <a href="/user-comparison" className="hover:text-blue-400 transition-colors duration-200">🔄 사용자 비교</a>
              <span>🎯 정확한 분석</span>
              <span>⚡ 빠른 처리</span>
              <span>🔒 안전한 업로드</span>
            </div>
          </div>
        </div>
      </div>

      {/* 메인 콘텐츠 */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* 히어로 섹션 */}
        <div className="text-center mb-16 relative">
          <div className="relative z-10">
            <h2 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-6 sm:mb-8 animate-fade-in px-2">
              프로와 비교하는
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-green-400 via-blue-400 to-purple-400 gradient-text-animate">
                AI 골프 스윙 분석
              </span>
            </h2>
            <p className="text-base sm:text-lg md:text-xl lg:text-2xl text-gray-300 max-w-4xl mx-auto leading-relaxed animate-slide-up px-4">
              영상 업로드 → AI 분석 → 정확한 점수 제공
            </p>
            
            {/* CTA 버튼들 */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center mt-8 animate-fade-in-delayed">
              <a
                href="/pro-comparison"
                className="group relative inline-block px-8 py-4 bg-gradient-to-r from-green-500 to-blue-600 text-white font-bold rounded-xl hover:from-green-600 hover:to-blue-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl overflow-hidden"
              >
                <span className="relative z-10 flex items-center justify-center">
                  <svg className="w-5 h-5 mr-2 group-hover:rotate-12 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  🚀 프로 비교 분석 시작
                </span>
                <div className="absolute inset-0 bg-gradient-to-r from-green-600 to-blue-700 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </a>
              
              <a
                href="/user-comparison"
                className="group relative inline-block px-8 py-4 bg-white/10 backdrop-blur-sm text-white font-bold rounded-xl border border-white/20 hover:bg-white/20 hover:border-white/30 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl overflow-hidden"
              >
                <span className="relative z-10 flex items-center justify-center">
                  <svg className="w-5 h-5 mr-2 group-hover:rotate-12 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  🔄 사용자 비교 분석
                </span>
                <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </a>
            </div>
          </div>
        </div>

        {/* 분석 단계 표시 */}
        <div className="mb-12 sm:mb-16 md:mb-20">
          <div className="flex items-center justify-center space-x-3 sm:space-x-6 md:space-x-8 px-4">
            <div className="flex items-center space-x-2 sm:space-x-3 md:space-x-4">
              <div className="w-8 h-8 sm:w-10 sm:h-10 bg-white/10 backdrop-blur-sm rounded-full flex items-center justify-center border border-white/20">
                <span className="text-white font-semibold text-xs sm:text-sm">1</span>
              </div>
              <span className="text-gray-300 font-medium text-xs sm:text-sm md:text-base">업로드</span>
            </div>
            
            <div className="w-6 sm:w-8 md:w-12 h-px bg-gradient-to-r from-transparent via-white/30 to-transparent"></div>
            
            <div className="flex items-center space-x-2 sm:space-x-3 md:space-x-4">
              <div className="w-8 h-8 sm:w-10 sm:h-10 bg-white/10 backdrop-blur-sm rounded-full flex items-center justify-center border border-white/20">
                <span className="text-white font-semibold text-xs sm:text-sm">2</span>
              </div>
              <span className="text-gray-300 font-medium text-xs sm:text-sm md:text-base">분석</span>
            </div>
            
            <div className="w-6 sm:w-8 md:w-12 h-px bg-gradient-to-r from-transparent via-white/30 to-transparent"></div>
            
            <div className="flex items-center space-x-2 sm:space-x-3 md:space-x-4">
              <div className="w-8 h-8 sm:w-10 sm:h-10 bg-white/10 backdrop-blur-sm rounded-full flex items-center justify-center border border-white/20">
                <span className="text-white font-semibold text-xs sm:text-sm">3</span>
              </div>
              <span className="text-gray-300 font-medium text-xs sm:text-sm md:text-base">결과</span>
            </div>
          </div>
        </div>

        {/* 분석 옵션 안내 */}
        <div className="text-center mb-8">
          <p className="text-lg text-gray-400">
            아래에서 원하는 분석 방법을 선택하세요
          </p>
        </div>

        {/* 서비스 소개 */}
        <div className="text-center mb-16">
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold text-white mb-6 sm:mb-8 animate-fade-in px-4">
            🎯 SSSwing으로 무엇을 할 수 있나요?
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-8 mt-6 sm:mt-8">
            <div className="group relative bg-white/5 backdrop-blur-sm p-6 sm:p-8 rounded-3xl border border-white/10 hover:border-white/20 transition-all duration-500 hover:-translate-y-1">
              <div className="text-center">
                <div className="w-12 h-12 sm:w-16 sm:h-16 bg-gradient-to-br from-green-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-4 sm:mb-6 group-hover:scale-110 transition-transform duration-300">
                  <svg className="w-6 h-6 sm:w-8 sm:h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="text-xl sm:text-2xl font-bold text-white mb-3 sm:mb-4">
                  프로와 비교 분석
                </h3>
                <p className="text-sm sm:text-base text-gray-400 mb-6 sm:mb-8 leading-relaxed">
                  영상 업로드 → AI 분석 → 정확한 점수 제공
                </p>
              </div>
            </div>
            
            <div className="group relative bg-white/5 backdrop-blur-sm p-6 sm:p-8 rounded-3xl border border-white/10 hover:border-white/20 transition-all duration-500 hover:-translate-y-1">
              <div className="text-center">
                <div className="w-12 h-12 sm:w-16 sm:h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4 sm:mb-6 group-hover:scale-110 transition-transform duration-300">
                  <svg className="w-6 h-6 sm:w-8 sm:h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </div>
                <h3 className="text-xl sm:text-2xl font-bold text-white mb-3 sm:mb-4">
                  사용자 영상 비교
                </h3>
                <p className="text-sm sm:text-base text-gray-400 mb-6 sm:mb-8 leading-relaxed">
                  두 영상 업로드 → 직접 비교 → 개선 과정 추적
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* 분석 옵션 선택 */}
        <div className="relative bg-white/5 backdrop-blur-sm rounded-3xl border border-white/10 p-8 mb-12">
          <div className="relative z-10">
            <h2 className="text-2xl sm:text-3xl font-bold text-white mb-6 sm:mb-8 text-center px-4">
              📊 어떤 분석을 원하시나요?
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-8">
              {/* 프로와 비교 분석 */}
              <div className="group relative bg-white/5 backdrop-blur-sm rounded-3xl border border-white/10 hover:border-white/20 transition-all duration-500 hover:-translate-y-1 p-6 sm:p-8">
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300">
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-4">
                    프로와 비교 분석
                  </h3>
                  <p className="text-gray-400 mb-8 leading-relaxed">
                    영상 업로드 → AI 분석 → 정확한 점수 제공
                  </p>
                  <a
                    href="/pro-comparison"
                    className="inline-block px-8 py-4 bg-gradient-to-r from-green-500 to-blue-600 text-white font-bold rounded-xl hover:from-green-600 hover:to-blue-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                  >
                    🚀 프로 비교 분석 시작
                  </a>
                </div>
              </div>

              {/* 사용자 영상 비교 분석 */}
              <div className="group relative bg-white/5 backdrop-blur-sm rounded-3xl border border-white/10 hover:border-white/20 transition-all duration-500 hover:-translate-y-1 p-8">
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300">
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-4">
                    사용자 영상 비교
                  </h3>
                  <p className="text-gray-400 mb-8 leading-relaxed">
                    두 영상 업로드 → 직접 비교 → 개선 과정 추적
                  </p>
                  <a
                    href="/user-comparison"
                    className="inline-block px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-bold rounded-xl hover:from-blue-600 hover:to-purple-700 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                  >
                    🔄 사용자 비교 분석 시작
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}