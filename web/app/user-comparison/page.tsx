'use client';

import { useState, useRef, useEffect } from 'react';
import { UploadArea } from '../../components/UploadArea';
import SideBySidePlayer from '../../components/SideBySidePlayer';
import { MergeDownload } from '../../components/MergeDownload';
import { createUserComparisonAnalysis, getUserComparisonResult } from '@/lib/api';

interface AnalysisResult {
  status: string;
  analysisId: string;
  scores?: {
    angle_score: number;
    speed_score: number;
    final_score: number;
  };
  userPoses?: any[];
  comparisonPoses?: any[];
  comparisonVideoUrl?: string;
  description?: string;
}

export default function UserComparisonPage() {
  const [video1, setVideo1] = useState<File | null>(null);
  const [video2, setVideo2] = useState<File | null>(null);
  const [description, setDescription] = useState('');
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const video1Ref = useRef<HTMLVideoElement>(null);
  const video2Ref = useRef<HTMLVideoElement>(null);
  const previewRef = useRef<HTMLVideoElement>(null);

  const handleVideo1Upload = (file: File) => {
    console.log('handleVideo1Upload called with:', file?.name) // 디버깅용
    setVideo1(file);
    setError(null);
  };

  const handleVideo2Upload = (file: File) => {
    console.log('handleVideo2Upload called with:', file?.name) // 디버깅용
    setVideo2(file);
    setError(null);
  };

  // 디버깅용 상태 출력
  console.log('Current state:', { video1: video1?.name, video2: video2?.name, description: description })

  const startAnalysis = async () => {
    if (!video1 || !video2) {
      setError('두 개의 영상 파일을 모두 업로드해주세요.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      console.log('=== 분석 요청 시작 ===');
      console.log('첫 번째 영상:', video1.name, '크기:', video1.size);
      console.log('두 번째 영상:', video2.name, '크기:', video2.size);

      console.log('FormData 생성 완료, API 요청 시작...');

      const result = await createUserComparisonAnalysis(
        video1,
        video2,
        description || '사용자 영상 비교 분석'
      );
      console.log('분석 요청 성공, 응답 데이터:', result);

      if (!result.analysisId) {
        throw new Error('분석 ID가 반환되지 않았습니다.');
      }

      console.log('분석 ID 확인:', result.analysisId);
      setAnalysisId(result.analysisId);
      
      // 분석 완료까지 폴링 시작
      console.log('폴링 시작...');
      pollAnalysisResult(result.analysisId);
      
    } catch (err) {
      console.error('분석 요청 실패:', err);
      setError(err instanceof Error ? err.message : '알 수 없는 오류가 발생했습니다.');
      setIsAnalyzing(false);
    }
  };

  const pollAnalysisResult = async (id: string) => {
    console.log('폴링 시작 - 분석 ID:', id);
    let pollCount = 0;
    const maxPolls = 200; // 시간 제한 없음 (200 * 3초 = 10분, 충분히 긴 시간)
    let pollInterval: NodeJS.Timeout | null = null;
    
    const stopPolling = () => {
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
        console.log('폴링 중단됨');
      }
    };
    
    pollInterval = setInterval(async () => {
      pollCount++;
      console.log(`폴링 시도 ${pollCount}/${maxPolls} - 분석 ID: ${id}`);
      
      // 최대 폴링 횟수 초과 시 강제 중단 (매우 긴 시간 후)
      if (pollCount > maxPolls) {
        console.log('최대 폴링 횟수 초과, 폴링 강제 중단');
        setError('분석 시간이 초과되었습니다. (10분 초과) 분석이 아직 진행 중일 수 있습니다. 아래 버튼을 클릭하여 결과를 확인해보세요.');
        setIsAnalyzing(false);
        stopPolling();
        return;
      }
      
      try {
        let responseData: any;
        try {
          responseData = await getUserComparisonResult(id);
        } catch (e) {
          console.log('분석 결과가 아직 준비되지 않음 (404)');
          if (pollCount >= maxPolls) {
            console.log('최대 폴링 횟수 도달, 폴링 중단');
            setError('분석 시간이 초과되었습니다. (10분 초과) 분석이 아직 진행 중일 수 있습니다. 아래 버튼을 클릭하여 결과를 확인해보세요.');
            setIsAnalyzing(false);
            stopPolling();
          }
          return; // 404일 때는 계속 폴링
        }
        const result = responseData;
        console.log('폴링 성공, 결과:', result);
        
        if (result.status === 'completed') {
          console.log('분석 완료!');
          setAnalysisResult(result.result);
          setIsAnalyzing(false);
          stopPolling();
        } else if (result.status === 'failed') {
          console.log('분석 실패:', result.result?.error);
          setError(result.result?.error || '분석에 실패했습니다.');
          setIsAnalyzing(false);
          stopPolling();
        } else if (result.status === 'processing') {
          console.log('분석 진행 중...');
          // processing 상태면 계속 폴링
        } else {
          console.log('알 수 없는 상태:', result.status);
          // 알 수 없는 상태도 최대 횟수 후 중단
          if (pollCount >= maxPolls) {
            setError('분석 상태를 확인할 수 없습니다. 잠시 후 다시 시도해주세요.');
            setIsAnalyzing(false);
            stopPolling();
          }
        }
      } catch (err) {
        console.error('폴링 중 오류:', err);
        if (pollCount >= maxPolls) {
          setError(err instanceof Error ? err.message : '분석 결과 조회에 실패했습니다.');
          setIsAnalyzing(false);
          stopPolling();
        }
      }
    }, 3000); // 3초마다 폴링 (분석 시간 고려)
  };


  const resetAnalysis = () => {
    setVideo1(null);
    setVideo2(null);
    setDescription('');
    setAnalysisId(null);
    setAnalysisResult(null);
    setIsAnalyzing(false);
    setError(null);
    // 폴링 중단 (컴포넌트 언마운트 시 자동으로 정리됨)
  };

  // 분석 ID가 바뀔 때 미리보기 비디오 강제 재로딩 (캐시/오디오 버퍼 잔존 방지)
  useEffect(() => {
    if (!analysisResult?.analysisId) return;
    const v = previewRef.current;
    try {
      v?.pause();
      v?.load();
      // 자동 재생 시도 (브라우저 정책상 실패할 수 있으므로 무시)
      v?.play().catch(() => {});
    } catch {}
  }, [analysisResult?.analysisId]);

  // 컴포넌트 언마운트 시 폴링 정리
  useEffect(() => {
    return () => {
      // 컴포넌트 언마운트 시 모든 타이머 정리
      console.log('컴포넌트 언마운트, 폴링 정리');
    };
  }, []);

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
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">사용자 영상 비교 분석</h1>
                <p className="text-sm text-gray-400">두 개의 골프 스윙 영상을 업로드하여 상세한 비교 분석을 받아보세요</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 메인 콘텐츠 */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">

        {/* 분석 폼 */}
        {!analysisResult && (
          <div className="bg-white/10 backdrop-blur-sm rounded-3xl border border-white/20 shadow-xl p-8 mb-8">
            <h2 className="text-2xl font-semibold text-white mb-6">
              📹 영상 업로드 및 분석 설정
            </h2>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-6">
              {/* 첫 번째 영상 업로드 */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium text-white">
                  🎯 첫 번째 영상 (기준)
                </h3>
                <UploadArea
                  onFileUpload={handleVideo1Upload}
                  acceptedFileTypes={['video/*']}
                  maxFileSize={100 * 1024 * 1024} // 100MB
                  placeholder="첫 번째 골프 스윙 영상을 업로드하세요"
                />
                {video1 && (
                  <div className="text-sm text-green-600">
                    ✅ {video1.name} 업로드 완료
                  </div>
                )}
              </div>

              {/* 두 번째 영상 업로드 */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium text-white">
                  🎯 두 번째 영상 (비교)
                </h3>
                <UploadArea
                  onFileUpload={handleVideo2Upload}
                  acceptedFileTypes={['video/*']}
                  maxFileSize={100 * 1024 * 1024} // 100MB
                  placeholder="두 번째 골프 스윙 영상을 업로드하세요"
                />
                {video2 && (
                  <div className="text-sm text-green-600">
                    ✅ {video2.name} 업로드 완료
                  </div>
                )}
              </div>
            </div>

             {/* 분석 시작 버튼 */}
             <div className="text-center">
               {/* 디버깅 정보 */}
               <div className="mb-4 p-3 bg-white/10 backdrop-blur-sm rounded-lg text-sm text-gray-300 border border-white/20">
                 <p><strong>첫 번째 영상:</strong> {video1 ? `✅ ${video1.name}` : '❌ 없음'}</p>
                 <p><strong>두 번째 영상:</strong> {video2 ? `✅ ${video2.name}` : '❌ 없음'}</p>
               </div>
              
              <button
                onClick={startAnalysis}
                disabled={!video1 || !video2 || isAnalyzing}
                className="px-8 py-4 bg-gradient-to-r from-green-500 to-blue-600 text-white font-semibold rounded-xl shadow-lg hover:from-green-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
              >
                {isAnalyzing ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    분석 중...
                  </span>
                ) : (
                  '🚀 분석 시작하기'
                )}
              </button>
              
              {/* 버튼 비활성화 이유 표시 */}
              {(!video1 || !video2) && (
                <div className="mt-2 text-sm text-gray-400">
                  {!video1 && !video2 && '두 개의 영상을 모두 업로드해주세요.'}
                  {video1 && !video2 && '두 번째 영상을 업로드해주세요.'}
                  {!video1 && video2 && '첫 번째 영상을 업로드해주세요.'}
                </div>
              )}
            </div>

            {/* 오류 메시지 */}
            {error && (
              <div className="mt-4 p-4 bg-red-500/20 backdrop-blur-sm border border-red-400/30 rounded-lg">
                <p className="text-red-300 text-center">{error}</p>
                {analysisId && (
                  <div className="mt-2 text-sm text-gray-300 text-center">
                    <p>분석 ID: {analysisId}</p>
                    <p>디버깅 정보: <a href={`${process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'}/debug/analysis/${analysisId}`} target="_blank" className="text-blue-400 hover:underline">Redis 데이터 확인</a></p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* 분석 결과 */}
        {analysisResult && (
          <div className="space-y-8">
            {/* 점수 결과 */}
            <div className="bg-white/10 backdrop-blur-sm rounded-3xl border border-white/20 shadow-xl p-8">
              <h2 className="text-2xl font-semibold text-white mb-6">
                📊 분석 결과
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div className="text-center p-6 bg-gradient-to-br from-blue-500/20 to-blue-600/30 backdrop-blur-sm rounded-xl border border-blue-400/30">
                  <div className="text-3xl font-bold text-blue-300 mb-2">
                    {analysisResult.scores?.angle_score?.toFixed(1)}%
                  </div>
                  <div className="text-sm text-blue-200">관절 각도 점수</div>
                </div>
                
                <div className="text-center p-6 bg-gradient-to-br from-green-500/20 to-green-600/30 backdrop-blur-sm rounded-xl border border-green-400/30">
                  <div className="text-3xl font-bold text-green-300 mb-2">
                    {analysisResult.scores?.speed_score?.toFixed(1)}%
                  </div>
                  <div className="text-sm text-green-200">속도 타이밍 점수</div>
                </div>
                
                <div className="text-center p-6 bg-gradient-to-br from-purple-500/20 to-purple-600/30 backdrop-blur-sm rounded-xl border border-purple-400/30">
                  <div className="text-3xl font-bold text-purple-300 mb-2">
                    {analysisResult.scores?.final_score?.toFixed(1)}%
                  </div>
                  <div className="text-sm text-purple-200">최종 종합 점수</div>
                </div>
              </div>

              <div className="text-center space-y-4">
                {/* 네비게이션 버튼들 */}
                <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                  <a
                    href="/"
                    className="px-6 py-3 bg-gradient-to-r from-green-500 to-blue-600 text-white font-medium rounded-lg hover:from-green-600 hover:to-blue-700 transition-all duration-200 transform hover:scale-105 shadow-lg"
                  >
                    🏠 홈으로 돌아가기
                  </a>
                  <a
                    href="/pro-comparison"
                    className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-200 transform hover:scale-105 shadow-lg"
                  >
                    🏆 프로 비교 분석
                  </a>
                </div>
                
                {/* 현재 분석 재시작 버튼 */}
                <div className="pt-4 border-t border-white/20">
                  <button
                    onClick={resetAnalysis}
                    className="px-6 py-3 bg-white/10 backdrop-blur-sm text-white font-medium rounded-lg hover:bg-white/20 border border-white/30 transition-all duration-200"
                  >
                    🔄 새로운 사용자 비교 분석
                  </button>
                </div>
              </div>
            </div>

            {/* 서버 합성 영상 자동 재생 미리보기 */}
            {analysisResult.comparisonVideoUrl && (
              <div className="bg-white/10 backdrop-blur-sm rounded-3xl border border-white/20 shadow-xl p-8">
                <h3 className="text-xl font-semibold text-white mb-4">🎬 합성 영상 미리보기</h3>
                {(() => {
                  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'
                  const mp4Url = `${apiBase}/download/combined/${analysisResult.analysisId}`
                  const webmUrl = `${apiBase}/download/combined-webm/${analysisResult.analysisId}`
                  const raw = analysisResult.comparisonVideoUrl || ''
                  const directBase = raw.startsWith('http') ? raw : `${apiBase}${raw}`
                  return (
                    <div className="w-full">
                      <video
                        ref={previewRef}
                        key={analysisResult.analysisId}
                        className="w-full rounded-lg border border-gray-200"
                        autoPlay
                        loop
                        muted
                        playsInline
                        crossOrigin="anonymous"
                        preload="metadata"
                        controls
                      >
                        {/* 서버가 선택한 URL 우선 시도 (비어있으면 무시) */}
                        {raw && <source src={directBase} />}
                        <source src={webmUrl} type="video/webm" />
                        <source src={mp4Url} type="video/mp4" />
                        {/* 브라우저가 지원하지 않을 때 대체 텍스트 */}
                        브라우저가 HTML5 비디오를 지원하지 않습니다.
                      </video>
                      <p className="mt-2 text-xs text-gray-400">자동 재생 중입니다. 소리가 필요하면 음소거를 해제하세요.</p>
                    </div>
                  )
                })()}
              </div>
            )}

            {/* 스켈레톤 비교 */}
            {analysisResult.userPoses && analysisResult.comparisonPoses && (
              <div className="bg-white/10 backdrop-blur-sm rounded-3xl border border-white/20 shadow-xl p-8">
                <h3 className="text-xl font-semibold text-white mb-4">
                  🦴 포즈 스켈레톤 비교
                </h3>
                {/* 디버깅 정보 */}
                <div className="mb-4 p-3 bg-white/10 backdrop-blur-sm rounded-lg text-sm text-gray-300 border border-white/20">
                  <p><strong>첫 번째 영상:</strong> {video1 ? `✅ ${video1.name}` : '❌ 없음'}</p>
                  <p><strong>두 번째 영상:</strong> {video2 ? `✅ ${video2.name}` : '❌ 없음'}</p>
                  <p><strong>사용자 포즈 데이터:</strong> {analysisResult.userPoses ? `✅ ${analysisResult.userPoses.length}개 프레임` : '❌ 없음'}</p>
                  <p><strong>비교 포즈 데이터:</strong> {analysisResult.comparisonPoses ? `✅ ${analysisResult.comparisonPoses.length}개 프레임` : '❌ 없음'}</p>
                </div>
                <SideBySidePlayer
                  leftVideoUrl={video1 ? URL.createObjectURL(video1) : ''}
                  rightVideoUrl={video2 ? URL.createObjectURL(video2) : ''}
                  leftPoses={analysisResult.userPoses}
                  rightPoses={analysisResult.comparisonPoses}
                />
              </div>
            )}

            {/* 비교 영상 다운로드 */}
            {analysisResult.comparisonVideoUrl && (
              <div className="bg-white/10 backdrop-blur-sm rounded-3xl border border-white/20 shadow-xl p-8">
                <h3 className="text-xl font-semibold text-white mb-4">
                  🎬 비교 영상 다운로드
                </h3>
                {(() => {
                  const raw = analysisResult.comparisonVideoUrl || ''
                  const apiBase2 = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'
                  const base = raw.startsWith('http') ? raw : `${apiBase2}${raw}`
                  const bust = base
                  return (
                    <MergeDownload
                      videoUrl={bust}
                      filename={`comparison_${analysisResult.analysisId}.mp4`}
                    />
                  )
                })()}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
