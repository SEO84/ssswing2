'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';

interface PoseData {
  frame: number;
  landmarks: { [key: string]: [number, number, number] };
}

interface SideBySidePlayerProps {
  leftVideoUrl: string;
  rightVideoUrl: string;
  leftPoses?: PoseData[];
  rightPoses?: PoseData[];
  onAnalysisComplete?: (scores: any) => void;
}

const SideBySidePlayer: React.FC<SideBySidePlayerProps> = ({
  leftVideoUrl,
  rightVideoUrl,
  leftPoses = [],
  rightPoses = [],
  onAnalysisComplete
}) => {
  const leftVideoRef = useRef<HTMLVideoElement>(null);
  const rightVideoRef = useRef<HTMLVideoElement>(null);
  const leftCanvasRef = useRef<HTMLCanvasElement>(null);
  const rightCanvasRef = useRef<HTMLCanvasElement>(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(0.5); // 0.25에서 0.5로 변경하여 원래대로 복원
  const [repeatCount, setRepeatCount] = useState(0);
  const [currentRepeat, setCurrentRepeat] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [showPoses, setShowPoses] = useState(true);
  
  // 스켈레톤 그리기 설정
  const [skeletonColor, setSkeletonColor] = useState('#00FF00'); // 초록색
  const [landmarkRadius, setLandmarkRadius] = useState(8); // 5에서 8로 증가하여 더 진하게
  const [connectionThickness, setConnectionThickness] = useState(4); // 2에서 4로 증가하여 더 진하게

  // 포즈 랜드마크 그리기 함수 (MediaPipe 대신 직접 구현)
  const drawPoseLandmarks = useCallback((ctx: CanvasRenderingContext2D, poses: PoseData[], frame: number, color: string) => {
    if (!poses || poses.length === 0) return;
    
    const currentPose = poses[frame] || poses[poses.length - 1];
    if (!currentPose || !currentPose.landmarks) return;

    const canvas = ctx.canvas;
    const landmarks = currentPose.landmarks;

    // 랜드마크 그리기 (더 안정적인 렌더링)
    Object.entries(landmarks).forEach(([key, [x, y, z]]) => {
      // 좌표 유효성 검사
      if (x < 0 || x > 1 || y < 0 || y > 1 || isNaN(x) || isNaN(y)) return;
      
      const pixelX = x * canvas.width;
      const pixelY = y * canvas.height;
      
      // 더 큰 원으로 그리기
      ctx.beginPath();
      ctx.arc(pixelX, pixelY, landmarkRadius * 1.5, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // 골프 스윙 관련 연결선 그리기
    const connections = [
      ['left_shoulder', 'right_shoulder'],
      ['left_shoulder', 'left_elbow'],
      ['right_shoulder', 'right_elbow'],
      ['left_elbow', 'left_wrist'],
      ['right_elbow', 'right_wrist'],
      ['left_shoulder', 'left_hip'],
      ['right_shoulder', 'right_hip'],
      ['left_hip', 'right_hip'],
      ['left_hip', 'left_knee'],
      ['right_hip', 'right_knee'],
      ['left_knee', 'left_ankle'],
      ['right_knee', 'right_ankle']
    ];

    ctx.strokeStyle = color;
    ctx.lineWidth = connectionThickness;

    connections.forEach(([start, end]) => {
      const startLandmark = landmarks[start];
      const endLandmark = landmarks[end];
      
      if (startLandmark && endLandmark) {
        const [startX, startY, startZ] = startLandmark;
        const [endX, endY, endZ] = endLandmark;
        
        // 좌표 유효성 검사
        if (startX < 0 || startX > 1 || startY < 0 || startY > 1 || 
            endX < 0 || endX > 1 || endY < 0 || endY > 1 ||
            isNaN(startX) || isNaN(startY) || isNaN(endX) || isNaN(endY)) return;
        
        const pixelStartX = startX * canvas.width;
        const pixelStartY = startY * canvas.height;
        const pixelEndX = endX * canvas.width;
        const pixelEndY = endY * canvas.height;
        
        ctx.beginPath();
        ctx.moveTo(pixelStartX, pixelStartY);
        ctx.lineTo(pixelEndX, pixelEndY);
        ctx.stroke();
      }
    });
  }, [landmarkRadius, connectionThickness]);

  // 비디오 프레임 업데이트
  const updateFrame = useCallback(() => {
    const leftVideo = leftVideoRef.current;
    const rightVideo = rightVideoRef.current;
    const leftCanvas = leftCanvasRef.current;
    const rightCanvas = rightCanvasRef.current;

    if (!leftVideo || !rightVideo || !leftCanvas || !rightCanvas) return;

    const leftCtx = leftCanvas.getContext('2d');
    const rightCtx = rightCanvas.getContext('2d');

    if (!leftCtx || !rightCtx) return;

    // 캔버스 크기 설정
    leftCanvas.width = leftVideo.videoWidth;
    leftCanvas.height = leftVideo.videoHeight;
    rightCanvas.width = rightVideo.videoWidth;
    rightCanvas.height = rightVideo.videoHeight;

    // 비디오 그리기
    leftCtx.drawImage(leftVideo, 0, 0, leftCanvas.width, leftCanvas.height);
    rightCtx.drawImage(rightVideo, 0, 0, rightCanvas.width, rightCanvas.height);

    // 포즈 랜드마크 그리기
    if (showPoses) {
      const leftLen = Array.isArray(leftPoses) ? leftPoses.length : 0;
      const rightLen = Array.isArray(rightPoses) ? rightPoses.length : 0;
      const baseLen = Math.max(leftLen, rightLen, 1);
      const ratio = leftVideo.duration ? (leftVideo.currentTime / leftVideo.duration) : 0;
      const currentFrameIndex = Math.min(baseLen - 1, Math.max(0, Math.floor(ratio * baseLen)));
      setCurrentFrame(currentFrameIndex);
      
      drawPoseLandmarks(leftCtx, leftPoses || [], currentFrameIndex, skeletonColor);
      drawPoseLandmarks(rightCtx, rightPoses || [], currentFrameIndex, skeletonColor);
    }
  }, [leftPoses, rightPoses, showPoses, skeletonColor, drawPoseLandmarks]);

  // 비디오 이벤트 핸들러
  const setupVideoEvents = useCallback(() => {
    const leftVideo = leftVideoRef.current;
    const rightVideo = rightVideoRef.current;

    if (!leftVideo || !rightVideo) return;

    const onTimeUpdate = () => {
      if (isPlaying) {
        updateFrame();
      }
    };

    const onPlay = () => {
      setIsPlaying(true);
      // applyPlaybackRate(); // 삭제됨
    };

    const onPause = () => {
      setIsPlaying(false);
    };

    const onEnded = () => {
      setIsPlaying(false);
      if (currentRepeat < repeatCount) {
        setCurrentRepeat(prev => prev + 1);
        leftVideo.currentTime = 0;
        rightVideo.currentTime = 0;
        leftVideo.play();
        rightVideo.play();
      }
    };

    // 비디오 로드 완료 후 재생 속도 적용
    const onLoadedMetadata = () => {
      const v1 = leftVideo as HTMLVideoElement;
      const v2 = rightVideo as HTMLVideoElement;
      v1.playbackRate = playbackSpeed;
      v2.playbackRate = playbackSpeed;
    };
    // 기타 속도 보강 이벤트 제거

    leftVideo.addEventListener('timeupdate', onTimeUpdate);
    rightVideo.addEventListener('timeupdate', onTimeUpdate);
    leftVideo.addEventListener('play', onPlay);
    rightVideo.addEventListener('play', onPlay);
    leftVideo.addEventListener('pause', onPause);
    rightVideo.addEventListener('pause', onPause);
    leftVideo.addEventListener('ended', onEnded);
    rightVideo.addEventListener('ended', onEnded);
    leftVideo.addEventListener('loadedmetadata', onLoadedMetadata);
    rightVideo.addEventListener('loadedmetadata', onLoadedMetadata);
    // 기타 속도 보강 이벤트 제거

    return () => {
      leftVideo.removeEventListener('timeupdate', onTimeUpdate);
      rightVideo.removeEventListener('timeupdate', onTimeUpdate);
      leftVideo.removeEventListener('play', onPlay);
      rightVideo.removeEventListener('play', onPlay);
      leftVideo.removeEventListener('pause', onPause);
      rightVideo.removeEventListener('pause', onPause);
      leftVideo.removeEventListener('ended', onEnded);
      rightVideo.removeEventListener('ended', onEnded);
      leftVideo.removeEventListener('loadedmetadata', onLoadedMetadata);
      rightVideo.removeEventListener('loadedmetadata', onLoadedMetadata);
      // 기타 속도 보강 이벤트 제거
    };
  }, [isPlaying, currentRepeat, repeatCount, updateFrame, playbackSpeed]);

  // 재생/일시정지 토글
  const togglePlay = () => {
    const leftVideo = leftVideoRef.current;
    const rightVideo = rightVideoRef.current;

    if (leftVideo && rightVideo) {
      if (isPlaying) {
        leftVideo.pause();
        rightVideo.pause();
      } else {
        leftVideo.play();
        rightVideo.play();
      }
    }
  };

  // 재생 속도 변경
  const changePlaybackSpeed = (speed: number) => {
    const leftVideo = leftVideoRef.current;
    const rightVideo = rightVideoRef.current;

    // 선택된 최신 speed로 즉시 적용 (state 반영 대기 없이)
    if (leftVideo) leftVideo.playbackRate = speed;
    if (rightVideo) rightVideo.playbackRate = speed;

    setPlaybackSpeed(speed);

    // 확인 로그
    setTimeout(() => {
      if (leftVideo) console.log('변경 후 왼쪽 비디오 재생 속도:', leftVideo.playbackRate);
      if (rightVideo) console.log('변경 후 오른쪽 비디오 재생 속도:', rightVideo.playbackRate);
    }, 100);
  };

  // 반복 재생 설정
  const setRepeat = (count: number) => {
    setRepeatCount(count);
    setCurrentRepeat(0);
  };

  // 비디오 이벤트 설정
  useEffect(() => {
    const cleanup = setupVideoEvents();
    return cleanup;
  }, [setupVideoEvents]);

  // playbackSpeed 변경 시 요소에 즉시 반영
  useEffect(() => {
    const v1 = leftVideoRef.current;
    const v2 = rightVideoRef.current;
    if (v1) v1.playbackRate = playbackSpeed;
    if (v2) v2.playbackRate = playbackSpeed;
  }, [playbackSpeed]);

  // 비디오 이벤트 설정

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white rounded-xl shadow-lg">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">포즈 스켈레톤 비교</h2>
        <p className="text-gray-600">서버 분석 결과 기반 포즈 스켈레톤 표시</p>
      </div>

      {/* 비디오 플레이어 컨테이너 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* 왼쪽 비디오 (프로 영상) */}
        <div className="relative">
          <h3 className="text-lg font-semibold text-gray-700 mb-3">프로 영상</h3>
          <div className="relative bg-black rounded-lg overflow-hidden">
            <video
              ref={leftVideoRef}
              src={leftVideoUrl}
              className="w-full h-auto"
              muted
              playsInline
            />
            <canvas
              ref={leftCanvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />
          </div>
        </div>

        {/* 오른쪽 비디오 (사용자 영상) */}
        <div className="relative">
          <h3 className="text-lg font-semibold text-gray-700 mb-3">사용자 영상</h3>
          <div className="relative bg-black rounded-lg overflow-hidden">
            <video
              ref={rightVideoRef}
              src={rightVideoUrl}
              className="w-full h-auto"
              muted
              playsInline
            />
            <canvas
              ref={rightCanvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />
          </div>
        </div>
      </div>

      {/* 컨트롤 패널 */}
      <div className="bg-gray-50 rounded-lg p-4 mb-6">
        <div className="flex flex-wrap items-center gap-4">
          {/* 재생/일시정지 버튼 */}
          <button
            onClick={togglePlay}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            {isPlaying ? '일시정지' : '재생'}
          </button>

          {/* 재생 속도 조절 */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">재생 속도:</span>
            <select
              value={playbackSpeed}
              onChange={(e) => changePlaybackSpeed(Number(e.target.value))}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm"
            >
              <option value={0.25}>0.25x</option>
              <option value={0.5}>0.5x (기본)</option>
              <option value={1}>1x</option>
              <option value={1.5}>1.5x</option>
              <option value={2}>2x</option>
            </select>
          </div>

          {/* 반복 설정 */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">반복:</span>
            <select
              value={repeatCount}
              onChange={(e) => setRepeat(Number(e.target.value))}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm"
            >
              <option value={0}>0회</option>
              <option value={1}>1회</option>
              <option value={2}>2회</option>
              <option value={3}>3회</option>
            </select>
            {repeatCount > 0 && (
              <span className="text-sm text-gray-500">
                ({currentRepeat}/{repeatCount})
              </span>
            )}
          </div>

          {/* 포즈 표시 토글 */}
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="showPoses"
              checked={showPoses}
              onChange={(e) => setShowPoses(e.target.checked)}
              className="w-4 h-4"
            />
            <label htmlFor="showPoses" className="text-sm text-gray-600">
              포즈 표시
            </label>
          </div>

          {/* 현재 프레임 정보 */}
          <div className="text-sm text-gray-600">
            프레임: {currentFrame + 1} / {Math.max(Array.isArray(leftPoses)?leftPoses.length:0, Array.isArray(rightPoses)?rightPoses.length:0)}
          </div>
        </div>
      </div>

      {/* 스켈레톤 설정 패널 */}
      <div className="bg-gray-50 rounded-lg p-4 mb-6">
        <h4 className="text-lg font-semibold text-gray-700 mb-3">스켈레톤 설정</h4>
        <div className="flex flex-wrap items-center gap-4">
          {/* 색상 설정 */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">색상:</span>
            <input
              type="color"
              value={skeletonColor}
              onChange={(e) => setSkeletonColor(e.target.value)}
              className="w-10 h-8 border border-gray-300 rounded cursor-pointer"
            />
          </div>

          {/* 랜드마크 크기 */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">랜드마크 크기:</span>
            <input
              type="range"
              min="3"
              max="10"
              value={landmarkRadius}
              onChange={(e) => setLandmarkRadius(Number(e.target.value))}
              className="w-20"
            />
            <span className="text-sm text-gray-500 w-8">{landmarkRadius}</span>
          </div>

          {/* 연결선 두께 */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">연결선 두께:</span>
            <input
              type="range"
              min="1"
              max="5"
              value={connectionThickness}
              onChange={(e) => setConnectionThickness(Number(e.target.value))}
              className="w-20"
            />
            <span className="text-sm text-gray-500 w-8">{connectionThickness}</span>
          </div>
        </div>
      </div>

      {/* 상태 정보 */}
      <div className="text-center text-sm text-gray-500 mb-4">
        {(Array.isArray(leftPoses) && leftPoses.length > 0) && (Array.isArray(rightPoses) && rightPoses.length > 0) ? (
          <span className="text-green-600">✅ 서버 분석 결과 기반 포즈 스켈레톤 표시</span>
        ) : (
          <span className="text-yellow-600">⚠️ 포즈 데이터가 없습니다. 분석을 먼저 실행해주세요.</span>
        )}
      </div>
    </div>
  );
};

export default SideBySidePlayer;


