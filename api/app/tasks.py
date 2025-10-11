"""
SSSwing3 Celery 태스크 모듈

이 모듈은 골프 스윙 분석을 위한 백그라운드 작업을 처리합니다.
Celery를 사용하여 비동기적으로 영상 분석을 수행하며, 다음과 같은 기능을 제공합니다:

1. 사용자 스윙 비교 분석
2. 영상 전처리 (슬로우 다운, 크롭핑)
3. 합성 영상 생성
4. 점수 계산 및 결과 저장

주요 특징:
- 초고속 스윙 대응을 위한 2배 슬로우 다운 변환
- MediaPipe 기반 스윙 키 프레임 검출
- FFmpeg를 우선 사용하고 폴백으로 OpenCV 사용
- WebM 변환을 통한 브라우저 호환성 향상
"""

from .celery_app import celery_app
from .video_processor import detect_swing_key_frames, crop_video_from_frame
from .scorer import score_all
from ssswing.video_generator import generate_comparison_video, create_dummy_video
import os
import shutil
import uuid
import json
import time
from datetime import datetime


@celery_app.task(name='app.tasks.run_user_comparison')
def run_user_comparison(video1_path: str, video2_path: str, project_root: str) -> dict:
    """
    사용자 비교 분석을 수행하고 결과 JSON을 반환하는 Celery 태스크
    
    Args:
        video1_path: 첫 번째 사용자 영상 경로
        video2_path: 두 번째 사용자 영상 경로
        project_root: 프로젝트 루트 디렉토리 경로
        
    Returns:
        dict: 분석 결과를 포함한 딕셔너리
        
    처리 단계:
        1. 초고속 스윙 대응을 위한 슬로우 다운 변환
        2. 스윙 시작점 검출 및 영상 크롭핑
        3. 합성 영상 생성 (MP4 + WebM)
        4. 점수 계산 및 결과 저장
    """

    # 0) 초고속 스윙 대응: 입력을 2배 느리게 변환한 사본으로 분석 수행
    #    프리뷰도 느린 파일을 사용하도록 output은 동일 규칙 유지
    
    # 분석 시작 시간 기록
    start_time = time.time()
    start_datetime = datetime.now()
    
    # 임시 디렉토리 및 분석 ID 추출
    temp_dir = os.path.dirname(video1_path)
    an_id = os.path.basename(temp_dir)
    
    # 진행 상황 추적을 위한 중간 결과 파일 생성
    progress_file = os.path.join(temp_dir, "progress.json")
    
    def update_progress(step: str, progress: int, elapsed: float = None):
        """진행 상황 업데이트 헬퍼 함수"""
        try:
            current_elapsed = time.time() - start_time
            with open(progress_file, "w", encoding="utf-8") as pf:
                json.dump({
                    "status": "processing",
                    "start_time": start_datetime.isoformat(),
                    "current_step": step,
                    "progress": progress,
                    "elapsed_time": round(current_elapsed, 1),
                    "step_elapsed": round(elapsed or 0, 1)
                }, pf, ensure_ascii=False)
        except Exception:
            pass
    
    # 초기 진행 상황 설정
    update_progress("분석 시작", 0)
    
    # 1) 슬로우 다운 변환 제거 - 원본 영상 직접 사용
    print("[INFO] 슬로우 다운 변환 생략 - 원본 영상 직접 사용")
    video1_path_slow = video1_path
    video2_path_slow = video2_path

    # 2) 스윙 키 프레임 검출 (시간 측정 포함)
    step_start = time.time()
    update_progress("스윙 키 프레임 검출 시작", 10)
    
    try:
        # 캐시 파일 경로 설정
        cache_b = os.path.join(temp_dir, f"keyframes_b_{an_id}.json")
        cache_t = os.path.join(temp_dir, f"keyframes_t_{an_id}.json")
        
        # 첫 번째 영상 키 프레임 검출 (캐시 확인)
        if os.path.exists(cache_b):
            print("[KeyFrames] 첫 번째 영상 캐시 사용")
            with open(cache_b, 'r') as f:
                cached_data = json.load(f)
                start_abs_b = cached_data.get('start_abs', 0)
        else:
            print("[KeyFrames] 첫 번째 영상 키 프레임 검출 중...")
            _, _, _, start_abs_b = detect_swing_key_frames(video1_path_slow)
            print(f"[KeyFrames] 첫 번째 영상 검출 완료: start={start_abs_b}")
            # 캐시 저장
            with open(cache_b, 'w') as f:
                json.dump({'start_abs': start_abs_b}, f)
        
        # 두 번째 영상 키 프레임 검출 (캐시 확인)
        if os.path.exists(cache_t):
            print("[KeyFrames] 두 번째 영상 캐시 사용")
            with open(cache_t, 'r') as f:
                cached_data = json.load(f)
                start_abs_t = cached_data.get('start_abs', 0)
        else:
            print("[KeyFrames] 두 번째 영상 키 프레임 검출 중...")
            _, _, _, start_abs_t = detect_swing_key_frames(video2_path_slow)
            print(f"[KeyFrames] 두 번째 영상 검출 완료: start={start_abs_t}")
            # 캐시 저장
            with open(cache_t, 'w') as f:
                json.dump({'start_abs': start_abs_t}, f)
        
        # 스윙 시작(웨글 제외) 직전 1프레임부터 시작하도록 프리롤 적용
        # 환경변수 PREROLL_FRAMES로 조정 가능 (기본 1)
        try:
            preroll_frames = max(0, int(os.getenv("PREROLL_FRAMES", "1")))
        except Exception:
            preroll_frames = 1
        
        # 키 프레임 검출 시간 측정
        keyframes_elapsed = time.time() - step_start
        update_progress("스윙 키 프레임 검출 완료", 25, keyframes_elapsed)
        print(f"[TIMING] 스윙 키 프레임 검출: {keyframes_elapsed:.1f}초")
        
        start_abs_b = max(0, int(start_abs_b) - preroll_frames)
        start_abs_t = max(0, int(start_abs_t) - preroll_frames)
        print(f"[INFO] PREROLL_FRAMES={preroll_frames} → 시작 프레임(B/T)={start_abs_b}/{start_abs_t}")
        
        print(f"[INFO] 스윙 시작 직전 프레임부터 시작: baseline 시작점={start_abs_b}, target 시작점={start_abs_t}")
        
        # 피니시 절대 프레임 계산 로직은 옵션A 롤백에 따라 제거
    except Exception as e:
        print(f"[WARNING] 스윙 시작점 감지 실패: {e} → 기본값 사용")
        # 검출 실패 시 기본값 사용 (스윙 시작 직전을 고려하여 충분히 앞에서 시작)
        start_abs_b, start_abs_t = 45, 45  # 1.5초 * 30fps

    # 임시 디렉토리 및 분석 ID 재확인
    temp_dir = os.path.dirname(video1_path)
    an_id = os.path.basename(temp_dir)

    # 크롭핑된 영상 파일 경로 설정
    cropped_b = os.path.join(temp_dir, f"video1_cropped_{an_id}.mp4")
    cropped_t = os.path.join(temp_dir, f"video2_cropped_{an_id}.mp4")
    
    # 영상 크롭핑 수행 (시작점부터 끝까지)
    ok_b = crop_video_from_frame(video1_path_slow, start_abs_b, cropped_b)
    ok_t = crop_video_from_frame(video2_path_slow, start_abs_t, cropped_t)
    
    # 크롭핑 성공 시 크롭된 파일 사용, 실패 시 원본 사용
    baseline_path = cropped_b if ok_b else video1_path_slow
    target_path = cropped_t if ok_t else video2_path_slow

    # 2) 합성 영상 생성 (크롭된 영상 사용하여 웨글 부분 제거)
    out_dir = os.path.join(project_root, "ssswing", "mp4")
    os.makedirs(out_dir, exist_ok=True)
    output = os.path.join(out_dir, f"combined_swing_{an_id}.mp4")
    
    # 기준 영상과 타겟 영상을 나란히 배치한 합성 영상 생성
    try:
        # 크롭된 영상을 사용하여 웨글 부분이 제거된 영상으로 합성
        success = generate_comparison_video(
            video1_path=baseline_path,  # 크롭된 영상 사용 (웨글 부분 제거됨)
            video2_path=target_path,    # 크롭된 영상 사용 (웨글 부분 제거됨)
            output_path=output,
            target_fps=30.0,
            max_duration=45.0  # 피니시까지 완전한 스윙 보장 (충분한 여유)
        )
        
        if success:
            print(f"[INFO] 합성 영상 생성 완료: {output}")
        else:
            print(f"[WARN] 합성 영상 생성 실패, 더미 영상 생성")
            # 실패 시 더미 영상 생성
            create_dummy_video(output, duration=5.0, fps=30.0)
        
        # 진행 상황 업데이트
        try:
            elapsed_time = time.time() - start_time
            with open(progress_file, "w", encoding="utf-8") as pf:
                json.dump({
                    "status": "processing",
                    "start_time": start_datetime.isoformat(),
                    "current_step": "합성 영상 생성 완료",
                    "progress": 50,
                    "elapsed_time": round(elapsed_time, 1)
                }, pf, ensure_ascii=False)
        except Exception:
            pass
        
    except Exception as e:
        print(f"[ERROR] 합성 영상 생성 실패: {e}")
        # 예외 발생 시 더미 영상 생성
        try:
            create_dummy_video(output, duration=5.0, fps=30.0)
            print(f"[INFO] 예외 발생으로 더미 영상 생성: {output}")
        except Exception as e2:
            print(f"[ERROR] 더미 영상 생성도 실패: {e2}")
    
    # 2-1) 브라우저 호환을 위한 WebM 변환 시도 (성공 시 webm 우선 사용)
    webm_path = os.path.join(out_dir, f"combined_swing_{an_id}.webm")
    use_webm = False
    
    try:
        import cv2
        # MP4 영상 정보 읽기
        cap = cv2.VideoCapture(output)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # WebM 인코더 설정 (VP90 코덱)
            fourcc = cv2.VideoWriter_fourcc(*'VP90')
            webm_out = cv2.VideoWriter(webm_path, fourcc, fps, (w, h))
            
            # 프레임별 변환
            frame_written = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                webm_out.write(frame)
                frame_written += 1
            
            # 리소스 해제
            cap.release()
            webm_out.release()
            
            # 변환 성공 여부 확인
            use_webm = frame_written > 0 and os.path.exists(webm_path)
    except Exception:
        use_webm = False

    # 3) 점수 계산
    # 기준 영상과 타겟 영상을 비교하여 점수 계산
    print("[INFO] 점수 계산 시작...")
    try:
        s = score_all(baseline_path, target_path, use_no_top_method=True)
        print(f"[INFO] 점수 계산 완료: {s}")
    except Exception as e:
        print(f"[ERROR] 점수 계산 실패: {e}")
        import traceback
        traceback.print_exc()
        s = {"joint_angle_score": 0.0, "swing_speed_score": 0.0, "total_score": 0.0}
    
    # 3-1) 키 이미지 저장 (프로/유저 각각 3장씩)
    try:
        # 프로젝트 루트를 Python 경로에 추가
        import sys
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # ssswing 패키지에서 직접 import
        from ssswing import save_pro_user_key_images, extract_landmarks_from_video, detect_swing_phases
        
        print("[INFO] 키 이미지 저장 시작...")
        
        # 스윙 단계 감지 (키 이미지 저장용)
        try:
            pro_landmarks, pro_aspect_ratio = extract_landmarks_from_video(baseline_path)
            user_landmarks, user_aspect_ratio = extract_landmarks_from_video(target_path)
            
            # video_path 전달로 샤프트 CV 보조 사용
            pro_phases = detect_swing_phases(pro_landmarks, video_aspect_ratio=pro_aspect_ratio, video_path=baseline_path)
            user_phases = detect_swing_phases(user_landmarks, video_aspect_ratio=user_aspect_ratio, video_path=target_path)
            
            print(f"[DEBUG] 프로 phases: {pro_phases}")
            print(f"[DEBUG] 유저 phases: {user_phases}")
            
        except Exception as e:
            print(f"[WARNING] 스윙 단계 감지 실패: {e}")
            # 기본값 사용
            pro_phases = {"address": 0, "top": 25, "finish": 50}
            user_phases = {"address": 0, "top": 22, "finish": 45}
        
        # 키 이미지 저장 디렉토리 설정
        key_images_dir = os.path.join(project_root, "ssswing", "swing_key_images")
        
        # 이미지 저장 실행
        saved_images = save_pro_user_key_images(
            baseline_path, target_path, 
            pro_phases, user_phases, 
            key_images_dir
        )
        
        print(f"[SUCCESS] 키 이미지 저장 완료: {len(saved_images)}장")
        
        # 진행 상황 업데이트
        try:
            elapsed_time = time.time() - start_time
            with open(progress_file, "w", encoding="utf-8") as pf:
                json.dump({
                    "status": "processing",
                    "start_time": start_datetime.isoformat(),
                    "current_step": "키 이미지 저장 완료",
                    "progress": 70,
                    "elapsed_time": round(elapsed_time, 1)
                }, pf, ensure_ascii=False)
        except Exception:
            pass
        
    except Exception as e:
        print(f"[ERROR] 키 이미지 저장 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 속도/프레임/초 단위 메타 로그 출력 (Celery 콘솔로 바로 보이도록 print 사용)
    speed_meta = s.get("speed_meta") or {}
    try:
        # 속도 메타데이터에서 필요한 정보 추출
        b = speed_meta.get("times", {}).get("baseline", {})      # 기준 영상 시간 정보
        t = speed_meta.get("times", {}).get("target", {})        # 타겟 영상 시간 정보
        bf = speed_meta.get("frames", {}).get("baseline", {})    # 기준 영상 프레임 정보
        tf = speed_meta.get("frames", {}).get("target", {})      # 타겟 영상 프레임 정보
        ratios = speed_meta.get("ratios", {})                     # 비율 정보
        seg = speed_meta.get("segment_scores", {})               # 구간별 점수
        
        # 기준 영상 속도 정보 로그 출력
        print(
            f"[Speed] baseline S/T/F: {bf.get('start')}(" \
            f"{b.get('start_s', 0):.3f}s)/{bf.get('top')}({b.get('top_s', 0):.3f}s)/" \
            f"{bf.get('finish')}({b.get('finish_s', 0):.3f}s) d1={b.get('d1_s', 0):.3f}s d2={b.get('d2_s', 0):.3f}s"
        )
        
        # 타겟 영상 속도 정보 및 점수 로그 출력
        print(
            f"[Speed] target  S/T/F: {tf.get('start')}({t.get('start_s', 0):.3f}s)/" \
            f"{tf.get('top')}({t.get('top_s', 0):.3f}s)/{tf.get('finish')}({t.get('finish_s', 0):.3f}s) " \
            f"d1={t.get('d1_s', 0):.3f}s d2={t.get('d2_s', 0):.3f}s | " \
            f"r1={ratios.get('r1', 0):.3f} r2={ratios.get('r2', 0):.3f} " \
            f"s1={seg.get('s1', 0):.1f} s2={seg.get('s2', 0):.1f} final={speed_meta.get('final_speed_score', 0):.1f}"
        )
    except Exception:
        pass
    
    # 점수 정보 구성
    scores = {
        "angle_score": s.get("joint_angle_score", 0.0),      # 관절 각도 점수
        "speed_score": s.get("swing_speed_score", 0.0),      # 스윙 속도 점수
        "final_score": s.get("total_score", 0.0)             # 종합 점수
    }

    # 최종 결과 구성
    print("[INFO] 최종 결과 구성 시작...")
    result = {
        "status": "completed",                                    # 상태: 완료
        "analysisId": an_id,                                      # 분석 ID
        "scores": scores,                                         # 점수 정보
        "comparisonVideoUrl": f"/api/download/combined-webm/{an_id}" if use_webm else f"/api/download/combined/{an_id}",  # 비교 영상 URL
        "description": "사용자 영상 비교 분석",                    # 설명
        "speedMeta": speed_meta,                                  # 속도 메타데이터
    }
    print(f"[INFO] 최종 결과 구성 완료: {result}")

    # 4) 결과 저장 (프런트엔드 폴링 호환)
    result_json_path = os.path.join(temp_dir, "result.json")
    try:
        # 결과를 JSON 파일로 저장
        with open(result_json_path, "w", encoding="utf-8") as rf:
            json.dump(result, rf, ensure_ascii=False)
    except Exception:
        pass
    
    # 최종 진행 상황 업데이트 (완료)
    try:
        elapsed_time = time.time() - start_time
        with open(progress_file, "w", encoding="utf-8") as pf:
            json.dump({
                "status": "completed",
                "start_time": start_datetime.isoformat(),
                "current_step": "분석 완료",
                "progress": 100,
                "elapsed_time": round(elapsed_time, 1),
                "total_time": round(elapsed_time, 1)
            }, pf, ensure_ascii=False)
    except Exception:
        pass

    print(f"[SUCCESS] 분석 완료! 결과 반환: {result}")
    return result


