"""
골프 스윙 분석을 위한 점수 계산 모듈

이 모듈은 골프 스윙 영상을 분석하여 다음과 같은 점수를 계산합니다:
1. 스윙 속도 점수: 기준 영상과 비교하여 스윙 타이밍의 일치도
2. 관절 각도 점수: 기준 영상과 비교하여 자세의 일치도
3. 종합 점수: 위 두 점수의 평균

주요 기능:
- MediaPipe를 사용한 포즈 랜드마크 추출
- 프레임별 관절 각도 계산
- 시간 기반 스윙 구간 분석
- 진행률 기반 동기화 분석
"""

import os
import sys
import logging
import numpy as np
import cv2
import mediapipe as mp

# 현재 파일의 디렉토리 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 경로 설정 (상위 디렉토리 2단계)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
# 프로젝트 루트를 Python 경로에 추가
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# MediaPipe 포즈 솔루션 초기화
mp_pose = mp.solutions.pose
# 로거 설정
logger = logging.getLogger(__name__)
from scipy.interpolate import interp1d
from scipy.signal import resample


def _hands_center_from_pose_landmarks(pose_landmarks):
    try:
        lm = pose_landmarks.landmark
        lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        if getattr(lw, 'visibility', 1.0) < 0.2 or getattr(rw, 'visibility', 1.0) < 0.2:
            return None
        return ((lw.x + rw.x) / 2.0, (lw.y + rw.y) / 2.0)
    except Exception:
        return None


def calc_speed_score_trajectory(landmarks_b, landmarks_t, phases_b, phases_t, fps_b=30, fps_t=30, num_points=20):
    """
    손 궤적 누적 거리 기반 속도 점수 계산 (Address→Finish 한 구간)
    - 기술적 차이 보정: 세그먼트 정렬, 비율 클리핑, 좌표 유사성 보정
    """
    # 1) 세그먼트 정렬 (겹치는 구간 사용)
    sb0, sb1 = int(phases_b.get('segment', [phases_b.get('address', 0), phases_b.get('finish', 0)])[0]), int(phases_b.get('segment', [phases_b.get('address', 0), phases_b.get('finish', 0)])[1])
    st0, st1 = int(phases_t.get('segment', [phases_t.get('address', 0), phases_t.get('finish', 0)])[0]), int(phases_t.get('segment', [phases_t.get('address', 0), phases_t.get('finish', 0)])[1])
    start = max(0, min(max(sb0, st0), min(len(landmarks_b)-1, len(landmarks_t)-1)))
    end = min(max(sb1, st1), min(len(landmarks_b)-1, len(landmarks_t)-1))
    if end - start < 3:
        start, end = min(sb0, st0), max(sb1, st1)
        end = min(end, min(len(landmarks_b)-1, len(landmarks_t)-1))

    # 2) 궤적 추출 (정규화 좌표 사용 → 해상도 의존성 제거)
    def extract_xy(landmarks, s, e):
        xs, ys = [], []
        for i in range(int(s), int(e)+1):
            pl = landmarks[i] if i < len(landmarks) else None
            if pl is None:
                xs.append(np.nan); ys.append(np.nan); continue
            try:
                lm = pl.landmark
                ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                mid_x = (ls.x + rs.x) / 2.0
                mid_y = (ls.y + rs.y) / 2.0
                shoulder_dx = (ls.x - rs.x)
                shoulder_dy = (ls.y - rs.y)
                shoulder_dist = float(np.hypot(shoulder_dx, shoulder_dy))
                # 가시성 체크 및 최소 어깨 폭 설정
                if getattr(ls, 'visibility', 1.0) < 0.3 or getattr(rs, 'visibility', 1.0) < 0.3:
                    shoulder_dist = max(shoulder_dist, 0.1)  # 가시성 낮으면 최소치로 완화
                shoulder_dist = max(shoulder_dist, 0.1)  # 최소 어깨 폭 0.1로 설정

                hc = _hands_center_from_pose_landmarks(pl)
                if hc is None:
                    xs.append(np.nan); ys.append(np.nan)
                else:
                    # 어깨폭(화면상 길이)으로 정규화하고, 어깨중점 기준으로 평행이동 제거
                    xs.append((float(hc[0]) - mid_x) / shoulder_dist)
                    ys.append((float(hc[1]) - mid_y) / shoulder_dist)
            except Exception:
                xs.append(np.nan); ys.append(np.nan)
        return np.column_stack([xs, ys]) if xs else np.zeros((0, 2))

    traj_b = extract_xy(landmarks_b, start, end)
    traj_t = extract_xy(landmarks_t, start, end)
    
    # 궤적 데이터 품질 검사 및 로깅
    logger.info(f"[Speed-Trajectory] Baseline trajectory: {len(traj_b)} points, Target trajectory: {len(traj_t)} points")
    
    if len(traj_b) < 2 or len(traj_t) < 2:
        logger.error(f"[Speed-Trajectory] FAILED - Insufficient trajectory data - Baseline: {len(traj_b)}, Target: {len(traj_t)}")
        logger.error(f"[Speed-Trajectory] REASON - 궤적 데이터 부족 (최소 2개 포인트 필요)")
        return None, {"error": "insufficient trajectory", "baseline_points": len(traj_b), "target_points": len(traj_t)}
    
    # NaN 값 검사 및 처리
    nan_count_b = np.sum(np.isnan(traj_b))
    nan_count_t = np.sum(np.isnan(traj_t))
    if nan_count_b > 0 or nan_count_t > 0:
        logger.warning(f"[Speed-Trajectory] NaN values detected - Baseline: {nan_count_b}, Target: {nan_count_t}")
        
        # NaN 값을 이전 유효한 값으로 채우기
        def fill_nan_values(traj):
            for i in range(1, len(traj)):
                if np.any(np.isnan(traj[i])):
                    traj[i] = traj[i-1]
            return traj
        
        traj_b = fill_nan_values(traj_b.copy())
        traj_t = fill_nan_values(traj_t.copy())

    # 프레임 수 통일 (리샘플링)
    min_frames = min(len(traj_b), len(traj_t))
    if len(traj_b) != len(traj_t):
        # NaN 값이 있는 경우를 고려하여 안전하게 리샘플링
        try:
            # X, Y 좌표를 각각 리샘플링
            traj_b_x = resample(traj_b[:, 0], min_frames) if len(traj_b) > min_frames else traj_b[:, 0]
            traj_b_y = resample(traj_b[:, 1], min_frames) if len(traj_b) > min_frames else traj_b[:, 1]
            traj_t_x = resample(traj_t[:, 0], min_frames) if len(traj_t) > min_frames else traj_t[:, 0]
            traj_t_y = resample(traj_t[:, 1], min_frames) if len(traj_t) > min_frames else traj_t[:, 1]
            
            # 리샘플링된 좌표를 다시 결합
            traj_b = np.column_stack([traj_b_x, traj_b_y])
            traj_t = np.column_stack([traj_t_x, traj_t_y])
        except Exception as e:
            logger.warning(f"[Speed-Trajectory] Resampling failed: {e}, using original trajectories")

    # 3) 누적거리 계산
    def cumulative_from_traj(traj):
        cum = np.zeros(len(traj))
        for i in range(1, len(traj)):
            if not np.any(np.isnan(traj[i])) and not np.any(np.isnan(traj[i-1])):
                cum[i] = cum[i-1] + float(np.linalg.norm(traj[i]-traj[i-1]))
            else:
                cum[i] = cum[i-1]
        return float(cum[-1]), cum

    td_b, cum_b = cumulative_from_traj(traj_b)
    td_t, cum_t = cumulative_from_traj(traj_t)
    if td_b <= 0:
        return 0.0, {"error": "baseline cumulative distance is zero"}

    # 4) 진행률 보간 (누적거리/좌표 모두)
    pr = np.linspace(0.0, 1.0, num_points)
    x = np.linspace(0.0, 1.0, len(cum_b))
    ib = interp1d(x, cum_b, kind='linear', fill_value='extrapolate')
    it = interp1d(np.linspace(0.0, 1.0, len(cum_t)), cum_t, kind='linear', fill_value='extrapolate')
    cb = ib(pr) / max(td_b, 1e-9)
    ct = it(pr) / max(td_t, 1e-9) if td_t > 0 else np.zeros_like(cb)

    # 좌표 보간 (유사성 측정용)
    ibx = interp1d(np.linspace(0.0, 1.0, len(traj_b)), traj_b[:, 0], fill_value='extrapolate')
    iby = interp1d(np.linspace(0.0, 1.0, len(traj_b)), traj_b[:, 1], fill_value='extrapolate')
    itx = interp1d(np.linspace(0.0, 1.0, len(traj_t)), traj_t[:, 0], fill_value='extrapolate')
    ity = interp1d(np.linspace(0.0, 1.0, len(traj_t)), traj_t[:, 1], fill_value='extrapolate')
    pb = np.column_stack([ibx(pr), iby(pr)])
    pt = np.column_stack([itx(pr), ity(pr)])

    # 5) 점수 계산 (비율 클리핑 ±5%)
    ratios, scores = [], []
    for i in range(num_points):
        r = float(ct[i] / max(cb[i], 1e-9)) if cb[i] > 0 else 1.0
        r = max(0.90, min(1.10, r))  # ±10% 허용으로 확장
        ratios.append(r)
        scores.append(float(100.0 * min(r, 1.0 / r)))

    score = float(np.mean(scores)) if scores else 0.0

    # 6) 좌표 유사성 보정: 평균 좌표 차이가 매우 작으면 100점으로 승격
    diffs = []
    for i in range(num_points):
        if not np.any(np.isnan(pb[i])) and not np.any(np.isnan(pt[i])):
            diffs.append(float(np.linalg.norm(pb[i] - pt[i])))
    coord_diff_mean = float(np.mean(diffs)) if diffs else None
    if coord_diff_mean is not None and coord_diff_mean < 0.015:  # 임계값 완화
        score = 100.0
        logger.info("[Speed-Trajectory] High similarity detected, score clipped to 100.0")

    meta = {
        "fps": {"baseline": fps_b, "target": fps_t},
        "total_distance": {"baseline": td_b, "target": td_t},
        "cumulative_ratios": ratios,
        "point_scores": scores,
        "coord_diff_mean": coord_diff_mean,
        "final_speed_score": score,
    }
    logger.info(f"[Speed-Traj-robust] tdB/T={td_b:.4f}/{td_t:.4f}, coordDiff={coord_diff_mean}, score={score:.1f}")
    return score, meta


def calc_speed_score_sliding_window(baseline_path: str, target_path: str) -> tuple[float, dict]:
    """
    슬라이딩 윈도우를 사용한 실시간 속도 차이 분석 함수
    
    특징:
    - 0.5초 윈도우로 전체 스윙 구간을 실시간 분석
    - 0.25초 이동 간격으로 부드러운 전환 보장
    - 구간 설정의 기술적 어려움을 우회
    - 최소 0.5초 이상의 비교 구간으로 점수 안정성 확보
    
    Args:
        baseline_path: 기준 영상 경로
        target_path: 타겟 영상 경로
        
    Returns:
        tuple[float, dict]: (속도 점수, 메타데이터)
    """
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    
    # 1. 스윙 키 타임 추출 (웨글 제거된 구간 사용)
    try:
        from .video_processor import detect_swing_key_frames, detect_swing_key_times
        
        # 웨글 제거된 프레임 정보 추출
        start_rel_b, _, finish_rel_b, waggle_end_b = detect_swing_key_frames(baseline_path)
        start_rel_t, _, finish_rel_t, waggle_end_t = detect_swing_key_frames(target_path)
        
        # 웨글 제거된 시작 프레임
        start_b = waggle_end_b if waggle_end_b is not None else 0
        start_t = waggle_end_t if waggle_end_t is not None else 0
        
        # 피니시 프레임
        finish_b = start_b + finish_rel_b if finish_rel_b is not None else start_b + 60
        finish_t = start_t + finish_rel_t if finish_rel_t is not None else start_t + 60
        
        logger.info(f"[Sliding-Window] 웨글 제거 - Baseline: {start_b}→{finish_b}, Target: {start_t}→{finish_t}")
        
    except Exception as e:
        logger.error(f"[Sliding-Window] 키 프레임 감지 실패: {e}")
        return 50.0, {"error": "key_frames_detection_failed", "reason": str(e)}
    
    # 2. 비디오 캡처 및 FPS 추출
    cap_b = cv2.VideoCapture(baseline_path)
    cap_t = cv2.VideoCapture(target_path)
    
    if not cap_b.isOpened() or not cap_t.isOpened():
        logger.error("[Sliding-Window] 비디오 파일을 열 수 없습니다")
        return 50.0, {"error": "video_open_failed"}
    
    fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 30.0
    fps_t = cap_t.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_t = int(cap_t.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 웨글 제거된 구간의 시간 계산
    b_start_s = start_b / fps_b
    b_finish_s = finish_b / fps_b
    t_start_s = start_t / fps_t
    t_finish_s = finish_t / fps_t
    
    # 총 지속 시간 (더 짧은 구간을 기준으로 함)
    b_duration = b_finish_s - b_start_s
    t_duration = t_finish_s - t_start_s
    total_duration = min(b_duration, t_duration)
    
    if total_duration < 0.5:
        logger.warning(f"[Sliding-Window] 스윙 구간이 너무 짧음: {total_duration:.2f}초")
        cap_b.release()
        cap_t.release()
        return 50.0, {"error": "swing_duration_too_short", "duration": total_duration}
    
    logger.info(f"[Sliding-Window] 분석 구간 - Baseline: {b_duration:.2f}초, Target: {t_duration:.2f}초, Total: {total_duration:.2f}초")
    
    # 3. 슬라이딩 윈도우 설정
    window_size = 0.5  # 0.5초 윈도우
    step_size = 0.25   # 0.25초 이동 (50% 오버랩)
    
    windows_b = []
    windows_t = []
    current_time = 0.0
    
    while current_time + window_size <= total_duration:
        # 각 윈도우의 프레임 범위 계산
        start_frame_b = int((start_b + current_time * fps_b))
        end_frame_b = int((start_b + (current_time + window_size) * fps_b))
        start_frame_t = int((start_t + current_time * fps_t))
        end_frame_t = int((start_t + (current_time + window_size) * fps_t))
        
        # 프레임 범위 검증
        start_frame_b = max(0, min(start_frame_b, total_frames_b - 1))
        end_frame_b = max(0, min(end_frame_b, total_frames_b - 1))
        start_frame_t = max(0, min(start_frame_t, total_frames_t - 1))
        end_frame_t = max(0, min(end_frame_t, total_frames_t - 1))
        
        if end_frame_b > start_frame_b and end_frame_t > start_frame_t:
            windows_b.append((start_frame_b, end_frame_b))
            windows_t.append((start_frame_t, end_frame_t))
        
        current_time += step_size
    
    if len(windows_b) < 2:
        logger.warning(f"[Sliding-Window] 윈도우 수 부족: {len(windows_b)}개")
        cap_b.release()
        cap_t.release()
        return 50.0, {"error": "insufficient_windows", "count": len(windows_b)}
    
    logger.info(f"[Sliding-Window] 윈도우 설정 완료: {len(windows_b)}개 윈도우")
    
    # 4. 각 윈도우별 속도 계산
    scores = []
    window_details = []
    
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for i, ((start_b, end_b), (start_t, end_t)) in enumerate(zip(windows_b, windows_t)):
            try:
                # 기준 영상 랜드마크 추출
                cap_b.set(cv2.CAP_PROP_POS_FRAMES, start_b)
                landmarks_b = []
                for frame_idx in range(start_b, end_b + 1):
                    ret, frame = cap_b.read()
                    if ret:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = pose.process(rgb)
                        landmarks_b.append(result.pose_landmarks)
                
                # 타겟 영상 랜드마크 추출
                cap_t.set(cv2.CAP_PROP_POS_FRAMES, start_t)
                landmarks_t = []
                for frame_idx in range(start_t, end_t + 1):
                    ret, frame = cap_t.read()
                    if ret:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = pose.process(rgb)
                        landmarks_t.append(result.pose_landmarks)
                
                # 손 궤적 기반 속도 계산
                def calc_velocity(landmarks, fps, window_idx, video_type):
                    if not landmarks or len(landmarks) < 2:
                        logger.warning(f"[Speed-Detail] 윈도우 {window_idx} {video_type}: 랜드마크 부족 ({len(landmarks) if landmarks else 0}개)")
                        return 0.0
                    
                    distances = 0.0
                    valid_pairs = 0
                    frame_distances = []
                    
                    for i in range(len(landmarks) - 1):
                        if landmarks[i] and landmarks[i+1]:
                            hc_i = _hands_center_from_pose_landmarks(landmarks[i])
                            hc_j = _hands_center_from_pose_landmarks(landmarks[i+1])
                            
                            if hc_i and hc_j:
                                dist = np.linalg.norm(np.array(hc_i) - np.array(hc_j))
                                distances += dist
                                valid_pairs += 1
                                frame_distances.append(dist)
                    
                    if valid_pairs > 0:
                        velocity = distances / (window_size * fps)
                        avg_frame_dist = distances / valid_pairs
                        logger.info(f"[Speed-Detail] 윈도우 {window_idx} {video_type}:")
                        logger.info(f"  - 프레임 수: {len(landmarks)}개, 유효 쌍: {valid_pairs}개")
                        logger.info(f"  - 총 이동거리: {distances:.4f}, 평균 프레임간 거리: {avg_frame_dist:.4f}")
                        logger.info(f"  - 윈도우 시간: {window_size}초, FPS: {fps}")
                        logger.info(f"  - 속도 계산: {distances:.4f} / ({window_size} × {fps}) = {velocity:.4f}")
                        return velocity
                    else:
                        logger.warning(f"[Speed-Detail] 윈도우 {window_idx} {video_type}: 유효한 프레임 쌍 없음")
                        return 0.0
                
                v_b = calc_velocity(landmarks_b, fps_b, i, "Baseline")
                v_t = calc_velocity(landmarks_t, fps_t, i, "Target")
                
                if v_b > 0 and v_t > 0:
                    ratio = v_t / v_b
                    score = 100.0 * min(ratio, 1.0 / ratio)
                    scores.append(float(np.clip(score, 0.0, 100.0)))
                    
                    window_details.append({
                        "window_idx": i,
                        "baseline_velocity": v_b,
                        "target_velocity": v_t,
                        "ratio": ratio,
                        "score": score
                    })
                    
                    logger.info(f"[Speed-Detail] 윈도우 {i} 최종 계산:")
                    logger.info(f"  - Baseline 속도: {v_b:.4f}")
                    logger.info(f"  - Target 속도: {v_t:.4f}")
                    logger.info(f"  - 비율 계산: {v_t:.4f} / {v_b:.4f} = {ratio:.4f}")
                    logger.info(f"  - 점수 계산: 100 × min({ratio:.4f}, {1/ratio:.4f}) = 100 × {min(ratio, 1/ratio):.4f} = {score:.1f}")
                else:
                    logger.warning(f"[Speed-Detail] 윈도우 {i}: 속도 계산 실패 (v_b={v_b:.4f}, v_t={v_t:.4f})")
                
            except Exception as e:
                logger.warning(f"[Sliding-Window] 윈도우 {i} 처리 실패: {e}")
                continue
    
    # 5. 평균 점수 계산
    logger.info(f"[Speed-Detail] 최종 점수 계산:")
    logger.info(f"  - 전체 윈도우 수: {len(windows_b)}개")
    logger.info(f"  - 유효한 점수 수: {len(scores)}개")
    
    if len(scores) < 2:
        logger.warning(f"[Speed-Detail] 유효한 점수 부족: {len(scores)}개, 중간값 사용")
        final_score = float(np.median(scores)) if scores else 50.0
        logger.info(f"  - 중간값 계산: {final_score:.1f}")
    else:
        final_score = float(np.mean(scores))
        logger.info(f"  - 평균 계산: {np.mean(scores):.4f} = {final_score:.1f}")
    
    # 각 윈도우 점수 상세 출력
    logger.info(f"[Speed-Detail] 윈도우별 점수 상세:")
    for i, score in enumerate(scores):
        logger.info(f"  - 윈도우 {i}: {score:.1f}점")
    
    # 통계 정보
    if len(scores) > 1:
        logger.info(f"[Speed-Detail] 점수 통계:")
        logger.info(f"  - 최고점: {max(scores):.1f}")
        logger.info(f"  - 최저점: {min(scores):.1f}")
        logger.info(f"  - 표준편차: {np.std(scores):.2f}")
        logger.info(f"  - 중간값: {np.median(scores):.1f}")
    
    # 6. 메타데이터 구성
    meta = {
        "method": "sliding_window",
        "fps": {"baseline": fps_b, "target": fps_t},
        "window_size_seconds": window_size,
        "step_size_seconds": step_size,
        "total_windows": len(windows_b),
        "valid_scores": len(scores),
        "window_details": window_details,
        "scores": scores,
        "final_speed_score": final_score,
        "analysis_duration": total_duration,
        "statistics": {
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "std_deviation": float(np.std(scores)) if len(scores) > 1 else 0,
            "median": float(np.median(scores)) if scores else 0
        }
    }
    
    logger.info(f"[Sliding-Window] 분석 완료 - 윈도우: {len(windows_b)}개, 유효 점수: {len(scores)}개, 최종 점수: {final_score:.1f}")
    
    cap_b.release()
    cap_t.release()
    return final_score, meta


def calc_speed_score_three_segments_trajectory(
    landmarks_b: list,
    landmarks_t: list,
    start_b: int,
    finish_b: int,
    start_t: int,
    finish_t: int,
    fps_b: float,
    fps_t: float,
    tolerance_ratio_clip: float = 0.10,
) -> tuple[float, dict]:
    """
    웨글 제거 후 Address→Finish 구간을 3등분하여, 각 구간의 손 궤적 이동거리/시간(속도)을 비율로 비교해 점수를 계산합니다.

    Args:
        landmarks_b: 기준 영상의 포즈 랜드마크 리스트
        landmarks_t: 타겟 영상의 포즈 랜드마크 리스트
        start_b, finish_b: 기준 영상의 시작/종료 프레임 (웨글 제거 후)
        start_t, finish_t: 타겟 영상의 시작/종료 프레임 (웨글 제거 후)
        fps_b, fps_t: 각 영상 FPS
        tolerance_ratio_clip: 비율 클리핑 허용 범위 (±10% 기본)

    Returns:
        (final_score, meta)
    """
    # 유효 길이 확인
    len_b = max(0, int(finish_b) - int(start_b))
    len_t = max(0, int(finish_t) - int(start_t))
    if len_b < 3 or len_t < 3:
        logger.warning("[Three-Seg] 유효 구간 길이 부족, 전체 시간 비교로 폴백")
        d_b = max(len_b / max(fps_b, 1e-6), 0.01)
        d_t = max(len_t / max(fps_t, 1e-6), 0.01)
        score = time_diff_score(d_b, d_t, tolerance_percent=20.0)
        return score, {
            "method": "fallback_full_duration",
            "duration_b": d_b,
            "duration_t": d_t,
        }

    # 손 중심 궤적 추출 (어깨 폭 정규화, 방향 무시 누적거리)
    def extract_xy_norm(landmarks: list, s: int, e: int):
        xs, ys = [], []
        for i in range(int(s), int(e) + 1):
            pl = landmarks[i] if i < len(landmarks) else None
            if pl is None:
                xs.append(np.nan); ys.append(np.nan); continue
            try:
                lm = pl.landmark
                ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                mid_x = (ls.x + rs.x) / 2.0
                mid_y = (ls.y + rs.y) / 2.0
                shoulder_dx = (ls.x - rs.x)
                shoulder_dy = (ls.y - rs.y)
                shoulder_dist = float(np.hypot(shoulder_dx, shoulder_dy))
                if getattr(ls, 'visibility', 1.0) < 0.3 or getattr(rs, 'visibility', 1.0) < 0.3:
                    shoulder_dist = max(shoulder_dist, 0.1)
                shoulder_dist = max(shoulder_dist, 0.1)

                hc = _hands_center_from_pose_landmarks(pl)
                if hc is None:
                    xs.append(np.nan); ys.append(np.nan)
                else:
                    xs.append((float(hc[0]) - mid_x) / shoulder_dist)
                    ys.append((float(hc[1]) - mid_y) / shoulder_dist)
            except Exception:
                xs.append(np.nan); ys.append(np.nan)
        return np.column_stack([xs, ys]) if xs else np.zeros((0, 2))

    def fill_nan_values_inplace(traj: np.ndarray) -> np.ndarray:
        for i in range(1, len(traj)):
            if np.any(np.isnan(traj[i])):
                traj[i] = traj[i - 1]
        return traj

    def cumulative_distance(traj: np.ndarray) -> np.ndarray:
        cum = np.zeros(len(traj))
        for i in range(1, len(traj)):
            if not np.any(np.isnan(traj[i])) and not np.any(np.isnan(traj[i - 1])):
                cum[i] = cum[i - 1] + float(np.linalg.norm(traj[i] - traj[i - 1]))
            else:
                cum[i] = cum[i - 1]
        return cum

    traj_b = extract_xy_norm(landmarks_b, start_b, finish_b)
    traj_t = extract_xy_norm(landmarks_t, start_t, finish_t)

    if len(traj_b) < 2 or len(traj_t) < 2:
        logger.error("[Three-Seg] 궤적 데이터 부족, 전체 시간 비교로 폴백")
        d_b = max(len_b / max(fps_b, 1e-6), 0.01)
        d_t = max(len_t / max(fps_t, 1e-6), 0.01)
        score = time_diff_score(d_b, d_t, tolerance_percent=20.0)
        return score, {"method": "fallback_full_duration", "duration_b": d_b, "duration_t": d_t}

    traj_b = fill_nan_values_inplace(traj_b.copy())
    traj_t = fill_nan_values_inplace(traj_t.copy())

    cum_b = cumulative_distance(traj_b)
    cum_t = cumulative_distance(traj_t)

    # 3등분 세그먼트 경계 (로컬 인덱스 기준)
    n_b = len(traj_b)
    n_t = len(traj_t)
    b_edges = [0, int(n_b / 3), int(2 * n_b / 3), n_b - 1]
    t_edges = [0, int(n_t / 3), int(2 * n_t / 3), n_t - 1]

    # 각 세그먼트의 이동거리/시간/속도 계산
    durations_b, durations_t = [], []
    distances_b, distances_t = [], []
    velocities_b, velocities_t = [], []

    for i in range(3):
        sb_local, eb_local = b_edges[i], b_edges[i + 1]
        st_local, et_local = t_edges[i], t_edges[i + 1]

        # 로컬 → 절대 프레임 인덱스 변환
        sb_abs = start_b + sb_local
        eb_abs = start_b + eb_local
        st_abs = start_t + st_local
        et_abs = start_t + et_local

        # 시간(초)
        dur_b = max((eb_abs - sb_abs) / max(fps_b, 1e-6), 0.01)
        dur_t = max((et_abs - st_abs) / max(fps_t, 1e-6), 0.01)

        # 거리(정규화 좌표 누적)
        dist_b = max(float(cum_b[eb_local] - cum_b[sb_local]), 0.0)
        dist_t = max(float(cum_t[et_local] - cum_t[st_local]), 0.0)

        # 속도(거리/시간)
        vel_b = dist_b / dur_b if dur_b > 0 else 0.0
        vel_t = dist_t / dur_t if dur_t > 0 else 0.0

        durations_b.append(dur_b); durations_t.append(dur_t)
        distances_b.append(dist_b); distances_t.append(dist_t)
        velocities_b.append(vel_b); velocities_t.append(vel_t)

    # 세그먼트별 비율/점수
    ratios, seg_scores = [], []
    clip_low = 1.0 - tolerance_ratio_clip
    clip_high = 1.0 + tolerance_ratio_clip
    # 구간별 상세 로그 (시간, 거리, 속도)
    for i in range(3):
        logger.info(
            f"[Three-Seg] {i+1}구간: Baseline 시간={durations_b[i]:.3f}s, 거리={distances_b[i]:.4f}, 속도={velocities_b[i]:.4f} | "
            f"Target 시간={durations_t[i]:.3f}s, 거리={distances_t[i]:.4f}, 속도={velocities_t[i]:.4f}"
        )
    for i in range(3):
        vb = velocities_b[i]
        vt = velocities_t[i]
        if vb <= 0 or vt <= 0:
            r = 1.0
            s = 50.0
        else:
            r_raw = float(vt / vb)
            r = max(clip_low, min(clip_high, r_raw))
            s = float(100.0 * min(r, 1.0 / r))
        ratios.append(r)
        seg_scores.append(s)

    final_score = float(np.mean(seg_scores))

    meta = {
        "method": "three_segments_trajectory",
        "segments": {
            "durations_b": durations_b,
            "durations_t": durations_t,
            "distances_b": distances_b,
            "distances_t": distances_t,
            "velocities_b": velocities_b,
            "velocities_t": velocities_t,
            "ratios": ratios,
            "segment_scores": seg_scores,
        },
        "final_speed_score": final_score,
        "tolerance_ratio_clip": tolerance_ratio_clip,
    }

    logger.info(
        f"[Three-Seg] 구간별 점수={['%.1f'%s for s in seg_scores]}, 최종={final_score:.1f} | 비율={['%.3f'%r for r in ratios]}"
    )
    return final_score, meta

def _hands_center_from_pose_landmarks(pose_landmarks):
    """포즈 랜드마크에서 손 중심점 추출"""
    try:
        lm = pose_landmarks.landmark
        lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        if getattr(lw, 'visibility', 1.0) < 0.2 or getattr(rw, 'visibility', 1.0) < 0.2:
            return None
        
        return ((lw.x + rw.x) / 2.0, (lw.y + rw.y) / 2.0)
    except Exception:
        return None


def time_diff_score(d_base: float, d_target: float, tolerance_percent: float = 20.0) -> float:
    """
    시간 차이를 기반으로 점수 계산 (스윙 분석용 관대한 허용 범위 적용)
    
    Args:
        d_base: 기준 시간
        d_target: 타겟 시간
        tolerance_percent: 허용 오차 백분율 (기본값: 20%)
        
    Returns:
        float: 0~100 범위의 점수
    """
    base = max(0.01, float(d_base))
    target = float(d_target)
    diff_percent = abs(target - base) / base * 100
    
    # 허용 범위 내에서는 100점 반환 (스윙이 같다고 판단)
    if diff_percent <= tolerance_percent:
        return 100.0
    
    # 허용 범위를 벗어난 경우에만 점수 감점
    # 더 관대한 감점 곡선 적용
    excess_percent = diff_percent - tolerance_percent
    penalty = min(excess_percent * 1.5, 100.0)  # 최대 100점 감점
    score = max(100.0 - penalty, 0.0)
    
    return float(score)


def calc_speed_score_full(b_start_s: float, b_finish_s: float, t_start_s: float, t_finish_s: float) -> tuple[float, dict]:
    """
    스윙 시작부터 피니시까지의 전체 속도만 측정하는 단일 구간 분석 함수
    
    Address에서 Finish까지의 전체 시간을 직접 비교하여 속도 일치도를 계산합니다.
    구간을 나누지 않고 전체 스윙의 속도를 단일 지표로 평가합니다.
    
    Args:
        b_start_s: 기준 영상의 스윙 시작 시간 (초)
        b_finish_s: 기준 영상의 스윙 종료 시간 (초)
        t_start_s: 대상 영상의 스윙 시작 시간 (초)
        t_finish_s: 대상 영상의 스윙 종료 시간 (초)
        
    Returns:
        tuple[float, dict]: (속도 점수, 메타데이터)
    """
    # 전체 스윙 시간 계산 (최소 0.01초로 제한)
    d_b = max(round(b_finish_s - b_start_s, 2), 0.01)
    d_t = max(round(t_finish_s - t_start_s, 2), 0.01)
    
    # 시간 차이 기반 점수 계산 (스윙 분석용 관대한 허용 범위 적용)
    score = time_diff_score(d_b, d_t, tolerance_percent=20.0)  # 20% 허용 범위
    
    # 메타데이터 구성
    meta = {
        "times": {
            "baseline": {
                "start_s": b_start_s,
                "finish_s": b_finish_s,
                "duration_s": d_b
            },
            "target": {
                "start_s": t_start_s,
                "finish_s": t_finish_s,
                "duration_s": d_t
            }
        },
        "final_speed_score": float(score),
        "method": "full_swing_duration"
    }
    
    # 상세 디버깅 로그
    ratio = d_t / d_b if d_b > 0 else 1.0
    diff_percent = abs(d_t - d_b) / d_b * 100 if d_b > 0 else 0.0
    
    logger.info(f"[Speed-Full] Baseline: {d_b:.2f}s, Target: {d_t:.2f}s, Ratio: {ratio:.3f}, Diff: {diff_percent:.1f}%, Tolerance: 20%, Score: {score:.1f}")
    
    # 점수가 낮은 경우 상세 정보 로깅
    if score < 90.0:
        logger.warning(f"[Speed-Full] Low score detected - Baseline: {d_b:.2f}s, Target: {d_t:.2f}s, Diff: {diff_percent:.1f}%, Score: {score:.1f}")
    
    return score, meta


def _extract_angles_per_frame(image, pose) -> dict | None:
    """
    단일 프레임에서 관절 각도를 추출하는 내부 함수
    
    Args:
        image: 분석할 이미지 (BGR 형식)
        pose: MediaPipe 포즈 객체
        
    Returns:
        dict: 관절별 각도 정보를 담은 딕셔너리
        None: 포즈 랜드마크를 찾을 수 없는 경우
    """
    # 이미지 크기 추출
    h, w = image.shape[:2]
    # BGR을 RGB로 변환 (MediaPipe는 RGB 형식 사용)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 포즈 랜드마크 검출
    result = pose.process(rgb)
    if not result.pose_landmarks:
        return None
    # 랜드마크 좌표 추출
    lm = result.pose_landmarks.landmark

    def p(idx):
        """
        랜드마크 인덱스에 해당하는 정규화된 좌표 반환
        
        Args:
            idx: MediaPipe 포즈 랜드마크 인덱스
            
        Returns:
            np.array: [x, y] 좌표 배열
        """
        # 정규화 좌표 사용 (0~1 범위)
        return np.array([lm[idx].x, lm[idx].y])

    def angle(a, b, c):
        """
        세 점으로 이루어진 각도 계산
        
        Args:
            a, b, c: 각도를 구성하는 세 점의 좌표
            
        Returns:
            float: 각도 (도 단위)
        """
        # 벡터 계산: ba = a - b, bc = c - b
        ba = a - b
        bc = c - b
        # 벡터의 크기 계산
        na = np.linalg.norm(ba)
        nc = np.linalg.norm(bc)
        # 0으로 나누기 방지
        if na == 0 or nc == 0:
            return np.nan
        # 코사인 각도 계산 및 클리핑
        cosv = np.clip(np.dot(ba, bc) / (na*nc), -1.0, 1.0)
        # 라디안을 도로 변환
        return float(np.degrees(np.arccos(cosv)))

    # 관절별 각도 계산
    A = {}
    # 왼쪽 팔꿈치 각도: 어깨-팔꿈치-손목
    A['left_elbow'] = angle(p(mp_pose.PoseLandmark.LEFT_SHOULDER), p(mp_pose.PoseLandmark.LEFT_ELBOW), p(mp_pose.PoseLandmark.LEFT_WRIST))
    # 오른쪽 팔꿈치 각도: 어깨-팔꿈치-손목
    A['right_elbow'] = angle(p(mp_pose.PoseLandmark.RIGHT_SHOULDER), p(mp_pose.PoseLandmark.RIGHT_ELBOW), p(mp_pose.PoseLandmark.RIGHT_WRIST))
    # 왼쪽 무릎 각도: 엉덩이-무릎-발목
    A['left_knee'] = angle(p(mp_pose.PoseLandmark.LEFT_HIP), p(mp_pose.PoseLandmark.LEFT_KNEE), p(mp_pose.PoseLandmark.LEFT_ANKLE))
    # 오른쪽 무릎 각도: 엉덩이-무릎-발목
    A['right_knee'] = angle(p(mp_pose.PoseLandmark.RIGHT_HIP), p(mp_pose.PoseLandmark.RIGHT_KNEE), p(mp_pose.PoseLandmark.RIGHT_ANKLE))
    # 왼쪽 어깨 각도: 팔꿈치-어깨-엉덩이
    A['left_shoulder'] = angle(p(mp_pose.PoseLandmark.LEFT_ELBOW), p(mp_pose.PoseLandmark.LEFT_SHOULDER), p(mp_pose.PoseLandmark.LEFT_HIP))
    # 오른쪽 어깨 각도: 팔꿈치-어깨-엉덩이
    A['right_shoulder'] = angle(p(mp_pose.PoseLandmark.RIGHT_ELBOW), p(mp_pose.PoseLandmark.RIGHT_SHOULDER), p(mp_pose.PoseLandmark.RIGHT_HIP))
    # 왼쪽 엉덩이 각도: 어깨-엉덩이-무릎
    A['left_hip'] = angle(p(mp_pose.PoseLandmark.LEFT_SHOULDER), p(mp_pose.PoseLandmark.LEFT_HIP), p(mp_pose.PoseLandmark.LEFT_KNEE))
    # 오른쪽 엉덩이 각도: 어깨-엉덩이-무릎
    A['right_hip'] = angle(p(mp_pose.PoseLandmark.RIGHT_SHOULDER), p(mp_pose.PoseLandmark.RIGHT_HIP), p(mp_pose.PoseLandmark.RIGHT_KNEE))

    return A


def _calc_angle_based_sync_score(cap_b, cap_t, pose_b, pose_t,
                                start_b, finish_b, start_t, finish_t,
                                total_b, total_t, max_angle_diff,
                                desired_samples: int):
    """
    각도 기반 동기화로 관절 점수를 계산하는 내부 함수
    
    Args:
        cap_b, cap_t: 비디오 캡처 객체
        pose_b, pose_t: MediaPipe 포즈 객체
        start_b, finish_b: 기준 영상의 시작/종료 프레임
        start_t, finish_t: 타겟 영상의 시작/종료 프레임
        total_b, total_t: 총 프레임 수
        max_angle_diff: 최대 각도 차이
        
    Returns:
        list: 점수 리스트
    """
    # 1단계: 각 영상의 전체 관절 각도 시퀀스 추출
    logger.info(f"[Angle-Detail] 각도 기반 동기화 사용")
    logger.info(f"  - Baseline 각도 시퀀스 추출: {start_b}-{finish_b} 프레임")
    baseline_angles = _extract_angle_sequence(cap_b, pose_b, start_b, finish_b, total_b, desired_samples)
    logger.info(f"  - Target 각도 시퀀스 추출: {start_t}-{finish_t} 프레임")
    target_angles = _extract_angle_sequence(cap_t, pose_t, start_t, finish_t, total_t, desired_samples)
    
    if not baseline_angles or not target_angles:
        logger.error(f"[Angle-Score] FAILED - Insufficient angle data - Baseline: {len(baseline_angles) if baseline_angles else 0}, Target: {len(target_angles) if target_angles else 0}")
        logger.error(f"[Angle-Score] REASON - 포즈 랜드마크 감지 실패 또는 프레임 범위 오류")
        # 점수 산정 불가 - 빈 리스트 반환
        return []
    
    logger.info(f"  - Baseline 각도 시퀀스: {len(baseline_angles)}개 프레임")
    logger.info(f"  - Target 각도 시퀀스: {len(target_angles)}개 프레임")
    
    # 2단계: 각도 기반 동기화 매칭
    logger.info(f"  - 각도 유사성 기반 매칭 시작")
    matched_pairs = _match_angles_by_similarity(baseline_angles, target_angles)
    logger.info(f"  - 매칭된 쌍: {len(matched_pairs)}개")
    
    # 3단계: 매칭된 쌍들로 점수 계산
    scores = []
    pair_count = 0
    
    for i, (baseline_data, target_data) in enumerate(matched_pairs):
        per_scores = []
        joint_details = []
        
        for joint_name in baseline_data['angles'].keys():
            vb = baseline_data['angles'][joint_name]
            vt = target_data['angles'][joint_name]
            
            if vb is None or vt is None or np.isnan(vb) or np.isnan(vt):
                continue
                
            # 각도 차이 계산
            diff = abs(float(vt) - float(vb))
            # 점수 계산: 100 - (차이/최대차이) * 100
            score = 100.0 - (diff / max(1e-6, max_angle_diff)) * 100.0
            clipped_score = float(np.clip(score, 0.0, 100.0))
            per_scores.append(clipped_score)
            
            joint_details.append({
                "joint": joint_name,
                "baseline_angle": vb,
                "target_angle": vt,
                "angle_diff": diff,
                "score": clipped_score
            })
        
        if per_scores:
            pair_score = float(np.mean(per_scores))
            scores.append(pair_score)
            pair_count += 1
            
            logger.debug(f"[Angle-Detail] 매칭 쌍 {i} (프레임 {baseline_data['frame_idx']} vs {target_data['frame_idx']}):")
            logger.debug(f"  - 유효 관절: {len(per_scores)}개")
            logger.debug(f"  - 평균 점수: {pair_score:.1f}")
            
            # 상세한 관절별 정보 (디버그 레벨)
            for detail in joint_details:
                logger.debug(f"    - {detail['joint']}: {detail['baseline_angle']:.1f}° vs {detail['target_angle']:.1f}° (차이: {detail['angle_diff']:.1f}°, 점수: {detail['score']:.1f})")
    
    logger.info(f"  - 매칭 쌍별 점수 계산 완료: {pair_count}개 쌍")
    
    # 매칭이 실패한 경우 오류 보고
    if not scores:
        logger.error(f"[Angle-Score] FAILED - No matched pairs found")
        logger.error(f"[Angle-Score] REASON - 각도 유사성 매칭 실패 (임계값 45도 초과)")
        # 점수 산정 불가 - 빈 리스트 반환
        return []
    
    return scores


def _extract_angle_sequence(cap, pose, start_frame, finish_frame, total_frames, desired_samples: int = 50):
    """
    영상에서 관절 각도 시퀀스를 추출하는 내부 함수
    
    Args:
        cap: 비디오 캡처 객체
        pose: MediaPipe 포즈 객체
        start_frame, finish_frame: 시작/종료 프레임
        total_frames: 총 프레임 수
        
    Returns:
        list: 각 프레임의 관절 각도 데이터 리스트
    """
    angles_sequence = []
    
    # 프레임 범위 검증
    if start_frame >= total_frames or finish_frame >= total_frames:
        return angles_sequence
    
    # 균등 간격 샘플링: Baseline/Target 동일 개수로 추출
    sample_count = max(1, int(desired_samples))
    indices = np.linspace(start_frame, finish_frame, num=sample_count, dtype=int)
    
    for frame_idx in indices:
        if frame_idx >= total_frames:
            break
            
        # 해당 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # 관절 각도 추출
        angles = _extract_angles_per_frame(frame, pose)
        if angles:
            angles_sequence.append({
                'frame_idx': frame_idx,
                'angles': angles
            })
    
    return angles_sequence


def _match_angles_by_similarity(baseline_angles, target_angles):
    """
    각도 유사성을 기반으로 프레임을 매칭하는 내부 함수
    
    Args:
        baseline_angles: 기준 영상의 각도 시퀀스
        target_angles: 타겟 영상의 각도 시퀀스
        
    Returns:
        list: 매칭된 (baseline_data, target_data) 쌍들의 리스트
    """
    matched_pairs = []
    
    # 기준 영상의 각 프레임에 대해 가장 유사한 타겟 프레임 찾기
    for baseline_data in baseline_angles:
        best_match = None
        best_similarity = float('inf')
        
        for target_data in target_angles:
            # 각도 유사성 계산 (평균 절대 차이)
            similarity = _calculate_angle_similarity(
                baseline_data['angles'], 
                target_data['angles']
            )
            
            if similarity < best_similarity:
                best_similarity = similarity
                best_match = target_data
        
        # 유사도가 임계값 이하인 경우만 매칭 (임계값 완화)
        if best_match and best_similarity < 45.0:  # 45도 이하 차이로 완화
            matched_pairs.append((baseline_data, best_match))
    
    return matched_pairs


def _calculate_angle_similarity(angles1, angles2):
    """
    두 관절 각도 딕셔너리 간의 유사성을 계산하는 내부 함수
    
    Args:
        angles1, angles2: 관절 각도 딕셔너리
        
    Returns:
        float: 평균 절대 각도 차이
    """
    total_diff = 0.0
    valid_joints = 0
    
    for joint_name in angles1.keys():
        if joint_name in angles2:
            a1 = angles1[joint_name]
            a2 = angles2[joint_name]
            
            if a1 is not None and a2 is not None and not (np.isnan(a1) or np.isnan(a2)):
                diff = abs(float(a1) - float(a2))
                total_diff += diff
                valid_joints += 1
    
    return total_diff / max(1, valid_joints)


def calc_speed_score(baseline_path: str, target_path: str,
                     start_b, top_b, finish_b, start_t, top_t, finish_t) -> tuple[float, dict]:
    """
    FPS 기반 프레임을 초로 환산하여 속도 점수를 계산하는 함수 (하위호환 메서드)
    
    Args:
        baseline_path: 기준 영상 경로
        target_path: 분석할 영상 경로
        start_b, top_b, finish_b: 기준 영상의 시작/최고점/종료 프레임
        start_t, top_t, finish_t: 타겟 영상의 시작/최고점/종료 프레임
        
    Returns:
        tuple: (속도 점수, 메타데이터)
    """
    # 비디오 캡처 객체 생성
    cap_b = cv2.VideoCapture(baseline_path)
    cap_t = cv2.VideoCapture(target_path)
    # FPS 추출 (기본값 30으로 설정)
    fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 0.0
    fps_t = cap_t.get(cv2.CAP_PROP_FPS) or 0.0
    # 리소스 해제
    cap_b.release(); cap_t.release()
    if fps_b <= 0: fps_b = 30.0
    if fps_t <= 0: fps_t = 30.0

    # 구간 길이(초) 계산 (최소값 가드 적용)
    # 구간1: 시작 → 최고점, 구간2: 최고점 → 종료
    d1_b = max((top_b - start_b) / fps_b, 1.0 / fps_b)
    d2_b = max((finish_b - top_b) / fps_b, 1.0 / fps_b)
    d1_t = max((top_t - start_t) / fps_t, 1.0 / fps_t)
    d2_t = max((finish_t - top_t) / fps_t, 1.0 / fps_t)

    # 사용자 요구: (비교/기준)*100을 100 기준의 절대 일치 점수로 해석
    # ratio=0.8 또는 1.2 → 80점: score = 100 * min(ratio, 1/ratio)
    def time_diff_score(d_base: float, d_target: float) -> float:
        """
        시간 차이를 기반으로 점수 계산
        
        Args:
            d_base: 기준 시간
            d_target: 타겟 시간
            
        Returns:
            float: 0~100 범위의 점수
        """
        # 100 * (1 - |target - base| / base) 공식 사용
        base = max(0.01, float(d_base))
        diff = abs(float(d_target) - base)
        return float(np.clip(100.0 * (1.0 - diff / base), 0.0, 100.0))

    # 각 구간별 점수 계산
    s1 = time_diff_score(d1_b, d1_t)  # 구간1 점수
    s2 = time_diff_score(d2_b, d2_t)  # 구간2 점수
    # 비율 계산
    r1 = d1_t / d1_b  # 구간1 비율
    r2 = d2_t / d2_b  # 구간2 비율
    # 최종 점수 (두 구간의 평균)
    score = float((s1 + s2) / 2.0)

    # 메타데이터 구성
    meta = {
        "fps": {"baseline": fps_b, "target": fps_t},  # FPS 정보
        "frames": {  # 프레임 정보
            "baseline": {"start": int(start_b), "top": int(top_b), "finish": int(finish_b)},
            "target": {"start": int(start_t), "top": int(top_t), "finish": int(finish_t)},
        },
        "times": {  # 시간 정보 (초 단위)
            "baseline": {
                "start_s": float(start_b / fps_b),
                "top_s": float(top_b / fps_b),
                "finish_s": float(finish_b / fps_b),
                "d1_s": float(d1_b),
                "d2_s": float(d2_b),
            },
            "target": {
                "start_s": float(start_t / fps_t),
                "top_s": float(top_t / fps_t),
                "finish_s": float(finish_t / fps_t),
                "d1_s": float(d1_t),
                "d2_s": float(d2_t),
            },
        },
        "ratios": {"r1": float(r1), "r2": float(r2)},  # 비율 정보
        "segment_scores": {"s1": float(s1), "s2": float(s2)},  # 구간별 점수
        "final_speed_score": float(score),  # 최종 속도 점수
    }

    # 로그 출력 (초 단위로 표시)
    logger.info(
        "[Speed] baseline(start/top/finish): %d(%.3fs)/%d(%.3fs)/%d(%.3fs), d1=%.3fs, d2=%.3fs",
        start_b, meta["times"]["baseline"]["start_s"],
        top_b, meta["times"]["baseline"]["top_s"],
        finish_b, meta["times"]["baseline"]["finish_s"],
        meta["times"]["baseline"]["d1_s"], meta["times"]["baseline"]["d2_s"],
    )
    logger.info(
        "[Speed] target  (start/top/finish): %d(%.3fs)/%d(%.3fs)/%d(%.3fs), d1=%.3fs, d2=%.3fs | r1=%.3f, r2=%.3f, s1=%.1f, s2=%.1f, final=%.1f",
        start_t, meta["times"]["target"]["start_s"],
        top_t, meta["times"]["target"]["top_s"],
        finish_t, meta["times"]["target"]["finish_s"],
        meta["times"]["target"]["d1_s"], meta["times"]["target"]["d2_s"],
        r1, r2, s1, s2, score,
    )

    return score, meta


def calc_speed_score_seconds(b_start_s: float, b_top_s: float, b_finish_s: float,
                             t_start_s: float, t_top_s: float, t_finish_s: float,
                             frames_meta: dict | None = None,
                             fps_meta: dict | None = None) -> tuple[float, dict]:
    """
    초 단위(0.01초 해상도)로 정확히 계산하는 속도 점수 함수
    
    Args:
        b_start_s, b_top_s, b_finish_s: 기준 영상의 시작/최고점/종료 시간(초)
        t_start_s, t_top_s, t_finish_s: 타겟 영상의 시작/최고점/종료 시간(초)
        frames_meta: 프레임 메타데이터 (선택사항)
        fps_meta: FPS 메타데이터 (선택사항)
        
    Returns:
        tuple: (속도 점수, 메타데이터)
        
    특징:
        - 구간1: start→top, 구간2: top→finish
        - 점수: 100 * min(ratio, 1/ratio)  where ratio = target_dur / baseline_dur
    """
    # 소수점 2자리로 반올림하는 헬퍼 함수
    round2 = lambda x: float(round(x + 1e-9, 2))

    # 각 구간의 지속 시간 계산 (최소값 0.01초 보장)
    d1_b = max(round2(b_top_s - b_start_s), 0.01)  # 기준 영상 구간1
    d2_b = max(round2(b_finish_s - b_top_s), 0.01)  # 기준 영상 구간2
    d1_t = max(round2(t_top_s - t_start_s), 0.01)   # 타겟 영상 구간1
    d2_t = max(round2(t_finish_s - t_top_s), 0.01)  # 타겟 영상 구간2

    def time_diff_score(d_base: float, d_target: float) -> float:
        """
        시간 차이를 기반으로 점수 계산
        
        Args:
            d_base: 기준 시간
            d_target: 타겟 시간
            
        Returns:
            float: 0~100 범위의 점수
        """
        base = max(0.01, float(d_base))
        diff = abs(float(d_target) - base)
        return float(np.clip(100.0 * (1.0 - diff / base), 0.0, 100.0))

    # 각 구간별 점수 계산
    s1 = time_diff_score(d1_b, d1_t)  # 구간1 점수
    s2 = time_diff_score(d2_b, d2_t)  # 구간2 점수
    # 비율 계산
    r1 = d1_t / d1_b  # 구간1 비율
    r2 = d2_t / d2_b  # 구간2 비율
    # 최종 점수 (두 구간의 평균)
    score = float((s1 + s2) / 2.0)

    # 메타데이터 구성
    meta = {
        "fps": fps_meta or {},  # FPS 정보
        "frames": frames_meta or {},  # 프레임 정보
        "times": {  # 시간 정보 (초 단위)
            "baseline": {
                "start_s": round2(b_start_s),
                "top_s": round2(b_top_s),
                "finish_s": round2(b_finish_s),
                "d1_s": d1_b,
                "d2_s": d2_b,
            },
            "target": {
                "start_s": round2(t_start_s),
                "top_s": round2(t_top_s),
                "finish_s": round2(t_finish_s),
                "d1_s": d1_t,
                "d2_s": d2_t,
            },
        },
        "ratios": {"r1": float(r1), "r2": float(r2)},  # 비율 정보
        "segment_scores": {"s1": float(s1), "s2": float(s2)},  # 구간별 점수
        "final_speed_score": float(score),  # 최종 속도 점수
    }

    # 로그 출력
    logger.info(
        "[Speed-sec] baseline(start/top/finish): %.2fs/%.2fs/%.2fs, d1=%.2fs, d2=%.2fs",
        meta["times"]["baseline"]["start_s"], meta["times"]["baseline"]["top_s"], meta["times"]["baseline"]["finish_s"],
        meta["times"]["baseline"]["d1_s"], meta["times"]["baseline"]["d2_s"],
    )
    logger.info(
        "[Speed-sec] target  (start/top/finish): %.2fs/%.2fs/%.2fs, d1=%.2fs, d2=%.2fs | r1=%.3f, r2=%.3f, s1=%.1f, s2=%.1f, final=%.1f",
        meta["times"]["target"]["start_s"], meta["times"]["target"]["top_s"], meta["times"]["target"]["finish_s"],
        meta["times"]["target"]["d1_s"], meta["times"]["target"]["d2_s"],
        r1, r2, s1, s2, score,
    )

    return score, meta


def calc_speed_score_no_top(b_address_s: float, b_impact_s: float, b_finish_s: float,
                           t_address_s: float, t_impact_s: float, t_finish_s: float,
                           frames_meta: dict | None = None,
                           fps_meta: dict | None = None) -> tuple[float, dict]:
    """
    백스윙 탑을 무시한 임팩트 중심 2구간 속도 점수 계산 함수
    
    구간 정의:
    - 구간1: Address → Impact (전체 백스윙 + 다운스윙)
    - 구간2: Impact → Finish (팔로우스루)
    
    Args:
        b_address_s, b_impact_s, b_finish_s: 기준 영상의 어드레스/임팩트/피니시 시간(초)
        t_address_s, t_impact_s, t_finish_s: 타겟 영상의 어드레스/임팩트/피니시 시간(초)
        frames_meta: 프레임 메타데이터 (선택사항)
        fps_meta: FPS 메타데이터 (선택사항)
        
    Returns:
        tuple: (속도 점수, 메타데이터)
        
    특징:
        - 백스윙 탑의 불안정성 제거
        - 임팩트 중심의 안정적인 구간 비교
        - 스윙 리듬의 70-80%를 커버하는 구간1이 핵심
    """
    # 소수점 2자리로 반올림하는 헬퍼 함수
    round2 = lambda x: float(round(x + 1e-9, 2))

    # 각 구간의 지속 시간 계산 (최소값 0.2초 보장)
    d1_b = max(round2(b_impact_s - b_address_s), 0.2)  # 기준 영상 구간1: Address → Impact
    d2_b = max(round2(b_finish_s - b_impact_s), 0.2)   # 기준 영상 구간2: Impact → Finish
    d1_t = max(round2(t_impact_s - t_address_s), 0.2)  # 타겟 영상 구간1: Address → Impact
    d2_t = max(round2(t_finish_s - t_impact_s), 0.2)   # 타겟 영상 구간2: Impact → Finish

    def time_diff_score(d_base: float, d_target: float) -> float:
        """
        시간 차이를 기반으로 점수 계산
        
        Args:
            d_base: 기준 시간
            d_target: 타겟 시간
            
        Returns:
            float: 0~100 범위의 점수
        """
        base = max(0.2, float(d_base))  # 최소 0.2초 보장
        diff = abs(float(d_target) - base)
        return float(np.clip(100.0 * (1.0 - diff / base), 0.0, 100.0))

    # 각 구간별 점수 계산
    s1 = time_diff_score(d1_b, d1_t)  # 구간1 점수: Address → Impact (스윙 리듬의 핵심)
    s2 = time_diff_score(d2_b, d2_t)  # 구간2 점수: Impact → Finish (팔로우스루 균형)
    
    # 비율 계산
    r1 = d1_t / d1_b  # 구간1 비율
    r2 = d2_t / d2_b  # 구간2 비율
    
    # 최종 점수 (두 구간의 평균)
    score = float((s1 + s2) / 2.0)

    # 메타데이터 구성
    meta = {
        "method": "no_top_impact_based",  # 사용된 방법 표시
        "fps": fps_meta or {},  # FPS 정보
        "frames": frames_meta or {},  # 프레임 정보
        "times": {  # 시간 정보 (초 단위)
            "baseline": {
                "address_s": round2(b_address_s),
                "impact_s": round2(b_impact_s),
                "finish_s": round2(b_finish_s),
                "d1_s": d1_b,  # Address → Impact
                "d2_s": d2_b,  # Impact → Finish
            },
            "target": {
                "address_s": round2(t_address_s),
                "impact_s": round2(t_impact_s),
                "finish_s": round2(t_finish_s),
                "d1_s": d1_t,  # Address → Impact
                "d2_s": d2_t,  # Impact → Finish
            },
        },
        "ratios": {"r1": float(r1), "r2": float(r2)},  # 비율 정보
        "segment_scores": {
            "s1": float(s1),  # 구간1: Address → Impact (스윙 리듬)
            "s2": float(s2)   # 구간2: Impact → Finish (팔로우스루)
        },
        "final_speed_score": float(score),  # 최종 속도 점수
        "segment_descriptions": {
            "s1": "Address → Impact (전체 백스윙 + 다운스윙)",
            "s2": "Impact → Finish (팔로우스루)"
        }
    }

    # 로그 출력
    logger.info(
        "[Speed-NoTop] baseline(address/impact/finish): %.2fs/%.2fs/%.2fs, d1=%.2fs, d2=%.2fs",
        meta["times"]["baseline"]["address_s"], meta["times"]["baseline"]["impact_s"], meta["times"]["baseline"]["finish_s"],
        meta["times"]["baseline"]["d1_s"], meta["times"]["baseline"]["d2_s"],
    )
    logger.info(
        "[Speed-NoTop] target  (address/impact/finish): %.2fs/%.2fs/%.2fs, d1=%.2fs, d2=%.2fs | r1=%.3f, r2=%.3f, s1=%.1f, s2=%.1f, final=%.1f",
        meta["times"]["target"]["address_s"], meta["times"]["target"]["impact_s"], meta["times"]["target"]["finish_s"],
        meta["times"]["target"]["d1_s"], meta["times"]["target"]["d2_s"],
        r1, r2, s1, s2, score,
    )

    return score, meta


def calc_joint_angle_score(baseline_path: str, target_path: str,
                           start_b, finish_b, start_t, finish_t,
                           num_samples: int = 50,
                           max_angle_diff: float = 180.0,
                           use_angle_based_sync: bool = True) -> float:
    """
    진행률(progress) 또는 각도 기반 동기화로 절대각도 차이 기반 점수(0~100) 계산
    
    Args:
        baseline_path: 기준 영상 경로
        target_path: 분석할 영상 경로
        start_b, finish_b: 기준 영상의 시작/종료 프레임
        start_t, finish_t: 타겟 영상의 시작/종료 프레임
        num_samples: 분석할 샘플 수 (기본값: 50)
        max_angle_diff: 최대 각도 차이 (기본값: 180도)
        use_angle_based_sync: 각도 기반 동기화 사용 여부 (기본값: True)
        
    Returns:
        float: 관절 각도 점수 (0~100)
        
    특징:
        - 각도 기반 동기화: 동일한 각도 구간끼리 매칭하여 속도 차이 보정
        - 진행률 기반 동기화: 기존 방식 (fallback)
        - per-joint per-sample score = 100 - (|vt - vb| / max_angle_diff) * 100
    """
    # 비디오 캡처 객체 생성
    cap_b = cv2.VideoCapture(baseline_path)
    cap_t = cv2.VideoCapture(target_path)
    # 총 프레임 수 추출
    total_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
    total_t = int(cap_t.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 동적 샘플링 개선: 프레임 수에 비례한 샘플링
    frame_range_b = finish_b - start_b
    frame_range_t = finish_t - start_t
    min_frame_range = min(frame_range_b, frame_range_t)
    
    # FPS를 고려한 동적 샘플링
    fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 30.0
    fps_t = cap_t.get(cv2.CAP_PROP_FPS) or 30.0
    avg_fps = (fps_b + fps_t) / 2.0
    
    # 프레임 수에 비례한 샘플 수 계산 (최소 20, 최대 100)
    dynamic_samples = max(20, min(100, int(min_frame_range * 0.3)))
    
    # FPS가 높으면 샘플 수 증가
    if avg_fps > 45.0:
        dynamic_samples = int(dynamic_samples * 1.2)
    elif avg_fps < 25.0:
        dynamic_samples = int(dynamic_samples * 0.8)
    
    # 사용자가 지정한 num_samples와 동적 샘플 수 중 더 적절한 값 선택
    final_num_samples = min(num_samples, dynamic_samples) if num_samples > 0 else dynamic_samples
    
    # 상세한 분석 시작 로그
    logger.info(f"[Angle-Detail] 관절 각도 분석 시작:")
    logger.info(f"  - Baseline: {total_b} 프레임 (분석 구간: {start_b}-{finish_b}, 범위: {frame_range_b} 프레임)")
    logger.info(f"  - Target: {total_t} 프레임 (분석 구간: {start_t}-{finish_t}, 범위: {frame_range_t} 프레임)")
    logger.info(f"  - FPS: Baseline={fps_b:.1f}, Target={fps_t:.1f}, 평균={avg_fps:.1f}")
    logger.info(f"  - 동적 샘플링: 최소 프레임 범위={min_frame_range}, 기본 샘플={dynamic_samples}")
    logger.info(f"  - 최종 샘플 수: {final_num_samples}개 (사용자 지정: {num_samples})")
    logger.info(f"  - 최대 각도 차이: {max_angle_diff}도, 각도 기반 동기화: {use_angle_based_sync}")

    # MediaPipe 포즈 객체 생성 (컨텍스트 매니저 사용)
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_b, \
         mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_t:

        scores = []  # 점수 저장 리스트
        
        if use_angle_based_sync:
            # 각도 기반 동기화: 동일한 각도 구간끼리 매칭
            # 각도 기반 동기화: 동일 샘플 수로 각도 시퀀스 추출 후 유사 매칭
            desired_samples = final_num_samples
            scores = _calc_angle_based_sync_score(
                cap_b, cap_t, pose_b, pose_t,
                start_b, finish_b, start_t, finish_t,
                total_b, total_t, max_angle_diff,
                desired_samples
            )
        else:
            # 기존 진행률 기반 동기화 (동적 샘플링 적용)
            logger.info(f"[Angle-Detail] 진행률 기반 동기화 사용")
            progress = np.linspace(0, 1, final_num_samples)
            logger.info(f"  - 진행률 배열: {len(progress)}개 샘플 (0.0 ~ 1.0)")
            
            sample_count = 0
            for i, pr in enumerate(progress):
                # 진행률에 따른 프레임 인덱스 계산
                fi_b = int(start_b + pr * (finish_b - start_b))  # 기준 영상 프레임
                fi_t = int(start_t + pr * (finish_t - start_t))  # 타겟 영상 프레임
                
                # 프레임 범위 검증
                if fi_b >= total_b or fi_t >= total_t:
                    logger.debug(f"[Angle-Detail] 샘플 {i}: 프레임 범위 초과 (fi_b={fi_b}, fi_t={fi_t})")
                    continue
                    
                # 해당 프레임으로 이동
                cap_b.set(cv2.CAP_PROP_POS_FRAMES, fi_b)
                cap_t.set(cv2.CAP_PROP_POS_FRAMES, fi_t)
                
                # 프레임 읽기
                rb, fb = cap_b.read(); rt, ft = cap_t.read()
                if not (rb and rt):
                    logger.debug(f"[Angle-Detail] 샘플 {i}: 프레임 읽기 실패")
                    continue
                    
                # 각 프레임에서 관절 각도 추출
                Ab = _extract_angles_per_frame(fb, pose_b)  # 기준 영상 각도
                At = _extract_angles_per_frame(ft, pose_t)  # 타겟 영상 각도
                if not Ab or not At:
                    logger.debug(f"[Angle-Detail] 샘플 {i}: 관절 각도 추출 실패")
                    continue
                    
                # 각 관절별 점수 계산
                per_scores = []
                joint_details = []
                
                for k in Ab.keys():
                    vb, vt = Ab.get(k), At.get(k)  # 기준값, 타겟값
                    # 유효성 검증
                    if vb is None or vt is None or np.isnan(vb) or np.isnan(vt):
                        continue
                    # 각도 차이 계산
                    diff = abs(float(vt) - float(vb))
                    # 점수 계산: 100 - (차이/최대차이) * 100
                    score = 100.0 - (diff / max(1e-6, max_angle_diff)) * 100.0
                    clipped_score = float(np.clip(score, 0.0, 100.0))
                    per_scores.append(clipped_score)
                    
                    joint_details.append({
                        "joint": k,
                        "baseline_angle": vb,
                        "target_angle": vt,
                        "angle_diff": diff,
                        "score": clipped_score
                    })
                    
                # 해당 프레임의 평균 점수 저장
                if per_scores:
                    frame_score = float(np.mean(per_scores))
                    scores.append(frame_score)
                    sample_count += 1
                    
                    logger.debug(f"[Angle-Detail] 샘플 {i} (진행률 {pr:.3f}):")
                    logger.debug(f"  - 프레임: Baseline={fi_b}, Target={fi_t}")
                    logger.debug(f"  - 유효 관절: {len(per_scores)}개")
                    logger.debug(f"  - 평균 점수: {frame_score:.1f}")
                    
                    # 상세한 관절별 정보 (디버그 레벨)
                    for detail in joint_details:
                        logger.debug(f"    - {detail['joint']}: {detail['baseline_angle']:.1f}° vs {detail['target_angle']:.1f}° (차이: {detail['angle_diff']:.1f}°, 점수: {detail['score']:.1f})")
            
            logger.info(f"[Angle-Detail] 진행률 기반 분석 완료: {sample_count}/{len(progress)} 샘플 성공")

    # 리소스 해제
    cap_b.release(); cap_t.release()
    
    # 최종 결과 계산 및 상세 로그
    logger.info(f"[Angle-Detail] 최종 관절 각도 점수 계산:")
    logger.info(f"  - 유효한 점수 수: {len(scores)}개")
    
    if scores:
        final_score = float(np.mean(scores))
        
        # 각 샘플별 점수 상세 출력
        logger.info(f"[Angle-Detail] 샘플별 점수 상세:")
        for i, score in enumerate(scores):
            logger.info(f"  - 샘플 {i}: {score:.1f}점")
        
        # 통계 정보
        if len(scores) > 1:
            logger.info(f"[Angle-Detail] 점수 통계:")
            logger.info(f"  - 최고점: {max(scores):.1f}")
            logger.info(f"  - 최저점: {min(scores):.1f}")
            logger.info(f"  - 표준편차: {np.std(scores):.2f}")
            logger.info(f"  - 중간값: {np.median(scores):.1f}")
        
        # 최종 점수 계산 과정
        logger.info(f"[Angle-Detail] 최종 점수 계산:")
        logger.info(f"  - 평균 계산: {np.mean(scores):.4f} = {final_score:.1f}")
        logger.info(f"  - 수식: (모든 샘플 점수의 합) / (샘플 수) = {sum(scores):.1f} / {len(scores)} = {final_score:.1f}")
        
        logger.info(f"[Angle-Score] SUCCESS - {len(scores)} samples, average score: {final_score:.1f}")
        return final_score
    else:
        logger.error(f"[Angle-Score] FAILED - No valid scores calculated")
        logger.error(f"[Angle-Score] REASON - 포즈 랜드마크 감지 실패 또는 각도 매칭 실패")
        # 점수 산정 불가 - None 반환하여 오류 상태 명시
        return None


def score_all(baseline_path: str, target_path: str, use_no_top_method: bool = False, save_key_images: bool = False) -> dict:
    """
    모든 점수를 계산하는 메인 함수
    
    Args:
        baseline_path: 기준 영상 경로
        target_path: 분석할 영상 경로
        use_no_top_method: 백스윙 탑 무시 방법 사용 여부 (기본값: False)
        save_key_images: 키 프레임 이미지 저장 여부 (기본값: False)
        
    Returns:
        dict: 모든 점수와 메타데이터를 포함한 딕셔너리
    """
    from .video_processor import detect_swing_key_frames, detect_swing_key_times
    from ssswing.swing_phase_detector import detect_swing_phases_no_top, detect_swing_phases_simple

    if use_no_top_method:
        # 백스윙 탑 무시 방법 사용
        logger.info("[INFO] 백스윙 탑 무시 방법을 사용합니다 (임팩트 중심 2구간 비교)")
        
        # MediaPipe를 사용한 랜드마크 추출 및 스윙 단계 감지
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        # 동일 파일일 경우: 속도 점수 100으로 단축 경로 (강화된 로직)
        same_file = False
        try:
            if os.path.exists(baseline_path) and os.path.exists(target_path):
                # 1차: os.path.samefile 시도
                same_file = os.path.samefile(baseline_path, target_path)
                logger.info(f"[Same-File] os.path.samefile success: {same_file}")
        except (OSError, ValueError) as e:
            logger.warning(f"[Same-File] os.path.samefile failed: {e}")
            try:
                # 2차: 절대 경로 비교
                abs_baseline = os.path.abspath(baseline_path)
                abs_target = os.path.abspath(target_path)
                same_file = (abs_baseline == abs_target)
                logger.info(f"[Same-File] abs path comparison: {same_file}")
                
                # 3차: 파일 크기와 이름 비교 (추가 검증)
                if not same_file:
                    size_baseline = os.path.getsize(baseline_path)
                    size_target = os.path.getsize(target_path)
                    name_baseline = os.path.basename(baseline_path)
                    name_target = os.path.basename(target_path)
                    
                    if size_baseline == size_target and name_baseline == name_target:
                        same_file = True
                        logger.info(f"[Same-File] Size and name match: {same_file}")
                        
            except Exception as e2:
                logger.error(f"[Same-File] All comparison methods failed: {e2}")
                same_file = False

        # 기준 영상 처리
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            cap_b = cv2.VideoCapture(baseline_path)
            fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 30.0
            landmarks_b = []
            
            while True:
                ret, frame = cap_b.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                landmarks_b.append(result.pose_landmarks)
            cap_b.release()
        
        # 타겟 영상 처리
        if same_file:
            fps_t = fps_b
            landmarks_t = landmarks_b
        else:
            with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                cap_t = cv2.VideoCapture(target_path)
                fps_t = cap_t.get(cv2.CAP_PROP_FPS) or 30.0
                landmarks_t = []
                
                while True:
                    ret, frame = cap_t.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb)
                    landmarks_t.append(result.pose_landmarks)
                cap_t.release()
        
        # 웨글 제거: detect_swing_key_frames로 웨글 종료 지점 감지
        from .video_processor import detect_swing_key_frames
        start_rel_b, top_rel_b, finish_rel_b, waggle_end_b = detect_swing_key_frames(baseline_path)
        start_rel_t, top_rel_t, finish_rel_t, waggle_end_t = detect_swing_key_frames(target_path)
        
        # 웨글 제거: start는 waggle_end부터 시작 (웨글 부분 제거)
        start_b = waggle_end_b if waggle_end_b is not None else 0
        start_t = waggle_end_t if waggle_end_t is not None else 0
        
        # 상대 프레임을 절대 프레임으로 변환
        finish_b = start_b + finish_rel_b if finish_rel_b is not None else start_b + 60
        finish_t = start_t + finish_rel_t if finish_rel_t is not None else start_t + 60
        
        # 기본값 설정 (검출 실패 시)
        if start_b is None: start_b, finish_b = 0, 60
        if start_t is None: start_t, finish_t = 0, 60
        
        logger.info(f"[Waggle-Removal] Baseline: waggle_end={waggle_end_b}, start={start_b}, finish={finish_b}")
        logger.info(f"[Waggle-Removal] Target: waggle_end={waggle_end_t}, start={start_t}, finish={finish_t}")
        
        # 웨글 제거된 프레임을 기준으로 시간 계산
        try:
            # FPS 추출
            cap_b = cv2.VideoCapture(baseline_path)
            cap_t = cv2.VideoCapture(target_path)
            fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 30.0
            fps_t = cap_t.get(cv2.CAP_PROP_FPS) or 30.0
            cap_b.release()
            cap_t.release()
            
            # 웨글 제거된 프레임을 초 단위로 변환
            sb_s = start_b / fps_b  # 웨글 제거된 시작 시간
            fb_s = finish_b / fps_b  # 피니시 시간
            st_s = start_t / fps_t  # 웨글 제거된 시작 시간
            ft_s = finish_t / fps_t  # 피니시 시간
            
            logger.info(f"[Waggle-Removed-Times] Baseline: start={sb_s:.2f}s, finish={fb_s:.2f}s")
            logger.info(f"[Waggle-Removed-Times] Target: start={st_s:.2f}s, finish={ft_s:.2f}s")
            
        except Exception as e:
            logger.warning(f"[Waggle-Removed-Times] Failed to calculate times: {e}")
            # 폴백: 기본값 사용
            sb_s = 0.0
            fb_s = 1.0
            st_s = 0.0
            ft_s = 1.0
        
        # 메타데이터 구성
        frames_meta = {
            "baseline": {"address": start_b, "finish": finish_b},
            "target": {"address": start_t, "finish": finish_t},
        }
        fps_meta = {"baseline": fps_b, "target": fps_t}
        
        # 3등분 궤적 기반 속도 점수 계산 (거리/시간 비율)
        try:
            if same_file:
                swing_speed_score, speed_meta = 100.0, {"reason": "identical_video_shortcircuit", "final_speed_score": 100.0}
            else:
                swing_speed_score, speed_meta = calc_speed_score_three_segments_trajectory(
                    landmarks_b=landmarks_b,
                    landmarks_t=landmarks_t,
                    start_b=start_b,
                    finish_b=finish_b,
                    start_t=start_t,
                    finish_t=finish_t,
                    fps_b=fps_b,
                    fps_t=fps_t,
                )
        except Exception as e:
            logger.error(f"[Speed-Score] FAILED - three_segments_trajectory 실패: {e}")
            logger.error(f"[Speed-Score] REASON - 세그먼트 궤적 기반 분석 오류")
            # 속도 점수 계산 실패 - None으로 설정
            swing_speed_score, speed_meta = None, {"error": "three_segments_calculation_failed", "reason": str(e)}
        
        # 속도 점수 계산 실패 시 오류 처리
        if swing_speed_score is None:
            logger.error(f"[Score-All] FAILED - 속도 점수 계산 실패")
            logger.error(f"[Score-All] REASON - 시간 계산 오류 또는 데이터 부족")
            # 속도 점수 없이 관절 각도 점수만으로 계산
            swing_speed_score = 0.0
            logger.warning(f"[Score-All] WARNING - 관절 각도 점수만으로 계산 예정")
        
        # 웨글 제거된 스타트와 피니시 이미지 저장
        if save_key_images:
            try:
                save_start_finish_images(baseline_path, "baseline", start_b, finish_b)
                save_start_finish_images(target_path, "target", start_t, finish_t)
            except Exception as e:
                logger.warning(f"웨글 제거된 스타트/피니시 이미지 저장 실패: {e}")
        
        # waggle 정보는 기본값으로 설정
        waggle_b = 0
        waggle_t = 0
        
    else:
        # 기존 방법 사용 (백스윙 탑 포함)
        logger.info("[INFO] 기존 방법을 사용합니다 (백스윙 탑 포함)")
        
        # 프레임 기반 검출 (웨글 제거 포함)
        start_rel_b, top_rel_b, finish_rel_b, waggle_end_b = detect_swing_key_frames(baseline_path)
        start_rel_t, top_rel_t, finish_rel_t, waggle_end_t = detect_swing_key_frames(target_path)
        
        # 웨글 제거: start는 waggle_end부터 시작 (웨글 부분 제거)
        start_b = waggle_end_b if waggle_end_b is not None else 0
        start_t = waggle_end_t if waggle_end_t is not None else 0
        
        # 상대 프레임을 절대 프레임으로 변환
        top_b = start_b + top_rel_b if top_rel_b is not None else start_b + 15
        finish_b = start_b + finish_rel_b if finish_rel_b is not None else start_b + 60
        top_t = start_t + top_rel_t if top_rel_t is not None else start_t + 15
        finish_t = start_t + finish_rel_t if finish_rel_t is not None else start_t + 60
        
        # 기본값 설정 (검출 실패 시)
        if start_b is None: start_b, top_b, finish_b = 0, 15, 60
        if start_t is None: start_t, top_t, finish_t = 0, 15, 60
        
        logger.info(f"[Waggle-Removal] Baseline: waggle_end={waggle_end_b}, start={start_b}, top={top_b}, finish={finish_b}")
        logger.info(f"[Waggle-Removal] Target: waggle_end={waggle_end_t}, start={start_t}, top={top_t}, finish={finish_t}")
        
        # 스타트와 피니시 이미지 저장 (웨글 제거 후)
        if save_key_images:
            try:
                save_start_finish_images(baseline_path, "baseline", start_b, finish_b)
                save_start_finish_images(target_path, "target", start_t, finish_t)
            except Exception as e:
                logger.warning(f"스타트/피니시 이미지 저장 실패: {e}")

        # 초 단위 정밀 키타임 (슬로우 업샘플 적용)
        # 타겟은 slow_factor=2로 보정하여 정확도 향상
        sb_s, tb_s, fb_s, _ = detect_swing_key_times(baseline_path, slow_factor=1.0)
        st_s, tt_s, ft_s, _ = detect_swing_key_times(target_path, slow_factor=2.0)

        # 초 기반 속도 점수 (0.01초 해상도)
        frames_meta = {
            "baseline": {"start": start_b, "top": top_b, "finish": finish_b},
            "target": {"start": start_t, "top": top_t, "finish": finish_t},
        }
        
        # FPS 메타데이터 추출
        capb = cv2.VideoCapture(baseline_path); fps_b = capb.get(cv2.CAP_PROP_FPS) or 0.0; capb.release()
        capt = cv2.VideoCapture(target_path); fps_t = capt.get(cv2.CAP_PROP_FPS) or 0.0; capt.release()
        fps_meta = {"baseline": fps_b, "target": fps_t}

        # 기존 속도 점수 계산
        try:
            swing_speed_score, speed_meta = calc_speed_score_seconds(
                sb_s, tb_s, fb_s,
                st_s, tt_s, ft_s,
                frames_meta=frames_meta,
                fps_meta=fps_meta,
            )
        except Exception as e:
            logger.error(f"[Speed-Score] FAILED - calc_speed_score_seconds 실패: {e}")
            logger.error(f"[Speed-Score] REASON - 시간 계산 오류 또는 데이터 부족")
            # 속도 점수 계산 실패 - 0점으로 설정
            swing_speed_score, speed_meta = 0.0, {"error": "speed_calculation_failed", "reason": str(e)}
    
    # 관절 각도 점수 계산 (환경변수로 동기화 방식 제어)
    use_angle_sync_env = os.getenv("USE_ANGLE_SYNC", "true").strip().lower()
    use_angle_sync = use_angle_sync_env in ("1", "true", "yes", "y")
    logger.info(f"[Angle-Detail] 동기화 방식: {'각도 기반' if use_angle_sync else '진행률 기반(1:1)'} (USE_ANGLE_SYNC={use_angle_sync_env})")
    joint_angle_score = calc_joint_angle_score(
        baseline_path, target_path, start_b, finish_b, start_t, finish_t,
        use_angle_based_sync=use_angle_sync
    )
    
    # 관절 각도 점수 계산 실패 시 오류 처리
    if joint_angle_score is None:
        logger.error(f"[Score-All] FAILED - 관절 각도 점수 계산 실패")
        logger.error(f"[Score-All] REASON - 포즈 랜드마크 감지 실패 또는 각도 매칭 실패")
        # 관절 각도 점수 없이 속도 점수만으로 계산
        joint_angle_score = 0.0
        total_score = swing_speed_score
        logger.warning(f"[Score-All] WARNING - 속도 점수만으로 계산: {total_score:.1f}")
    else:
        # 종합 점수 계산 (두 점수의 평균)
        total_score = float((swing_speed_score + joint_angle_score) / 2.0)
    
    # 최종 점수 검증
    if swing_speed_score == 0.0 and joint_angle_score == 0.0:
        logger.error(f"[Score-All] CRITICAL FAILURE - 모든 점수 계산 실패")
        logger.error(f"[Score-All] REASON - 속도 점수와 관절 각도 점수 모두 계산 불가")
        logger.error(f"[Score-All] ACTION - 분석 불가능, 사용자에게 오류 보고 필요")

    # 결과 반환
    result = {
        "swing_speed_score": round(swing_speed_score, 1),      # 스윙 속도 점수
        "joint_angle_score": round(joint_angle_score, 1),      # 관절 각도 점수
        "total_score": round(total_score, 2),                  # 종합 점수
        "speed_meta": speed_meta,  # 속도 분석 메타데이터
        "method_used": "no_top_impact_based" if use_no_top_method else "traditional_with_top"  # 사용된 방법
    }
    
    # 오류 정보 추가
    if swing_speed_score == 0.0 and joint_angle_score == 0.0:
        result["error"] = "analysis_failed"
        result["error_reason"] = "모든 점수 계산 실패 - 포즈 랜드마크 감지 실패 또는 데이터 부족"
        result["error_details"] = {
            "speed_score_failed": swing_speed_score == 0.0,
            "angle_score_failed": joint_angle_score == 0.0
        }
    elif swing_speed_score == 0.0:
        result["warning"] = "speed_score_failed"
        result["warning_reason"] = "속도 점수 계산 실패 - 시간 계산 오류 또는 데이터 부족"
    elif joint_angle_score == 0.0:
        result["warning"] = "angle_score_failed"
        result["warning_reason"] = "관절 각도 점수 계산 실패 - 포즈 랜드마크 감지 실패 또는 각도 매칭 실패"
    
    # 프레임 정보 (방법에 따라 다르게 구성)
    if use_no_top_method:
        result["frames"] = {
            "baseline": {"address": start_b, "finish": finish_b, "waggle_end": waggle_b},
            "target": {"address": start_t, "finish": finish_t, "waggle_end": waggle_t},
        }
    else:
        result["frames"] = {
            "baseline": {"start": start_b, "top": top_b, "finish": finish_b, "waggle_end": waggle_b},
            "target": {"start": start_t, "top": top_t, "finish": finish_t, "waggle_end": waggle_t},
        }
    
    return result


def score_all_no_top(baseline_path: str, target_path: str, save_key_images: bool = True) -> dict:
    """
    백스윙 탑을 무시한 스윙 분석을 수행하는 편의 함수
    
    Args:
        baseline_path: 기준 영상 경로
        target_path: 분석할 영상 경로
        
    Returns:
        dict: 백스윙 탑 무시 방법으로 계산된 모든 점수와 메타데이터
    """
    return score_all(baseline_path, target_path, use_no_top_method=True, save_key_images=save_key_images)


def save_swing_key_frames_limited_debug(video_path: str, start_frame: int, top_rel: int, finish_rel: int) -> None:
    """
    디버깅용 키 프레임 이미지 저장 함수
    
    Args:
        video_path: 영상 파일 경로
        start_frame: 시작 프레임
        top_rel: 백스윙 탑 상대 프레임
        finish_rel: 피니시 상대 프레임
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"[Debug-Images] 영상을 열 수 없습니다: {video_path}")
            return
        
        # 출력 디렉토리 생성
        debug_dir = os.path.join(PROJECT_ROOT, 'ssswing', 'debug_key_images')
        os.makedirs(debug_dir, exist_ok=True)
        
        # 저장할 프레임 정보
        frames_to_save = [
            ("start", start_frame),
            ("top", start_frame + top_rel if top_rel else start_frame + 15),
            ("finish", start_frame + finish_rel if finish_rel else start_frame + 60)
        ]
        
        video_name = os.path.basename(video_path).replace('.mp4', '').replace('.webm', '')
        
        for frame_name, frame_idx in frames_to_save:
            if frame_idx is None or frame_idx < 0:
                logger.warning(f"[Debug-Images] {frame_name} 프레임이 유효하지 않습니다: {frame_idx}")
                continue
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                filename = f"{video_name}_{frame_name}_frame_{frame_idx}.jpg"
                out_path = os.path.join(debug_dir, filename)
                cv2.imwrite(out_path, frame)
                logger.info(f"[Debug-Images] {frame_name} 프레임 저장 완료: {out_path}")
            else:
                logger.warning(f"[Debug-Images] {frame_name} 프레임 {frame_idx}을 읽을 수 없습니다")
        
        cap.release()
        
    except Exception as e:
        logger.error(f"[Debug-Images] 키 프레임 저장 중 오류 발생: {e}")


def save_swing_key_frames_impact_limited(video_path: str, video_name: str,
                                        address_frame: int, impact_frame: int,
                                        finish_frame: int) -> None:
    """
    임팩트 이미지를 저장하지 않고, 시작(start)과 피니시(finish) 2장만 저장합니다.

    Args:
        video_path: 영상 파일 경로
        video_name: 영상 구분명 (baseline 또는 target)
        address_frame: 어드레스 프레임
        impact_frame: 임팩트 프레임
        finish_frame: 피니시 프레임
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"영상을 열 수 없습니다: {video_path}")
            return

        # 출력 디렉토리: PROJECT_ROOT/ssswing/swing_key_images/(pro|user)
        # video_name이 'baseline'이면 pro, 'target'이면 user로 매핑
        subdir = 'pro' if str(video_name).lower() == 'baseline' else 'user'
        out_dir = os.path.join(PROJECT_ROOT, 'ssswing', 'swing_key_images', subdir)
        os.makedirs(out_dir, exist_ok=True)

        # 파일명: start/finish만 저장 (임팩트 제거)
        frames_to_save = [
            ("start", address_frame),
            ("finish", finish_frame),
        ]

        for frame_name, frame_idx in frames_to_save:
            if frame_idx is None or frame_idx < 0:
                logger.warning(f"{video_name} {frame_name} 프레임이 유효하지 않습니다: {frame_idx}")
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # 파일명: 기존 규칙과 동일 (덮어쓰기 허용)
                prefix = 'pro' if subdir == 'pro' else 'user'
                filename = f"{prefix}_{frame_name}.jpg"
                out_path = os.path.join(out_dir, filename)
                cv2.imwrite(out_path, frame)
                logger.info(f"{video_name} {frame_name} 프레임 저장 완료: {out_path} (프레임 {frame_idx})")
            else:
                logger.warning(f"{video_name} {frame_name} 프레임 {frame_idx}을 읽을 수 없습니다")

        cap.release()
    except Exception as e:
        logger.error(f"임팩트 중심 키 프레임 저장 중 오류 발생: {e}")
        try:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
        except:
            pass

def save_start_finish_images(video_path: str, video_name: str, start_frame: int, finish_frame: int) -> None:
    """
    웨글 제거 후 스타트와 피니시 이미지만 저장하는 함수
    
    Args:
        video_path: 영상 파일 경로
        video_name: 영상 구분명 (baseline 또는 target)
        start_frame: 스윙 시작 프레임 (웨글 제거 후)
        finish_frame: 스윙 피니시 프레임
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"영상을 열 수 없습니다: {video_path}")
            return

        # 출력 디렉토리: PROJECT_ROOT/ssswing/swing_key_images/(pro|user)
        subdir = 'pro' if str(video_name).lower() == 'baseline' else 'user'
        out_dir = os.path.join(PROJECT_ROOT, 'ssswing', 'swing_key_images', subdir)
        os.makedirs(out_dir, exist_ok=True)

        # 스타트와 피니시 이미지 저장
        frames_to_save = [
            ("start", start_frame),
            ("finish", finish_frame),
        ]

        for frame_name, frame_idx in frames_to_save:
            if frame_idx is None or frame_idx < 0:
                logger.warning(f"{video_name} {frame_name} 프레임이 유효하지 않습니다: {frame_idx}")
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # 파일명: 웨글 제거 후 스타트/피니시
                prefix = 'pro' if subdir == 'pro' else 'user'
                filename = f"{prefix}_{frame_name}_waggle_removed.jpg"
                out_path = os.path.join(out_dir, filename)
                cv2.imwrite(out_path, frame)
                logger.info(f"{video_name} {frame_name} 프레임 저장 완료 (웨글 제거 후): {out_path} (프레임 {frame_idx})")
            else:
                logger.warning(f"{video_name} {frame_name} 프레임 {frame_idx}을 읽을 수 없습니다")

        cap.release()
    except Exception as e:
        logger.error(f"스타트/피니시 이미지 저장 중 오류 발생: {e}")
        try:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
        except:
            pass


def save_swing_key_frames_limited(video_path: str, video_name: str, start_frame: int, top_frame: int, 
                                 finish_frame: int) -> None:
    """
    스윙 키 프레임들을 제한적으로 이미지로 저장합니다 (프로3장, 유저3장만).
    
    Args:
        video_path: 영상 파일 경로
        video_name: 영상 구분명 (baseline 또는 target)
        start_frame: 스윙 시작 프레임
        top_frame: 백스윙 탑 프레임
        finish_frame: 스윙 피니쉬 프레임
    
    저장되는 이미지 (3장만):
    - {video_name}_start_frame.jpg: 스윙 시작 시점
    - {video_name}_top_frame.jpg: 백스윙 탑 시점
    - {video_name}_finish_frame.jpg: 스윙 피니쉬 시점
    
    특징:
    - 웨글 이미지는 저장하지 않음 (로직적으로만 인식)
    - 프로와 유저 영상에서 각각 3장씩만 저장
    - API 폴더에 깔끔하게 정리된 이미지만 저장
    """
    try:
        # 영상 파일 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"영상을 열 수 없습니다: {video_path}")
            return
        
        # 저장할 프레임 정보 (3장만: 스타트, 백스윙탑, 피니시)
        frames_to_save = {
            "start": start_frame,    # 스타트 (어드레스)
            "top": top_frame,        # 백스윙탑
            "finish": finish_frame   # 피니시
        }
        
        # 각 키 프레임을 이미지로 저장
        for frame_name, frame_idx in frames_to_save.items():
            if frame_idx is None or frame_idx < 0:
                logger.warning(f"{video_name} {frame_name} 프레임이 유효하지 않습니다: {frame_idx}")
                continue
                
            # 지정된 프레임으로 이동
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # 파일명 생성 (타임스탬프 포함하여 중복 방지)
                import time
                timestamp = int(time.time())
                filename = f"{video_name}_{frame_name}_frame_{timestamp}.jpg"
                
                # 이미지 저장
                cv2.imwrite(filename, frame)
                logger.info(f"{video_name} {frame_name} 프레임 저장 완료: {filename} (프레임 {frame_idx})")
            else:
                logger.warning(f"{video_name} {frame_name} 프레임 {frame_idx}을 읽을 수 없습니다")
        
        cap.release()
        
    except Exception as e:
        logger.error(f"스윙 키 프레임 저장 중 오류 발생: {e}")
        # 영상 파일이 열려있다면 해제
        try:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
        except:
            pass


