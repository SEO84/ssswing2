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
                                total_b, total_t, max_angle_diff):
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
    baseline_angles = _extract_angle_sequence(cap_b, pose_b, start_b, finish_b, total_b)
    target_angles = _extract_angle_sequence(cap_t, pose_t, start_t, finish_t, total_t)
    
    if not baseline_angles or not target_angles:
        return []
    
    # 2단계: 각도 기반 동기화 매칭
    matched_pairs = _match_angles_by_similarity(baseline_angles, target_angles)
    
    # 3단계: 매칭된 쌍들로 점수 계산
    scores = []
    for baseline_data, target_data in matched_pairs:
        per_scores = []
        for joint_name in baseline_data['angles'].keys():
            vb = baseline_data['angles'][joint_name]
            vt = target_data['angles'][joint_name]
            
            if vb is None or vt is None or np.isnan(vb) or np.isnan(vt):
                continue
                
            # 각도 차이 계산
            diff = abs(float(vt) - float(vb))
            # 점수 계산: 100 - (차이/최대차이) * 100
            score = 100.0 - (diff / max(1e-6, max_angle_diff)) * 100.0
            per_scores.append(float(np.clip(score, 0.0, 100.0)))
        
        if per_scores:
            scores.append(float(np.mean(per_scores)))
    
    return scores


def _extract_angle_sequence(cap, pose, start_frame, finish_frame, total_frames):
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
    
    # 샘플링 간격 계산 (최대 50개 샘플)
    frame_range = finish_frame - start_frame
    sample_interval = max(1, frame_range // 50)
    
    for frame_idx in range(start_frame, finish_frame + 1, sample_interval):
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
        
        # 유사도가 임계값 이하인 경우만 매칭
        if best_match and best_similarity < 30.0:  # 30도 이하 차이
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

    # MediaPipe 포즈 객체 생성 (컨텍스트 매니저 사용)
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_b, \
         mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_t:

        scores = []  # 점수 저장 리스트
        
        if use_angle_based_sync:
            # 각도 기반 동기화: 동일한 각도 구간끼리 매칭
            scores = _calc_angle_based_sync_score(
                cap_b, cap_t, pose_b, pose_t,
                start_b, finish_b, start_t, finish_t,
                total_b, total_t, max_angle_diff
            )
        else:
            # 기존 진행률 기반 동기화
            progress = np.linspace(0, 1, num_samples)
            
            for pr in progress:
                # 진행률에 따른 프레임 인덱스 계산
                fi_b = int(start_b + pr * (finish_b - start_b))  # 기준 영상 프레임
                fi_t = int(start_t + pr * (finish_t - start_t))  # 타겟 영상 프레임
                
                # 프레임 범위 검증
                if fi_b >= total_b or fi_t >= total_t:
                    continue
                    
                # 해당 프레임으로 이동
                cap_b.set(cv2.CAP_PROP_POS_FRAMES, fi_b)
                cap_t.set(cv2.CAP_PROP_POS_FRAMES, fi_t)
                
                # 프레임 읽기
                rb, fb = cap_b.read(); rt, ft = cap_t.read()
                if not (rb and rt):
                    continue
                    
                # 각 프레임에서 관절 각도 추출
                Ab = _extract_angles_per_frame(fb, pose_b)  # 기준 영상 각도
                At = _extract_angles_per_frame(ft, pose_t)  # 타겟 영상 각도
                if not Ab or not At:
                    continue
                    
                # 각 관절별 점수 계산
                per_scores = []
                for k in Ab.keys():
                    vb, vt = Ab.get(k), At.get(k)  # 기준값, 타겟값
                    # 유효성 검증
                    if vb is None or vt is None or np.isnan(vb) or np.isnan(vt):
                        continue
                    # 각도 차이 계산
                    diff = abs(float(vt) - float(vb))
                    # 점수 계산: 100 - (차이/최대차이) * 100
                    score = 100.0 - (diff / max(1e-6, max_angle_diff)) * 100.0
                    per_scores.append(float(np.clip(score, 0.0, 100.0)))
                    
                # 해당 프레임의 평균 점수 저장
                if per_scores:
                    scores.append(float(np.mean(per_scores)))

    # 리소스 해제
    cap_b.release(); cap_t.release()
    # 전체 점수의 평균 반환 (점수가 없으면 0.0)
    return float(np.mean(scores)) if scores else 0.0


def score_all(baseline_path: str, target_path: str) -> dict:
    """
    모든 점수를 계산하는 메인 함수
    
    Args:
        baseline_path: 기준 영상 경로
        target_path: 분석할 영상 경로
        
    Returns:
        dict: 모든 점수와 메타데이터를 포함한 딕셔너리
    """
    from .video_processor import detect_swing_key_frames, detect_swing_key_times

    # 프레임 기반 검출 (각도 계산용 프레임 범위)
    start_b, top_b, finish_b, waggle_b = detect_swing_key_frames(baseline_path)
    start_t, top_t, finish_t, waggle_t = detect_swing_key_frames(target_path)
    # 기본값 설정 (검출 실패 시)
    if start_b is None: start_b, top_b, finish_b = 0, 15, 60
    if start_t is None: start_t, top_t, finish_t = 0, 15, 60

    # 스윙 키 프레임 이미지 저장 중단 (이미지 저장 제한)
    # save_swing_key_frames_limited(baseline_path, "baseline", start_b, top_b, finish_b)
    # save_swing_key_frames_limited(target_path, "target", start_t, top_t, finish_t)

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

    # 속도 점수 계산
    swing_speed_score, speed_meta = calc_speed_score_seconds(
        sb_s, tb_s, fb_s,
        st_s, tt_s, ft_s,
        frames_meta=frames_meta,
        fps_meta=fps_meta,
    )
    
    # 관절 각도 점수 계산
    joint_angle_score = calc_joint_angle_score(baseline_path, target_path, start_b, finish_b, start_t, finish_t)
    
    # 종합 점수 계산 (두 점수의 평균)
    total_score = float((swing_speed_score + joint_angle_score) / 2.0)

    # 결과 반환
    return {
        "swing_speed_score": round(swing_speed_score, 1),      # 스윙 속도 점수
        "joint_angle_score": round(joint_angle_score, 1),      # 관절 각도 점수
        "total_score": round(total_score, 2),                  # 종합 점수
        "frames": {  # 프레임 정보
            "baseline": {"start": start_b, "top": top_b, "finish": finish_b, "waggle_end": waggle_b},
            "target": {"start": start_t, "top": top_t, "finish": finish_t, "waggle_end": waggle_t},
        },
        "speed_meta": speed_meta,  # 속도 분석 메타데이터
    }


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


