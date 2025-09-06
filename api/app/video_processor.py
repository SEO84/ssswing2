import os
import sys
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 삭제된 모듈들 import 제거
# from ssswing.video_generator import generate_comparison_video
# from ssswing.video_utils import draw_pose_on_frame
import mediapipe as mp

mp_pose = mp.solutions.pose


def detect_camera_view(pose_landmarks) -> str:
    """어깨 좌우 x 거리로 정면/사이드 시점 판별."""
    ls = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    dist_x = abs(ls.x - rs.x)
    return 'front' if dist_x > 0.25 else 'side'


def detect_waggle_and_swing_start(video_path: str, window_size: int = 10, movement_threshold: float = 0.01) -> int:
    """웨글 종료(=스윙 시작) 프레임 추정. 카메라 시점에 따라 사용 축을 달리함.
    front: y 변동, side: x 변동.
    """
    cap = cv2.VideoCapture(video_path)
    mp_p = mp_pose

    hand_x, hand_y, frame_idx = [], [], []
    camera_view = None
    with mp_p.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                if camera_view is None:
                    camera_view = detect_camera_view(result.pose_landmarks)
                rw = result.pose_landmarks.landmark[mp_p.PoseLandmark.RIGHT_WRIST]
                hand_x.append(rw.x)
                hand_y.append(rw.y)
                frame_idx.append(fi)
            fi += 1
    cap.release()

    if len(hand_x) < window_size + 2:
        return 0

    coord = np.array(hand_y if camera_view == 'front' else hand_x)
    # 이동 평균으로 스무딩 후 변화량 확인
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(coord, kernel, mode='valid')
    diffs = np.abs(np.diff(smoothed))
    for i, val in enumerate(diffs):
        if val > movement_threshold:
            # 스무딩에 의한 오프셋 보정
            return frame_idx[min(i + window_size, len(frame_idx)-1)]
    return 0


def detect_swing_key_frames(video_path: str):
    """카메라 시점 분기 + 웨글 제거를 포함한 start/top/finish 감지(강화).
    - top: 축 극값 보조 + 속도(미분) 0 교차점 우선
    - finish: 속도 크기 하강 + 자세(오른팔 각도) 조건 + 시간 상한
    반환: (start_rel=0, top_rel, finish_rel, waggle_end_abs)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 1) 웨글 종료/스윙 시작(웨글 기준) 추정
    waggle_end = detect_waggle_and_swing_start(video_path)
    # 1-2) 테이크백 시작(손 위치 이동 + 팔꿈치 각도 변화) 추정
    takeback_start = detect_takeback_start(video_path)
    # 두 기준 중 더 뒤쪽(보수적) 프레임을 테이크백 시작 후보로 사용
    waggle_end = int(waggle_end)
    takeback_start = int(takeback_start)
    takeback_start_abs = max(waggle_end, takeback_start)
    waggle_end = max(0, min(waggle_end, total-1))
    takeback_start_abs = max(0, min(takeback_start_abs, total-1))

    # 1-3) 어드레스 구간(정지 상태) 시작 프레임을 테이크백 직전으로부터 역방향으로 탐색
    address_abs = detect_address_before_takeback(video_path, takeback_start_abs)
    robust_start = address_abs  # 최종 자르기 시작 프레임은 어드레스 시작

    # 2) 시점과 손목 궤적으로 top/finish 산정
    mp_p = mp_pose
    hand_x, hand_y, hand_z, frame_idx = [], [], [], []
    camera_view = None
    cap = cv2.VideoCapture(video_path)
    # 최적화된 MediaPipe 설정 (속도 우선)
    with mp_p.Pose(
        static_image_mode=False,
        model_complexity=1,  # 2에서 1로 변경 (속도 향상)
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,  # 0.5에서 0.6으로 증가 (정확도 향상)
        min_tracking_confidence=0.6    # 0.5에서 0.6으로 증가 (정확도 향상)
    ) as pose:
        fi = 0
        # 프레임 샘플링으로 속도 향상 (매 2프레임마다 처리)
        sample_rate = 2
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 샘플링 적용
            if fi % sample_rate == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks:
                    if camera_view is None:
                        camera_view = detect_camera_view(result.pose_landmarks)
                    rw = result.pose_landmarks.landmark[mp_p.PoseLandmark.RIGHT_WRIST]
                    hand_x.append(rw.x)
                    hand_y.append(rw.y)
                    hand_z.append(getattr(rw, 'z', 0.0))
                    frame_idx.append(fi)
            fi += 1
    cap.release()

    if not frame_idx:
        return 0, 15, min(60, total-1)

    # start 이후 구간에서 top 계산
    # front: y 최소값(손목이 가장 위), side: start 대비 x 편차가 최대인 지점
    try:
        start_pos = next(i for i, f in enumerate(frame_idx) if f >= robust_start)
    except StopIteration:
        start_pos = 0

    # top: 속도 0 교차점 + 기하학적 극값 보조
    MIN_TOP_FRAMES = 10
    x_seg = np.asarray(hand_x[start_pos:], dtype=float)
    y_seg = np.asarray(hand_y[start_pos:], dtype=float)
    z_seg = np.asarray(hand_z[start_pos:], dtype=float) if hand_z else np.zeros_like(x_seg)
    if len(x_seg) >= 2:
        vx = np.diff(x_seg, prepend=x_seg[0])
        vy = np.diff(y_seg, prepend=y_seg[0])
        vz = np.diff(z_seg, prepend=z_seg[0])
    else:
        vx = np.zeros_like(x_seg)
        vy = np.zeros_like(y_seg)
        vz = np.zeros_like(z_seg)

    def _smooth(arr: np.ndarray, k: int = 5) -> np.ndarray:
        if k <= 1:
            return arr
        kernel = np.ones(k, dtype=float) / float(k)
        return np.convolve(arr, kernel, mode='same')

    def _smooth_local(arr: np.ndarray, k: int = 5) -> np.ndarray:
        if k <= 1:
            return arr
        kernel = np.ones(k, dtype=float) / float(k)
        return np.convolve(arr, kernel, mode='same')
    vx_s, vy_s = _smooth_local(vx, 5), _smooth_local(vy, 5)

    if camera_view == 'front':
        rel0 = int(np.argmin(y_seg)) if y_seg.size else 0
        zero = rel0
        for i in range(max(1, rel0 - 8), min(len(vy_s) - 1, rel0 + 8)):
            if vy_s[i - 1] < 0 <= vy_s[i] or vy_s[i - 1] > 0 >= vy_s[i]:
                zero = i
                break
        top_rel = max(zero, MIN_TOP_FRAMES)
    else:
        base_x = x_seg[0] if x_seg.size else 0.0
        rel0 = int(np.argmax(np.abs(x_seg - base_x))) if x_seg.size else 0
        zero = rel0
        for i in range(max(1, rel0 - 8), min(len(vx_s) - 1, rel0 + 8)):
            if vx_s[i - 1] < 0 <= vx_s[i] or vx_s[i - 1] > 0 >= vx_s[i]:
                zero = i
                break
        # 3D 거리 보조(시작점 대비)
        r0 = np.array([x_seg[0], y_seg[0], z_seg[0]])
        R = np.stack([x_seg, y_seg, z_seg], axis=1)
        dist = np.linalg.norm(R - r0, axis=1)
        dist_idx = int(np.argmax(dist)) if dist.size else zero
        top_rel = max(int(np.median([zero, rel0, dist_idx])), MIN_TOP_FRAMES)

    # FPS 기반으로 다운스윙 최소/기본 길이 보장
    cap2 = cv2.VideoCapture(video_path)
    fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
    cap2.release()
    min_down = max(int(0.35 * fps), 10)      # 최소 0.35초
    default_down = max(int(0.75 * fps), 20)  # 기본 0.75초

    end_rel = len(frame_idx) - start_pos - 1
    remain = end_rel - top_rel
    if remain < min_down:
        # top을 앞쪽으로 당겨 최소 다운스윙 길이 확보 시도
        pull = min(min_down - remain, max(top_rel - 1, 0))
        top_rel = max(0, top_rel - pull)
        remain = (len(frame_idx) - start_pos - 1) - top_rel
    # finish: 속도 크기 하강 + 자세(오른팔 각도) 조건 + 상한
    vmag = np.sqrt(vx_s**2 + vy_s**2 + vz**2)
    vmax = float(np.max(vmag)) if vmag.size else 1.0
    thresh = max(0.12 * vmax, 1e-4)
    hold = 6

    def _right_arm_angles_at(global_idx: int) -> tuple[float, float]:
        try:
            return _compute_right_arm_angles(video_path, global_idx)
        except Exception:
            return 180.0, 60.0

    finish_rel = min(end_rel, top_rel + default_down)
    for i in range(top_rel + 1, min(end_rel - hold, top_rel + default_down)):
        if np.all(vmag[i:i + hold] < thresh):
            elbow, shoulder = _right_arm_angles_at(start_pos + i)
            if elbow >= 150.0:  # 팔 펴짐 조건
                finish_rel = i
                break
    
    # 스윙 키 프레임 저장 중단 (이미지 저장 제한)
    # save_swing_key_frames_limited_debug(video_path, robust_start, top_rel, finish_rel)
    
    # start는 0으로 통일, waggle_end는 원본 절대 프레임
    return 0, top_rel, finish_rel, robust_start


# ===== 시간 기반 동기화 유틸 =====
def get_frame_time(video_path: str, frame_number: int) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(frame_number) / float(fps)


def get_frame_at_time(video_path: str, target_time: float) -> int:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return max(int(round(target_time * fps)), 0)


def detect_waggle_end_time(video_path: str) -> float:
    waggle_end_frame = detect_waggle_and_swing_start(video_path)
    return get_frame_time(video_path, waggle_end_frame)


def detect_synchronized_start(baseline_path: str, target_path: str):
    """기준 영상의 웨글 종료 시간을 타겟 영상에 투영하여 타겟 시작 프레임(절대)을 산출."""
    baseline_end_time = detect_waggle_end_time(baseline_path)
    synced_target_start_frame = get_frame_at_time(target_path, baseline_end_time)
    return baseline_end_time, synced_target_start_frame


def detect_swing_key_frames_from(video_path: str, start_abs: int):
    """주어진 절대 시작 프레임부터 상대 top/finish 산출(강화).
    반환: (0, top_rel, finish_rel)
    """
    mp_p = mp_pose
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hand_x, hand_y, frame_idx = [], [], []
    camera_view = None
    with mp_p.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                if camera_view is None:
                    camera_view = detect_camera_view(result.pose_landmarks)
                rw = result.pose_landmarks.landmark[mp_p.PoseLandmark.RIGHT_WRIST]
                hand_x.append(rw.x)
                hand_y.append(rw.y)
                frame_idx.append(fi)
            fi += 1
    cap.release()

    if not frame_idx:
        return 0, 10, min(55, total-1)

    try:
        start_pos = next(i for i, f in enumerate(frame_idx) if f >= start_abs)
    except StopIteration:
        start_pos = 0

    MIN_TOP_FRAMES = 10
    x_seg = np.asarray(hand_x[start_pos:], dtype=float)
    y_seg = np.asarray(hand_y[start_pos:], dtype=float)
    if len(x_seg) >= 2:
        vx = np.diff(x_seg, prepend=x_seg[0])
        vy = np.diff(y_seg, prepend=y_seg[0])
    else:
        vx = np.zeros_like(x_seg)
        vy = np.zeros_like(y_seg)
    vx_s, vy_s = _smooth(vx, 5), _smooth(vy, 5)
    if camera_view == 'front':
        rel0 = int(np.argmin(y_seg)) if y_seg.size else 0
        zero = rel0
        for i in range(max(1, rel0 - 8), min(len(vy_s) - 1, rel0 + 8)):
            if vy_s[i - 1] < 0 <= vy_s[i] or vy_s[i - 1] > 0 >= vy_s[i]:
                zero = i
                break
        top_rel = max(zero, MIN_TOP_FRAMES)
    else:
        base_x = x_seg[0] if x_seg.size else 0.0
        rel0 = int(np.argmax(np.abs(x_seg - base_x))) if x_seg.size else 0
        zero = rel0
        for i in range(max(1, rel0 - 8), min(len(vx_s) - 1, rel0 + 8)):
            if vx_s[i - 1] < 0 <= vx_s[i] or vx_s[i - 1] > 0 >= vx_s[i]:
                zero = i
                break
        top_rel = max(zero, MIN_TOP_FRAMES)

    # finish: 속도 하강 + 상한
    cap2 = cv2.VideoCapture(video_path)
    fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
    cap2.release()
    min_down = max(int(0.35 * fps), 10)
    default_down = max(int(0.75 * fps), 20)
    end_rel = len(frame_idx) - start_pos - 1
    vmag = np.sqrt(vx_s**2 + vy_s**2)
    vmax = float(np.max(vmag)) if vmag.size else 1.0
    thresh = max(0.12 * vmax, 1e-4)
    hold = 6
    remain = end_rel - top_rel
    if remain < min_down:
        pull = min(min_down - remain, max(top_rel - 1, 0))
        top_rel = max(0, top_rel - pull)
    finish_rel = min(end_rel, top_rel + default_down)
    for i in range(top_rel + 1, min(end_rel - hold, top_rel + default_down)):
        if np.all(vmag[i:i + hold] < thresh):
            finish_rel = i
            break
    return 0, top_rel, finish_rel


def detect_swing_key_times(video_path: str, slow_factor: float = 2.0):
    """
    시간 도메인(초)에서 키 타임을 검출하는 함수 (속도 기반, 강건)
    
    Args:
        video_path: 비디오 파일 경로
        slow_factor: 슬로우 업샘플링 배수 (기본값: 2.0)
        
    Returns:
        tuple: (start_s, top_s, finish_s, waggle_end_s)
            - start_s: 시작 시간 (초)
            - top_s: 최고점 시간 (초)
            - finish_s: 종료 시간 (초)
            - waggle_end_s: 웨글 종료 시간 (초)
            
    특징:
        - 프레임 좌표 시퀀스를 시간축으로 보간해 slow_factor 배로 초해상도 업샘플링
        - top: front는 y 최소 부근의 vy=0 교차점, side는 |x-x0| 최대 부근의 vx=0 교차점
        - finish: top 이후 속도 크기(|v|)의 이동평균이 임계값 이하로 N샘플 이상 유지되는 최초 시점
    """
    # 1) 기존 어드레스/웨글 기반 시작 프레임 계산
    start_rel, top_rel_dummy, finish_rel_dummy, waggle_abs = detect_swing_key_frames(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # 시작 시간을 초 단위로 변환
    start_time = float(waggle_abs) / float(fps)

    # 2) 손목 궤적 추출 (프레임 단위)
    mp_p = mp_pose
    hand_x, hand_y, frame_idx = [], [], []
    camera_view = None
    
    cap = cv2.VideoCapture(video_path)
    # 최적화된 MediaPipe 설정 (속도 우선)
    with mp_p.Pose(
        static_image_mode=False,
        model_complexity=1,  # 2에서 1로 변경 (속도 향상)
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,  # 0.5에서 0.6으로 증가 (정확도 향상)
        min_tracking_confidence=0.6    # 0.5에서 0.6으로 증가 (정확도 향상)
    ) as pose:
        fi = 0
        # 프레임 샘플링으로 속도 향상 (매 2프레임마다 처리)
        sample_rate = 2
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 샘플링 적용
            if fi % sample_rate == 0:
                # BGR을 RGB로 변환
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 포즈 랜드마크 검출
                result = pose.process(rgb)
                if result.pose_landmarks:
                    # 카메라 시점 감지
                    if camera_view is None:
                        camera_view = detect_camera_view(result.pose_landmarks)
                    # 오른손목 랜드마크 추출
                    rw = result.pose_landmarks.landmark[mp_p.PoseLandmark.RIGHT_WRIST]
                    # 좌표 저장
                    hand_x.append(rw.x)
                    hand_y.append(rw.y)
                    frame_idx.append(fi)
            fi += 1
    cap.release()

    # 충분한 데이터가 없는 경우 기본값 반환 (초 단위)
    if not frame_idx:
        return start_time, start_time + 0.5, min(start_time + 1.25, start_time + len(frame_idx)/max(fps,1.0)), start_time

    # 프레임 인덱스를 시간으로 변환
    times = np.array(frame_idx, dtype=float) / float(fps)
    hand_x = np.asarray(hand_x, dtype=float)
    hand_y = np.asarray(hand_y, dtype=float)

    # 3) start 이후 구간에서 시간축 보간(슬로우 효과)
    try:
        st_pos = int(np.searchsorted(times, start_time, side='left'))
    except Exception:
        st_pos = 0
    
    # 시작점 이후의 시간 세그먼트 추출
    t_seg = times[st_pos:]
    if t_seg.size == 0:
        t_seg = times
        st_pos = 0
    
    # 타임 스텝 계산 (슬로우 팩터 적용)
    if len(t_seg) >= 2:
        dt = np.median(np.diff(t_seg)) / max(slow_factor, 1.0)
    else:
        dt = (1.0 / fps) / max(slow_factor, 1.0)
    
    # 고해상도 시간축 생성
    t_high = np.arange(t_seg[0], times[-1] + 1e-9, dt)
    x_seg = hand_x[st_pos:]
    y_seg = hand_y[st_pos:]
    
    # 선형 보간으로 고해상도 좌표 생성
    x_high = np.interp(t_high, t_seg, x_seg)
    y_high = np.interp(t_high, t_seg, y_seg)

    # 파생량(속도) 계산 + 스무딩(간단한 이동평균)
    if t_high.size >= 2:
        vx = np.gradient(x_high, t_high)  # x 방향 속도
        vy = np.gradient(y_high, t_high)  # y 방향 속도
    else:
        vx = np.zeros_like(x_high)
        vy = np.zeros_like(y_high)

    def moving_avg(arr: np.ndarray, k: int) -> np.ndarray:
        """이동 평균을 계산하는 함수"""
        if k <= 1:
            return arr
        kernel = np.ones(k, dtype=float) / float(k)
        return np.convolve(arr, kernel, mode='same')

    # 창 크기는 30~60Hz 샘플링을 가정해 5~9샘플 정도
    win = max(5, int(round(0.08 / max(np.median(np.diff(t_high)) if t_high.size > 1 else 0.033, 1e-3))))
    vx_s = moving_avg(vx, win)
    vy_s = moving_avg(vy, win)

    # 4) top 검출
    MIN_TOP_TIME = max(10.0 / fps, 0.15)  # 최소 top 시간
    
    if camera_view == 'front':
        # 정면 시점: y 최소값 부근에서 속도 0 교차점 찾기
        rel_idx0 = int(np.argmin(y_high)) if y_high.size else 0
        # 속도 0 교차점 찾기(부호 변화)
        zero_idx = rel_idx0
        for i in range(max(1, rel_idx0 - win), min(len(vy_s) - 1, rel_idx0 + win)):
            if vy_s[i - 1] < 0 <= vy_s[i] or vy_s[i - 1] > 0 >= vy_s[i]:
                zero_idx = i
                break
        rel_idx = zero_idx
    else:
        # 사이드 시점: x 편차 최대값 부근에서 속도 0 교차점 찾기
        base_x = x_high[0] if x_high.size else 0.0
        rel_idx0 = int(np.argmax(np.abs(x_high - base_x))) if x_high.size else 0
        zero_idx = rel_idx0
        for i in range(max(1, rel_idx0 - win), min(len(vx_s) - 1, rel_idx0 + win)):
            if vx_s[i - 1] < 0 <= vx_s[i] or vx_s[i - 1] > 0 >= vx_s[i]:
                zero_idx = i
                break
        rel_idx = zero_idx
    
    # top 시간 계산 및 최소값 검증
    top_time = t_high[rel_idx] if t_high.size else (start_time + MIN_TOP_TIME)
    if top_time - start_time < MIN_TOP_TIME:
        top_time = start_time + MIN_TOP_TIME

    # 5) finish 검출: 속도 크기(|v|)가 임계값 이하로 내려간 최초 시점
    vmag = np.sqrt(vx_s**2 + vy_s**2)  # 속도 크기
    
    # 스케일-불변 임계값: 초기 구간의 80퍼센타일을 기반으로 10~20% 수준으로 설정
    try:
        ref = np.percentile(vmag[: max(10, len(vmag)//5)], 80)
    except Exception:
        ref = float(np.max(vmag)) if vmag.size else 0.1
    thresh = max(0.1 * ref, 1e-3)
    
    # 연속 샘플 수 계산 (시간 기반)
    hold_samples = max(5, int(round(0.12 / max(np.median(np.diff(t_high)) if t_high.size > 1 else 0.033, 1e-3))))
    
    # finish 지점 검출
    finish_idx = None
    for i in range(rel_idx + 1, len(vmag) - hold_samples):
        seg = vmag[i : i + hold_samples]
        if np.all(seg < thresh):  # 연속으로 속도가 낮음
            finish_idx = i
            break
    
    # 다운스윙 길이 제약 조건
    MIN_DOWN = 0.35  # 최소 다운스윙 시간 (초)
    DEFAULT_DOWN = 0.75  # 기본 다운스윙 시간 (초)
    end_time = times[-1]
    
    if finish_idx is not None:
        finish_time = t_high[finish_idx]
    else:
        # finish를 찾지 못한 경우 기본값 사용
        remain = end_time - top_time
        if remain < MIN_DOWN:
            pull = MIN_DOWN - remain
            top_time = max(start_time + MIN_TOP_TIME, top_time - pull)
        finish_time = min(end_time, top_time + DEFAULT_DOWN)

    return start_time, top_time, finish_time, start_time


def detect_takeback_start(video_path: str, window: int = 5, movement_threshold: float = 0.005) -> int:
    """
    손 위치(오른손목) 이동 + 팔꿈치 각도 변화가 동시에 증가하는 최초 프레임을 테이크백 시작으로 판단하는 함수
    
    Args:
        video_path: 비디오 파일 경로
        window: 분석 윈도우 크기 (기본값: 5)
        movement_threshold: 움직임 감지 임계값 (기본값: 0.005)
        
    Returns:
        int: 테이크백 시작 프레임 인덱스 (절대)
    """
    cap = cv2.VideoCapture(video_path)
    mp_p = mp_pose

    # 데이터 저장 리스트
    hand_positions = []  # [x,y,z] 정규화 좌표
    elbow_angles = []    # deg (도)
    frame_idx = []

    # MediaPipe 포즈 객체 생성
    with mp_p.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR을 RGB로 변환
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 포즈 랜드마크 검출
            result = pose.process(rgb)
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                # 필요한 랜드마크 추출
                rw = lm[mp_p.PoseLandmark.RIGHT_WRIST]    # 오른손목
                re = lm[mp_p.PoseLandmark.RIGHT_ELBOW]    # 오른팔꿈치
                rs = lm[mp_p.PoseLandmark.RIGHT_SHOULDER] # 오른어깨

                # 손 위치 (3D 좌표)
                hand_pos = np.array([rw.x, rw.y, rw.z])

                # 팔꿈치 각도 계산 (어깨-팔꿈치-손목)
                ba = np.array([rs.x, rs.y, rs.z]) - np.array([re.x, re.y, re.z])  # 어깨→팔꿈치
                bc = np.array([rw.x, rw.y, rw.z]) - np.array([re.x, re.y, re.z])  # 손목→팔꿈치
                ba_norm = ba / (np.linalg.norm(ba) + 1e-8)  # 정규화
                bc_norm = bc / (np.linalg.norm(bc) + 1e-8)  # 정규화
                cos_angle = np.clip(np.dot(ba_norm, bc_norm), -1.0, 1.0)  # 코사인 각도
                angle = float(np.degrees(np.arccos(cos_angle)))  # 도 단위로 변환

                # 데이터 저장
                hand_positions.append(hand_pos)
                elbow_angles.append(angle)
                frame_idx.append(fi)
            fi += 1
    cap.release()

    # 충분한 데이터가 없는 경우 기본값 반환
    if len(hand_positions) < window + 2:
        return 0

    # numpy 배열로 변환
    hand_positions = np.asarray(hand_positions, dtype=float)
    elbow_angles = np.asarray(elbow_angles, dtype=float)

    # 손 위치 이동 거리 (정규화 좌표)
    hand_diff = np.linalg.norm(hand_positions[1:] - hand_positions[:-1], axis=1)
    # 팔꿈치 각도 변화량
    angle_diff = np.abs(elbow_angles[1:] - elbow_angles[:-1])

    # 두 조건을 동시에 만족하는 첫 번째 지점 찾기
    for i in range(len(hand_diff)):
        if hand_diff[i] > movement_threshold and angle_diff[i] > movement_threshold * 100.0:
            return frame_idx[i + 1]
    
    return 0


def detect_address_before_takeback(video_path: str, takeback_start_abs: int,
                                   stable_window: int = 10,
                                   stable_threshold: float = 0.0015,
                                   prepad_frames: int = 2) -> int:
    """
    테이크백 시작 직전의 어드레스(정지) 구간 시작 프레임을 절대 인덱스로 반환하는 함수
    
    Args:
        video_path: 비디오 파일 경로
        takeback_start_abs: 테이크백 시작 절대 프레임
        stable_window: 안정성 판단 윈도우 크기 (기본값: 10)
        stable_threshold: 안정성 임계값 (기본값: 0.0015)
        prepad_frames: 앞쪽으로 이동할 프레임 수 (기본값: 2)
        
    Returns:
        int: 어드레스 시작 절대 프레임 인덱스
        
    특징:
        - stable_window 동안 손목 이동 평균이 stable_threshold 이하인 마지막 지점을 찾고
        - 약간 앞쪽(prepad)으로 이동하여 안전한 시작점 확보
    """
    cap = cv2.VideoCapture(video_path)
    mp_p = mp_pose
    
    # 손목 좌표와 프레임 인덱스 저장
    hand = []
    idxs = []
    
    with mp_p.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR을 RGB로 변환
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 포즈 랜드마크 검출
            result = pose.process(rgb)
            if result.pose_landmarks:
                # 오른손목 랜드마크 추출
                rw = result.pose_landmarks.landmark[mp_p.PoseLandmark.RIGHT_WRIST]
                # 2D 좌표 저장
                hand.append([rw.x, rw.y])
                idxs.append(fi)
            fi += 1
    cap.release()

    # 데이터가 없는 경우 기본값 반환
    if not hand:
        return max(0, takeback_start_abs - 10)

    # numpy 배열로 변환
    hand = np.asarray(hand)
    
    # 연속 프레임 간 이동 거리 계산
    diffs = np.linalg.norm(hand[1:] - hand[:-1], axis=1)
    
    # 이동 평균 계산
    if len(diffs) < stable_window:
        return max(0, takeback_start_abs - 10)
    
    kernel = np.ones(stable_window) / stable_window
    mov = np.convolve(diffs, kernel, mode='same')

    # 테이크백 시작 절대 프레임에 해당하는 인덱스 찾기
    try:
        tb_pos = next(i for i, f in enumerate(idxs) if f >= takeback_start_abs)
    except StopIteration:
        tb_pos = len(idxs) - 1

    # tb_pos 이전에서 가장 최근의 정지 구간 찾기
    candidate = 0
    for i in range(max(0, tb_pos - 1)):
        if mov[i] <= stable_threshold:
            candidate = i
    
    # prepad_frames만큼 앞쪽으로 이동하여 안전한 시작점 확보
    address_abs = idxs[max(0, candidate - prepad_frames)]
    return address_abs


def crop_video_from_frame(input_path: str, start_frame: int, output_path: str) -> bool:
    """
    입력 영상을 start_frame부터 끝까지 잘라 저장하는 함수
    
    Args:
        input_path: 입력 영상 경로
        start_frame: 시작 프레임 번호
        output_path: 출력 영상 경로
        
    Returns:
        bool: 성공 여부
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False
    
    # 비디오 속성 추출
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 시작 프레임 범위 검증
    start_frame = max(0, min(start_frame, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # 프레임 복사
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    # 리소스 해제
    cap.release()
    out.release()
    return True


def slow_down_video(input_path: str, output_path: str, factor: float = 2.0) -> bool:
    """
    동영상 재생을 느리게 만드는 간단한 폴백 구현
    
    Args:
        input_path: 입력 영상 경로
        output_path: 출력 영상 경로
        factor: 슬로우 팩터 (기본값: 2.0)
        
    Returns:
        bool: 성공 여부
        
    특징:
        - factor=2.0이면 각 프레임을 2번씩 기록하여 2배 느리게 보이도록 함
        - 코덱은 mp4v 사용
        - 정수 배수에 대해서만 프레임 중복 기록 (2.0, 3.0 등)
    """
    try:
        factor = max(1.0, float(factor))
    except Exception:
        factor = 2.0

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False
    
    # 비디오 속성 추출
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 정수 배수에 대해서만 프레임 중복 기록 (2.0, 3.0 등)
    repeat = max(1, int(round(factor)))
    success_frames = 0
    
    # 프레임 읽기 및 중복 기록
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # repeat 횟수만큼 프레임 기록
        for _ in range(repeat):
            out.write(frame)
        success_frames += 1

    # 리소스 해제
    cap.release()
    out.release()
    
    # 성공 여부 확인
    return success_frames > 0 and os.path.exists(output_path)


def _compute_right_arm_angles(video_path: str, global_frame_idx: int) -> tuple[float, float]:
    """
    주어진 절대 프레임에서 오른팔(팔꿈치) 각도와 어깨 각도를 계산하는 함수
    
    Args:
        video_path: 비디오 파일 경로
        global_frame_idx: 전역 프레임 인덱스
        
    Returns:
        tuple: (elbow_deg, shoulder_deg)
            - elbow_deg: 팔꿈치 각도 (도)
            - shoulder_deg: 어깨 굽힘 각도 (도)
    """
    mp_p = mp_pose
    cap = cv2.VideoCapture(video_path)
    try:
        # 지정된 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, global_frame_idx))
        ret, frame = cap.read()
        if not ret:
            return 180.0, 60.0
        
        # BGR을 RGB로 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with mp_p.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # 포즈 랜드마크 검출
            result = pose.process(rgb)
            if not result.pose_landmarks:
                return 180.0, 60.0
            
            # 필요한 랜드마크 추출
            lm = result.pose_landmarks.landmark
            rs = lm[mp_p.PoseLandmark.RIGHT_SHOULDER]  # 오른어깨
            re = lm[mp_p.PoseLandmark.RIGHT_ELBOW]     # 오른팔꿈치
            rw = lm[mp_p.PoseLandmark.RIGHT_WRIST]     # 오른손목
            rh = lm[mp_p.PoseLandmark.RIGHT_HIP]       # 오른엉덩이

            def angle(a, b, c):
                """세 점으로 이루어진 각도 계산 (3D)"""
                a = np.array([a.x, a.y, a.z])
                b = np.array([b.x, b.y, b.z])
                c = np.array([c.x, c.y, c.z])
                ba = a - b  # b→a 벡터
                bc = c - b  # b→c 벡터
                na = np.linalg.norm(ba) + 1e-8  # 벡터 크기
                nc = np.linalg.norm(bc) + 1e-8  # 벡터 크기
                cosv = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)  # 코사인 각도
                return float(np.degrees(np.arccos(cosv)))  # 도 단위로 변환

            # 각도 계산
            elbow = angle(rs, re, rw)           # 팔꿈치 각도 (어깨-팔꿈치-손목)
            shoulder = angle(re, rs, rh)        # 어깨 굽힘 각도 유사치 (팔꿈치-어깨-엉덩이)
            return elbow, shoulder
    finally:
        cap.release()


# ===== 디버깅 및 검증 유틸 =====

def save_swing_key_frames_limited_debug(video_path: str, start_frame: int, top_rel: int, finish_rel: int) -> None:
    """
    스윙 키 프레임들을 제한적으로 디버깅용 이미지로 저장합니다 (웨글 제외).
    
    Args:
        video_path: 영상 파일 경로
        start_frame: 스윙 시작 프레임 (절대)
        top_rel: 백스윙 탑 프레임 (상대)
        finish_rel: 스윙 피니쉬 프레임 (상대)
    
    저장되는 이미지 (3장만):
    - swing_start_frame.jpg: 스윙 시작 시점
    - swing_top_frame.jpg: 백스윙 탑 시점
    - swing_finish_frame.jpg: 스윙 피니쉬 시점
    
    특징:
    - 웨글 이미지는 저장하지 않음 (로직적으로만 인식)
    - 프로와 유저 영상에서 각각 3장씩만 저장
    - API 폴더에 깔끔하게 정리된 이미지만 저장
    """
    try:
        # 영상 파일 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] 영상을 열 수 없습니다: {video_path}")
            return
        
        # 저장할 프레임 정보 (3장만: 스타트, 백스윙탑, 피니시)
        frames_to_save = {
            "start": start_frame,           # 스타트 (어드레스)
            "top": start_frame + top_rel,   # 백스윙탑
            "finish": start_frame + finish_rel  # 피니시
        }
        
        # 각 키 프레임을 이미지로 저장
        for frame_name, frame_idx in frames_to_save.items():
            if frame_idx is None or frame_idx < 0:
                print(f"[WARN] {frame_name} 프레임이 유효하지 않습니다: {frame_idx}")
                continue
                
            # 지정된 프레임으로 이동
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # 파일명 생성 (타임스탬프 포함하여 중복 방지)
                import time
                timestamp = int(time.time())
                filename = f"{frame_name}_{timestamp}.jpg"
                
                # 이미지 저장
                cv2.imwrite(filename, frame)
                print(f"[DEBUG] {frame_name} 프레임 저장 완료: {filename} (프레임 {frame_idx})")
            else:
                print(f"[WARN] {frame_name} 프레임 {frame_idx}을 읽을 수 없습니다")
        
        cap.release()
        
    except Exception as e:
        print(f"[ERROR] 스윙 키 프레임 저장 중 오류 발생: {e}")
        # 영상 파일이 열려있다면 해제
        try:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
        except:
            pass

