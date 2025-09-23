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
    with mp_p.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as pose:
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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    # FPS 기반 시간 제약 강화
    MIN_TOP_TIME_SEC = 0.15  # 최소 top 시간 (초)
    MIN_FINISH_TIME_SEC = 0.5  # 최소 finish 시간 (초)
    MAX_FINISH_TIME_SEC = 2.0  # 최대 finish 시간 (초)
    
    MIN_TOP_FRAMES = max(5, int(MIN_TOP_TIME_SEC * fps))
    MIN_FINISH_FRAMES = max(15, int(MIN_FINISH_TIME_SEC * fps))
    MAX_FINISH_FRAMES = int(MAX_FINISH_TIME_SEC * fps)

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
    # MediaPipe 설정 (정확도 우선)
    with mp_p.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as pose:
        fi = 0
        prev_left_heel = None
        prev_left_toe = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 입력 최소 해상도 보장(너비/높이 중 한 변이 480 미만이면 업샘플)
            h, w = frame.shape[:2]
            if max(h, w) < 640 or min(h, w) < 480:
                scale = max(640 / max(h, w), 480 / max(1, min(h, w)))
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

            # 전처리: CLAHE(대비 향상) + 가우시안 블러(노이즈 완화)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
            except Exception:
                enhanced = gray
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            enhanced_bgr = cv2.GaussianBlur(enhanced_bgr, (3, 3), 0)

            rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                if camera_view is None:
                    camera_view = detect_camera_view(result.pose_landmarks)
                rw = result.pose_landmarks.landmark[mp_p.PoseLandmark.RIGHT_WRIST]
                # 왼발 힐/토 디버깅 로그 추가
                try:
                    lh = result.pose_landmarks.landmark[mp_p.PoseLandmark.LEFT_HEEL]
                    lt = result.pose_landmarks.landmark[mp_p.PoseLandmark.LEFT_FOOT_INDEX]
                    # 누락 시 이전 프레임 기반 보간/유지
                    if getattr(lh, 'visibility', 0.0) < 0.4 and prev_left_heel is not None:
                        lh.x = prev_left_heel.x if getattr(lh, 'x', None) is None else (prev_left_heel.x + lh.x) / 2.0
                        lh.y = prev_left_heel.y if getattr(lh, 'y', None) is None else (prev_left_heel.y + lh.y) / 2.0
                        lh.visibility = max(lh.visibility, prev_left_heel.visibility)
                    if getattr(lt, 'visibility', 0.0) < 0.4 and prev_left_toe is not None:
                        lt.x = prev_left_toe.x if getattr(lt, 'x', None) is None else (prev_left_toe.x + lt.x) / 2.0
                        lt.y = prev_left_toe.y if getattr(lt, 'y', None) is None else (prev_left_toe.y + lt.y) / 2.0
                        lt.visibility = max(lt.visibility, prev_left_toe.visibility)
                    prev_left_heel = lh
                    prev_left_toe = lt
                    print(f"[LeftFoot] f={fi} heel vis={getattr(lh,'visibility',0):.2f} x={lh.x:.3f} y={lh.y:.3f} | toe vis={getattr(lt,'visibility',0):.2f} x={lt.x:.3f} y={lt.y:.3f}")
                    if getattr(lh, 'visibility', 0.0) < 0.5:
                        print("[LeftFoot][WARN] Left Heel visibility < 0.5 → 가림/조명/해상도 점검 필요")
                except Exception:
                    pass

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

    # top: 속도 0 교차점 + 기하학적 극값 보조 (FPS 기반)
    MIN_TOP_FRAMES = max(5, int(0.15 * fps))  # 0.15초를 프레임으로 변환
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

    # 가우시안 스무딩 적용 (지터 완화)
    def _gaussian_smooth(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        if arr.size == 0:
            return arr
        # OpenCV GaussianBlur는 2D 입력이 필요 → (N,1)로 확장 후 복원
        a2 = arr.reshape(-1, 1).astype(np.float32)
        sm = cv2.GaussianBlur(a2, (5, 1), sigmaX=sigma, sigmaY=0)
        return sm.reshape(-1)
    vx_s, vy_s = _gaussian_smooth(vx, 1.0), _gaussian_smooth(vy, 1.0)

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
    with mp_p.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as pose:
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

    # FPS 기반 MIN_TOP_FRAMES 사용
    MIN_TOP_FRAMES = max(5, int(0.15 * fps))  # 0.15초를 프레임으로 변환
    x_seg = np.asarray(hand_x[start_pos:], dtype=float)
    y_seg = np.asarray(hand_y[start_pos:], dtype=float)
    if len(x_seg) >= 2:
        vx = np.diff(x_seg, prepend=x_seg[0])
        vy = np.diff(y_seg, prepend=y_seg[0])
    else:
        vx = np.zeros_like(x_seg)
        vy = np.zeros_like(y_seg)
    # 간단한 이동평균으로 스무딩
    def simple_smooth(arr, window):
        if len(arr) < window:
            return arr
        result = np.zeros_like(arr)
        for i in range(len(arr)):
            start = max(0, i - window // 2)
            end = min(len(arr), i + window // 2 + 1)
            result[i] = np.mean(arr[start:end])
        return result
    
    vx_s, vy_s = simple_smooth(vx, 5), simple_smooth(vy, 5)
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
    시간 도메인(초)에서 키 타임을 검출하는 함수 (간소화된 버전)
    
    Args:
        video_path: 비디오 파일 경로
        slow_factor: 슬로우 업샘플링 배수 (기본값: 2.0)
        
    Returns:
        tuple: (start_s, top_s, finish_s, waggle_end_s)
            - start_s: 시작 시간 (초)
            - top_s: 최고점 시간 (초)
            - finish_s: 종료 시간 (초)
            - waggle_end_s: 웨글 종료 시간 (초)
    """
    # 1) 기존 어드레스/웨글 기반 시작 프레임 계산
    start_rel, top_rel, finish_rel, waggle_abs = detect_swing_key_frames(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # 시작 시간을 초 단위로 변환
    start_time = float(waggle_abs) / float(fps)
    
    # detect_swing_key_frames에서 이미 계산된 상대 프레임을 사용
    if top_rel is None or finish_rel is None:
        # 기본값 사용
        top_time = start_time + 0.5
        finish_time = start_time + 1.0
        print(f"[Key-Times] Using default values - top_rel: {top_rel}, finish_rel: {finish_rel}")
    else:
        # 상대 프레임을 절대 프레임으로 변환 후 초 단위로 변환
        top_abs = waggle_abs + top_rel
        finish_abs = waggle_abs + finish_rel
        top_time = float(top_abs) / float(fps)
        finish_time = float(finish_abs) / float(fps)
        print(f"[Key-Times] Calculated times - start: {start_time:.2f}s, top: {top_time:.2f}s, finish: {finish_time:.2f}s")
    
    return start_time, top_time, finish_time, start_time


def detect_takeback_start(video_path: str, window: int = 5, movement_threshold: float = 0.005) -> int:
    """
    손 위치(오른손목) 이동 + 팔꿈치 각도 변화가 동시에 증가하는 최초 프레임을 테이크백 시작으로 판단하는 함수
    
    Args:
        video_path: 비디오 파일 경로
        window: 분석 윈도우 크기 (기본값: 5)
        movement_threshold: 움직임 임계값 (기본값: 0.005)
        
    Returns:
        int: 테이크백 시작 프레임 번호
    """
    mp_p = mp_pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    hand_positions = []
    elbow_angles = []
    frame_indices = []
    
    with mp_p.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                
                # 오른손목 위치
                right_wrist = landmarks[mp_p.PoseLandmark.RIGHT_WRIST]
                hand_positions.append((right_wrist.x, right_wrist.y))
                
                # 오른팔꿈치 각도 계산
                right_shoulder = landmarks[mp_p.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = landmarks[mp_p.PoseLandmark.RIGHT_ELBOW]
                
                # 벡터 계산
                vec1 = np.array([right_elbow.x - right_shoulder.x, right_elbow.y - right_shoulder.y])
                vec2 = np.array([right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y])
                
                # 각도 계산
                if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    elbow_angles.append(angle)
                else:
                    elbow_angles.append(0.0)
                
                frame_indices.append(frame_idx)
            
            frame_idx += 1
    
    cap.release()
    
    if len(hand_positions) < window * 2:
        return 0
    
    # 손 위치 이동량과 팔꿈치 각도 변화량 계산
    hand_movements = []
    elbow_changes = []
    
    for i in range(window, len(hand_positions) - window):
        # 손 위치 이동량 (이전 윈도우와 현재 윈도우의 평균 위치 차이)
        prev_hand = np.mean(hand_positions[i-window:i], axis=0)
        curr_hand = np.mean(hand_positions[i:i+window], axis=0)
        hand_movement = np.linalg.norm(curr_hand - prev_hand)
        hand_movements.append(hand_movement)
        
        # 팔꿈치 각도 변화량
        prev_elbow = np.mean(elbow_angles[i-window:i])
        curr_elbow = np.mean(elbow_angles[i:i+window])
        elbow_change = abs(curr_elbow - prev_elbow)
        elbow_changes.append(elbow_change)
    
    # 두 조건을 모두 만족하는 첫 번째 지점 찾기
    for i, (hand_mov, elbow_chg) in enumerate(zip(hand_movements, elbow_changes)):
        if hand_mov > movement_threshold and elbow_chg > 0.1:  # 각도 변화 임계값
            return frame_indices[i + window]
    
    # 조건을 만족하는 지점이 없으면 중간 지점 반환
    return frame_indices[len(frame_indices) // 2] if frame_indices else 0


def detect_address_before_takeback(video_path: str, takeback_start_frame: int) -> int:
    """
    테이크백 시작 프레임 이전에서 어드레스 구간(정지 상태)의 시작 프레임을 찾는 함수
    
    Args:
        video_path: 비디오 파일 경로
        takeback_start_frame: 테이크백 시작 프레임
        
    Returns:
        int: 어드레스 시작 프레임
    """
    # 간단한 구현: 테이크백 시작 프레임에서 10프레임 이전을 어드레스로 설정
    address_frame = max(0, takeback_start_frame - 10)
    return address_frame


def crop_video_from_frame(input_path: str, start_frame: int, output_path: str) -> bool:
    """
    입력 영상을 start_frame부터 끝까지 잘라 저장하는 함수
    
    Args:
        input_path: 입력 영상 경로
        start_frame: 시작 프레임 번호 (포함)
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
    
    # 프레임 복사 (영상 끝까지)
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
        
        with mp_p.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        ) as pose:
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

