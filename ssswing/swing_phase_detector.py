"""
골프 스윙 단계 감지 모듈 (Swing Phase Detector, view-agnostic)

주요 단계 및 검출 기준
- 어드레스(Address): 손 중심이 일정 시간(정적 창) 동안 거의 움직이지 않다가 손 속도가 임계치를 넘기기 직전의 프레임.
- 백스윙 탑(Backswing Top): 손 y가 상승→하강으로 전환되는 zero‑cross 후보를 생성하고,
  샤프트 수평성(CV 우선·포즈 보조), 팔 각도, 가시성, 각속도/유지 안정성을 점수화해 최적 프레임 선택.
- 임팩트(Impact): 손 속도 최대 + 손 높이(=y) 최소를 결합한 점수 최대 프레임.
- 피니시(Finish): 팔 펴짐, 손 위치(어깨 이하도 허용), 척추 기울기, 손 변위가 충족되고 0.6초 이상 연속 유지되면 선택.
  실패 시 임팩트 이후 일정 프레임으로 폴백.

최신 개선사항 및 특징 (2025-09)
- 뷰-불변(정면/측면 무관) 검출: Mediapipe 정규화 좌표와 포즈/샤프트 추정 각도 활용.
- 백스윙 탑 정밀화: 수평성·팔 각도·연속성·각속도 안정성 융합.
- 피니시 낮게 인정: 다양한 실제 스윙 스타일 수용.
- 동작 기반 + 정적성: 시작/종료를 분산·속도 임계로 견고하게 검출.

활용 예시 및 응용
- 스윙 주요 구간(어드레스/탑/임팩트/피니시) 자동 태깅·분석.
- 다양한 촬영 각도에서 일관 검출 및 단계별 속도·자세 피드백 제공.

작성자: AI Assistant
개선일: 2025-09
"""

import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark

# -------------------- 기본 유틸 --------------------
def _get(lms, idx):
    return lms.landmark[idx] if (lms is not None and hasattr(lms, "landmark") and len(lms.landmark) > idx) else None

def _midpoint(p, q):
    return type("P", (), {"x": (p.x + q.x) / 2.0, "y": (p.y + q.y) / 2.0, "z": (getattr(p, "z", 0.0) + getattr(q, "z", 0.0)) / 2.0})

def _hands_center(lms):
    lw = _get(lms, PL.LEFT_WRIST)
    rw = _get(lms, PL.RIGHT_WRIST)
    if lw is None or rw is None: 
        return None
    return _midpoint(lw, rw)

def _shoulders(lms):
    ls, rs = _get(lms, PL.LEFT_SHOULDER), _get(lms, PL.RIGHT_SHOULDER)
    return ls, rs

def _hips(lms):
    lh, rh = _get(lms, PL.LEFT_HIP), _get(lms, PL.RIGHT_HIP)
    return lh, rh

def _angle(a, b, c):
    if any(p is None for p in (a,b,c)): return np.nan
    v1 = np.array([a.x - b.x, a.y - b.y])
    v2 = np.array([c.x - b.x, c.y - b.y])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return np.nan
    cos = np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

def _line_angle_deg(p, q):
    dy, dx = q.y - p.y, q.x - p.x
    return float(np.degrees(np.arctan2(dy, dx)))

def _vertical_dev_deg(p, q):
    return abs(90.0 - abs(_line_angle_deg(p, q)))

def _safe_idx(i, lo, hi):
    return max(lo, min(hi, i))

def _shoulder_yaw_value(lms) -> float:
    """어깨 축 회전(yaw) 근사치: 좌/우 어깨 z차를 어깨 폭으로 정규화.
    정면에서도 팔 가려질 때 사용할 수 있는 회전량 지표.
    """
    ls, rs = _shoulders(lms)
    if ls is None or rs is None:
        return np.nan
    dx = float(ls.x - rs.x)
    dy = float(ls.y - rs.y)
    dist = float(np.hypot(dx, dy)) + 1e-6
    return float((getattr(ls, 'z', 0.0) - getattr(rs, 'z', 0.0)) / dist)

def _shoulder_axis_angle_deg(lms) -> float:
    """어깨 축의 화면상 기울기 각도(deg). 좌→우 어깨 벡터의 각도.
    빨간선(어깨선)이 시계/반시계로 회전하는 변화를 포착.
    """
    ls, rs = _shoulders(lms)
    if ls is None or rs is None:
        return np.nan
    return _line_angle_deg(ls, rs)

def _unwrap_angles_deg(a: np.ndarray) -> np.ndarray:
    """-180~180 범위 각도를 연속적인 곡선으로 언랩(unwarp)하여 급격한 점프 제거."""
    if a is None or len(a) == 0:
        return np.asarray([], dtype=float)
    a = np.asarray(a, dtype=float)
    out = a.copy()
    for i in range(1, len(out)):
        d = out[i] - out[i-1]
        if d > 180.0:
            out[i:] -= 360.0
        elif d < -180.0:
            out[i:] += 360.0
    return out

def _pelvis_axis_angle_deg(lms) -> float:
    """골반(힙) 축의 화면상 각도(deg). 좌→우 엉덩이 벡터 각도."""
    lh, rh = _hips(lms)
    if lh is None or rh is None:
        return np.nan
    return _line_angle_deg(lh, rh)

def _angle_diff_deg(a: float, b: float) -> float:
    """두 각도(deg)의 최소 차이(0..180)."""
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return np.nan
    d = abs(float(a) - float(b)) % 360.0
    return d if d <= 180.0 else 360.0 - d

# -------------------- 보조: 각도/프레임/ROI 유틸 --------------------
def _horizontal_deviation_deg(angle_deg: float) -> float:
    """수평(0° 또는 180°)로부터의 최소 편차(deg)를 반환."""
    a = abs(float(angle_deg))
    return min(a, abs(180.0 - a))

def _ema(arr: list[float] | np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """지수이동평균(EMA) 스무딩."""
    if arr is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(arr, dtype=float)
    out = np.empty_like(arr)
    m = None
    for i, v in enumerate(arr):
        m = v if m is None else alpha * v + (1.0 - alpha) * m
        out[i] = m
    return out

def _compute_y_velocity(y_arr: np.ndarray, fps: int, smooth_alpha: float = 0.35) -> np.ndarray:
    """EMA 스무딩 후 프레임별 미분(속도)."""
    if y_arr is None or len(y_arr) < 2:
        return np.array([])
    y_s = _ema(y_arr, alpha=smooth_alpha)
    dy = np.diff(y_s, prepend=y_s[0])
    dy[np.abs(dy) < 1e-4] = 0.0
    return dy

def _backswing_candidate_indices_from_dy(dy: np.ndarray):
    """
    dy: smoothed y derivative (len = N)
    백스윙 탑 후보는 dy가 음수→양수로 바뀌는 지점.
    """
    if len(dy) < 2:
        return []
    cand = []
    for i in range(1, len(dy)):
        if dy[i-1] < 0 and dy[i] >= 0:
            cand.append(i)
    return cand

def _score_top_candidate(idx_map, j, landmarks_data, video_path, fps):
    """zero-cross 후보 점수화: 높이·팔 모양·샤프트 수평·가시성 반영 (높을수록 좋음)."""
    i = idx_map[j]
    lms = landmarks_data[i]
    if lms is None:
        return -999.0
    ls, rs = _shoulders(lms)
    lw, rw = _get(lms, PL.LEFT_WRIST), _get(lms, PL.RIGHT_WRIST)
    le, re = _get(lms, PL.LEFT_ELBOW), _get(lms, PL.RIGHT_ELBOW)
    if any(p is None for p in (ls, rs, lw, rw, le, re)):
        return -999.0

    hands_y = min(lw.y, rw.y)
    shoulders_y = min(ls.y, rs.y)
    hands_above = 1.0 if hands_y < shoulders_y * 0.95 else 0.0

    left_elbow_angle = _angle(ls, le, lw)
    right_elbow_angle = _angle(rs, re, rw)
    elbow_score = 0.0
    if not np.isnan(left_elbow_angle) and not np.isnan(right_elbow_angle):
        elbow_score = 0.5 * (min(max((left_elbow_angle - 110) / 40.0, 0.0), 1.0) +
                             min(max((180 - right_elbow_angle) / 60.0, 0.0), 1.0))

    shaft_score = 0.0
    ang_cv = None
    if video_path:
        ang_cv = _estimate_shaft_angle_cv(video_path, i, lms)
    if ang_cv is not None:
        dev = _horizontal_deviation_deg(ang_cv)
        shaft_score = max(0.0, 1.0 - dev / 40.0)
    else:
        mid_s = _midpoint(ls, rs)
        ang_pose = _line_angle_deg(mid_s, _midpoint(lw, rw))
        dev_pose = _horizontal_deviation_deg(ang_pose)
        shaft_score = max(0.0, 1.0 - dev_pose / 60.0)

    vis_vals = []
    for p in (PL.LEFT_WRIST, PL.RIGHT_WRIST, PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER):
        lm = _get(lms, p)
        if lm is not None and hasattr(lm, 'visibility'):
            vis_vals.append(float(lm.visibility))
    vis_mean = float(np.mean(vis_vals)) if vis_vals else 1.0
    vis_penalty = 0.0 if vis_mean >= 0.5 else (0.5 - vis_mean)

    score = 1.6 * hands_above + 1.2 * elbow_score + 1.6 * shaft_score - vis_penalty
    return float(score)

def _get_frame_at(video_path: str, frame_index: int):
    """지정 프레임을 읽어 (frame, w, h)를 반환. 실패 시 (None, 0, 0)."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] 비디오 파일을 열 수 없습니다: {video_path}")
            return None, 0, 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index >= total_frames:
            print(f"[ERROR] 프레임 인덱스 초과: {frame_index} >= {total_frames}")
            cap.release()
            return None, 0, 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        if not ok:
            print(f"[ERROR] 프레임 읽기 실패: frame {frame_index}")
            return None, 0, 0
        
        print(f"[DEBUG] 프레임 읽기 성공: frame {frame_index}, size {w}x{h}")
        return frame, w, h
    except Exception as e:
        print(f"[ERROR] 프레임 읽기 예외: {e}")
        return None, 0, 0

def _norm_to_pixel(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    """정규화 좌표(0..1)를 픽셀 좌표로 변환."""
    return int(round(x * max(w, 1))), int(round(y * max(h, 1)))

def _estimate_shaft_angle_cv(video_path: str | None, frame_index: int, lms) -> float | None:
    """
    간단 CV 기반 샤프트 각도 추정 (deg).
    - 손목 주변 ROI에서 Canny+HoughLinesP로 긴 직선을 검출해 대표 각도를 반환.
    - 실패 시 None 반환.
    """
    if not video_path or lms is None:
        return None
    try:
        import cv2
        frame, w, h = _get_frame_at(video_path, frame_index)
        if frame is None or w == 0 or h == 0:
            return None

        # 손목 좌표로 ROI 설정
        rw = _get(lms, PL.RIGHT_WRIST)
        lw = _get(lms, PL.LEFT_WRIST)
        if rw is None and lw is None:
            return None
        pts = []
        if rw is not None:
            pts.append(_norm_to_pixel(rw.x, rw.y, w, h))
        if lw is not None:
            pts.append(_norm_to_pixel(lw.x, lw.y, w, h))
        cx = int(sum(p[0] for p in pts) / len(pts))
        cy = int(sum(p[1] for p in pts) / len(pts))

        # ROI 확장: 손목 중심 + 어깨→손 방향으로 연장(헤드 포함 시도) - 확대
        # 기본 박스 확대(가로 1.8x, 세로 1.2x) -> 더 확대
        roi_w = max(120, int(w * 0.6))  # 0.45에서 0.6으로 확대
        roi_h = max(100, int(h * 0.4))  # 0.30에서 0.4로 확대
        x0 = max(0, cx - roi_w // 2)
        y0 = max(0, cy - roi_h // 2)
        x1 = min(w - 1, x0 + roi_w)
        y1 = min(h - 1, y0 + roi_h)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 대비 향상
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # 적응적 Canny (중앙값 기반)
        v = float(np.median(gray))
        low = int(max(10, 0.66 * v))
        high = int(min(255, 1.33 * v))
        edges = cv2.Canny(gray, low, high)

        # HoughLinesP로 선 검출
        # 노이즈 감소를 위한 morphology 연산 추가
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=40, minLineLength=int(min(roi_w, roi_h) * 0.35), maxLineGap=24)
        if lines is None:
            return None

        # 상위 n개 선분 중 수평 편차가 가장 작은 각도 선택(몸통에서 먼 선 우선)
        diag = float(np.hypot(roi_w, roi_h))
        cand = []
        for l in lines.reshape(-1, 4):
            x1p, y1p, x2p, y2p = map(int, l)
            dx = float(x2p - x1p); dy = float(y2p - y1p)
            length = float(np.hypot(dx, dy))
            if length < max(50.0, 0.30 * diag):
                continue
            ang = float(np.degrees(np.arctan2(dy, dx)))
            dev = _horizontal_deviation_deg(ang)
            # 몸통 중심에서 먼 선에 가중(간단히 x 좌표 큰 쪽 우대)
            far_bonus = 0.05 * max(x1p, x2p) / max(1, roi_w)
            cand.append((dev - far_bonus, ang, length))
        if not cand:
            return None
        cand.sort(key=lambda t: (t[0], -t[2]))
        return cand[0][1]
    except Exception:
        return None

def _estimate_shaft_and_head(video_path: str | None, frame_index: int, lms):
    """샤프트 각도와 헤드 끝점(대략)을 추정. 실패 시 (None, None)"""
    try:
        import cv2
    except Exception:
        return None, None
    ang = _estimate_shaft_angle_cv(video_path, frame_index, lms)
    if ang is None:
        return None, None
    # 헤드 추정: ROI에서 샤프트 방향으로 가장 먼 에지 포인트(간략화)
    frame, w, h = _get_frame_at(video_path, frame_index)
    if frame is None:
        return ang, None
    rw = _get(lms, PL.RIGHT_WRIST); lw = _get(lms, PL.LEFT_WRIST)
    if rw is None and lw is None:
        return ang, None
    cx = np.mean([p.x for p in [rw, lw] if p is not None])
    cy = np.mean([p.y for p in [rw, lw] if p is not None])
    px, py = _norm_to_pixel(cx, cy, w, h)
    # 헤드 위치를 샤프트 방향으로 일정 거리 투영(프레임 폭의 20%)
    length_pix = int(0.20 * w)
    rad = np.deg2rad(ang)
    hx = int(px + length_pix * np.cos(rad))
    hy = int(py + length_pix * np.sin(rad))
    return ang, (hx, hy)

# -------------------- 뷰-불변 자세 판정 --------------------
def is_address_pose(lms):
    """기본 어드레스 정적 자세(가벼운 체크). 최종 어드레스는 동작기반(가속직전)에서 확정."""
    ls, rs = _shoulders(lms)
    lh, rh = _hips(lms)
    lk, rk = _get(lms, PL.LEFT_KNEE), _get(lms, PL.RIGHT_KNEE)

    # 상체 숙임(측면/정면 모두에서 과도하지 않게)
    ang = _angle(ls, lh, lk if lk is not None else lh)
    # 어깨 수평 기울기/엉덩이 수평 기울기 너무 크지 않게
    shoulder_tilt = abs((ls.y if ls else 0) - (rs.y if rs else 0))
    hip_tilt = abs((lh.y if lh else 0) - (rh.y if rh else 0))

    ANG_MIN = 135.0     # 과도한 굽힘 배제
    TILT_MAX = 0.10     # 좌우 높이 차이 제한

    return (not np.isnan(ang) and ang > ANG_MIN) and (shoulder_tilt < TILT_MAX) and (hip_tilt < TILT_MAX)

def is_backswing_top_pose(lms):
    """손 y 상승 후 하강 시작 시점 + 샤프트 유사각 + 팔 각도."""
    ls, rs = _shoulders(lms)
    le, re = _get(lms, PL.LEFT_ELBOW), _get(lms, PL.RIGHT_ELBOW)
    lw, rw = _get(lms, PL.LEFT_WRIST), _get(lms, PL.RIGHT_WRIST)
    if any(p is None for p in (ls, rs, le, re, lw, rw)): 
        return False

    # 손이 어깨보다 높다(미디어파이프 y는 아래로 증가하므로 "더 작음"이 높음) - 완화: 0.90
    hands_y = min(lw.y, rw.y)
    shoulders_y = min(ls.y, rs.y)
    hands_above = hands_y < shoulders_y * 0.90

    # 팔꿈치 각도 (리드팔 비교적 펴지고 트레일팔 굽음 경향) - 뷰와 무관하게 양쪽을 유연히 체크
    left_elbow_angle = _angle(ls, le, lw)
    right_elbow_angle = _angle(rs, re, rw)
    elbow_ok = (left_elbow_angle > 140.0) or (right_elbow_angle < 125.0)

    # 샤프트 유사각: 어깨중점→손중점 선이 수평에 가깝다
    mid_s = _midpoint(ls, rs)
    mid_w = _midpoint(lw, rw)
    shaft_like = abs(_line_angle_deg(mid_s, mid_w))
    shaft_parallel = shaft_like < 25.0

    score = 0
    score += 1 if hands_above else 0
    score += 1 if elbow_ok else 0
    score += 1 if shaft_parallel else 0

    return score >= 2

def is_finish_pose(lms):
    """피니쉬: 팔 펴짐 + 손 위치 + 상체 기울기 + 낮은 자세 허용."""
    ls, rs = _shoulders(lms)
    lh, rh = _hips(lms)
    le, re = _get(lms, PL.LEFT_ELBOW), _get(lms, PL.RIGHT_ELBOW)
    lw, rw = _get(lms, PL.LEFT_WRIST), _get(lms, PL.RIGHT_WRIST)
    if any(p is None for p in (ls, rs, lh, rh, le, re, lw, rw)):
        return False

    # 팔 펴짐 (강화: 150도 기준)
    left_elbow_angle = _angle(ls, le, lw)
    right_elbow_angle = _angle(rs, re, rw)
    elbows_extended = (left_elbow_angle > 150.0) and (right_elbow_angle > 150.0)

    # 손 높이 - 완화: 어깨 높이 이하도 허용 (<= 1.0)
    hands_y = min(lw.y, rw.y)
    shoulders_y = min(ls.y, rs.y)
    hands_high = hands_y <= shoulders_y * 1.0

    # 척추 기울기 (힙→숄더 수직 편차)
    spine_dev = _vertical_dev_deg(_midpoint(lh, rh), _midpoint(ls, rs))
    spine_tilt_ok = spine_dev > 10.0

    # 손이 몸 중심의 타깃쪽(정면/측면 불변 지표로는 어렵기 때문에, 단순히 몸 중심 x 대비 충분히 이동)
    body_cx = (ls.x + rs.x + lh.x + rh.x) / 4.0
    hands_cx = (lw.x + rw.x) / 2.0
    hands_shifted = abs(hands_cx - body_cx) > 0.06

    score = 0.0
    score += 1.0 if elbows_extended else 0
    score += 1.0 if hands_high else 0
    score += 1.0 if spine_tilt_ok else 0
    score += 1.0 if hands_shifted else 0

    return score >= 3.0  # 완화: 3.5 -> 3.0

# -------------------- 동작 기반(정적→가속) 스윙 시작 검출 --------------------
def _detect_swing_start(landmarks_data, fps=30, static_win=15, vel_win=3):
    """
    - 정적 창(static_win) 동안 손 중심의 분산이 낮다가
    - 직후 vel_win 동안 평균 속도가 임계치를 넘으면 그 첫 프레임을 '스윙 시작'으로 본다.
    """
    N = len(landmarks_data)
    if N == 0: 
        return 0

    # 손 중심 궤적
    hand_xy = []
    for i in range(N):
        hc = _hands_center(landmarks_data[i])
        if hc is None:
            hand_xy.append((np.nan, np.nan))
        else:
            hand_xy.append((hc.x, hc.y))

    hand_xy = np.array(hand_xy)

    # 정적성 임계치(뷰 무관, 픽셀 정규화 좌표 기준)
    # y는 더 민감하니 살짝 타이트
    STATIC_VAR_X = 1e-4  # 약 0.01^2
    STATIC_VAR_Y = 6e-5  # 약 0.0075^2

    # 속도 임계치(프레임 간 L1)
    VEL_THR = 0.02  # 0.02 ~ 0.03 권장

    # 이동 평균 속도
    v = np.zeros(N)
    for i in range(1, N):
        if not np.any(np.isnan(hand_xy[i])) and not np.any(np.isnan(hand_xy[i-1])):
            v[i] = abs(hand_xy[i,0]-hand_xy[i-1,0]) + abs(hand_xy[i,1]-hand_xy[i-1,1])
        else:
            v[i] = 0.0
    # vel_win 평균
    if vel_win > 1:
        vv = np.convolve(v, np.ones(vel_win)/vel_win, mode="same")
    else:
        vv = v

    for i in range(static_win, N-vel_win):
        sx, sy = hand_xy[i-static_win:i, 0], hand_xy[i-static_win:i, 1]
        if np.any(np.isnan(sx)) or np.any(np.isnan(sy)):
            continue
        if (np.var(sx) < STATIC_VAR_X) and (np.var(sy) < STATIC_VAR_Y):
            # 정적 이후 가속 시작?
            if np.mean(vv[i:i+vel_win]) > VEL_THR:
                return i  # 이 프레임이 '스윙 시작'
    # 폴백: 최대 속도 지점
    return int(np.argmax(v))

# -------------------- 개선된 백스윙 탑 검출 (통합 로직) --------------------
def _diagnose_environment():
    """환경 및 라이브러리 진단"""
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        import scipy
        
        print("[DEBUG] 환경 진단:")
        print(f"  - OpenCV: {cv2.__version__}")
        print(f"  - MediaPipe: {mp.__version__}")
        print(f"  - NumPy: {np.__version__}")
        print(f"  - SciPy: {scipy.__version__}")
        
        # MediaPipe 호환성 체크
        if hasattr(mp.solutions.pose, 'PoseLandmark'):
            print("  - MediaPipe Pose: OK")
        else:
            print("  - MediaPipe Pose: WARNING - 구조 변경 가능성")
            
        # OpenCV 기능 체크
        if hasattr(cv2, 'HoughLinesP'):
            print("  - OpenCV HoughLinesP: OK")
        else:
            print("  - OpenCV HoughLinesP: ERROR")
            
        return True
    except Exception as e:
        print(f"[ERROR] 환경 진단 실패: {e}")
        return False


def detect_top(landmarks_data, fps, start_frame=0, end_frame=None, video_path: str | None = None):
    """
    뷰별 특화된 백스윙탑 검출
    """
    if end_frame is None:
        end_frame = len(landmarks_data)

    # 0. 환경 진단 (첫 실행시만)
    if not hasattr(detect_top, '_env_diagnosed'):
        _diagnose_environment()
        detect_top._env_diagnosed = True

    # 1. 데이터 품질 검증
    quality_score = _validate_landmarks_quality(landmarks_data[start_frame:end_frame])
    print(f"[DEBUG] 랜드마크 품질 점수: {quality_score:.2f}")
    
    if quality_score < 0.3:
        print("[WARNING] 랜드마크 품질이 낮습니다. MediaPipe 설정 조정이 필요할 수 있습니다.")
    
    # FPS 검증 및 적응적 처리
    if video_path:
        actual_fps = _get_video_fps(video_path)
        if actual_fps and abs(actual_fps - fps) > 5:
            print(f"[WARNING] FPS 불일치: 입력 {fps}, 실제 {actual_fps:.1f}")
            fps = actual_fps  # 실제 FPS 사용
    
    # 1. 뷰 타입 정확한 판별
    view_type = _detect_view_type_robust(landmarks_data[start_frame:end_frame])
    print(f"[DEBUG] 감지된 뷰 타입: {view_type}")
    
    if view_type == 'front':
        return _detect_top_front_view(landmarks_data, fps, start_frame, end_frame, video_path)
    else:  # side view
        return _detect_top_side_view(landmarks_data, fps, start_frame, end_frame, video_path)


def _get_video_fps(video_path):
    """비디오의 실제 FPS를 반환"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
    except Exception as e:
        print(f"[ERROR] FPS 읽기 실패: {e}")
        return None


def _validate_landmarks_quality(landmarks_subset):
    """랜드마크 데이터 품질 검증 (0~1 점수)"""
    if not landmarks_subset:
        return 0.0
    
    valid_frames = 0
    total_visibility = 0.0
    key_joints_found = 0
    
    for lms in landmarks_subset:
        if lms is None:
            continue
            
        valid_frames += 1
        
        # 핵심 관절들의 가시성 확인
        key_joints = [
            PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER,
            PL.LEFT_WRIST, PL.RIGHT_WRIST,
            PL.LEFT_ELBOW, PL.RIGHT_ELBOW
        ]
        
        frame_visibility = []
        for joint in key_joints:
            joint_data = _get(lms, joint)
            if joint_data and hasattr(joint_data, 'visibility'):
                frame_visibility.append(joint_data.visibility)
        
        if frame_visibility:
            avg_visibility = np.mean(frame_visibility)
            total_visibility += avg_visibility
            
            # 핵심 관절이 충분히 보이는지 확인
            if avg_visibility > 0.6:
                key_joints_found += 1
    
    if valid_frames == 0:
        return 0.0
    
    # 품질 점수 계산
    frame_ratio = valid_frames / len(landmarks_subset)
    avg_visibility = total_visibility / valid_frames if valid_frames > 0 else 0
    key_joint_ratio = key_joints_found / valid_frames if valid_frames > 0 else 0
    
    quality_score = (frame_ratio * 0.3 + avg_visibility * 0.4 + key_joint_ratio * 0.3)
    return min(quality_score, 1.0)


def _detect_view_type_robust(landmarks_subset):
    """
    강화된 뷰 타입 판별 - 전체 프레임 분석 및 다중 지표 활용
    """
    shoulder_width_changes = []
    z_variations = []
    yaw_values = []
    
    # 전체 프레임 분석 (초반 30프레임 제한 제거)
    for lms in landmarks_subset:
        if lms is None:
            continue
            
        ls, rs = _shoulders(lms)
        if ls and rs:
            # 어깨 폭 (정면에서는 회전시 변함, 측면에서는 일정)
            shoulder_width = abs(ls.x - rs.x)
            shoulder_width_changes.append(shoulder_width)
            
            # Z 좌표 차이 (정면에서 회전시 z 차이 발생)
            z_diff = abs(getattr(ls, 'z', 0) - getattr(rs, 'z', 0))
            z_variations.append(z_diff)
            
            # 어깨 yaw 값 (정면에서 회전시 변화)
            yaw_val = _shoulder_yaw_value(lms)
            if not np.isnan(yaw_val):
                yaw_values.append(yaw_val)
    
    if not shoulder_width_changes:
        return 'side'  # 기본값
    
    # 다중 지표로 판별
    width_std = np.std(shoulder_width_changes)
    z_std = np.std(z_variations) if z_variations else 0
    yaw_mean = np.mean(np.abs(yaw_values)) if yaw_values else 0
    
    print(f"[DEBUG] 뷰 판별 지표 - width_std: {width_std:.4f}, z_std: {z_std:.4f}, yaw_mean: {yaw_mean:.4f}")
    
    # 정면뷰 판별 기준 (더 관대하게 조정)
    is_front = (width_std > 0.008) or (z_std > 0.012) or (yaw_mean > 0.04)
    
    return 'front' if is_front else 'side'


def _detect_top_front_view(landmarks_data, fps, start_frame, end_frame, video_path):
    """
    정면뷰 전용 백스윙탑 검출
    - 어깨 회전 최대점
    - 손목의 몸 중심으로부터 최대 이탈점
    """
    idx_map = []
    shoulder_angles = []
    hand_distances = []  # 몸 중심으로부터의 거리
    
    for i in range(start_frame, end_frame):
        lms = landmarks_data[i]
        if lms is None:
            continue
            
        ls, rs = _shoulders(lms)
        lh, rh = _hips(lms)
        lw, rw = _get(lms, PL.LEFT_WRIST), _get(lms, PL.RIGHT_WRIST)
        
        if not all([ls, rs, lh, rh, lw, rw]):
            continue
            
        idx_map.append(i)
        
        # 어깨 회전각 (정면에서 핵심 지표)
        shoulder_angle = _line_angle_deg(ls, rs)
        shoulder_angles.append(shoulder_angle)
        
        # 손목의 몸 중심으로부터 거리
        body_center_x = (ls.x + rs.x + lh.x + rh.x) / 4
        hands_center_x = (lw.x + rw.x) / 2
        distance = abs(hands_center_x - body_center_x)
        hand_distances.append(distance)
    
    if len(idx_map) < 10:
        return None
    
    # 어깨 각도 unwrap 및 스무딩
    from scipy import ndimage
    shoulder_angles = _unwrap_angles_deg(np.array(shoulder_angles))
    shoulder_smooth = ndimage.gaussian_filter1d(shoulder_angles, sigma=1.5)
    
    # 손 거리 스무딩
    hand_distances = ndimage.gaussian_filter1d(np.array(hand_distances), sigma=1.0)
    
    # 탐색 범위 (정면은 빨리 회전하므로 좀 더 앞쪽)
    search_start = int(len(idx_map) * 0.15)
    search_end = int(len(idx_map) * 0.70)
    
    candidates = []
    
    # 방법 1: 어깨 회전 극값 (개선된 파라미터)
    from scipy.signal import find_peaks
    
    # FPS 적응적 distance 설정
    min_distance = max(5, int(0.15*fps))
    
    # 회전 최대점들 찾기 (prominence 낮춤)
    peaks_max, props_max = find_peaks(
        shoulder_smooth[search_start:search_end], 
        distance=min_distance, 
        prominence=1.5  # 3.0에서 1.5로 낮춤
    )
    
    peaks_min, props_min = find_peaks(
        -shoulder_smooth[search_start:search_end], 
        distance=min_distance, 
        prominence=1.5  # 3.0에서 1.5로 낮춤
    )
    
    # 후보 수 제한 (상위 3개만)
    max_candidates = []
    
    # 최대값 후보들
    for i, peak in enumerate(peaks_max):
        actual_idx = peak + search_start
        prominence = props_max['prominences'][i]
        max_candidates.append(('rotation_max', actual_idx, prominence))
    
    # 최소값 후보들
    for i, peak in enumerate(peaks_min):
        actual_idx = peak + search_start
        prominence = props_min['prominences'][i]
        max_candidates.append(('rotation_min', actual_idx, prominence))
    
    # prominence 기준으로 상위 3개만 선택
    max_candidates.sort(key=lambda x: x[2], reverse=True)
    candidates.extend(max_candidates[:3])
    
    # 방법 2: 손 거리 최대점
    distance_peak = np.argmax(hand_distances[search_start:search_end]) + search_start
    peak_value = hand_distances[distance_peak]
    if peak_value > 0.08:  # 충분히 멀어진 경우만
        candidates.append(('hand_distance', distance_peak, peak_value * 10))
    
    # 후보 평가
    result = _evaluate_front_candidates(candidates, idx_map, landmarks_data, shoulder_smooth, hand_distances, video_path)
    
    # 폴백 적용
    if result is None:
        print("[DEBUG] 정면뷰 후보 없음 - 폴백 실행")
        result = _enhanced_fallback_top(landmarks_data, idx_map, start_frame, end_frame, video_path)
    
    return result


def _detect_top_side_view(landmarks_data, fps, start_frame, end_frame, video_path):
    """
    측면뷰 전용 백스윙탑 검출  
    - 손목 높이 최대점
    - 팔 각도와 샤프트 수평성
    """
    idx_map = []
    hand_heights = []
    arm_angles = []
    
    for i in range(start_frame, end_frame):
        lms = landmarks_data[i]
        if lms is None:
            continue
            
        hc = _hands_center(lms)
        if hc is None:
            continue
            
        idx_map.append(i)
        hand_heights.append(hc.y)
        
        # 팔 각도 (어깨-팔꿈치-손목)
        ls, rs = _shoulders(lms)
        le, re = _get(lms, PL.LEFT_ELBOW), _get(lms, PL.RIGHT_ELBOW)
        lw, rw = _get(lms, PL.LEFT_WRIST), _get(lms, PL.RIGHT_WRIST)
        
        if ls and le and lw:  # 왼팔 각도 우선
            angle = _angle(ls, le, lw)
            arm_angles.append(angle if not np.isnan(angle) else 180.0)
        elif rs and re and rw:  # 오른팔 대체
            angle = _angle(rs, re, rw)
            arm_angles.append(angle if not np.isnan(angle) else 180.0)
        else:
            arm_angles.append(180.0)
    
    if len(idx_map) < 10:
        return None
    
    # 스무딩
    from scipy import ndimage
    hand_heights = np.array(hand_heights)
    hand_smooth = ndimage.gaussian_filter1d(hand_heights, sigma=1.5)
    arm_smooth = ndimage.gaussian_filter1d(np.array(arm_angles), sigma=1.0)
    
    # 탐색 범위 (측면은 높이 변화가 명확하므로 넓게)
    search_start = int(len(idx_map) * 0.20)
    search_end = int(len(idx_map) * 0.80)
    
    candidates = []
    
    # 방법 1: 손 높이 최고점 (측면뷰에서 가장 확실) - 개선된 파라미터
    from scipy.signal import find_peaks
    
    # FPS 적응적 distance 설정
    min_distance = max(5, int(0.2*fps))
    
    height_peaks, height_props = find_peaks(
        -hand_smooth[search_start:search_end],  # 음수로 변환하여 최소값을 피크로
        distance=min_distance,
        prominence=0.02  # 0.015에서 0.02로 높임
    )
    
    # 상위 3개만 선택
    height_candidates = []
    for i, peak in enumerate(height_peaks):
        actual_idx = peak + search_start
        prominence = height_props['prominences'][i]
        height_candidates.append(('height_peak', actual_idx, prominence * 50))
    
    # prominence 기준으로 상위 3개만 선택
    height_candidates.sort(key=lambda x: x[2], reverse=True)
    candidates.extend(height_candidates[:3])
    
    # 방법 2: 팔 각도 변곡점 (펴짐 → 굽힘 전환) - 개선된 threshold
    arm_velocity = np.gradient(arm_smooth)
    for i in range(search_start + 1, search_end - 1):
        # 팔이 펴지다가 굽히기 시작하는 지점 (threshold 높임)
        if arm_velocity[i-1] > 0.2 and arm_velocity[i] < -0.2:  # 0.1에서 0.2로 높임
            strength = abs(arm_velocity[i] - arm_velocity[i-1])
            candidates.append(('arm_inflection', i, strength * 5))
    
    # 후보 평가  
    result = _evaluate_side_candidates(candidates, idx_map, landmarks_data, hand_smooth, arm_smooth, video_path)
    
    # 폴백 적용
    if result is None:
        print("[DEBUG] 측면뷰 후보 없음 - 폴백 실행")
        result = _enhanced_fallback_top(landmarks_data, idx_map, start_frame, end_frame, video_path)
    
    return result


def _evaluate_front_candidates(candidates, idx_map, landmarks_data, shoulder_smooth, hand_distances, video_path=None):
    """정면뷰 후보 평가 (CV 샤프트 통합)"""
    if not candidates:
        return None
    
    print(f"[DEBUG] 정면뷰 후보 수: {len(candidates)}")
    
    best_frame = None
    best_score = -1
    
    for cand_type, cand_idx, strength in candidates:
        if cand_idx >= len(idx_map):
            continue
            
        score = strength  # 기본 강도
        
        # 회전량 보너스 (정면에서 중요)
        rotation_change = abs(shoulder_smooth[cand_idx] - shoulder_smooth[max(0, cand_idx-5)])
        score += min(rotation_change * 0.2, 2.0)
        
        # 손 거리 보너스
        score += hand_distances[cand_idx] * 8.0
        
        # CV 샤프트 수평성 보너스 (video_path가 있을 때)
        if video_path:
            try:
                frame_idx = idx_map[cand_idx]
                ang_cv = _estimate_shaft_angle_cv(video_path, frame_idx, landmarks_data[frame_idx])
                if ang_cv is not None:
                    dev = _vertical_dev_deg(type('P', (), {'x': 0, 'y': 0}), 
                                          type('P', (), {'x': 1, 'y': 0}))  # 수평선과의 차이
                    if dev < 30:  # 30도 이내면 수평에 가까움
                        score += 2.0
                        print(f"[DEBUG] CV 샤프트 보너스: {dev:.1f}도")
            except Exception as e:
                print(f"[DEBUG] CV 샤프트 분석 실패: {e}")
        
        # 포즈 검증 필터 (score < 0.5면 무시)
        pose_score = _evaluate_top_pose_simple(landmarks_data[idx_map[cand_idx]])
        if pose_score < 0.5:
            print(f"[DEBUG] 포즈 검증 실패: {pose_score:.2f}")
            continue
        
        score += pose_score * 1.0  # 포즈 보너스
        
        # 타이밍 적절성
        timing_ratio = cand_idx / len(idx_map)
        if 0.2 <= timing_ratio <= 0.6:
            score += 1.0
        
        print(f"[DEBUG] 후보 {cand_type}: score={score:.2f}, timing={timing_ratio:.2f}")
        
        if score > best_score:
            best_score = score
            best_frame = idx_map[cand_idx]
    
    print(f"[DEBUG] 정면뷰 최종 선택: frame={best_frame}, score={best_score:.2f}")
    return best_frame


def _evaluate_side_candidates(candidates, idx_map, landmarks_data, hand_smooth, arm_smooth, video_path=None):
    """측면뷰 후보 평가 (각속도 체크 및 팔 각도 범위 확대)"""
    if not candidates:
        return None
    
    print(f"[DEBUG] 측면뷰 후보 수: {len(candidates)}")
    
    best_frame = None
    best_score = -1
    
    for cand_type, cand_idx, strength in candidates:
        if cand_idx >= len(idx_map):
            continue
            
        score = strength  # 기본 강도
        
        # 높이 보너스 (측면에서 핵심)
        height_percentile = np.percentile(hand_smooth, 15)  # 상위 15% 높이
        if hand_smooth[cand_idx] <= height_percentile:
            score += 3.0
        
        # 팔 각도 적절성 (100~170도 범위로 확대)
        arm_angle = arm_smooth[cand_idx]
        if 100 <= arm_angle <= 170:  # 120~160에서 100~170으로 확대
            score += 2.0
        
        # 각속도 체크 (탑에서 각속도 < 1.0)
        arm_velocity = abs(np.gradient(arm_smooth)[cand_idx])
        if arm_velocity < 1.0:
            score += 1.5
            print(f"[DEBUG] 각속도 보너스: {arm_velocity:.2f}")
        
        # CV 샤프트 수평성 보너스 (video_path가 있을 때)
        if video_path:
            try:
                frame_idx = idx_map[cand_idx]
                ang_cv = _estimate_shaft_angle_cv(video_path, frame_idx, landmarks_data[frame_idx])
                if ang_cv is not None:
                    dev = _vertical_dev_deg(type('P', (), {'x': 0, 'y': 0}), 
                                          type('P', (), {'x': 1, 'y': 0}))  # 수평선과의 차이
                    if dev < 30:  # 30도 이내면 수평에 가까움
                        score += 2.0
                        print(f"[DEBUG] CV 샤프트 보너스: {dev:.1f}도")
            except Exception as e:
                print(f"[DEBUG] CV 샤프트 분석 실패: {e}")
        
        # 포즈 검증 필터 (score < 0.5면 무시)
        pose_score = _evaluate_top_pose_simple(landmarks_data[idx_map[cand_idx]])
        if pose_score < 0.5:
            print(f"[DEBUG] 포즈 검증 실패: {pose_score:.2f}")
            continue
        
        score += pose_score * 1.0  # 포즈 보너스
        
        # 타이밍 적절성  
        timing_ratio = cand_idx / len(idx_map)
        if 0.25 <= timing_ratio <= 0.7:
            score += 1.0
        
        print(f"[DEBUG] 후보 {cand_type}: score={score:.2f}, timing={timing_ratio:.2f}, arm_angle={arm_angle:.1f}")
        
        if score > best_score:
            best_score = score
            best_frame = idx_map[cand_idx]
    
    print(f"[DEBUG] 측면뷰 최종 선택: frame={best_frame}, score={best_score:.2f}")
    return best_frame


def _enhanced_fallback_top(landmarks_data, idx_map, start_frame, end_frame, video_path=None):
    """강화된 폴백 로직 - y min + 샤프트 dev min 결합"""
    if not idx_map:
        return None
    
    print("[DEBUG] 폴백 로직 실행")
    
    # 1순위: 손 높이 최소값 (기본)
    hand_heights = []
    for i in range(len(idx_map)):
        hc = _hands_center(landmarks_data[idx_map[i]])
        if hc:
            hand_heights.append(hc.y)
        else:
            hand_heights.append(float('inf'))
    
    height_min_idx = np.argmin(hand_heights)
    height_min_frame = idx_map[height_min_idx]
    
    # 2순위: CV 샤프트 수평성 (video_path가 있을 때)
    if video_path:
        best_cv_frame = None
        best_cv_score = float('inf')
        
        # 탐색 범위: 전체의 20%~80%
        search_start = int(len(idx_map) * 0.20)
        search_end = int(len(idx_map) * 0.80)
        
        for i in range(search_start, min(search_end, len(idx_map))):
            try:
                frame_idx = idx_map[i]
                ang_cv = _estimate_shaft_angle_cv(video_path, frame_idx, landmarks_data[frame_idx])
                if ang_cv is not None:
                    # 수평선과의 차이 계산 (ang_cv 기반)
                    dev = _horizontal_deviation_deg(ang_cv)
                    if dev < best_cv_score:
                        best_cv_score = dev
                        best_cv_frame = frame_idx
            except Exception:
                continue
        
        if best_cv_frame is not None and best_cv_score < 45:  # 45도 이내면 수평에 가까움
            print(f"[DEBUG] CV 폴백 선택: frame={best_cv_frame}, dev={best_cv_score:.1f}도")
            return best_cv_frame
    
    # 3순위: 손 높이 최소값
    print(f"[DEBUG] 높이 폴백 선택: frame={height_min_frame}")
    return height_min_frame


def _evaluate_top_pose_simple(lms):
    """간단하고 신뢰성 높은 탑 포즈 평가 (0~1 점수)"""
    if lms is None:
        return 0.0
    
    ls, rs = _shoulders(lms)
    lw, rw = _get(lms, PL.LEFT_WRIST), _get(lms, PL.RIGHT_WRIST)
    
    if any(p is None for p in [ls, rs, lw, rw]):
        return 0.0
    
    score = 0.0
    
    # 1) 손이 어깨보다 높은가 (가장 중요)
    hands_y = min(lw.y, rw.y)
    shoulders_y = min(ls.y, rs.y)
    if hands_y < shoulders_y * 1.05:  # 약간의 여유
        score += 0.4
    
    # 2) 손목들이 몸 중심에서 벗어나 있는가
    body_center_x = (ls.x + rs.x) / 2
    hands_center_x = (lw.x + rw.x) / 2
    if abs(hands_center_x - body_center_x) > 0.1:
        score += 0.3
    
    # 3) 가시성 충분한가
    visibility = []
    for p in [ls, rs, lw, rw]:
        if hasattr(p, 'visibility'):
            visibility.append(p.visibility)
    if visibility and np.mean(visibility) > 0.6:
        score += 0.3
    
    return min(score, 1.0)



# -------------------- 개선된 임팩트 검출 (통합 로직) --------------------
def detect_impact(landmarks_data, velocities, fps, start_frame, end_frame):
    """
    임팩트 검출(개선 버전): 속도 스무딩 + 최소 속도 임계 + 뷰별 가중 + 손높이 상대화 + 근접 재탐색
    
    핵심 아이디어:
      - 속도는 EMA로 스무딩하여 노이즈 완화
      - 너무 느린 프레임은 제외(VEL_MIN)
      - 정면/측면 뷰에 따라 (속도):(손높이 역수) 가중을 달리 부여
      - 손높이는 어드레스 대비 상대 높이(hand_y / hand_y_address)로 정규화하여 카메라 각도 보정
      - 1차 후보 이후 ±win 창에서 손높이 최소 프레임로 미세조정
    """
    N = len(landmarks_data)
    if N == 0:
        return None

    # 검색 구간 보정: 너무 이른/늦은 영역 제외 (0.15s ~ end)
    start_frame = max(0, int(start_frame))
    end_frame = min(int(end_frame), N)
    start_frame = max(start_frame, int(_safe_idx(_detect_swing_start(landmarks_data, fps=fps) + int(0.15 * fps), 0, N - 1)))

    # 뷰 타입 판별(가중치 조정용)
    try:
        view_type = _detect_view_type_robust(landmarks_data[start_frame:end_frame])
    except Exception:
        view_type = 'side'

    # 속도 스무딩(EMA)
    velocities = _ema(velocities, alpha=0.35) if velocities is not None else []

    # 어드레스 손높이(상대화 기준) 추정
    try:
        addr_idx = _detect_swing_start(landmarks_data, fps=fps)
        hc_addr = _hands_center(landmarks_data[_safe_idx(addr_idx, 0, N - 1)])
        hand_y_base = float(hc_addr.y) if hc_addr is not None else 0.7
    except Exception:
        hand_y_base = 0.7
    hand_y_base = max(hand_y_base, 1e-3)

    VEL_MIN = 0.02
    impact_candidates: list[tuple[float, int, float, float]] = []  # (score, idx, velocity, hand_y_rel)

    for i in range(start_frame, min(end_frame, N)):
        lms = landmarks_data[i]
        if lms is None:
            continue
        hc = _hands_center(lms)
        if hc is None:
            continue
        velocity = float(velocities[i]) if i < len(velocities) else 0.0
        if velocity < VEL_MIN:
            continue
        hand_y = float(hc.y)
        hand_y_rel = hand_y / hand_y_base

        # 뷰별 가중치 결합 점수
        if view_type == 'front':
            score = velocity * 0.7 + (1.0 / (hand_y_rel + 1e-6)) * 0.3
        else:  # side (default)
            score = velocity * 0.8 + (1.0 / (hand_y_rel + 1e-6)) * 0.2

        impact_candidates.append((float(score), i, velocity, hand_y_rel))

    if not impact_candidates:
        return None

    # 1차 선택: 결합 점수 최대 프레임
    score_max_idx = max(impact_candidates, key=lambda t: t[0])[1]

    # 2차: 시간 창 기반 투표/평균(변형 포즈 허용)
    win_sec = 0.10  # ±0.1s 창
    win = max(1, int(win_sec * fps / 2.0))
    lo = max(start_frame, score_max_idx - win)
    hi = min(end_frame, score_max_idx + win + 1)

    local = [t for t in impact_candidates if lo <= t[1] < hi]
    if local:
        v_max = max(t[2] for t in local)
        # 속도 상위(>=90% max) + 손높이 상대값 허용(<=1.05)
        local_candidates = [t for t in local if t[2] >= 0.9 * v_max and t[3] <= 1.05]
        if local_candidates:
            # 단순 투표(동률 시 손높이 상대값 최소 프레임)
            counts: dict[int, int] = {}
            for _, idx, _, _ in local_candidates:
                counts[idx] = counts.get(idx, 0) + 1
            best_idx = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            # 근접 후보 중에서도 손이 가장 낮은 프레임으로 한 번 더 보정
            best_group = [t for t in local_candidates if t[1] == best_idx]
            if best_group:
                best = min(best_group, key=lambda t: t[3])
                return int(best[1])

        # 후보 없으면 손높이 상대값 최소 프레임로 미세조정(기존 로직)
        strong = [t for t in local if t[2] >= 0.9 * v_max]
        if strong:
            best = min(strong, key=lambda t: t[3])
            return int(best[1])

    return int(score_max_idx)

# -------------------- 개선된 피니시 검출 (통합 로직) --------------------
def detect_finish(landmarks_data, fps, impact_frame):
    """
    피니시 검출: 포즈 충족 + 최소 0.6초 연속 유지
    """
    stable_win = int(0.6 * fps)
    consec = 0
    min_search_start = impact_frame + int(0.5 * fps)
    
    for i in range(max(impact_frame, min_search_start), len(landmarks_data)):
        lms = landmarks_data[i]
        if lms is None:
            consec = 0
            continue
            
        if is_finish_pose(lms):
            consec += 1
            if consec >= stable_win:
                return i - stable_win + 1
        else:
            consec = 0
    
    return None

# -------------------- 메인: 단계 검출 (통합 로직 적용) --------------------
def detect_swing_phases(landmarks_data, total_frames=None, video_aspect_ratio=None, fps=30, video_path: str | None = None):
    """
    Args:
        landmarks_data: mediapipe pose landmarks 리스트 (프레임 순서)
        total_frames: 전체 프레임 수(선택)
        video_aspect_ratio: W/H. None이면 1.0 처리
        fps: 프레임레이트(기본 30)
    Returns:
        dict: address/takeaway_end/backswing/top/downswing_start/impact/followthrough/finish/finish_plus_1_5s
    """
    # 빈 입력 처리
    if not landmarks_data:
        num_frames = total_frames if total_frames is not None else 5
        return {
            "address": 0, "takeaway_end": min(1, num_frames - 4),
            "backswing": [min(1, num_frames - 4), min(2, num_frames - 3)],
            "top": min(2, num_frames - 3), "downswing_start": min(1, num_frames - 4),
            "impact": min(3, num_frames - 2),
            "followthrough": [min(2, num_frames - 3), min(num_frames - 1, num_frames - 1)],
            "finish": min(4, num_frames - 1),
            "finish_plus_1_5s": min(num_frames - 1, num_frames - 1)
        }

    try:
        num_frames = len(landmarks_data)
        if total_frames is None:
             total_frames = num_frames
        
        ar = 1.0 if video_aspect_ratio is None else float(video_aspect_ratio)
        if ar < 1.0:      lim = int(num_frames * 0.70)
        elif ar > 1.5:    lim = int(num_frames * 0.80)
        else:             lim = int(num_frames * 0.75)
        video_length_limit = max(1, min(lim, num_frames))

        # 0) 스윙 시작 검출 -> 어드레스 = 시작-1
        swing_start = _detect_swing_start(landmarks_data, fps=fps)
        address_frame = _safe_idx(swing_start - 1, 0, num_frames - 1)

        # 1) 백스윙 탑: 개선된 통합 로직 사용
        top_frame = detect_top(landmarks_data, fps, address_frame, video_length_limit, video_path=video_path)
        
        if top_frame is None:
            # 폴백: 손목 y 최소 프레임 - 강화: 최소 간격 address_frame + 20
            y_list = []
            for i in range(address_frame, video_length_limit):
                hc = _hands_center(landmarks_data[i])
                if hc is not None:
                    y_list.append((hc.y, i))
            if y_list:
                top_frame = min(y_list, key=lambda t: t[0])[1]
            else:
                top_frame = _safe_idx(address_frame + 20, 0, num_frames - 1)

        # 2) 임팩트: 개선된 통합 로직 사용
        # 손 중심 속도 계산
        hc_xy = []
        for i in range(num_frames):
            hc = _hands_center(landmarks_data[i])
            hc_xy.append((np.nan, np.nan) if hc is None else (hc.x, hc.y))
        hc_xy = np.array(hc_xy)
        
        velocities = np.zeros(num_frames)
        for i in range(1, num_frames):
            if not np.any(np.isnan(hc_xy[i])) and not np.any(np.isnan(hc_xy[i-1])):
                velocities[i] = abs(hc_xy[i,0]-hc_xy[i-1,0]) + abs(hc_xy[i,1]-hc_xy[i-1,1])
        
        # 임팩트 검출 (탑 이후 ~ 영상 90%)
        impact_search_start = _safe_idx(top_frame, 0, num_frames - 1)
        impact_search_end = min(int(num_frames * 0.9), num_frames)
        
        impact_frame = detect_impact(landmarks_data, velocities, fps, impact_search_start, impact_search_end)
        
        if impact_frame is None:
            # 폴백: 탑 이후 속도 최대 프레임
            impact_frame = _safe_idx(top_frame + 10, 0, num_frames - 1)

        # 3) 피니시: 개선된 통합 로직 사용
        finish_frame = detect_finish(landmarks_data, fps, impact_frame)
        
        if finish_frame is None:
            # 폴백: 임팩트 이후 첫 finish 포즈 또는 임팩트+20 (강화)
            for i in range(impact_frame, num_frames):
                if is_finish_pose(landmarks_data[i]):
                    finish_frame = i
                    break
            if finish_frame is None:
                finish_frame = _safe_idx(impact_frame + 20, 0, num_frames - 1)

        # 4) 논리 순서 및 최소 간격 보정
        MIN_GAP = 3
        if top_frame <= address_frame:
            top_frame = _safe_idx(address_frame + MIN_GAP, 0, num_frames - 1)
        if impact_frame <= top_frame:
            impact_frame = _safe_idx(top_frame + MIN_GAP, 0, num_frames - 1)
        if finish_frame <= impact_frame:
            finish_frame = _safe_idx(impact_frame + MIN_GAP, 0, num_frames - 1)

        # 5) 보조 구간 산출
        takeaway_end = (address_frame + top_frame) // 2
        downswing_start = (top_frame + impact_frame) // 2

        result = {
            "address": address_frame,
            "takeaway_end": takeaway_end,
            "backswing": [address_frame, top_frame],
            "top": top_frame,
            "downswing_start": downswing_start,
            "impact": impact_frame,
            "followthrough": [impact_frame, finish_frame],
            "finish": finish_frame,
            "finish_plus_1_5s": _safe_idx(finish_frame + int(1.5 * fps), 0, num_frames - 1)
        }

        # 간단 로그 (원한다면 제거 가능)
        print("[INFO] Address:", result["address"],
              " Top:", result["top"],
              " Impact:", result["impact"],
              " Finish:", result["finish"])
        
        return result

    except Exception as e:
        print(f"[ERROR] detect_swing_phases failed: {e}")
        nf = len(landmarks_data) if landmarks_data else 5
        return {
            "address": 0,
            "takeaway_end": min(1, nf - 1),
            "backswing": [0, min(2, nf - 1)],
            "top": min(2, nf - 1),
            "downswing_start": min(1, nf - 1),
            "impact": min(3, nf - 1),
            "followthrough": [min(3, nf - 1), min(4, nf - 1)],
            "finish": min(4, nf - 1),
            "finish_plus_1_5s": min(4, nf - 1)
        }


# -------------------- 간단 버전: Address/Finish만 감지 --------------------
def detect_swing_phases_simple(landmarks_data, fps=30, video_path: str | None = None):
    """
    Address와 Finish만 감지하는 간단한 스윙 단계 감지 함수
    - 시작: 정적→가속 전환 직전 프레임(_detect_swing_start - 1)
    - 종료: finish 포즈가 0.6초 이상 유지되는 첫 프레임(detect_finish)
    """
    if not landmarks_data or len(landmarks_data) < 5:
        num_frames = 5
        return {
            "address": 0, "address_s": 0.0,
            "finish": min(4, num_frames - 1), "finish_s": min(4, num_frames - 1) / max(1, fps),
            "segment": [0, min(4, num_frames - 1)]
        }

    try:
        num_frames = len(landmarks_data)

        swing_start = _detect_swing_start(landmarks_data, fps=fps)
        address_frame = _safe_idx(swing_start - 1, 0, num_frames - 1)

        finish_frame = detect_finish(landmarks_data, fps, address_frame)
        if finish_frame is None:
            finish_frame = _safe_idx(address_frame + int(1.0 * fps), 0, num_frames - 1)

        MIN_GAP = int(0.5 * fps)
        if finish_frame <= address_frame:
            finish_frame = _safe_idx(address_frame + MIN_GAP, 0, num_frames - 1)

        address_s = address_frame / max(1, fps)
        finish_s = finish_frame / max(1, fps)

        result = {
            "address": address_frame, "address_s": address_s,
            "finish": finish_frame, "finish_s": finish_s,
            "segment": [address_frame, finish_frame]
        }
        print(f"[INFO] Simple Phases - Address: {address_frame} ({address_s:.2f}s), Finish: {finish_frame} ({finish_s:.2f}s)")
        return result
    except Exception as e:
        print(f"[ERROR] detect_swing_phases_simple failed: {e}")
        return {
            "address": 0, "address_s": 0.0,
            "finish": 30, "finish_s": 1.0,
            "segment": [0, 30]
        }


# -------------------- 백스윙 탑 무시 버전: 임팩트 중심 2구간 비교 --------------------
def detect_swing_phases_no_top(landmarks_data, total_frames=None, video_aspect_ratio=None, fps=30, video_path: str | None = None):
    """
    백스윙 탑을 무시한 스윙 단계 감지 함수
    
    임팩트 중심 2구간 비교 방식:
    - 구간1: Address → Impact (전체 백스윙 + 다운스윙)
    - 구간2: Impact → Finish (팔로우스루)
    
    Args:
        landmarks_data: mediapipe pose landmarks 리스트 (프레임 순서)
        total_frames: 전체 프레임 수(선택)
        video_aspect_ratio: W/H. None이면 1.0 처리
        fps: 프레임레이트(기본 30)
        video_path: 비디오 파일 경로 (선택사항)
        
    Returns:
        dict: address/impact/finish 및 구간 정보
    """
    # 빈 입력 처리
    if not landmarks_data:
        num_frames = total_frames if total_frames is not None else 5
        return {
            "address": 0, "address_s": 0.0,
            "impact": min(2, num_frames - 1), "impact_s": min(2, num_frames - 1) / fps,
            "finish": min(4, num_frames - 1), "finish_s": min(4, num_frames - 1) / fps,
            "segment1": [0, min(2, num_frames - 1)],  # Address → Impact
            "segment2": [min(2, num_frames - 1), min(4, num_frames - 1)]  # Impact → Finish
        }

    try:
        num_frames = len(landmarks_data)
        if total_frames is None:
            total_frames = num_frames
        
        # 비디오 길이 제한 설정
        ar = 1.0 if video_aspect_ratio is None else float(video_aspect_ratio)
        if ar < 1.0:      lim = int(num_frames * 0.70)
        elif ar > 1.5:    lim = int(num_frames * 0.80)
        else:             lim = int(num_frames * 0.75)
        video_length_limit = max(1, min(lim, num_frames))

        # 1) 스윙 시작 검출 -> 어드레스 = 시작-1
        swing_start = _detect_swing_start(landmarks_data, fps=fps)
        address_frame = _safe_idx(swing_start - 1, 0, num_frames - 1)

        # 2) 손 중심 속도 계산 (임팩트 검출용)
        hc_xy = []
        for i in range(num_frames):
            hc = _hands_center(landmarks_data[i])
            hc_xy.append((np.nan, np.nan) if hc is None else (hc.x, hc.y))
        hc_xy = np.array(hc_xy)
        
        velocities = np.zeros(num_frames)
        for i in range(1, num_frames):
            if not np.any(np.isnan(hc_xy[i])) and not np.any(np.isnan(hc_xy[i-1])):
                velocities[i] = abs(hc_xy[i,0]-hc_xy[i-1,0]) + abs(hc_xy[i,1]-hc_xy[i-1,1])

        # 3) 임팩트 검출 (Address 이후 ~ 영상 90%)
        impact_search_start = address_frame
        impact_search_end = min(int(num_frames * 0.9), num_frames)
        
        impact_frame = detect_impact(landmarks_data, velocities, fps, impact_search_start, impact_search_end)
        
        if impact_frame is None:
            # 폴백: Address + 0.3초 (기본 다운스윙 시간)
            impact_frame = _safe_idx(address_frame + int(0.3 * fps), 0, num_frames - 1)

        # 4) 피니시 검출 (임팩트 이후)
        finish_frame = detect_finish(landmarks_data, fps, impact_frame)
        
        if finish_frame is None:
            # 폴백: 임팩트 + 0.5초 (기본 팔로우스루 시간)
            finish_frame = _safe_idx(impact_frame + int(0.5 * fps), 0, num_frames - 1)

        # 5) 최소 간격 보정
        MIN_GAP = int(0.2 * fps)  # 0.2초 최소 간격
        if impact_frame <= address_frame:
            impact_frame = _safe_idx(address_frame + MIN_GAP, 0, num_frames - 1)
        if finish_frame <= impact_frame:
            finish_frame = _safe_idx(impact_frame + MIN_GAP, 0, num_frames - 1)

        # 6) 초 단위 환산
        def frame_to_sec(frame, fps): 
            return frame / fps
            
        address_s = frame_to_sec(address_frame, fps)
        impact_s = frame_to_sec(impact_frame, fps)
        finish_s = frame_to_sec(finish_frame, fps)

        result = {
            "address": address_frame, "address_s": address_s,
            "impact": impact_frame, "impact_s": impact_s,
            "finish": finish_frame, "finish_s": finish_s,
            "segment1": [address_frame, impact_frame],  # Address → Impact
            "segment2": [impact_frame, finish_frame]    # Impact → Finish
        }

        # 로그 출력
        print("[INFO] No-Top Method - Address:", result["address"], f"({address_s:.2f}s)",
              " Impact:", result["impact"], f"({impact_s:.2f}s)",
              " Finish:", result["finish"], f"({finish_s:.2f}s)")
        
        return result

    except Exception as e:
        print(f"[ERROR] detect_swing_phases_no_top failed: {e}")
        nf = len(landmarks_data) if landmarks_data else 5
        return {
            "address": 0, "address_s": 0.0,
            "impact": min(2, nf - 1), "impact_s": min(2, nf - 1) / fps,
            "finish": min(4, nf - 1), "finish_s": min(4, nf - 1) / fps,
            "segment1": [0, min(2, nf - 1)],
            "segment2": [min(2, nf - 1), min(4, nf - 1)]
        }