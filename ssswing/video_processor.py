"""
골프 스윙 분석을 위한 비디오 처리 모듈

이 모듈은 비디오에서 랜드마크를 추출하고 처리하는 함수들을 제공합니다.
주요 기능:
- 비디오에서 MediaPipe 랜드마크 추출
- 세로 영상 자동 회전 처리
- 영상 비율 감지
- 랜드마크 데이터 전처리

작성자: AI Assistant
개선일: 2024년
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from .math_utils import robust_argmin, robust_argmax, calculate_velocity_profile, find_swing_phases_by_velocity

mp_pose = mp.solutions.pose
pose_landmark_enum = mp.solutions.pose.PoseLandmark

# 환경변수 기반 튜닝 포인트 (필요 시 조정)
POSE_LONG_SIDE = int(os.getenv("POSE_LONG_SIDE", "1280"))  # 입력 프레임 업샘플 기준 긴 변(px)
POSE_MIN_DET = float(os.getenv("POSE_MIN_DET", "0.5"))     # 1차 트래킹 모드 detection 임계
POSE_MIN_TRK = float(os.getenv("POSE_MIN_TRK", "0.5"))     # 1차 트래킹 모드 tracking 임계
POSE_MODEL_COMPLEXITY = int(os.getenv("POSE_MODEL_COMPLEXITY", "2"))  # 0: lite, 1: full, 2: heavy

# 발 가시성 기준 및 재검출 조건
FOOT_VIS_THRESHOLD = float(os.getenv("POSE_FOOT_VISIBILITY", "0.4"))  # 제안: 0.3~0.4
LEFTFOOT_REQUIRED_POINTS = int(os.getenv("POSE_LEFTFOOT_REQUIRED_POINTS", "2"))  # 27/29/31 중 만족해야 하는 개수
REDETECT_STREAK = int(os.getenv("POSE_REDETECT_STREAK", "3"))          # 연속 소실 프레임 수 임계
LOWER_ROI_RATIO = float(os.getenv("POSE_LOWER_ROI_RATIO", "0.45"))     # 하체 ROI 높이 비율(하단부터)

# 스무딩 윈도우 (1이면 비활성)
SMOOTH_WINDOW = int(os.getenv("POSE_SMOOTH_WINDOW", "3"))


class SimpleLandmark:
    """
    MediaPipe NormalizedLandmark 대체용 간단 구조체.
    ROI 기반 재투영 이후 전체 프레임 정규 좌표계로 매핑할 때 사용.
    """
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _resize_to_long_side(frame: np.ndarray, target_long_side: int) -> np.ndarray:
    """
    긴 변 기준으로 프레임을 리사이즈(업샘플 포함)합니다.
    비율은 유지합니다.
    """
    h, w = frame.shape[:2]
    if max(h, w) == target_long_side:
        return frame
    if h >= w:
        scale = target_long_side / float(h)
    else:
        scale = target_long_side / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _is_left_foot_confident(landmarks) -> bool:
    """
    왼발 핵심 포인트(발목 27, 뒤꿈치 29, 검지 31)의 유효/가시성 확인.
    3점 중 LEFTFOOT_REQUIRED_POINTS 이상 충족 시 신뢰 OK로 간주.
    """
    try:
        idxs = [27, 29, 31]
        ok = 0
        for i in idxs:
            lm = landmarks[i]
            if lm is None:
                continue
            vis = getattr(lm, "visibility", 0.0)
            x = getattr(lm, "x", None)
            y = getattr(lm, "y", None)
            if x is None or y is None:
                continue
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                continue
            if vis >= FOOT_VIS_THRESHOLD:
                ok += 1
        return ok >= max(1, LEFTFOOT_REQUIRED_POINTS)
    except Exception:
        return False


def _remap_roi_landmarks_to_full(roi_landmarks, roi_xywh, full_w, full_h):
    """
    ROI에서 검출된 정규화 좌표(0..1)를 전체 프레임 정규화 좌표(0..1)로 매핑.
    """
    rx, ry, rw, rh = roi_xywh
    out = []
    for lm in roi_landmarks:
        if lm is None:
            out.append(None)
            continue
        x_full = (lm.x * rw + rx) / max(1.0, full_w)
        y_full = (lm.y * rh + ry) / max(1.0, full_h)
        out.append(SimpleLandmark(x_full, y_full, getattr(lm, "z", 0.0), getattr(lm, "visibility", 0.0)))
    return out


def _get_lower_body_bbox(landmarks, frame_w: int, frame_h: int, margin: float = 0.08):
    """
    마지막 유효 프레임의 하체(엉덩이/무릎/발목) 좌표를 이용해 바운딩박스 추정.
    margin 비율로 여유를 두고, 화면 범위로 클램프합니다.
    """
    if not landmarks:
        return None
    idxs = [23, 24, 25, 26, 27, 28]  # 좌우 엉덩이/무릎/발목
    xs = []
    ys = []
    for i in idxs:
        lm = landmarks[i] if i < len(landmarks) else None
        if lm is None:
            continue
        x = getattr(lm, "x", None)
        y = getattr(lm, "y", None)
        if x is None or y is None:
            continue
        xs.append(x * frame_w)
        ys.append(y * frame_h)
    if not xs or not ys:
        return None
    x1 = max(0, int(min(xs)))
    x2 = min(frame_w - 1, int(max(xs)))
    y1 = max(0, int(min(ys)))
    y2 = min(frame_h - 1, int(max(ys)))
    # 약간 확장, 특히 아래쪽으로 더 여유
    mx = int(round((x2 - x1 + 1) * margin))
    my = int(round((y2 - y1 + 1) * (margin * 1.5)))
    x1 = max(0, x1 - mx)
    x2 = min(frame_w - 1, x2 + mx)
    y1 = max(0, y1 - my)
    y2 = min(frame_h - 1, y2 + my * 2)
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def _enhance_contrast_sharpen(bgr_img: np.ndarray) -> np.ndarray:
    """
    ROI 대비 강화: CLAHE(L 채널) + 언샤프 마스크 적용.
    """
    try:
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        enh = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        # 언샤프: 가우시안 블러 후 가중 합성
        blur = cv2.GaussianBlur(enh, (0, 0), sigmaX=1.0)
        sharp = cv2.addWeighted(enh, 1.5, blur, -0.5, 0)
        return sharp
    except Exception:
        return bgr_img


def extract_landmarks_from_video(video_path):
    """
    주어진 비디오에서 mediapipe를 이용해 각 프레임의 랜드마크 리스트를 추출합니다.
    세로 영상은 자동으로 회전하여 처리합니다.
    
    Args:
        video_path (str): 입력 영상 경로
        
    Returns:
        tuple: (landmarks_list, global_video_aspect_ratio)
    """
    # 1차: 트래킹 모드(영상용) 포즈 추정기 (세그멘테이션 활성화)
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=POSE_MIN_DET,
        min_tracking_confidence=POSE_MIN_TRK,
        model_complexity=POSE_MODEL_COMPLEXITY,
        enable_segmentation=True,
        smooth_segmentation=True,
    )
    # 2차: 정적(고정) 모드 재검출기 (필요 시 on-demand 생성)
    pose_static = None
    cap = cv2.VideoCapture(video_path)
    
    # 실제 영상 비율 계산
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    global_video_aspect_ratio = width / height if height != 0 else 1.0
    print(f"[INFO] Detected video aspect ratio: {global_video_aspect_ratio:.2f}")
    
    landmarks_list = []
    missing_streak = 0
    last_good_lower = None  # 마지막으로 왼발 신뢰 OK였던 프레임의 하체 landmarks 보관
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 세로 영상(포트레이트) 처리: 회전
        if global_video_aspect_ratio < 1.0:  # height > width
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 시계방향 90도 회전 (세로 -> 가로)
            print(f"[DEBUG] Rotated frame for portrait video")
        # 입력 업샘플 (긴 변 기준)
        proc_frame = _resize_to_long_side(frame, POSE_LONG_SIDE)

        image = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 1차 결과
        if results.pose_landmarks:
            current_landmarks = results.pose_landmarks.landmark
        else:
            current_landmarks = [None] * 33

        # 왼발 신뢰도 평가 및 연속 소실 카운트
        if not _is_left_foot_confident(current_landmarks):
            missing_streak += 1
        else:
            missing_streak = 0
            # 마지막 유효 하체 landmarks 캐시
            try:
                last_good_lower = [
                    current_landmarks[i] if i < len(current_landmarks) else None
                    for i in range(33)
                ]
            except Exception:
                pass

        # 연속 소실 시: 정적(heavy) 재검출 시도
        if missing_streak >= REDETECT_STREAK:
            try:
                if pose_static is None:
                    pose_static = mp_pose.Pose(
                        static_image_mode=True,
                        min_detection_confidence=0.3,  # 제안값: 낮춰서 더 많이 탐지
                        min_tracking_confidence=0.3,
                        model_complexity=2,
                        enable_segmentation=True,
                        smooth_segmentation=True,
                    )

                # 2-1) 전체 프레임 재검출
                re_results = pose_static.process(image)
                if re_results and re_results.pose_landmarks:
                    re_landmarks = re_results.pose_landmarks.landmark
                else:
                    re_landmarks = None

                # 2-2) 하체 ROI 폴백 (하단 영역 확대)
                if (re_landmarks is None) or (not _is_left_foot_confident(re_landmarks)):
                    h2, w2 = proc_frame.shape[:2]
                    # 마지막 유효 하체 기준 ROI 우선, 없으면 하단 고정 ROI 사용
                    xywh = _get_lower_body_bbox(last_good_lower, w2, h2, margin=0.1) if last_good_lower else None
                    if xywh is None:
                        roi_h = int(round(h2 * LOWER_ROI_RATIO))
                        roi_y = h2 - roi_h
                        roi_x = 0
                        roi_w = w2
                    else:
                        roi_x, roi_y, roi_w, roi_h = xywh
                        # 프레임 하단과 겹치도록 조금 아래로 내림(가능한 경우)
                        shift = int(round(roi_h * 0.2))
                        roi_y = min(max(0, roi_y + shift), h2 - roi_h)

                    lower_roi = proc_frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                    lower_roi = _enhance_contrast_sharpen(lower_roi)
                    # ROI 자체도 업샘플 1.5배
                    scale = 1.6
                    roi_resized = cv2.resize(lower_roi, (int(roi_w * scale), int(roi_h * scale)), interpolation=cv2.INTER_CUBIC)
                    roi_img = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                    re2 = pose_static.process(roi_img)
                    if re2 and re2.pose_landmarks:
                        # ROI 정규좌표 → 전체 프레임 정규좌표로 매핑
                        re2_lm = re2.pose_landmarks.landmark
                        # ROI가 리사이즈 되었으므로, 정규좌표는 리사이즈된 ROI 기준. 먼저 리사이즈 기준을 원 ROI로 역스케일
                        # 정규좌표(0..1)는 크기 상관 없이 동일하게 해석되므로, 리사이즈의 영향은 없음. 정규좌표*원ROI폭/높이로 환산 후 전체로 투영.
                        remapped = _remap_roi_landmarks_to_full(
                            re2_lm,
                            (roi_x, roi_y, roi_w, roi_h),
                            w2, h2,
                        )
                        # 재검증
                        if _is_left_foot_confident(remapped):
                            current_landmarks = remapped
                            missing_streak = 0
                        else:
                            # 최종 실패 시 기존 결과 유지
                            pass
                    elif re_landmarks is not None and _is_left_foot_confident(re_landmarks):
                        current_landmarks = re_landmarks
                        missing_streak = 0
                else:
                    if _is_left_foot_confident(re_landmarks):
                        current_landmarks = re_landmarks
                        missing_streak = 0
            except Exception:
                # 재검출 실패는 무시하고 1차 결과 사용
                pass

        landmarks_list.append(current_landmarks)
    
    cap.release()
    try:
        pose.close()
    except Exception:
        pass
    if pose_static is not None:
        try:
            pose_static.close()
        except Exception:
            pass

    # 스무딩(기본 활성화; 윈도우 1이면 비활성)
    if SMOOTH_WINDOW and SMOOTH_WINDOW > 1:
        try:
            landmarks_list = smooth_landmarks_data(landmarks_list, window_size=SMOOTH_WINDOW)
        except Exception:
            # 스무딩 실패 시 원본 반환
            pass

    return landmarks_list, global_video_aspect_ratio  # aspect_ratio 반환 추가


def find_backswing_top_by_wrist_height(landmarks_data, address_frame, video_length_limit):
    """
    손목 높이 변화를 기반으로 백스윙 탑을 찾습니다.
    
    Args:
        landmarks_data: 랜드마크 데이터 리스트
        address_frame: 어드레스 프레임 인덱스
        video_length_limit: 비디오 길이 제한
        
    Returns:
        int: 백스윙 탑 프레임 인덱스
    """
    if not landmarks_data or len(landmarks_data) < 2:
        return address_frame + 1
    
    # 어드레스 이후 구간에서 손목 높이 추출
    start_idx = max(0, address_frame)
    end_idx = min(len(landmarks_data), video_length_limit)
    
    wrist_y_values = []
    valid_frames = []
    
    for i in range(start_idx, end_idx):
        landmarks = landmarks_data[i]
        if landmarks and len(landmarks) > 0:
            # 오른손목 랜드마크 (MediaPipe PoseLandmark.RIGHT_WRIST = 16)
            right_wrist = landmarks[16] if len(landmarks) > 16 else None
            if right_wrist and hasattr(right_wrist, 'y'):
                wrist_y_values.append(right_wrist.y)
                valid_frames.append(i)
            else:
                wrist_y_values.append(np.nan)
                valid_frames.append(i)
        else:
            wrist_y_values.append(np.nan)
            valid_frames.append(i)
    
    if not wrist_y_values:
        return address_frame + 1
    
    # NaN 값 제거
    valid_indices = [i for i, y in enumerate(wrist_y_values) if not np.isnan(y)]
    if not valid_indices:
        return address_frame + 1
    
    valid_wrist_y = [wrist_y_values[i] for i in valid_indices]
    valid_frame_indices = [valid_frames[i] for i in valid_indices]
    
    # 손목 높이가 최소인 지점 찾기 (Y 좌표: 위로 갈수록 작아짐)
    min_y_idx = robust_argmin(valid_wrist_y, "wrist_y", 0)
    if min_y_idx < len(valid_frame_indices):
        return valid_frame_indices[min_y_idx]
    
    return address_frame + 1


def preprocess_landmarks_data(landmarks_data):
    """
    랜드마크 데이터를 전처리합니다.
    
    Args:
        landmarks_data: 원본 랜드마크 데이터 리스트
        
    Returns:
        list: 전처리된 랜드마크 데이터 리스트
    """
    if not landmarks_data:
        return []
    
    processed_data = []
    
    for landmarks in landmarks_data:
        if landmarks is None:
            processed_data.append([None] * 33)
            continue
        
        # 각 랜드마크의 유효성 검사 및 정규화
        processed_landmarks = []
        for i, landmark in enumerate(landmarks):
            if landmark is None:
                processed_landmarks.append(None)
            else:
                # 좌표값 유효성 검사
                if hasattr(landmark, 'x') and hasattr(landmark, 'y') and hasattr(landmark, 'z'):
                    # 좌표값이 유효한 범위 내에 있는지 확인
                    if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                        processed_landmarks.append(landmark)
                    else:
                        processed_landmarks.append(None)
                else:
                    processed_landmarks.append(None)
        
        # 33개의 랜드마크가 있는지 확인
        while len(processed_landmarks) < 33:
            processed_landmarks.append(None)
        
        processed_data.append(processed_landmarks)
    
    return processed_data


def validate_video_file(video_path):
    """
    비디오 파일의 유효성을 검사합니다.
    
    Args:
        video_path (str): 비디오 파일 경로
        
    Returns:
        tuple: (is_valid, error_message, video_info)
    """
    if not video_path:
        return False, "비디오 경로가 제공되지 않았습니다.", None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, f"비디오 파일을 열 수 없습니다: {video_path}", None
    
    # 비디오 정보 추출
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    # 기본 검증
    if frame_count <= 0:
        return False, "유효한 프레임이 없습니다.", None
    
    if width <= 0 or height <= 0:
        return False, "유효하지 않은 비디오 해상도입니다.", None
    
    video_info = {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'aspect_ratio': width / height if height > 0 else 1.0,
        'duration': frame_count / fps if fps > 0 else 0.0
    }
    
    return True, "비디오 파일이 유효합니다.", video_info


def extract_key_frames_from_video(video_path, target_frames=None):
    """
    비디오에서 특정 프레임들을 추출합니다.
    
    Args:
        video_path (str): 비디오 파일 경로
        target_frames (list): 추출할 프레임 인덱스 리스트 (선택사항)
        
    Returns:
        dict: 프레임 인덱스를 키로 하는 프레임 이미지 딕셔너리
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    frames = {}
    
    if target_frames is None:
        # 모든 프레임 추출
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames[frame_idx] = frame
            frame_idx += 1
    else:
        # 지정된 프레임들만 추출
        for frame_idx in target_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames[frame_idx] = frame
    
    cap.release()
    return frames


def calculate_frame_statistics(landmarks_data):
    """
    랜드마크 데이터의 통계 정보를 계산합니다.
    
    Args:
        landmarks_data: 랜드마크 데이터 리스트
        
    Returns:
        dict: 통계 정보
    """
    if not landmarks_data:
        return {}
    
    total_frames = len(landmarks_data)
    valid_frames = 0
    invalid_frames = 0
    
    for landmarks in landmarks_data:
        if landmarks and len(landmarks) > 0:
            # 첫 번째 랜드마크가 유효한지 확인
            if landmarks[0] is not None and hasattr(landmarks[0], 'x'):
                valid_frames += 1
            else:
                invalid_frames += 1
        else:
            invalid_frames += 1
    
    statistics = {
        'total_frames': total_frames,
        'valid_frames': valid_frames,
        'invalid_frames': invalid_frames,
        'valid_ratio': valid_frames / total_frames if total_frames > 0 else 0.0
    }
    
    return statistics


def smooth_landmarks_data(landmarks_data, window_size=3):
    """
    랜드마크 데이터를 이동 평균으로 스무딩합니다.
    
    Args:
        landmarks_data: 원본 랜드마크 데이터 리스트
        window_size: 이동 평균 윈도우 크기
        
    Returns:
        list: 스무딩된 랜드마크 데이터 리스트
    """
    if not landmarks_data or len(landmarks_data) < window_size:
        return landmarks_data
    
    smoothed_data = []
    
    for i in range(len(landmarks_data)):
        # 윈도우 범위 계산
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(landmarks_data), i + window_size // 2 + 1)
        
        # 윈도우 내의 유효한 랜드마크들 수집
        window_landmarks = []
        for j in range(start_idx, end_idx):
            if landmarks_data[j] and len(landmarks_data[j]) > 0:
                window_landmarks.append(landmarks_data[j])
        
        if not window_landmarks:
            smoothed_data.append(landmarks_data[i])
            continue
        
        # 각 랜드마크 위치에 대해 평균 계산
        smoothed_frame = []
        for landmark_idx in range(33):  # MediaPipe Pose는 33개 랜드마크
            valid_coords = []
            
            for frame_landmarks in window_landmarks:
                if (len(frame_landmarks) > landmark_idx and 
                    frame_landmarks[landmark_idx] is not None and
                    hasattr(frame_landmarks[landmark_idx], 'x')):
                    
                    landmark = frame_landmarks[landmark_idx]
                    valid_coords.append([landmark.x, landmark.y, landmark.z])
            
            if valid_coords:
                # 평균 좌표 계산
                avg_coords = np.mean(valid_coords, axis=0)
                
                # 새로운 랜드마크 객체 생성 (간단한 구현)
                class SmoothedLandmark:
                    def __init__(self, x, y, z):
                        self.x = x
                        self.y = y
                        self.z = z
                        self.visibility = 1.0  # 기본값
                
                smoothed_landmark = SmoothedLandmark(avg_coords[0], avg_coords[1], avg_coords[2])
                smoothed_frame.append(smoothed_landmark)
            else:
                smoothed_frame.append(None)
        
        smoothed_data.append(smoothed_frame)
    
    return smoothed_data
