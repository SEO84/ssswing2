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
from .math_utils import robust_argmin, robust_argmax, calculate_velocity_profile, find_swing_phases_by_velocity

mp_pose = mp.solutions.pose
pose_landmark_enum = mp.solutions.pose.PoseLandmark


def extract_landmarks_from_video(video_path):
    """
    주어진 비디오에서 mediapipe를 이용해 각 프레임의 랜드마크 리스트를 추출합니다.
    세로 영상은 자동으로 회전하여 처리합니다.
    
    Args:
        video_path (str): 입력 영상 경로
        
    Returns:
        tuple: (landmarks_list, global_video_aspect_ratio)
    """
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)  # confidence 높임 (세로 영상 안정화)
    cap = cv2.VideoCapture(video_path)
    
    # 실제 영상 비율 계산
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    global_video_aspect_ratio = width / height if height != 0 else 1.0
    print(f"[INFO] Detected video aspect ratio: {global_video_aspect_ratio:.2f}")
    
    landmarks_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 세로 영상(포트레이트) 처리: 회전
        if global_video_aspect_ratio < 1.0:  # height > width
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 시계방향 90도 회전 (세로 -> 가로)
            print(f"[DEBUG] Rotated frame for portrait video")
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            landmarks_list.append(results.pose_landmarks.landmark)
        else:
            landmarks_list.append([None]*33)
    
    cap.release()
    pose.close()
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
