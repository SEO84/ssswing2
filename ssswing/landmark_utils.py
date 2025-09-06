"""
골프 스윙 분석을 위한 랜드마크 유틸리티 모듈

이 모듈은 MediaPipe 랜드마크 데이터를 처리하는 유틸리티 함수들을 제공합니다.
주요 기능:
- 랜드마크 안전 추출
- 좌표 변환
- 랜드마크 유효성 검사
- 좌표 계산

작성자: AI Assistant
개선일: 2024년
"""

import numpy as np


def get_landmark(landmarks, landmark_enum):
    """
    랜드마크 리스트에서 특정 랜드마크 객체를 안전하게 가져옵니다.
    
    Args:
        landmarks: MediaPipe 랜드마크 리스트
        landmark_enum: MediaPipe 포즈 랜드마크 enum
        
    Returns:
        MediaPipe 랜드마크 객체 또는 None
    """
    idx = landmark_enum.value
    if landmarks is None or len(landmarks) <= idx or landmarks[idx] is None:
        return None
    return landmarks[idx]


def get_valid_coords(landmark):
    """
    랜드마크 객체에서 유효한 (x, y, z) 좌표를 반환합니다.
    
    Args:
        landmark: MediaPipe 랜드마크 객체
        
    Returns:
        numpy.array: [x, y, z] 좌표 배열, 유효하지 않으면 [nan, nan, nan]
    """
    if landmark is not None and hasattr(landmark, 'x') and hasattr(landmark, 'y') and hasattr(landmark, 'z'):
        return np.array([landmark.x, landmark.y, landmark.z])
    return np.array([np.nan, np.nan, np.nan])


def get_landmark_coords(landmarks, landmark_enum):
    """
    랜드마크 리스트에서 특정 랜드마크의 좌표를 안전하게 가져옵니다.
    
    Args:
        landmarks: MediaPipe 랜드마크 리스트
        landmark_enum: MediaPipe 포즈 랜드마크 enum
        
    Returns:
        numpy.array: [x, y, z] 좌표 배열, 유효하지 않으면 [nan, nan, nan]
    """
    landmark = get_landmark(landmarks, landmark_enum)
    return get_valid_coords(landmark)


def is_landmark_visible(landmark, threshold=0.5):
    """
    랜드마크가 충분히 보이는지 확인합니다.
    
    Args:
        landmark: MediaPipe 랜드마크 객체
        threshold: visibility 임계값 (기본값: 0.5)
        
    Returns:
        bool: 랜드마크가 보이는지 여부
    """
    if landmark is None:
        return False
    
    if hasattr(landmark, 'visibility'):
        return landmark.visibility >= threshold
    
    return True  # visibility 속성이 없으면 보인다고 가정


def get_landmark_distance(landmark1, landmark2):
    """
    두 랜드마크 사이의 3D 거리를 계산합니다.
    
    Args:
        landmark1, landmark2: MediaPipe 랜드마크 객체
        
    Returns:
        float: 두 랜드마크 사이의 거리, 계산 불가능한 경우 np.nan
    """
    coords1 = get_valid_coords(landmark1)
    coords2 = get_valid_coords(landmark2)
    
    if np.isnan(coords1).any() or np.isnan(coords2).any():
        return np.nan
    
    return np.linalg.norm(coords1 - coords2)


def get_landmark_distance_2d(landmark1, landmark2):
    """
    두 랜드마크 사이의 2D 거리를 계산합니다 (X, Y 평면).
    
    Args:
        landmark1, landmark2: MediaPipe 랜드마크 객체
        
    Returns:
        float: 두 랜드마크 사이의 2D 거리, 계산 불가능한 경우 np.nan
    """
    coords1 = get_valid_coords(landmark1)[:2]  # X, Y만 사용
    coords2 = get_valid_coords(landmark2)[:2]
    
    if np.isnan(coords1).any() or np.isnan(coords2).any():
        return np.nan
    
    return np.linalg.norm(coords1 - coords2)


def get_landmark_height_diff(landmark1, landmark2):
    """
    두 랜드마크의 높이 차이를 계산합니다 (Y 좌표 차이).
    
    Args:
        landmark1, landmark2: MediaPipe 랜드마크 객체
        
    Returns:
        float: 높이 차이 (landmark2.y - landmark1.y), 계산 불가능한 경우 np.nan
    """
    if landmark1 is None or landmark2 is None:
        return np.nan
    
    if not hasattr(landmark1, 'y') or not hasattr(landmark2, 'y'):
        return np.nan
    
    return landmark2.y - landmark1.y


def get_landmark_width_diff(landmark1, landmark2):
    """
    두 랜드마크의 너비 차이를 계산합니다 (X 좌표 차이).
    
    Args:
        landmark1, landmark2: MediaPipe 랜드마크 객체
        
    Returns:
        float: 너비 차이 (landmark2.x - landmark1.x), 계산 불가능한 경우 np.nan
    """
    if landmark1 is None or landmark2 is None:
        return np.nan
    
    if not hasattr(landmark1, 'x') or not hasattr(landmark2, 'x'):
        return np.nan
    
    return landmark2.x - landmark1.x


def get_landmark_depth_diff(landmark1, landmark2):
    """
    두 랜드마크의 깊이 차이를 계산합니다 (Z 좌표 차이).
    
    Args:
        landmark1, landmark2: MediaPipe 랜드마크 객체
        
    Returns:
        float: 깊이 차이 (landmark2.z - landmark1.z), 계산 불가능한 경우 np.nan
    """
    if landmark1 is None or landmark2 is None:
        return np.nan
    
    if not hasattr(landmark1, 'z') or not hasattr(landmark2, 'z'):
        return np.nan
    
    return landmark2.z - landmark1.z


def get_landmark_center(landmarks, landmark_enums):
    """
    여러 랜드마크의 중심점을 계산합니다.
    
    Args:
        landmarks: MediaPipe 랜드마크 리스트
        landmark_enums: 중심점을 계산할 랜드마크 enum 리스트
        
    Returns:
        numpy.array: 중심점 [x, y, z] 좌표, 계산 불가능한 경우 [nan, nan, nan]
    """
    valid_coords = []
    
    for enum in landmark_enums:
        coords = get_landmark_coords(landmarks, enum)
        if not np.isnan(coords).any():
            valid_coords.append(coords)
    
    if not valid_coords:
        return np.array([np.nan, np.nan, np.nan])
    
    return np.mean(valid_coords, axis=0)


def get_landmark_center_2d(landmarks, landmark_enums):
    """
    여러 랜드마크의 2D 중심점을 계산합니다 (X, Y 평면).
    
    Args:
        landmarks: MediaPipe 랜드마크 리스트
        landmark_enums: 중심점을 계산할 랜드마크 enum 리스트
        
    Returns:
        numpy.array: 중심점 [x, y] 좌표, 계산 불가능한 경우 [nan, nan]
    """
    valid_coords = []
    
    for enum in landmark_enums:
        coords = get_landmark_coords(landmarks, enum)[:2]  # X, Y만 사용
        if not np.isnan(coords).any():
            valid_coords.append(coords)
    
    if not valid_coords:
        return np.array([np.nan, np.nan])
    
    return np.mean(valid_coords, axis=0)


def validate_landmarks(landmarks, required_landmarks=None):
    """
    랜드마크 데이터의 유효성을 검사합니다.
    
    Args:
        landmarks: MediaPipe 랜드마크 리스트
        required_landmarks: 필수 랜드마크 enum 리스트 (선택사항)
        
    Returns:
        bool: 랜드마크 데이터가 유효한지 여부
    """
    if landmarks is None or len(landmarks) == 0:
        return False
    
    # 기본 검사: 최소 33개의 랜드마크가 있어야 함 (MediaPipe Pose)
    if len(landmarks) < 33:
        return False
    
    # 필수 랜드마크 검사
    if required_landmarks:
        for enum in required_landmarks:
            landmark = get_landmark(landmarks, enum)
            if landmark is None:
                return False
    
    return True


def count_visible_landmarks(landmarks, landmark_enums, threshold=0.5):
    """
    보이는 랜드마크의 개수를 계산합니다.
    
    Args:
        landmarks: MediaPipe 랜드마크 리스트
        landmark_enums: 검사할 랜드마크 enum 리스트
        threshold: visibility 임계값 (기본값: 0.5)
        
    Returns:
        int: 보이는 랜드마크의 개수
    """
    visible_count = 0
    
    for enum in landmark_enums:
        landmark = get_landmark(landmarks, enum)
        if is_landmark_visible(landmark, threshold):
            visible_count += 1
    
    return visible_count
