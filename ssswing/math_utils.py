"""
골프 스윙 분석을 위한 수학 계산 유틸리티 모듈

이 모듈은 골프 스윙 분석에 필요한 수학적 계산 함수들을 제공합니다.
주요 기능:
- 각도 계산
- 좌표 변환
- 통계 계산
- 벡터 연산

작성자: AI Assistant
개선일: 2024년
"""

import numpy as np
import math


def calculate_angle(a, b, c):
    """
    세 점 a, b, c 사이의 각도를 계산합니다 (b가 꼭지점).
    
    Args:
        a, b, c: 각도를 구성하는 세 점 (x, y 속성을 가진 객체)
        
    Returns:
        float: 각도 (도 단위), 계산 불가능한 경우 np.nan
    """
    # 각 점이 유효한지 확인 (None 이 아니고, x, y 속성 있는지)
    if not all(p is not None and hasattr(p, 'x') and hasattr(p, 'y') for p in [a, b, c]):
        return np.nan  # 유효하지 않으면 NaN 반환

    # 2D 벡터 계산 (필요시 visibility나 z 고려)
    vec_ba = np.array([a.x - b.x, a.y - b.y])
    vec_bc = np.array([c.x - b.x, c.y - b.y])

    # 내적을 이용한 각도 계산
    dot_product = np.dot(vec_ba, vec_bc)
    norm_ba = np.linalg.norm(vec_ba)
    norm_bc = np.linalg.norm(vec_bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan  # 길이가 0인 벡터가 있으면 각도 계산 불가

    cos_theta = dot_product / (norm_ba * norm_bc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 수치 오류 방지

    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_shoulder_rotation(landmarks, pose_landmark_enum):
    """
    어깨 라인의 수평 대비 각도 계산 (X, Y 평면 기준)
    
    Args:
        landmarks: MediaPipe 랜드마크 리스트
        pose_landmark_enum: MediaPipe 포즈 랜드마크 enum
        
    Returns:
        float: 어깨 회전 각도 (도 단위), 계산 불가능한 경우 np.nan
    """
    from .landmark_utils import get_landmark, get_valid_coords
    
    left_shoulder = get_landmark(landmarks, pose_landmark_enum.LEFT_SHOULDER)
    right_shoulder = get_landmark(landmarks, pose_landmark_enum.RIGHT_SHOULDER)

    if left_shoulder is None or right_shoulder is None:
        return np.nan

    ls_coords = get_valid_coords(left_shoulder)[:2]  # X, Y만 사용
    rs_coords = get_valid_coords(right_shoulder)[:2]

    if np.isnan(ls_coords).any() or np.isnan(rs_coords).any():
        return np.nan

    delta_y = rs_coords[1] - ls_coords[1]
    delta_x = rs_coords[0] - ls_coords[0]

    if delta_x == 0:  # 수직선 방지
        return 90.0 if delta_y > 0 else -90.0 if delta_y < 0 else 0.0

    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def calculate_hip_rotation(landmarks, pose_landmark_enum):
    """
    힙 라인의 수평 대비 각도 계산 (X, Y 평면 기준)
    
    Args:
        landmarks: MediaPipe 랜드마크 리스트
        pose_landmark_enum: MediaPipe 포즈 랜드마크 enum
        
    Returns:
        float: 힙 회전 각도 (도 단위), 계산 불가능한 경우 np.nan
    """
    from .landmark_utils import get_landmark, get_valid_coords
    
    left_hip = get_landmark(landmarks, pose_landmark_enum.LEFT_HIP)
    right_hip = get_landmark(landmarks, pose_landmark_enum.RIGHT_HIP)

    if left_hip is None or right_hip is None:
        return np.nan

    lh_coords = get_valid_coords(left_hip)[:2]  # X, Y만 사용
    rh_coords = get_valid_coords(right_hip)[:2]

    if np.isnan(lh_coords).any() or np.isnan(rh_coords).any():
        return np.nan

    delta_y = rh_coords[1] - lh_coords[1]
    delta_x = rh_coords[0] - lh_coords[0]

    if delta_x == 0:  # 수직선 방지
        return 90.0 if delta_y > 0 else -90.0 if delta_y < 0 else 0.0

    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def robust_argmin(arr, arr_name, fallback_value):
    """
    배열의 최소값 인덱스를 안전하게 찾습니다.
    
    Args:
        arr: numpy 배열
        arr_name: 배열 이름 (디버깅용)
        fallback_value: 최소값을 찾을 수 없을 때 반환할 값
        
    Returns:
        int: 최소값의 인덱스
    """
    if arr is None or len(arr) == 0:
        print(f"[WARNING] {arr_name} 배열이 비어있습니다. fallback_value 사용: {fallback_value}")
        return fallback_value
    
    try:
        return int(np.argmin(arr))
    except Exception as e:
        print(f"[ERROR] {arr_name} 배열에서 최소값 인덱스 찾기 실패: {e}")
        return fallback_value


def robust_argmax(arr, arr_name, fallback_value):
    """
    배열의 최대값 인덱스를 안전하게 찾습니다.
    
    Args:
        arr: numpy 배열
        arr_name: 배열 이름 (디버깅용)
        fallback_value: 최대값을 찾을 수 없을 때 반환할 값
        
    Returns:
        int: 최대값의 인덱스
    """
    if arr is None or len(arr) == 0:
        print(f"[WARNING] {arr_name} 배열이 비어있습니다. fallback_value 사용: {fallback_value}")
        return fallback_value
    
    try:
        return int(np.argmax(arr))
    except Exception as e:
        print(f"[ERROR] {arr_name} 배열에서 최대값 인덱스 찾기 실패: {e}")
        return fallback_value


def calculate_velocity_profile(landmarks_data, address_frame, video_length_limit):
    """
    랜드마크 데이터로부터 속도 프로파일을 계산합니다.
    
    Args:
        landmarks_data: 랜드마크 데이터 리스트
        address_frame: 어드레스 프레임 인덱스
        video_length_limit: 비디오 길이 제한
        
    Returns:
        dict: 속도 프로파일 정보
    """
    if not landmarks_data or len(landmarks_data) < 2:
        return {}
    
    # 어드레스 이후 구간에서 속도 계산
    start_idx = max(0, address_frame)
    end_idx = min(len(landmarks_data), video_length_limit)
    
    velocities = []
    frame_indices = []
    
    for i in range(start_idx, end_idx - 1):
        current_landmarks = landmarks_data[i]
        next_landmarks = landmarks_data[i + 1]
        
        if (current_landmarks and next_landmarks and 
            len(current_landmarks) > 16 and len(next_landmarks) > 16):
            
            # 오른손목 랜드마크 사용
            current_wrist = current_landmarks[16]
            next_wrist = next_landmarks[16]
            
            if (current_wrist and next_wrist and 
                hasattr(current_wrist, 'x') and hasattr(next_wrist, 'x')):
                
                # 속도 계산 (간단한 유클리드 거리)
                dx = next_wrist.x - current_wrist.x
                dy = next_wrist.y - current_wrist.y
                dz = next_wrist.z - current_wrist.z
                
                velocity = np.sqrt(dx**2 + dy**2 + dz**2)
                velocities.append(velocity)
                frame_indices.append(i)
            else:
                velocities.append(0.0)
                frame_indices.append(i)
        else:
            velocities.append(0.0)
            frame_indices.append(i)
    
    if not velocities:
        return {}
    
    # 속도 프로파일 분석
    max_velocity_idx = robust_argmax(velocities, "velocities", 0)
    max_velocity_frame = frame_indices[max_velocity_idx] if max_velocity_idx < len(frame_indices) else address_frame
    
    # 가속/감속 지점 찾기
    acceleration_points = []
    deceleration_points = []
    
    for i in range(1, len(velocities)):
        if velocities[i] > velocities[i-1] * 1.1:  # 10% 이상 증가
            acceleration_points.append(frame_indices[i])
        elif velocities[i] < velocities[i-1] * 0.9:  # 10% 이상 감소
            deceleration_points.append(frame_indices[i])
    
    return {
        'velocities': velocities,
        'frame_indices': frame_indices,
        'max_velocity_frame': max_velocity_frame,
        'max_velocity': velocities[max_velocity_idx] if max_velocity_idx < len(velocities) else 0.0,
        'acceleration_points': acceleration_points,
        'deceleration_points': deceleration_points
    }


def find_swing_phases_by_velocity(velocity_profile, address_frame):
    """
    속도 프로파일을 기반으로 스윙 단계를 찾습니다.
    
    Args:
        velocity_profile: 속도 프로파일 딕셔너리
        address_frame: 어드레스 프레임 인덱스
        
    Returns:
        dict: 속도 기반 스윙 단계 정보
    """
    if not velocity_profile:
        return {}
    
    velocities = velocity_profile.get('velocities', [])
    frame_indices = velocity_profile.get('frame_indices', [])
    max_velocity_frame = velocity_profile.get('max_velocity_frame', address_frame)
    acceleration_points = velocity_profile.get('acceleration_points', [])
    deceleration_points = velocity_profile.get('deceleration_points', [])
    
    # 속도 기반 스윙 단계 추정
    phases = {}
    
    # 최대 속도 지점을 임팩트로 추정
    if max_velocity_frame > address_frame:
        phases['velocity_impact'] = max_velocity_frame
    
    # 가속 시작 지점을 백스윙 탑으로 추정
    if acceleration_points:
        # 어드레스 이후 첫 번째 가속 지점
        for point in acceleration_points:
            if point > address_frame:
                phases['velocity_top'] = point
                break
    
    # 감속 시작 지점을 피니시로 추정
    if deceleration_points:
        # 임팩트 이후 첫 번째 감속 지점
        for point in deceleration_points:
            if 'velocity_impact' in phases and point > phases['velocity_impact']:
                phases['velocity_finish'] = point
                break
    
    return phases
