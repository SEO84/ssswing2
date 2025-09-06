"""
SSSwing3 포즈 추출 및 관절 각도 계산 모듈

이 모듈은 MediaPipe를 사용하여 골프 스윙 영상에서 포즈 랜드마크를 추출하고
주요 관절 각도를 계산합니다.

주요 기능:
1. 비디오에서 포즈 랜드마크 추출
2. 관절 각도 계산 (3점 각도)
3. 좌우 대칭 관절 분석
4. JSON 형태로 결과 저장

계산되는 각도:
- hip_knee_ankle: 엉덩이-무릎-발목 각도
- shoulder_hip_knee: 어깨-엉덩이-무릎 각도
"""

import mediapipe as mp
import cv2
import json


def calculate_joint_angles(landmarks):
    """
    포즈 랜드마크에서 주요 관절 각도를 계산하는 함수
    
    Args:
        landmarks: 포즈 랜드마크 리스트 [(x, y, z), ...]
        
    Returns:
        dict: 관절별 각도 정보를 담은 딕셔너리
        
    특징:
        - Robust angle calculation: 유효하지 않은 포인트는 np.nan 반환
        - 좌우 대칭 관절 분석
        - 3점 각도 계산 (중간점을 중심으로)
    """
    # Robust angle calculation: returns np.nan if any point is None or invalid
    import numpy as np
    from .math_utils import calculate_angle as angle
    
    def safe_get(landmarks, idx):
        """
        안전한 랜드마크 추출 함수
        
        Args:
            landmarks: 랜드마크 리스트
            idx: 인덱스
            
        Returns:
            랜드마크 좌표 또는 None (유효하지 않은 경우)
        """
        if (
            isinstance(landmarks, list)
            and idx < len(landmarks)
            and landmarks[idx] is not None
        ):
            return landmarks[idx]
        return None

    # 관절 각도 계산 결과 저장
    angles = {}
    
    # 좌측 관절 각도
    angles["hip_knee_ankle_left"] = angle(
        safe_get(landmarks, 23),  # 좌측 엉덩이
        safe_get(landmarks, 25),  # 좌측 무릎
        safe_get(landmarks, 27)   # 좌측 발목
    )
    angles["shoulder_hip_knee_left"] = angle(
        safe_get(landmarks, 11),  # 좌측 어깨
        safe_get(landmarks, 23),  # 좌측 엉덩이
        safe_get(landmarks, 25)   # 좌측 무릎
    )
    
    # 우측 관절 각도
    angles["hip_knee_ankle_right"] = angle(
        safe_get(landmarks, 24),  # 우측 엉덩이
        safe_get(landmarks, 26),  # 우측 무릎
        safe_get(landmarks, 28)   # 우측 발목
    )
    angles["shoulder_hip_knee_right"] = angle(
        safe_get(landmarks, 12),  # 우측 어깨
        safe_get(landmarks, 24),  # 우측 엉덩이
        safe_get(landmarks, 26)   # 우측 무릎
    )
    
    return angles


def extract_angles(video_path, output_json):
    """
    비디오에서 포즈 랜드마크를 추출하고 관절 각도를 계산하여 JSON으로 저장하는 함수
    
    Args:
        video_path: 분석할 비디오 파일 경로
        output_json: 결과를 저장할 JSON 파일 경로
        
    특징:
        - 프레임별 포즈 랜드마크 추출
        - 각 프레임에서 관절 각도 계산
        - 결과를 JSON 형태로 저장
        - MediaPipe 포즈 추출 모듈 활용
    """
    from .video_utils import extract_pose_landmarks
    
    # 결과 데이터 저장 딕셔너리
    data = {}
    
    # 비디오에서 포즈 랜드마크 추출
    pose_landmarks_list = extract_pose_landmarks(video_path)
    
    # 각 프레임별로 관절 각도 계산
    for frame_idx, pose_landmarks in enumerate(pose_landmarks_list):
        if pose_landmarks is not None:
            # (x, y, z) 좌표로 변환
            landmarks = [(lm.x, lm.y, lm.z) for lm in pose_landmarks.landmark]
            # 해당 프레임의 관절 각도 계산
            data[frame_idx] = calculate_joint_angles(landmarks)
    
    # 결과를 JSON 파일로 저장
    import json
    with open(output_json, 'w') as f:
        json.dump(data, f)

