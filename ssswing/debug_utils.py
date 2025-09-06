"""
SSSwing3 디버깅 유틸리티 모듈

이 모듈은 골프 스윙 분석 과정에서 디버깅을 위한 유틸리티 함수를 제공합니다.
주요 기능:
1. 특정 프레임을 이미지 파일로 저장
2. 스윙 분석 과정의 중간 결과 확인
3. 문제가 발생한 프레임의 시각적 검증

사용 라이브러리:
- OpenCV (cv2): 영상 읽기 및 이미지 저장

사용 예시:
- 스윙 단계 검출 결과 검증
- 포즈 추정 품질 확인
- 분석 알고리즘 디버깅
"""

import cv2
import os

# 디버그 이미지 저장 제한 설정
DEBUG_IMAGE_LIMIT = 6  # 최대 6장까지만 저장 (프로/사용자 각각 3장씩: 스타트, 백스윙탑, 피니시)
_saved_image_count = 0

def save_debug_frame(video_path, frame_index, out_path):
    """
    디버그 프레임 저장 중단 (이미지 저장 제한).
    
    Args:
        video_path (str): 입력 영상 파일 경로
        frame_index (int): 저장할 프레임의 인덱스 (0부터 시작)
        out_path (str): 출력 이미지 파일 경로
    
    Returns:
        bool: 항상 False (이미지 저장 중단)
    """
    # 이미지 저장 완전 중단
    print(f"[INFO] 디버그 이미지 저장 중단: {out_path}")
    return False


def reset_debug_image_counter():
    """
    디버그 이미지 저장 카운터를 리셋합니다.
    새로운 분석 세션을 시작할 때 호출합니다.
    """
    global _saved_image_count
    _saved_image_count = 0
    print(f"[INFO] 디버그 이미지 카운터 리셋 완료")


def get_debug_image_count():
    """
    현재까지 저장된 디버그 이미지 수를 반환합니다.
    
    Returns:
        int: 저장된 디버그 이미지 수
    """
    global _saved_image_count
    return _saved_image_count
