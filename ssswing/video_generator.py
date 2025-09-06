"""
골프 스윙 비교 분석을 위한 합성 영상 생성 모듈

이 모듈은 프로 스윙과 사용자 스윙을 나란히 배치한 합성 영상을 생성합니다.
주요 기능:
1. 두 영상을 나란히 배치한 합성 영상 생성
2. 프레임 동기화 및 크기 조정
3. MP4 및 WebM 형식 지원
4. 고품질 인코딩을 위한 FFmpeg 우선 사용
"""

import os
import cv2
import numpy as np
import subprocess
import shlex
from typing import Tuple, Optional


def get_video_info(video_path: str) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    """
    영상 파일의 기본 정보를 추출합니다.
    
    Args:
        video_path: 영상 파일 경로
        
    Returns:
        Tuple: (fps, width, height, frame_count) 또는 (None, None, None, None) if 실패
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()
        return fps, width, height, frame_count
    except Exception:
        return None, None, None, None


def generate_comparison_video(
    video1_path: str, 
    video2_path: str, 
    output_path: str,
    target_fps: float = 30.0,
    max_duration: float = 15.0
) -> bool:
    """
    두 영상을 나란히 배치한 합성 영상을 생성합니다.
    
    Args:
        video1_path: 첫 번째 영상 경로 (왼쪽)
        video2_path: 두 번째 영상 경로 (오른쪽)
        output_path: 출력 영상 경로
        target_fps: 목표 프레임레이트
        max_duration: 최대 영상 길이 (초)
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 1) MediaPipe 스켈레톤이 포함된 OpenCV 합성 시도 (우선)
        if _generate_with_opencv_skeleton(video1_path, video2_path, output_path, target_fps, max_duration):
            print(f"[INFO] OpenCV 스켈레톤 포함 합성 영상 생성 성공: {output_path}")
            return True
        
        # 2) FFmpeg를 사용한 고품질 합성 시도
        if _generate_with_ffmpeg(video1_path, video2_path, output_path, target_fps, max_duration):
            print(f"[INFO] FFmpeg로 합성 영상 생성 성공: {output_path}")
            return True
        
        # 3) FFmpeg 실패 시 OpenCV 폴백 사용
        if _generate_with_opencv(video1_path, video2_path, output_path, target_fps, max_duration):
            print(f"[INFO] OpenCV로 합성 영상 생성 성공: {output_path}")
            return True
        
        print(f"[ERROR] 모든 합성 방법 실패")
        return False
        
    except Exception as e:
        print(f"[ERROR] 합성 영상 생성 중 예외 발생: {e}")
        return False


def _calculate_target_resolution(w1: int, h1: int, w2: int, h2: int) -> Tuple[int, int]:
    """
    두 영상을 원본 비율을 유지하면서 50:50으로 나누어 배치했을 때 최적의 출력 해상도를 계산합니다.
    """
    # 목표 높이 설정 (최소 720p 보장)
    target_height = max(h1, h2, 720)
    
    # 전체 화면을 50:50으로 나누기
    total_width = 1280  # 고정 전체 너비 (640 + 640)
    target_width_per_video = 640  # 각 영상당 640픽셀
    
    return total_width, target_height


def _resize_with_aspect_ratio_and_padding(frame, target_width: int, target_height: int) -> np.ndarray:
    """
    원본 비율을 유지하면서 검정색 패딩을 추가하여 프레임을 리사이징합니다.
    """
    h, w = frame.shape[:2]
    
    # 원본 비율 계산
    aspect_ratio = w / h
    target_aspect_ratio = target_width / target_height
    
    if aspect_ratio > target_aspect_ratio:
        # 가로가 더 긴 경우: 높이를 맞추고 가로에 패딩
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
        if new_width > target_width:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
    else:
        # 세로가 더 긴 경우: 가로를 맞추고 세로에 패딩
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
        if new_height > target_height:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
    
    # 프레임 리사이징
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # 검정색 배경 생성
    padded_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # 중앙에 배치
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    padded_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
    
    return padded_frame


def _transform_landmarks_to_resized_frame(landmarks, original_width: int, original_height: int, 
                                        target_width: int, target_height: int):
    """
    원본 프레임 크기에서 추출된 랜드마크를 리사이징된 프레임 크기에 맞게 좌표를 변환합니다.
    
    Args:
        landmarks: MediaPipe pose landmarks
        original_width, original_height: 원본 프레임 크기
        target_width, target_height: 리사이징된 프레임 크기
        
    Returns:
        변환된 랜드마크 (새로운 객체 생성)
    """
    if not landmarks:
        return None
    
    try:
        # 새로운 랜드마크 객체 생성 (원본 보존)
        import mediapipe as mp
        from mediapipe.framework.formats import landmark_pb2
        transformed_landmarks = landmark_pb2.NormalizedLandmarkList()
        
        # 스케일 팩터 계산
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        for landmark in landmarks.landmark:
            new_landmark = transformed_landmarks.landmark.add()
            
            # 정규화된 좌표를 픽셀 좌표로 변환 후 다시 정규화
            pixel_x = landmark.x * original_width
            pixel_y = landmark.y * original_height
            
            # 리사이징된 프레임 크기에 맞게 변환
            new_pixel_x = pixel_x * scale_x
            new_pixel_y = pixel_y * scale_y
            
            # 다시 정규화된 좌표로 변환
            new_landmark.x = new_pixel_x / target_width
            new_landmark.y = new_pixel_y / target_height
            new_landmark.z = landmark.z  # Z 좌표는 그대로 유지
            new_landmark.visibility = landmark.visibility
        
        return transformed_landmarks
    except Exception as e:
        print(f"[WARN] 랜드마크 변환 실패: {e}")
        return landmarks  # 실패 시 원본 반환


def _smooth_landmarks(current_landmarks, previous_landmarks, smoothing_factor: float = 0.7):
    """
    이전 프레임과 현재 프레임의 랜드마크를 평활화하여 지터를 줄입니다.
    
    Args:
        current_landmarks: 현재 프레임의 랜드마크
        previous_landmarks: 이전 프레임의 랜드마크
        smoothing_factor: 스무딩 강도 (0.0~1.0, 높을수록 더 부드러움)
        
    Returns:
        평활화된 랜드마크
    """
    if not current_landmarks or not previous_landmarks:
        return current_landmarks
    
    try:
        # 새로운 랜드마크 객체 생성
        import mediapipe as mp
        from mediapipe.framework.formats import landmark_pb2
        smoothed_landmarks = landmark_pb2.NormalizedLandmarkList()
        
        for i, (curr_lm, prev_lm) in enumerate(zip(current_landmarks.landmark, previous_landmarks.landmark)):
            new_landmark = smoothed_landmarks.landmark.add()
            
            # 가중 평균으로 스무딩
            new_landmark.x = curr_lm.x * (1 - smoothing_factor) + prev_lm.x * smoothing_factor
            new_landmark.y = curr_lm.y * (1 - smoothing_factor) + prev_lm.y * smoothing_factor
            new_landmark.z = curr_lm.z * (1 - smoothing_factor) + prev_lm.z * smoothing_factor
            new_landmark.visibility = max(curr_lm.visibility, prev_lm.visibility)
        
        return smoothed_landmarks
    except Exception as e:
        print(f"[WARN] 랜드마크 스무딩 실패: {e}")
        return current_landmarks  # 실패 시 현재 랜드마크 반환


def _smooth_landmarks_with_history(landmark_history):
    """
    히스토리 기반 랜드마크 스무딩 (더 안정적)
    """
    if not landmark_history:
        return None
    
    try:
        import mediapipe as mp
        from mediapipe.framework.formats import landmark_pb2
        
        # 새로운 랜드마크 객체 생성
        smoothed_landmarks = landmark_pb2.NormalizedLandmarkList()
        
        # 각 관절별로 히스토리 평균 계산
        for joint_idx in range(len(landmark_history[0].landmark)):
            new_landmark = smoothed_landmarks.landmark.add()
            
            x_sum = y_sum = z_sum = vis_sum = 0
            valid_count = 0
            
            for landmarks in landmark_history:
                if joint_idx < len(landmarks.landmark):
                    lm = landmarks.landmark[joint_idx]
                    if lm.visibility > 0.5:  # 유효한 랜드마크만 사용
                        x_sum += lm.x
                        y_sum += lm.y
                        z_sum += lm.z
                        vis_sum += lm.visibility
                        valid_count += 1
            
            if valid_count > 0:
                new_landmark.x = x_sum / valid_count
                new_landmark.y = y_sum / valid_count
                new_landmark.z = z_sum / valid_count
                new_landmark.visibility = vis_sum / valid_count
            else:
                # 기본값 설정
                new_landmark.x = 0.5
                new_landmark.y = 0.5
                new_landmark.z = 0.0
                new_landmark.visibility = 0.0
        
        return smoothed_landmarks
        
    except Exception as e:
        print(f"[WARN] 히스토리 기반 스무딩 실패: {e}")
        return landmark_history[-1] if landmark_history else None


def _validate_landmarks(landmarks, threshold: float = 0.6):  # 임계값 상향
    """
    랜드마크의 유효성을 검사합니다 (더 엄격한 기준).
    """
    if not landmarks:
        return False
    
    # 핵심 관절들만 체크 (골프 스윙에 중요한 부위)
    key_joints = [11, 12, 13, 14, 15, 16]  # 어깨, 팔꿈치, 손목
    valid_count = 0
    
    for joint_idx in key_joints:
        if joint_idx < len(landmarks.landmark):
            if landmarks.landmark[joint_idx].visibility > threshold:
                valid_count += 1
    
    # 핵심 관절의 70% 이상이 유효해야 함
    return valid_count >= len(key_joints) * 0.7


def _validate_landmarks_relaxed(landmarks, threshold: float = 0.4):  # 더 관대한 기준
    """
    랜드마크의 유효성을 검사합니다 (깜빡거림 방지를 위한 관대한 기준).
    """
    if not landmarks:
        return False
    
    # 핵심 관절들만 체크 (골프 스윙에 중요한 부위)
    key_joints = [11, 12, 13, 14, 15, 16]  # 어깨, 팔꿈치, 손목
    valid_count = 0
    
    for joint_idx in key_joints:
        if joint_idx < len(landmarks.landmark):
            if landmarks.landmark[joint_idx].visibility > threshold:
                valid_count += 1
    
    # 핵심 관절의 50% 이상이 유효하면 OK (깜빡거림 방지)
    return valid_count >= len(key_joints) * 0.5


def _generate_with_ffmpeg(
    video1_path: str, 
    video2_path: str, 
    output_path: str,
    target_fps: float,
    max_duration: float
) -> bool:
    """
    FFmpeg를 사용하여 고품질 합성 영상을 생성합니다.
    """
    try:
        # 영상 정보 가져오기
        fps1, w1, h1, _ = get_video_info(video1_path)
        fps2, w2, h2, _ = get_video_info(video2_path)
        
        if not all([fps1, w1, h1, fps2, w2, h2]):
            return False
        
        # 50:50 비율로 정확히 나누기
        target_height = max(h1, h2, 720)
        total_width = 1280  # 고정 전체 너비
        target_width_per_video = 640  # 각 영상당 640픽셀
        
        # FFmpeg 필터 구성: 원본 비율 유지하면서 검정색 패딩 추가
        filter_complex = (
            f"[0:v]scale={target_width_per_video}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width_per_video}:{target_height}:(ow-iw)/2:(oh-ih)/2:black[v1];"
            f"[1:v]scale={target_width_per_video}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width_per_video}:{target_height}:(ow-iw)/2:(oh-ih)/2:black[v2];"
            f"[v1][v2]hstack=inputs=2"
        )
        
        # FFmpeg 명령어 실행
        cmd = [
            "ffmpeg", "-y",
            "-i", video1_path,
            "-i", video2_path,
            "-filter_complex", filter_complex,
            "-r", str(target_fps),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            # 파일 크기 확인
            if os.path.getsize(output_path) > 1024:  # 1KB 이상
                return True
        
        return False
        
    except Exception as e:
        print(f"[WARN] FFmpeg 합성 실패: {e}")
        return False


def _generate_with_opencv(
    video1_path: str, 
    video2_path: str, 
    output_path: str,
    target_fps: float,
    max_duration: float
) -> bool:
    """
    OpenCV를 사용하여 합성 영상을 생성합니다 (FFmpeg 실패 시 폴백).
    """
    try:
        # 영상 열기
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        if not cap1.isOpened() or not cap2.isOpened():
            return False
        
        # 영상 정보 가져오기
        fps1, w1, h1, _ = get_video_info(video1_path)
        fps2, w2, h2, _ = get_video_info(video2_path)
        
        # 50:50 비율로 정확히 나누기
        target_height = max(h1, h2, 720)
        total_width = 1280  # 고정 전체 너비
        target_width_per_video = 640  # 각 영상당 640픽셀
        
        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (total_width, total_height))
        
        if not out.isOpened():
            return False
        
        # 프레임별 합성
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            # 두 영상 중 하나라도 끝나면 종료
            if not ret1 or not ret2:
                break
            
            # 프레임 크기 조정 (원본 비율 유지 + 검정색 패딩)
            frame1_resized = _resize_with_aspect_ratio_and_padding(frame1, target_width_per_video, target_height)
            frame2_resized = _resize_with_aspect_ratio_and_padding(frame2, target_width_per_video, target_height)
            
            # 두 프레임을 나란히 배치
            combined_frame = np.hstack((frame1_resized, frame2_resized))
            
            out.write(combined_frame)
        
        # 리소스 해제
        cap1.release()
        cap2.release()
        out.release()
        
        # 파일 생성 확인
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            return True
        
        return False
        
    except Exception as e:
        print(f"[ERROR] OpenCV 합성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def _generate_with_opencv_skeleton(
    video1_path: str, 
    video2_path: str, 
    output_path: str,
    target_fps: float,
    max_duration: float
) -> bool:
    """
    OpenCV를 사용하여 MediaPipe 스켈레톤이 포함된 합성 영상을 생성합니다.
    """
    try:
        import mediapipe as mp
        
        # MediaPipe Pose 초기화 (깜빡거림 방지를 위한 안정적인 설정)
        mp_pose = mp.solutions.pose
        pose1 = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 안정성 우선
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,  # 더 관대하게 (깜빡거림 방지)
            min_tracking_confidence=0.5
        )
        
        pose2 = mp_pose.Pose(  # 각 영상마다 별도의 인스턴스 사용
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,  # 더 관대하게
            min_tracking_confidence=0.5
        )
        
        # 영상 열기
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        if not cap1.isOpened() or not cap2.isOpened():
            return False
        
        # 영상 정보 가져오기
        fps1, w1, h1, frame_count1 = get_video_info(video1_path)
        fps2, w2, h2, frame_count2 = get_video_info(video2_path)

        # 프레임 동기화를 위한 설정
        duration1 = frame_count1 / fps1 if fps1 > 0 else 0
        duration2 = frame_count2 / fps2 if fps2 > 0 else 0
        min_duration = min(duration1, duration2, max_duration)
        
        # 각 영상의 프레임 간격 계산 (동기화)
        frame_interval1 = fps1 / target_fps if fps1 > 0 else 1
        frame_interval2 = fps2 / target_fps if fps2 > 0 else 1
        
        # 50:50 비율로 정확히 나누기
        target_height = max(h1, h2, 720)
        total_width = 1280  # 고정 전체 너비
        target_width_per_video = 640  # 각 영상당 640픽셀
        
        print(f"[DEBUG] 동기화 설정 - FPS1: {fps1}, FPS2: {fps2}, 목표: {target_fps}")
        print(f"[DEBUG] 프레임 간격 - 영상1: {frame_interval1:.2f}, 영상2: {frame_interval2:.2f}")
        print(f"[DEBUG] 출력 해상도: {total_width}x{target_height} (50:50 비율, 원본 비율 유지 + 검정색 패딩)")
        
        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (total_width, target_height))
        
        if not out.isOpened():
            return False
        
        # 프레임별 합성
        output_frame_count = 0
        max_frames = int(min_duration * target_fps)
        current_frame1 = 0.0
        current_frame2 = 0.0
        
        # 랜드마크 히스토리 (더 안정적인 스무딩을 위해)
        landmark_history1 = []
        landmark_history2 = []
        history_size = 5  # 최근 N개 프레임 평균
        
        print(f"[INFO] 스켈레톤 합성 시작 - 최대 {max_frames} 프레임")
        print(f"[DEBUG] 깜빡거림 방지 설정 - 신뢰도: 0.5, 관대한 유효성 검사")
        
        while output_frame_count < max_frames:
            # 동기화된 프레임 읽기
            cap1.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame1))
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame2))
            
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            # 프레임 리사이징 (원본 비율 유지 + 검정색 패딩)
            frame1_resized = _resize_with_aspect_ratio_and_padding(frame1, target_width_per_video, target_height)
            frame2_resized = _resize_with_aspect_ratio_and_padding(frame2, target_width_per_video, target_height)
            
            # 리사이징된 프레임에서 포즈 추정 (좌표 정확도 ↑)
            frame1_rgb = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)
            results1 = pose1.process(frame1_rgb)
            
            frame2_rgb = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB)
            results2 = pose2.process(frame2_rgb)
            
            # 랜드마크 처리 (더 안정적인 방식)
            current_landmarks1 = results1.pose_landmarks
            current_landmarks2 = results2.pose_landmarks
            
            # 첫 번째 영상 랜드마크 처리
            if current_landmarks1:
                # 유효성 검사 (더 관대한 기준)
                if _validate_landmarks_relaxed(current_landmarks1):
                    # 히스토리에 추가
                    landmark_history1.append(current_landmarks1)
                    if len(landmark_history1) > history_size:
                        landmark_history1.pop(0)
                    
                    # 히스토리가 충분하면 스무딩 적용
                    if len(landmark_history1) >= 3:
                        smoothed_landmarks1 = _smooth_landmarks_with_history(landmark_history1)
                    else:
                        smoothed_landmarks1 = current_landmarks1
                else:
                    # 유효하지 않으면 이전 랜드마크 사용 (깜빡거림 방지)
                    if landmark_history1:
                        smoothed_landmarks1 = landmark_history1[-1]
                    else:
                        smoothed_landmarks1 = current_landmarks1
            else:
                # 검출 실패 시 이전 랜드마크 사용
                if landmark_history1:
                    smoothed_landmarks1 = landmark_history1[-1]
                else:
                    smoothed_landmarks1 = None
                
            # 두 번째 영상 랜드마크 처리 (동일한 로직)
            if current_landmarks2:
                if _validate_landmarks_relaxed(current_landmarks2):
                    landmark_history2.append(current_landmarks2)
                    if len(landmark_history2) > history_size:
                        landmark_history2.pop(0)
                    
                    if len(landmark_history2) >= 3:
                        smoothed_landmarks2 = _smooth_landmarks_with_history(landmark_history2)
                    else:
                        smoothed_landmarks2 = current_landmarks2
                else:
                    if landmark_history2:
                        smoothed_landmarks2 = landmark_history2[-1]
                    else:
                        smoothed_landmarks2 = current_landmarks2
            else:
                if landmark_history2:
                    smoothed_landmarks2 = landmark_history2[-1]
                else:
                    smoothed_landmarks2 = None
            
            # 스켈레톤 그리기 (더 두꺼운 선으로 안정성 확보)
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec_point = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=6, circle_radius=3)
            drawing_spec_line = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)
            
            if smoothed_landmarks1:
                mp_drawing.draw_landmarks(
                    frame1_resized, smoothed_landmarks1, mp_pose.POSE_CONNECTIONS,
                    drawing_spec_point, drawing_spec_line
                )
            
            if smoothed_landmarks2:
                mp_drawing.draw_landmarks(
                    frame2_resized, smoothed_landmarks2, mp_pose.POSE_CONNECTIONS,
                    drawing_spec_point, drawing_spec_line
                )
            
            # 두 프레임을 나란히 배치
            combined_frame = np.hstack((frame1_resized, frame2_resized))
            
            out.write(combined_frame)
            output_frame_count += 1
            
            # 다음 프레임 위치 계산
            current_frame1 += frame_interval1
            current_frame2 += frame_interval2
            
            if output_frame_count % 30 == 0:
                print(f"[INFO] 진행률: {output_frame_count}/{max_frames} ({output_frame_count/max_frames*100:.1f}%)")
        
        # 리소스 해제
        cap1.release()
        cap2.release()
        out.release()
        pose1.close()
        pose2.close()
        
        print(f"[INFO] 스켈레톤 합성 완료 - 총 {output_frame_count} 프레임 처리")
        
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1024
        
    except Exception as e:
        print(f"[ERROR] OpenCV 스켈레톤 포함 합성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_dummy_video(output_path: str, duration: float = 5.0, fps: float = 30.0) -> bool:
    """
    테스트용 더미 영상을 생성합니다.
    """
    try:
        # 기본 크기
        width, height = 1280, 720
        
        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            return False
        
        # 더미 프레임 생성
        total_frames = int(duration * fps)
        
        for i in range(total_frames):
            # 시간에 따라 색상 변화
            hue = int((i / total_frames) * 180)
            frame = np.full((height, width, 3), [hue, 255, 255], dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            
            # 텍스트 추가
            text = f"더미 영상 - {i+1}/{total_frames}"
            cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1024
        
    except Exception as e:
        print(f"[ERROR] 더미 영상 생성 실패: {e}")
        return False