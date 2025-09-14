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
from typing import Tuple, Optional
from scipy.signal import savgol_filter  # Savgol filter for additional smoothing

def get_video_info(video_path: str) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    """
    영상 파일의 기본 정보를 추출합니다.
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
    """
    try:
        if _generate_with_opencv_skeleton(video1_path, video2_path, output_path, target_fps, max_duration):
            print(f"[INFO] OpenCV 스켈레톤 포함 합성 영상 생성 성공: {output_path}")
            return True
        
        if _generate_with_ffmpeg(video1_path, video2_path, output_path, target_fps, max_duration):
            print(f"[INFO] FFmpeg로 합성 영상 생성 성공: {output_path}")
            return True
        
        if _generate_with_opencv(video1_path, video2_path, output_path, target_fps, max_duration):
            print(f"[INFO] OpenCV로 합성 영상 생성 성공: {output_path}")
            return True
        
        print(f"[ERROR] 모든 합성 방법 실패")
        return False
        
    except Exception as e:
        print(f"[ERROR] 합성 영상 생성 중 예외 발생: {e}")
        return False


def _get_person_bounding_box(landmarks, frame_height: int, frame_width: int, margin_ratio: float = 1.0):
    """
    줌 제거 - 항상 None을 반환하여 원본 프레임을 그대로 사용합니다.
    """
    # 줌 완전 제거 - 원본 영상 크기 그대로 사용
    return None


def _crop_and_resize_frame(frame, bbox, target_width: int, target_height: int, landmarks=None) -> np.ndarray:
    """
    바운딩 박스가 있으면 크롭하고, 없으면 원본 프레임을 그대로 사용합니다. 랜드마크가 제공되면 패딩 전에 그립니다.
    """
    if bbox is None:
        # 줌 제거 - 원본 프레임을 그대로 사용
        cropped = frame
    else:
        x, y, width, height = bbox
        cropped = frame[y:y+height, x:x+width]
    
    # 원본 비율을 유지하면서 리사이징 (줌인/아웃 방지하면서 비율 보존)
    h, w = cropped.shape[:2]
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
    
    resized_frame = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    if landmarks:
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        
        base_thickness = max(2, min(8, int(new_height / 100)))
        base_radius = max(2, min(6, int(new_height / 150)))
        
        drawing_spec_point = mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=base_thickness, circle_radius=base_radius
        )
        drawing_spec_line = mp_drawing.DrawingSpec(
            color=(255, 0, 0), thickness=base_thickness
        )
        
        mp_drawing.draw_landmarks(
            resized_frame, landmarks, mp_pose.POSE_CONNECTIONS,
            drawing_spec_point, drawing_spec_line
        )
    
    # 검정색 배경 생성 및 패딩 추가
    padded_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # 중앙에 배치
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    padded_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
    
    return padded_frame


def _transform_landmarks_to_cropped_frame(landmarks, original_width: int, original_height: int, 
                                         bbox, target_width: int, target_height: int):
    """
    원본 프레임에서 추출된 랜드마크를 크롭된 프레임에 맞게 좌표를 변환합니다.
    """
    if not landmarks:
        return None
    
    try:
        from mediapipe.framework.formats import landmark_pb2
        transformed_landmarks = landmark_pb2.NormalizedLandmarkList()
        
        if bbox is None:
            crop_x, crop_y = 0, 0
            crop_width, crop_height = original_width, original_height
        else:
            crop_x, crop_y, crop_width, crop_height = bbox
        
        for landmark in landmarks.landmark:
            new_landmark = transformed_landmarks.landmark.add()
            
            pixel_x = landmark.x * original_width
            pixel_y = landmark.y * original_height
            
            cropped_x = pixel_x - crop_x
            cropped_y = pixel_y - crop_y
            
            new_landmark.x = cropped_x / crop_width if crop_width > 0 else 0.5
            new_landmark.y = cropped_y / crop_height if crop_height > 0 else 0.5
            new_landmark.z = landmark.z
            new_landmark.visibility = landmark.visibility
        
        return transformed_landmarks
    except Exception as e:
        print(f"[WARN] 랜드마크 변환 실패: {e}")
        return landmarks


def _smooth_landmarks_with_history(landmark_history, use_savgol: bool = True):
    """
    히스토리 기반 랜드마크 스무딩 (Savgol filter 추가 옵션).
    """
    if not landmark_history:
        return None
    
    try:
        from mediapipe.framework.formats import landmark_pb2
        smoothed_landmarks = landmark_pb2.NormalizedLandmarkList()
        
        num_joints = len(landmark_history[0].landmark)
        history_length = len(landmark_history)
        
        x_array = np.zeros((num_joints, history_length))
        y_array = np.zeros((num_joints, history_length))
        z_array = np.zeros((num_joints, history_length))
        vis_array = np.zeros((num_joints, history_length))
        
        for t, landmarks in enumerate(landmark_history):
            for j in range(num_joints):
                if j < len(landmarks.landmark):
                    lm = landmarks.landmark[j]
                    x_array[j, t] = lm.x
                    y_array[j, t] = lm.y
                    z_array[j, t] = lm.z
                    vis_array[j, t] = lm.visibility
        
        if use_savgol and history_length >= 5:
            for j in range(num_joints):
                x_array[j] = savgol_filter(x_array[j], window_length=5, polyorder=2)
                y_array[j] = savgol_filter(y_array[j], window_length=5, polyorder=2)
                z_array[j] = savgol_filter(z_array[j], window_length=5, polyorder=2)
        
        for j in range(num_joints):
            new_landmark = smoothed_landmarks.landmark.add()
            valid_mask = vis_array[j] > 0.5
            valid_count = np.sum(valid_mask)
            
            if valid_count > 0:
                new_landmark.x = np.mean(x_array[j][valid_mask])
                new_landmark.y = np.mean(y_array[j][valid_mask])
                new_landmark.z = np.mean(z_array[j][valid_mask])
                new_landmark.visibility = np.mean(vis_array[j][valid_mask])
            else:
                new_landmark.x = 0.5
                new_landmark.y = 0.5
                new_landmark.z = 0.0
                new_landmark.visibility = 0.0
        
        return smoothed_landmarks
        
    except Exception as e:
        print(f"[WARN] 히스토리 기반 스무딩 실패: {e}")
        return landmark_history[-1] if landmark_history else None


def _smooth_bounding_box(bbox_history):
    """
    바운딩 박스 히스토리 기반 스무딩 (흔들거림 방지)
    """
    if not bbox_history:
        return None
    
    try:
        x_sum = y_sum = w_sum = h_sum = 0
        valid_count = 0
        
        for bbox in bbox_history:
            if bbox is not None:
                x, y, w, h = bbox
                x_sum += x
                y_sum += y
                w_sum += w
                h_sum += h
                valid_count += 1
        
        if valid_count > 0:
            return (
                int(x_sum / valid_count),
                int(y_sum / valid_count),
                int(w_sum / valid_count),
                int(h_sum / valid_count)
            )
        else:
            return None
            
    except Exception as e:
        print(f"[WARN] 바운딩 박스 스무딩 실패: {e}")
        return bbox_history[-1] if bbox_history else None


def _validate_landmarks_relaxed(landmarks, threshold: float = 0.4):
    """
    랜드마크의 유효성을 검사합니다 (깜빡거림 방지를 위한 관대한 기준).
    """
    if not landmarks:
        return False
    
    key_joints = [11, 12, 13, 14, 15, 16]  # 어깨, 팔꿈치, 손목
    valid_count = 0
    
    for joint_idx in key_joints:
        if joint_idx < len(landmarks.landmark):
            if landmarks.landmark[joint_idx].visibility > threshold:
                valid_count += 1
    
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
        fps1, w1, h1, _ = get_video_info(video1_path)
        fps2, w2, h2, _ = get_video_info(video2_path)
        
        if not all([fps1, w1, h1, fps2, w2, h2]):
            return False
        
        target_height = max(h1, h2, 720)
        total_width = 1280
        target_width_per_video = 640
        
        filter_complex = (
            f"[0:v]scale={target_width_per_video}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width_per_video}:{target_height}:(ow-iw)/2:(oh-ih)/2:black[v1];"
            f"[1:v]scale={target_width_per_video}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width_per_video}:{target_height}:(ow-iw)/2:(oh-ih)/2:black[v2];"
            f"[v1][v2]hstack=inputs=2"
        )
        
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
            if os.path.getsize(output_path) > 1024:
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
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        if not cap1.isOpened() or not cap2.isOpened():
            return False
        
        fps1, w1, h1, _ = get_video_info(video1_path)
        fps2, w2, h2, _ = get_video_info(video2_path)
        
        target_height = max(h1, h2, 720)
        total_width = 1280
        target_width_per_video = 640
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (total_width, target_height))
        
        if not out.isOpened():
            return False
        
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            frame1_resized = _crop_and_resize_frame(frame1, None, target_width_per_video, target_height)
            frame2_resized = _crop_and_resize_frame(frame2, None, target_width_per_video, target_height)
            
            combined_frame = np.hstack((frame1_resized, frame2_resized))
            
            out.write(combined_frame)
        
        cap1.release()
        cap2.release()
        out.release()
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            return True
        
        return False
        
    except Exception as e:
        print(f"[ERROR] OpenCV 합성 실패: {e}")
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
        mp_pose = mp.solutions.pose
        
        pose1 = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5  # Increased for stability
        )
        
        pose2 = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5
        )
        
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        if not cap1.isOpened() or not cap2.isOpened():
            return False
        
        fps1, w1, h1, frame_count1 = get_video_info(video1_path)
        fps2, w2, h2, frame_count2 = get_video_info(video2_path)

        duration1 = frame_count1 / fps1 if fps1 > 0 else 0
        duration2 = frame_count2 / fps2 if fps2 > 0 else 0
        min_duration = min(duration1, duration2, max_duration)
        
        frame_interval1 = fps1 / target_fps if fps1 > 0 else 1
        frame_interval2 = fps2 / target_fps if fps2 > 0 else 1
        
        target_height = max(h1, h2, 720)
        total_width = 1280
        target_width_per_video = 640
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (total_width, target_height))
        
        if not out.isOpened():
            return False
        
        output_frame_count = 0
        max_frames = int(min_duration * target_fps)
        current_frame1 = 0.0
        current_frame2 = 0.0
        
        landmark_history1 = []
        landmark_history2 = []
        history_size = 5
        
        bbox_history1 = []
        bbox_history2 = []
        bbox_history_size = 50  # 최대 강화된 스무딩으로 크기 변동 완전 방지
        
        print(f"[INFO] 스켈레톤 합성 시작 - 최대 {max_frames} 프레임")
        print(f"[DEBUG] 개선: min_tracking_confidence 0.5, 줌 완전 제거 (원본 크기 그대로), 원본 비율 유지, 스무딩 50프레임")
        
        while output_frame_count < max_frames:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame1))
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame2))
            
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            results1 = pose1.process(frame1_rgb)
            
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            results2 = pose2.process(frame2_rgb)
            
            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            
            current_bbox1 = _get_person_bounding_box(results1.pose_landmarks, h1, w1)
            current_bbox2 = _get_person_bounding_box(results2.pose_landmarks, h2, w2)
            
            if current_bbox1 is not None:
                bbox_history1.append(current_bbox1)
                if len(bbox_history1) > bbox_history_size:
                    bbox_history1.pop(0)
            
            if current_bbox2 is not None:
                bbox_history2.append(current_bbox2)
                if len(bbox_history2) > bbox_history_size:
                    bbox_history2.pop(0)
            
            bbox1 = _smooth_bounding_box(bbox_history1) if bbox_history1 else current_bbox1
            bbox2 = _smooth_bounding_box(bbox_history2) if bbox_history2 else current_bbox2
            
            current_landmarks1 = results1.pose_landmarks
            current_landmarks2 = results2.pose_landmarks
            
            if current_landmarks1:
                if _validate_landmarks_relaxed(current_landmarks1):
                    landmark_history1.append(current_landmarks1)
                    if len(landmark_history1) > history_size:
                        landmark_history1.pop(0)
                    
                    if len(landmark_history1) >= 3:
                        smoothed_landmarks1 = _smooth_landmarks_with_history(landmark_history1)
                    else:
                        smoothed_landmarks1 = current_landmarks1
                else:
                    if landmark_history1:
                        smoothed_landmarks1 = landmark_history1[-1]
                    else:
                        smoothed_landmarks1 = current_landmarks1
            else:
                if landmark_history1:
                    smoothed_landmarks1 = landmark_history1[-1]
                else:
                    smoothed_landmarks1 = None
            
            if smoothed_landmarks1:
                transformed_landmarks1 = _transform_landmarks_to_cropped_frame(
                    smoothed_landmarks1, w1, h1, bbox1, target_width_per_video, target_height
                )
            else:
                transformed_landmarks1 = None
                
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
            
            if smoothed_landmarks2:
                transformed_landmarks2 = _transform_landmarks_to_cropped_frame(
                    smoothed_landmarks2, w2, h2, bbox2, target_width_per_video, target_height
                )
            else:
                transformed_landmarks2 = None
            
            frame1_resized = _crop_and_resize_frame(frame1, bbox1, target_width_per_video, target_height, transformed_landmarks1)
            frame2_resized = _crop_and_resize_frame(frame2, bbox2, target_width_per_video, target_height, transformed_landmarks2)
            
            combined_frame = np.hstack((frame1_resized, frame2_resized))
            
            out.write(combined_frame)
            output_frame_count += 1
            
            current_frame1 += frame_interval1
            current_frame2 += frame_interval2
            
            if output_frame_count % 30 == 0:
                print(f"[INFO] 진행률: {output_frame_count}/{max_frames} ({output_frame_count/max_frames*100:.1f}%)")
        
        cap1.release()
        cap2.release()
        out.release()
        pose1.close()
        pose2.close()
        
        print(f"[INFO] 스켈레톤 합성 완료 - 총 {output_frame_count} 프레임 처리")
        
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1024
        
    except Exception as e:
        print(f"[ERROR] OpenCV 스켈레톤 포함 합성 실패: {e}")
        return False


def create_dummy_video(output_path: str, duration: float = 5.0, fps: float = 30.0) -> bool:
    """
    테스트용 더미 영상을 생성합니다.
    """
    try:
        width, height = 1280, 720
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            return False
        
        total_frames = int(duration * fps)
        
        for i in range(total_frames):
            hue = int((i / total_frames) * 180)
            frame = np.full((height, width, 3), [hue, 255, 255], dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            
            text = f"더미 영상 - {i+1}/{total_frames}"
            cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1024
        
    except Exception as e:
        print(f"[ERROR] 더미 영상 생성 실패: {e}")
        return False