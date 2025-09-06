"""
SSSwing3 메인 실행 파일

이 모듈은 골프 스윙 분석 시스템의 메인 진입점입니다.
다음과 같은 기능을 제공합니다:

1. 골프 스윙 영상 비교 분석
2. 관절 각도 추출 및 점수 계산
3. 구간별 속도 분석
4. 비교 영상 생성 및 자동 재생

주요 특징:
- CLI 및 GUI 인터페이스 지원
- MediaPipe 기반 포즈 추출
- FFmpeg를 통한 고품질 영상 처리
- 다양한 재생 속도 지원
"""

import os
import sys

# 모든 C++/absl/tensorflow/mediapipe stderr 워닝 억제 (환경변수로 제어)
# SSSWING_SUPPRESS_STDERR=1로 설정하면 모든 stderr 출력을 억제
if os.environ.get('SSSWING_SUPPRESS_STDERR', '0') == '1':
    sys.stderr.flush()
    f = open(os.devnull, 'w')
    os.dup2(f.fileno(), sys.stderr.fileno())

# TensorFlow C++ 백엔드 워닝 제거
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Python 경고 메시지 억제
import warnings
warnings.filterwarnings('ignore')  # 대부분의 파이썬 워닝 무시

# absl 로깅 레벨 설정 (TensorFlow 관련)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

# 필요한 라이브러리 import
import cv2
import tkinter as tk
from tkinter import filedialog
import subprocess
import argparse

# 미디어파이프를 한글이 없는 별도 경로에서 우선 로드 (경로가 존재할 때만)
# Windows 환경에서 한글 경로 문제를 해결하기 위한 설정
ALT_SITE_PACKAGES = r"C:\\PyLib311"
if os.path.isdir(ALT_SITE_PACKAGES) and ALT_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, ALT_SITE_PACKAGES)

# 로컬 모듈 import
from .video_utils import draw_pose_on_frame
from .save_4key_section_images import save_pro_user_key_images
from . import pose_extractor
from . import phase_based_section_score
from . import score_calculator
from . import swing_phase_detector





def process_swing_video(pro_video_path, user_video_path):
    """
    스윙 영상 병합 처리 함수
    
    Args:
        pro_video_path: 프로 스윙 영상 경로
        user_video_path: 사용자 스윙 영상 경로
        
    Returns:
        str: 병합된 영상 파일 경로
        
    특징:
        - OpenCV 기반 병합으로 잘림 방지
        - ssswing/mp4 폴더에 combined_swing.mp4로 저장
    """
    # OpenCV 프레임 병합 로직 제거 (FFmpeg로 직접 처리)
    print("⚙️ 영상 병합 처리 중...")

    # 최종 출력 경로 설정 (ssswing/mp4)
    output_dir = os.path.join(os.path.dirname(__file__), 'mp4')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'combined_swing.mp4')

    # OpenCV 기반 병합 함수로 변경 (잘림 방지)
    from . import video_generator
    video_generator.generate_comparison_video(
        pro_video_path,
        user_video_path,
        output_path
    )

    return output_path  # 병합된 파일 경로 반환


def select_video_files():
    """
    tkinter를 사용하여 비디오 파일을 선택하는 함수
    
    Returns:
        tuple: (pro_video_path, user_video_path) 선택된 파일 경로들
        
    특징:
        - GUI 파일 선택 대화상자 제공
        - MP4 파일 우선 표시
        - 프로 영상과 사용자 영상을 순차적으로 선택
    """
    # tkinter로 파일 선택 대화 상자 띄우기
    root = tk.Tk()
    root.withdraw()  # 루트 창 숨기기 (파일 대화상자만 보이게)

    # 두 개의 비디오 파일을 선택하도록 함
    print("🔍 프로 스윙 영상을 선택하세요.")
    pro_video_path = filedialog.askopenfilename(
        title="프로 스윙 영상 선택",
        filetypes=(("MP4 Files", "*.mp4"), ("All Files", "*.*"))
    )

    print("🔍 유저 스윙 영상을 선택하세요.")
    user_video_path = filedialog.askopenfilename(
        title="유저 스윙 영상 선택",
        filetypes=(("MP4 Files", "*.mp4"), ("All Files", "*.*"))
    )

    return pro_video_path, user_video_path


# 영상 선택 및 분석 시작
def main():
    """
    메인 실행 함수
    
    골프 스윙 분석의 전체 워크플로우를 실행합니다:
    1. 영상 파일 선택 (CLI 우선, GUI 폴백)
    2. 관절 각도 JSON 추출
    3. 구간별 시간 추출
    4. 점수 계산
    5. 비교 영상 생성
    6. 자동 재생 (옵션)
    """
    try:

        # 명령행 인수 파싱
        parser = argparse.ArgumentParser(description='골프 스윙 비교/점수화 실행기')
        parser.add_argument('--pro', type=str, help='프로 스윙 영상 경로 (mp4)')
        parser.add_argument('--user', type=str, help='유저 스윙 영상 경로 (mp4)')
        parser.add_argument('--no-play', action='store_true', help='생성된 비교 영상을 자동 재생하지 않음')
        parser.add_argument('--align-start', action='store_true', help='프로 영상의 어드레스 프레임에 맞춰 시작 프레임 정렬')
        args, _ = parser.parse_known_args()



        # 1. 영상 선택 (CLI 우선, GUI 폴백)
        if args.pro and args.user:
            # 명령행 인수로 제공된 경우
            pro_video_path = args.pro
            user_video_path = args.user
        else:
            # GUI를 통한 파일 선택
            pro_video_path, user_video_path = select_video_files()
        
        # 파일 선택 검증
        if not pro_video_path or not user_video_path:
            print('❌ 두 영상 모두 선택해야 합니다.')
            return
        if not os.path.exists(pro_video_path):
            print(f'❌ 프로 영상 경로를 찾을 수 없습니다: {pro_video_path}')
            return
        if not os.path.exists(user_video_path):
            print(f'❌ 유저 영상 경로를 찾을 수 없습니다: {user_video_path}')
            return

        # 2. 관절 각도 JSON 추출
        pro_json = 'angles_pro.json'
        user_json = 'angles_user.json'
        print('⚙️ 프로/유저 관절 각도 추출 중...')
        
        try:
            pose_extractor.extract_angles(pro_video_path, pro_json)
            pose_extractor.extract_angles(user_video_path, user_json)
            print(f'✔️ 관절 각도 JSON 저장 완료: {pro_json}, {user_json}')
        except Exception as e:
            print(f'❌ 관절 각도 추출 중 오류 발생: {e}')
            print('⚠️ 기본값으로 진행합니다.')
            # 기본 JSON 파일 생성
            import json
            default_data = {"angles": []}
            with open(pro_json, 'w') as f:
                json.dump(default_data, f)
            with open(user_json, 'w') as f:
                json.dump(default_data, f)
        
        # 디버그 이미지 카운터 리셋
        from .debug_utils import reset_debug_image_counter
        reset_debug_image_counter()

        # 3. 섹션별 구간 시간 추출 및 스윙 단계 감지
        print('⚙️ 프로/유저 구간별 시간 추출 중...')
        
        try:
            pro_section_times = phase_based_section_score.get_section_times(pro_video_path)
            user_section_times = phase_based_section_score.get_section_times(user_video_path)
        except Exception as e:
            print(f'❌ 구간별 시간 추출 중 오류 발생: {e}')
            print('⚠️ 기본값으로 진행합니다.')
            pro_section_times = [1.0, 1.0]  # 기본값
            user_section_times = [1.0, 1.0]  # 기본값
        
        # 스윙 단계 감지 (키 이미지 저장용)
        print('⚙️ 프로/유저 스윙 단계 감지 중...')
        
        try:
            from .video_processor import extract_landmarks_from_video
            pro_landmarks, pro_aspect_ratio = extract_landmarks_from_video(pro_video_path)
            user_landmarks, user_aspect_ratio = extract_landmarks_from_video(user_video_path)
            
            print(f"[DEBUG] 프로 랜드마크 데이터 길이: {len(pro_landmarks) if pro_landmarks else 0}")
            print(f"[DEBUG] 유저 랜드마크 데이터 길이: {len(user_landmarks) if user_landmarks else 0}")
            print(f"[DEBUG] 프로 영상 비율: {pro_aspect_ratio}")
            print(f"[DEBUG] 유저 영상 비율: {user_aspect_ratio}")
            
            pro_phases = swing_phase_detector.detect_swing_phases(pro_landmarks, video_aspect_ratio=pro_aspect_ratio)
            user_phases = swing_phase_detector.detect_swing_phases(user_landmarks, video_aspect_ratio=user_aspect_ratio)
            
            print(f"[DEBUG] 프로 phases 원본: {pro_phases}")
            print(f"[DEBUG] 유저 phases 원본: {user_phases}")
            
        except Exception as e:
            print(f'❌ 스윙 단계 감지 중 오류 발생: {e}')
            print('⚠️ 기본값으로 진행합니다.')
            pro_phases = {"address": 0, "top": 25, "finish": 50}
            user_phases = {"address": 0, "top": 22, "finish": 45}
        
        print(f"[DEBUG] 프로 phases: {pro_phases}")
        print(f"[DEBUG] 유저 phases: {user_phases}")
        
        # 키 이미지 저장 (프로/사용자 각각 3장씩, swing_key_images 폴더에)
        print('⚙️ 프로/유저 키 이미지 저장 중...')
        
        # swing_key_images 폴더에 저장
        key_images_dir = os.path.join(os.path.dirname(__file__), 'swing_key_images')
        
        # phases 데이터 검증 및 정규화
        def validate_and_normalize_phases(phases, video_type="unknown"):
            """phases 데이터를 검증하고 정규화합니다."""
            print(f"[DEBUG] {video_type} phases 원본 검증: {phases}")
            
            # None이거나 빈 딕셔너리인 경우 기본값 사용
            if not phases:
                print(f"[WARNING] {video_type} phases가 비어있어 기본값을 사용합니다.")
                return {"address": 0, "top": 25, "finish": 50}
            
            # 기본 변환
            normalized = {
                "address": phases.get("address", 0),
                "top": phases.get("top", 25),
                "finish": phases.get("finish", 50)
            }
            
            # 특수한 경우 처리
            for key in ["top", "finish"]:
                value = normalized[key]
                if isinstance(value, list):
                    normalized[key] = value[0] if value else 25 if key == "top" else 50
                    print(f"[INFO] {video_type} {key} 리스트 값 변환: {value} -> {normalized[key]}")
                elif value is None:
                    normalized[key] = 25 if key == "top" else 50
                    print(f"[WARNING] {video_type} {key} 값이 None이어 기본값 사용: {normalized[key]}")
                else:
                    try:
                        normalized[key] = int(float(value))
                        if normalized[key] < 0:
                            normalized[key] = 0
                            print(f"[WARNING] {video_type} {key} 음수 값 수정: {value} -> 0")
                    except (ValueError, TypeError) as e:
                        default_val = 25 if key == "top" else 50
                        normalized[key] = default_val
                        print(f"[WARNING] {video_type} {key} 값 변환 실패: {value} -> 기본값 {default_val}")
            
            # address 값 정규화
            try:
                normalized["address"] = int(float(normalized["address"]))
                if normalized["address"] < 0:
                    normalized["address"] = 0
                    print(f"[WARNING] {video_type} address 음수 값 수정: {normalized['address']} -> 0")
            except (ValueError, TypeError) as e:
                normalized["address"] = 0
                print(f"[WARNING] {video_type} address 값 변환 실패: {normalized['address']} -> 0")
            
            print(f"[DEBUG] {video_type} phases 정규화 완료: {normalized}")
            return normalized
        
        # phases 데이터 정규화
        pro_phases = validate_and_normalize_phases(pro_phases, "프로")
        user_phases = validate_and_normalize_phases(user_phases, "유저")
        
        # 키 이미지 저장 (중복 방지)
        print(f"[DEBUG] 이미지 저장 함수 호출:")
        print(f"  - 프로 비디오: {pro_video_path}")
        print(f"  - 유저 비디오: {user_video_path}")
        print(f"  - 프로 phases: {pro_phases}")
        print(f"  - 유저 phases: {user_phases}")
        print(f"  - 출력 디렉토리: {key_images_dir}")
        
        saved_images = save_pro_user_key_images(
            pro_video_path, user_video_path, 
            pro_phases, user_phases, 
            key_images_dir
        )
        
        # 저장 결과 확인
        if len(saved_images) == 6:  # 프로 3장 + 유저 3장
            print(f'✔️ 키 이미지 저장 완료: {len(saved_images)}장')
            print(f'  - 프로: start, top, finish (3장)')
            print(f'  - 유저: start, top, finish (3장)')
        else:
            print(f'⚠️ 키 이미지 저장 불완전: {len(saved_images)}장 (예상: 6장)')

        # 4. 점수 계산
        print('⚙️ 점수 계산 중...')
        # 시작 프레임 설정: 기본은 풀영상(0,0). --align-start가 있을 때만 프로 어드레스 정렬 사용
        start_frame_pro = 0
        start_frame_user = 0
        if args.align_start:
            # swing_segmenter가 삭제되어 기본값 사용
            start_frame_pro = 0
        
        # 각도 점수 계산
        angle_score = score_calculator.calc_angle_score(pro_json, user_json)
        # 구간별 속도 점수 계산
        section_scores, section_avg = score_calculator.calc_section_speed_score(pro_section_times, user_section_times)
        # 최종 점수 계산
        final_score = score_calculator.calc_final_score(angle_score, section_avg)

        # 5. 결과 출력
        score_calculator.print_score_result(angle_score, section_scores, section_avg, final_score)

        # 6. 랜드마크 포함 비교 영상 생성 및 저장
        print('⚙️ 랜드마크 포함 비교 영상 생성 중...')
        output_dir = os.path.join(os.path.dirname(__file__), 'mp4')
        os.makedirs(output_dir, exist_ok=True)
        output_video = os.path.join(output_dir, 'combined_swing.mp4')
        
        # video_generator가 삭제되어 비교 영상 생성 건너뜀
        print("⚠️ video_generator가 삭제되어 비교 영상 생성을 건너뜁니다.")
        print(f'✔️ 비교 영상 저장 완료: {output_video}')
        
        # 영상 자동 재생 (옵션)
        if not args.no_play:
            play_video_with_speeds(output_video)
        else:
            print('▶️ 자동 재생은 건너뜁니다 (--no-play).')

    except Exception as e:
        print(f'❌ 오류 발생: {e}')
        import traceback
        traceback.print_exc()


def play_video_with_speeds(video_path):
    """
    다양한 속도로 영상을 재생하는 함수
    
    Args:
        video_path: 재생할 영상 파일 경로
        
    특징:
        - 1배속, 0.5배속, 0.25배속 순환 재생
        - Q 또는 ESC 키로 종료
        - 각 속도별로 영상 끝까지 재생 후 다음 속도로 전환
    """
    speeds = [1.0, 0.5, 0.25]  # 1배속, 0.5배속, 0.25배속
    speed_names = ["1배속", "0.5배속(슬로우)", "0.25배속(초슬로우)"]
    idx = 0
    
    while True:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 현재 속도에 맞는 프레임 지연 시간 계산
        delay = int(1000 / (fps * speeds[idx])) if fps > 0 else 40
        
        print(f"▶️ {speed_names[idx]}로 재생 중... (종료: Q 또는 ESC)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Swing Video", frame)
            key = cv2.waitKey(delay)
            if key == 27 or key == ord('q'):  # ESC 또는 Q 키
                cap.release()
                cv2.destroyAllWindows()
                return
        
        cap.release()
        # 다음 속도로 전환 (순환)
        idx = (idx + 1) % len(speeds)
    
    cv2.destroyAllWindows()


# 스크립트 직접 실행 시 메인 함수 호출
if __name__ == "__main__":
    main()
