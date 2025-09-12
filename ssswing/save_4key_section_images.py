import os
import sys
import cv2
import numpy as np
import traceback

# Suppress absl, TensorFlow, and C++ backend warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass
import warnings
warnings.filterwarnings('ignore')

# Redirect C++/stderr warnings (MediaPipe, TensorFlow, OpenCV)
class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr


def validate_video_file(video_path):
    """
    비디오 파일의 유효성을 검사합니다.
    
    Args:
        video_path (str): 비디오 파일 경로
        
    Returns:
        tuple: (is_valid, error_message, video_info)
    """
    print(f"[DEBUG] 비디오 파일 검증 시작: {video_path}")
    
    # 1. 파일 존재 여부 확인
    if not os.path.exists(video_path):
        return False, f"파일이 존재하지 않습니다: {video_path}", None
    
    # 2. 파일 크기 확인
    try:
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            return False, "파일 크기가 0입니다", None
        print(f"[DEBUG] 파일 크기: {file_size} bytes")
    except Exception as e:
        return False, f"파일 크기 확인 실패: {e}", None
    
    # 3. OpenCV로 비디오 열기 테스트
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "OpenCV로 비디오를 열 수 없습니다", None
        
        # 비디오 정보 추출
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 첫 번째 프레임 읽기 테스트
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False, "첫 번째 프레임을 읽을 수 없습니다", None
        
        cap.release()
        
        video_info = {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'frame_shape': frame.shape
        }
        
        print(f"[DEBUG] 비디오 정보: {video_info}")
        return True, None, video_info
        
    except Exception as e:
        return False, f"비디오 검증 중 오류: {e}", None


def validate_output_directory(output_dir):
    """
    출력 디렉토리의 쓰기 권한을 검사합니다.
    
    Args:
        output_dir (str): 출력 디렉토리 경로
        
    Returns:
        tuple: (is_valid, error_message)
    """
    print(f"[DEBUG] 출력 디렉토리 검증 시작: {output_dir}")
    
    try:
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        print(f"[DEBUG] 디렉토리 생성/확인 완료: {output_dir}")
        
        # 쓰기 권한 테스트
        test_file = os.path.join(output_dir, "test_write.tmp")
        test_content = "test_write_permission"
        
        # 파일 쓰기 테스트
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # 파일 읽기 테스트
        with open(test_file, 'r', encoding='utf-8') as f:
            read_content = f.read()
        
        # 파일 삭제 테스트
        os.remove(test_file)
        
        if read_content != test_content:
            return False, "파일 쓰기/읽기 테스트 실패"
        
        print(f"[DEBUG] 출력 디렉토리 권한 검증 완료: {output_dir}")
        return True, None
        
    except PermissionError as e:
        return False, f"권한 오류: {e}"
    except Exception as e:
        return False, f"디렉토리 검증 중 오류: {e}"


def create_dummy_image(name, video_type, width=640, height=480):
    """
    더미 이미지를 생성합니다.
    
    Args:
        name (str): 이미지 이름 (start, top, finish)
        video_type (str): 비디오 타입 (pro, user)
        width (int): 이미지 너비
        height (int): 이미지 높이
        
    Returns:
        numpy.ndarray: 더미 이미지
    """
    try:
        # 색상 정의
        colors = {
            'start': (100, 150, 200),    # 파란색
            'top': (200, 100, 150),      # 보라색
            'finish': (150, 200, 100)    # 초록색
        }
        
        # 이미지 생성
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = colors.get(name, (100, 100, 100))
        
        # 텍스트 추가
        text = f"{video_type.upper()} {name.upper()}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # 텍스트 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 텍스트 위치 계산 (중앙 정렬)
        text_x = (width - text_width) // 2
        text_y = (height + text_height) // 2
        
        # 텍스트 그리기
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        print(f"[DEBUG] 더미 이미지 생성 완료: {name} ({width}x{height})")
        return img
        
    except Exception as e:
        print(f"[ERROR] 더미 이미지 생성 실패: {e}")
        # 기본 이미지 반환
        return np.zeros((height, width, 3), dtype=np.uint8)


def safe_imwrite(file_path, image):
    """
    안전한 이미지 저장 함수
    
    Args:
        file_path (str): 저장할 파일 경로
        image (numpy.ndarray): 저장할 이미지
        
    Returns:
        bool: 저장 성공 여부
    """
    try:
        # 이미지 유효성 검사
        if image is None:
            print(f"[ERROR] 이미지가 None입니다: {file_path}")
            return False
        
        if not isinstance(image, np.ndarray):
            print(f"[ERROR] 이미지가 numpy 배열이 아닙니다: {file_path}")
            return False
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            print(f"[ERROR] 이미지 형태가 올바르지 않습니다: {image.shape}")
            return False
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 이미지 저장
        success = cv2.imwrite(file_path, image)
        
        if success:
            # 파일 존재 확인
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"[SUCCESS] 이미지 저장 완료: {file_path} ({file_size} bytes)")
                return True
            else:
                print(f"[ERROR] 파일이 생성되지 않았습니다: {file_path}")
                return False
        else:
            print(f"[ERROR] cv2.imwrite 실패: {file_path}")
            return False
            
    except PermissionError as e:
        print(f"[ERROR] 권한 오류로 이미지 저장 실패: {file_path} - {e}")
        return False
    except Exception as e:
        print(f"[ERROR] 이미지 저장 중 예외 발생: {file_path} - {e}")
        print(f"[DEBUG] 예외 상세: {traceback.format_exc()}")
        return False


def save_4key_section_images(video_path, output_dir, phases, video_type="user"):
    """
    영상에서 phases dict(address, impact/top, finish)로 지정된 프레임 이미지를 output_dir에 저장합니다.
    프로와 사용자 영상에 대해 각각 3장씩만 저장합니다.
    
    Args:
        video_path (str): 입력 영상 경로
        output_dir (str): 저장 폴더
        phases (dict): {'address': idx, 'impact' 또는 'top': idx, 'finish': idx}
        video_type (str): "pro" 또는 "user" (파일명 구분용)
    Returns:
        dict: 저장된 파일 경로
    """
    print(f"[INFO] {video_type.upper()} 키 이미지 저장 시작:")
    print(f"  - 영상 경로: {video_path}")
    print(f"  - 저장 폴더: {output_dir}")
    print(f"  - 스윙 단계: {phases}")
    
    # 1. phases 데이터 검증 및 정규화
    if phases is None:
        print("[WARNING] phases가 None입니다. 기본값을 사용합니다.")
        phases = {"address": 0, "top": 25, "finish": 50}
    
    # 안전한 값 추출 함수
    def safe_get_phase_value(phases, key, default):
        """안전하게 phases에서 값을 추출"""
        try:
            value = phases.get(key, default)
            
            # 리스트인 경우 첫 번째 값 사용
            if isinstance(value, list):
                return value[0] if value else default
            
            # None인 경우 기본값 사용
            if value is None:
                return default
            
            # 숫자로 변환
            return int(float(value))
            
        except (ValueError, TypeError) as e:
            print(f"[WARNING] phases[{key}] 값 변환 실패: {value} -> 기본값 {default} 사용")
            return default
    
    # 저장할 키 프레임 정의 (start, finish만 저장) - 임팩트/탑 제거
    key_frames = {
        'start': safe_get_phase_value(phases, 'address', 0),
        'finish': safe_get_phase_value(phases, 'finish', 50)
    }
    
    print(f"[INFO] 저장할 키 프레임: {key_frames}")
    
    # 2. 출력 디렉토리 검증
    is_valid_dir, dir_error = validate_output_directory(output_dir)
    if not is_valid_dir:
        print(f"[ERROR] 출력 디렉토리 검증 실패: {dir_error}")
        return {}
    
    # 3. 비디오 파일 검증
    is_valid_video, video_error, video_info = validate_video_file(video_path)
    if not is_valid_video:
        print(f"[WARNING] 비디오 파일 검증 실패: {video_error}")
        print(f"[INFO] 더미 이미지 생성으로 대체합니다.")
        
        # 더미 이미지 생성
        saved = {}
        for name in key_frames.keys():
            try:
                img = create_dummy_image(name, video_type)
                out_path = os.path.join(output_dir, f"{video_type}_{name}.jpg")
                
                if safe_imwrite(out_path, img):
                    saved[name] = out_path
                else:
                    print(f"[ERROR] {video_type.upper()} {name} 더미 이미지 저장 실패")
                    
            except Exception as e:
                print(f"[ERROR] {video_type.upper()} {name} 더미 이미지 생성 오류: {e}")
        
        print(f"[INFO] 총 {len(saved)}장의 더미 이미지 저장 완료")
        return saved
    
    # 4. 프레임 인덱스 유효성 검사 및 조정
    total_frames = video_info['total_frames']
    fps_info = video_info.get('fps') or 0
    print(f"[INFO] 영상 정보: {video_info}")
    
    # 프레임 인덱스 조정 (영상 길이를 초과하지 않도록) + start는 직전 프레임로 보정
    adjusted_frames = {}
    for name, target_idx in key_frames.items():
        # 숫자 보정
        try:
            target_idx = int(target_idx)
        except Exception:
            target_idx = 0
        
        # 어드레스 직전 프레임 보정: 고FPS면 -2, 아니면 -1
        if name == 'start':
            offset = 2 if fps_info and fps_info >= 50 else 1
            before_idx = max(0, target_idx - offset)
            if before_idx != target_idx:
                print(f"[INFO] start 프레임 보정: {target_idx} -> {before_idx} (offset={offset})")
            target_idx = before_idx
        
        if target_idx >= total_frames:
            adjusted_idx = max(0, total_frames - 1)  # 마지막 프레임 사용
            print(f"[WARNING] {name} 프레임 인덱스 조정: {target_idx} -> {adjusted_idx} (영상 길이: {total_frames})")
            adjusted_frames[name] = adjusted_idx
        else:
            adjusted_frames[name] = target_idx
    
    print(f"[INFO] 조정된 프레임 인덱스: {adjusted_frames}")
    
    # 5. 실제 영상에서 키 프레임 추출 및 저장
    saved = {}
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("비디오를 열 수 없습니다")
        
        frame_idx = 0
        target_frames = set(adjusted_frames.values())
        
        print(f"[INFO] 영상에서 키 프레임 추출 중...")
        print(f"[INFO] 대상 프레임 인덱스: {target_frames}")
        
        while cap.isOpened() and len(saved) < len(adjusted_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"[INFO] 영상 읽기 완료 (총 {frame_idx} 프레임 처리)")
                break
            
            # 현재 프레임이 저장할 키 프레임인지 확인
            for name, target_idx in adjusted_frames.items():
                if frame_idx == target_idx and name not in saved:
                    try:
                        out_path = os.path.join(output_dir, f"{video_type}_{name}.jpg")
                        
                        if safe_imwrite(out_path, frame):
                            saved[name] = out_path
                            print(f"[SUCCESS] {video_type.upper()} {name} 이미지 저장 (프레임 {frame_idx}): {out_path}")
                        else:
                            print(f"[ERROR] {video_type.upper()} {name} 이미지 저장 실패")
                            
                    except Exception as e:
                        print(f"[ERROR] {video_type.upper()} {name} 이미지 저장 중 오류: {e}")
            
            frame_idx += 1
            
    except Exception as e:
        print(f"[ERROR] 영상 프레임 읽기 중 오류: {e}")
        print(f"[DEBUG] 예외 상세: {traceback.format_exc()}")
    finally:
        if 'cap' in locals():
            cap.release()
    
    # 6. 누락된 프레임에 대해 더미 이미지 생성
    if len(saved) < len(adjusted_frames):
        missing_frames = set(adjusted_frames.keys()) - set(saved.keys())
        print(f"[WARNING] 저장되지 않은 프레임: {missing_frames}")
        
        for name in missing_frames:
            try:
                print(f"[INFO] {name} 프레임에 대해 더미 이미지 생성")
                
                img = create_dummy_image(name, video_type)
                out_path = os.path.join(output_dir, f"{video_type}_{name}.jpg")
                
                if safe_imwrite(out_path, img):
                    saved[name] = out_path
                    print(f"[SUCCESS] {video_type.upper()} {name} 더미 이미지 저장: {out_path}")
                else:
                    print(f"[ERROR] {video_type.upper()} {name} 더미 이미지 저장 실패")
                    
            except Exception as e:
                print(f"[ERROR] {video_type.upper()} {name} 더미 이미지 생성 오류: {e}")
    
    # 7. 저장 결과 요약
    if len(saved) == len(adjusted_frames):
        print(f"[SUCCESS] {video_type.upper()} 키 이미지 {len(saved)}장 모두 저장 완료")
    else:
        missing = set(adjusted_frames.keys()) - set(saved.keys())
        print(f"[WARNING] {video_type.upper()} 키 이미지 {len(saved)}장만 저장됨 (누락: {missing})")
    
    return saved


def save_pro_user_key_images(pro_video_path, user_video_path, pro_phases, user_phases, output_dir):
    """
    프로와 사용자 영상에서 각각 시작(어드레스), 백스윙탑, 피니시 3장씩만 이미지를 저장합니다.
    프로 이미지는 pro/ 서브폴더에, 사용자 이미지는 user/ 서브폴더에 저장됩니다.
    
    Args:
        pro_video_path (str): 프로 영상 경로
        user_video_path (str): 사용자 영상 경로
        pro_phases (dict): 프로 영상의 스윙 단계 정보
        user_phases (dict): 사용자 영상의 스윙 단계 정보
        output_dir (str): 기본 저장 폴더 (swing_key_images)
    
    Returns:
        dict: 저장된 파일 경로들
    """
    print(f"[INFO] 프로와 사용자 키 이미지 저장 시작...")
    print(f"[INFO] 프로 phases: {pro_phases}")
    print(f"[INFO] 유저 phases: {user_phases}")
    print(f"[INFO] 출력 디렉토리: {output_dir}")
    
    # 프로와 사용자 각각의 서브폴더 생성
    pro_output_dir = os.path.join(output_dir, "pro")
    user_output_dir = os.path.join(output_dir, "user")
    
    # 폴더가 없으면 생성
    os.makedirs(pro_output_dir, exist_ok=True)
    os.makedirs(user_output_dir, exist_ok=True)
    
    # 기존 이미지 삭제 (항상 새로 생성)
    try:
        existing_pro = [f for f in os.listdir(pro_output_dir) if f.endswith('.jpg')]
        existing_user = [f for f in os.listdir(user_output_dir) if f.endswith('.jpg')]
        
        if existing_pro or existing_user:
            print(f"[INFO] 기존 이미지 삭제 중...")
            if existing_pro:
                print(f"  - pro 폴더: {existing_pro}")
                for file in existing_pro:
                    file_path = os.path.join(pro_output_dir, file)
                    os.remove(file_path)
                    print(f"    - 삭제: {file}")
            if existing_user:
                print(f"  - user 폴더: {existing_user}")
                for file in existing_user:
                    file_path = os.path.join(user_output_dir, file)
                    os.remove(file_path)
                    print(f"    - 삭제: {file}")
    except Exception as e:
        print(f"[WARNING] 기존 이미지 삭제 중 오류: {e}")
    
    print(f"[INFO] 새로운 키 이미지 생성 시작...")
    
    # 프로 영상에서 3장 저장 (pro/ 폴더에)
    pro_saved = save_4key_section_images(pro_video_path, pro_output_dir, pro_phases, "pro")
    
    # 사용자 영상에서 3장 저장 (user/ 폴더에)
    user_saved = save_4key_section_images(user_video_path, user_output_dir, user_phases, "user")
    
    # 전체 저장 결과 합치기 (프로와 유저 구분하여 키 생성)
    all_saved = {}
    
    # 프로 이미지 추가
    for name, path in pro_saved.items():
        all_saved[f"pro_{name}"] = path
    
    # 유저 이미지 추가
    for name, path in user_saved.items():
        all_saved[f"user_{name}"] = path
    
    # 저장 결과 요약
    print(f"[INFO] 키 이미지 저장 완료 요약:")
    print(f"  - 프로 이미지: {len(pro_saved)}장")
    print(f"  - 유저 이미지: {len(user_saved)}장")
    print(f"  - 총 이미지: {len(all_saved)}장")
    
    # 파일 존재 확인
    for name, path in all_saved.items():
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            print(f"    ✓ {name}: {path} ({file_size} bytes)")
        else:
            print(f"    ✗ {name}: 파일이 존재하지 않음")
    
    return all_saved
