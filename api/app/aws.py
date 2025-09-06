"""
SSSwing3 AWS S3 연동 모듈

이 모듈은 AWS S3와의 연동을 담당하며, 다음과 같은 기능을 제공합니다:
1. S3 presigned URL 생성 (파일 업로드용)
2. 프로 스윙 템플릿 목록 조회
3. S3 객체 다운로드
4. 공개 URL 생성

주요 특징:
- 환경 변수를 통한 설정 관리
- 개발 환경에서의 더미 URL 제공
- 에러 처리 및 로깅
- 자동 디렉토리 생성
"""

import boto3
from botocore.config import Config
import uuid
import logging
import os
from dotenv import load_dotenv

# .env 로드 (aws.py가 import될 때 즉시 환경변수를 준비)
load_dotenv()

# 로거 설정
logger = logging.getLogger(__name__)

# 환경변수에서 AWS 설정 로드 (dotenv는 main에서 로드)
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")           # AWS 리전 (기본값: 서울)
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "ssswing-videos")    # S3 버킷명
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")              # AWS 액세스 키
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")      # AWS 시크릿 키

# AWS 인증 정보가 있을 때만 S3 클라이언트 생성
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    # boto3 S3 클라이언트 생성
    s3 = boto3.client(
        's3',                                    # S3 서비스
        region_name=AWS_REGION,                  # 리전 설정
        aws_access_key_id=AWS_ACCESS_KEY_ID,     # 액세스 키
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,  # 시크릿 키
        config=Config(signature_version='s3v4')  # S3v4 서명 버전 사용
    )
else:
    # 인증 정보가 없는 경우 (개발 환경)
    s3 = None


def presign_put_url(filename: str, content_type: str, prefix: str = "uploads/"):
    """
    업로드를 위한 S3 presigned PUT URL 생성 함수
    
    Args:
        filename: 업로드할 파일명
        content_type: 파일의 MIME 타입
        prefix: S3 키 접두사 (기본값: "uploads/")
        
    Returns:
        dict: S3 키와 presigned URL을 포함한 딕셔너리
            - key: S3 객체 키
            - url: presigned PUT URL
            
    특징:
        - 고유한 파일명 생성 (UUID 기반)
        - 1시간 유효한 presigned URL
        - 개발 환경에서는 더미 URL 제공
    """
    # 고유한 S3 키 생성 (UUID + 원본 파일명)
    key = f"{prefix}{uuid.uuid4().hex}_{filename}"

    if not s3:
        # 개발환경(로컬)에서 AWS 키 없을 때 더미 URL 반환
        return {"key": key, "url": f"https://dummy-s3.com/{key}"}

    try:
        # S3 presigned PUT URL 생성
        url = s3.generate_presigned_url(
            'put_object',                                    # PUT 작업
            Params={
                'Bucket': AWS_S3_BUCKET,                     # 버킷명
                'Key': key,                                  # 객체 키
                'ContentType': content_type                  # 콘텐츠 타입
            },
            ExpiresIn=3600                                   # 1시간 유효
        )
        return {"key": key, "url": url}
    except Exception as e:
        logger.error(f"Presigned URL 생성 실패: {e}")
        # 에러 발생 시 더미 URL 반환
        return {"key": key, "url": f"https://dummy-s3.com/{key}"}


def public_url_for_key(key: str):
    """
    S3 공개 URL 생성 함수 (정적 퍼블릭 접근을 가정)
    
    Args:
        key: S3 객체 키
        
    Returns:
        str: S3 공개 URL 또는 더미 URL
        
    특징:
        - S3 버킷이 퍼블릭 읽기 권한을 가지고 있다고 가정
        - 개발 환경에서는 더미 URL 제공
    """
    if not s3:
        return f"https://dummy-s3.com/{key}"
    
    try:
        # S3 공개 URL 생성 (버킷이 퍼블릭 읽기 권한을 가진 경우)
        return f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
    except Exception as e:
        logger.error(f"공개 URL 생성 실패: {e}")
        return f"https://dummy-s3.com/{key}"


def get_pro_templates() -> dict:
    """
    S3의 pro/ 폴더에서 프로 스윙 템플릿 목록을 가져오는 함수
    
    Returns:
        dict: {template_id: s3_key} 형태의 딕셔너리
        
    특징:
        - pro/ 폴더 내의 비디오 파일들을 자동으로 스캔
        - 파일명을 기반으로 템플릿 ID 자동 생성
        - S3 연결 실패 시 기본 템플릿 제공
        - 지원 형식: MP4, MOV, MKV, AVI
    """
    logger.info("프로 템플릿 목록 가져오기 시작")

    if not s3:
        logger.warning("S3 클라이언트가 없습니다. 기본값을 반환합니다.")
        # 개발 환경용 기본 템플릿
        return {
            'pro_iron_side': 'pro/Adam Scott iron.mp4',
            'pro_driver_side': 'pro/Rory McIlroy driver.mp4',
        }

    try:
        # S3 pro/ 폴더 내 객체 목록 조회 (최대 200개)
        response = s3.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix='pro/', MaxKeys=200)
        templates: dict[str, str] = {}
        
        # 각 객체를 순회하며 템플릿 정보 추출
        for obj in response.get('Contents', []):
            key = obj['Key']
            
            # 폴더는 건너뛰기
            if key.endswith('/'):
                continue
                
            # 비디오 파일만 처리
            if not key.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
                continue
                
            # 파일명에서 템플릿 ID 생성
            filename = key.split('/')[-1]
            template_id = f"pro_{os.path.splitext(filename)[0].lower().replace(' ', '_')}"
            templates[template_id] = key
        
        # 템플릿이 없는 경우 기본 템플릿 제공
        if not templates:
            templates = {
                'pro_driver_side': 'pro/Rory McIlroy driver.mp4'
            }
        
        return templates
        
    except Exception as e:
        logger.error(f"프로 템플릿 목록 가져오기 실패: {e}")
        # 에러 발생 시 기본 템플릿 반환
        return {
            'pro_driver_side': 'pro/Rory McIlroy driver.mp4'
        }


def download_s3_object_to_path(key: str, dest_path: str) -> bool:
    """
    S3 객체를 로컬 경로로 다운로드하는 함수
    
    Args:
        key: S3 객체 키
        dest_path: 로컬 저장 경로
        
    Returns:
        bool: 다운로드 성공 여부
        
    특징:
        - 자동으로 디렉토리 생성
        - 파일 존재 여부로 성공 판단
        - 에러 발생 시 로깅
    """
    if not s3:
        return False
        
    try:
        bucket = AWS_S3_BUCKET
        
        # 목적지 디렉토리 자동 생성
        import os
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # S3에서 파일 다운로드
        s3.download_file(bucket, key, dest_path)
        
        # 다운로드 성공 여부 확인 (파일 존재 여부)
        return os.path.exists(dest_path)
        
    except Exception as e:
        logger.error(f"S3 다운로드 실패: {key} -> {dest_path}: {e}")
        return False


