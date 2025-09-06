"""
SSSwing3 API 메인 애플리케이션

이 모듈은 골프 스윙 분석을 위한 FastAPI 기반 REST API를 제공합니다.
주요 기능:
1. 골프 스윙 영상 업로드 및 분석
2. 프로 스윙과 사용자 스윙 비교 분석
3. 분석 결과 다운로드 및 상태 조회
4. AWS S3 연동을 통한 파일 관리

API 구조:
- /videos/presign: S3 업로드용 presigned URL 생성
- /analysis/templates: 프로 스윙 템플릿 목록 조회
- /analysis/user-comparison: 사용자 간 스윙 비교 분석
- /analysis: 프로 스윙과 사용자 스윙 비교 분석
- /download/*: 분석 결과 및 입력 영상 다운로드
- /analysis/*: 분석 상태 및 결과 조회
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os
import sys
import uuid
import shutil
from pydantic import BaseModel
from dotenv import load_dotenv

# 프로젝트 루트를 파이썬 경로에 추가하여 `ssswing` 패키지 import 가능하게 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 필요한 모듈들 import
from ssswing.video_generator import generate_comparison_video
from .scorer import score_all
from .video_processor import detect_swing_key_frames, crop_video_from_frame
from .tasks import run_user_comparison
from .aws import presign_put_url, get_pro_templates, download_s3_object_to_path, public_url_for_key

# 환경 변수 로드
load_dotenv()

class Settings(BaseModel):
    """
    애플리케이션 설정을 관리하는 Pydantic 모델
    
    환경 변수에서 설정값을 로드하며, 기본값을 제공합니다.
    """
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-2")  # AWS 리전
    AWS_S3_BUCKET: str = os.getenv("AWS_S3_BUCKET", "ssswing-videos")  # S3 버킷명
    AWS_ACCESS_KEY_ID: str | None = os.getenv("AWS_ACCESS_KEY_ID")  # AWS 액세스 키
    AWS_SECRET_ACCESS_KEY: str | None = os.getenv("AWS_SECRET_ACCESS_KEY")  # AWS 시크릿 키
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6381/0")  # Redis 연결 URL
    TMP_DIR: str = os.getenv("TMP_DIR", os.path.join(PROJECT_ROOT, "temp_uploads"))  # 임시 파일 디렉토리


# 설정 인스턴스 생성
settings = Settings()

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(title="ssswing3 API", version="0.1.0")

# CORS 미들웨어 추가 (크로스 오리진 요청 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용
    allow_credentials=True,  # 쿠키 및 인증 헤더 허용
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)


@app.get("/")
def root():
    """
    루트 엔드포인트 - API 상태 확인용
    
    Returns:
        dict: API 서비스 정보
    """
    return {"ok": True, "service": "ssswing3", "version": "0.1.0"}


@app.post("/videos/presign")
async def presign_video(
    request: Request,
    filename: str | None = Form(None),
    contentType: str | None = Form(None),
):
    """
    S3 업로드용 presigned URL 생성 엔드포인트
    
    JSON 또는 Form 데이터 모두 지원하여 클라이언트의 요청 방식을 유연하게 처리합니다.
    
    Args:
        request: FastAPI Request 객체
        filename: 업로드할 파일명 (Form 파라미터)
        contentType: 파일의 MIME 타입 (Form 파라미터)
        
    Returns:
        dict: S3 presigned URL 정보
        
    Raises:
        HTTPException: 필수 파라미터가 누락된 경우 또는 서버 오류
    """
    try:
        # Content-Type에 따라 요청 데이터 파싱 방식 결정
        ct = request.headers.get("content-type", "")
        if ct.startswith("application/json"):
            # JSON 요청인 경우
            data = await request.json()
            filename = data.get("filename")
            contentType = data.get("contentType")
        # form-data일 경우, FastAPI가 위의 Form 파라미터로 채움
        
        # 필수 파라미터 검증
        if not filename or not contentType:
            raise HTTPException(status_code=422, detail="filename/contentType가 필요합니다")
        
        # S3 presigned URL 생성
        return presign_put_url(filename=filename, content_type=contentType)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/templates")
def list_templates():
    """
    S3의 pro/ 폴더에 있는 프로 스윙 템플릿 목록을 반환하는 엔드포인트
    
    Returns:
        dict: 템플릿 목록과 개수
            - templates: 템플릿 정보 리스트
            - count: 템플릿 총 개수
    """
    # S3에서 프로 템플릿 목록 가져오기
    templates = get_pro_templates()
    
    # 프런트엔드와 동일한 스키마로 변환
    template_list = []
    for template_id, s3_key in templates.items():
        filename = s3_key.split('/')[-1]  # 파일명 추출
        display = os.path.splitext(filename)[0]  # 확장자 제거
        template_list.append({
            "id": template_id,
            "name": display,
            "s3Key": s3_key,
            "description": f"{display} 스윙 템플릿",
        })
    
    return {"templates": template_list, "count": len(template_list)}


@app.post("/analysis/user-comparison")
async def user_comparison(video1: UploadFile = File(...), video2: UploadFile = File(...)):
    """
    사용자 간 스윙 비교 분석 엔드포인트
    
    두 개의 사용자 영상을 업로드받아 비교 분석을 수행합니다.
    
    Args:
        video1: 첫 번째 사용자 영상 파일
        video2: 두 번째 사용자 영상 파일
        
    Returns:
        dict: 분석 ID (프런트엔드에서 폴링용)
        
    Raises:
        HTTPException: 파일 업로드 또는 처리 중 오류 발생 시
    """
    try:
        # 고유한 분석 ID 생성
        an_id = str(uuid.uuid4())
        temp_dir = os.path.join(settings.TMP_DIR, an_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 업로드된 영상 파일 저장
        v1 = os.path.join(temp_dir, f"video1_{an_id}.mp4")
        v2 = os.path.join(temp_dir, f"video2_{an_id}.mp4")
        
        with open(v1, "wb") as f:
            shutil.copyfileobj(video1.file, f)
        with open(v2, "wb") as f:
            shutil.copyfileobj(video2.file, f)

        # Celery 비동기 태스크 큐잉 (백그라운드에서 분석 수행)
        run_user_comparison.delay(v1, v2, PROJECT_ROOT)

        # 초기 응답은 analysisId만 반환 (프런트엔드에서 폴링하여 진행상황 확인)
        return {"analysisId": an_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analysis")
async def create_pro_comparison(
    request: Request,
    userVideo: UploadFile | None = File(None),
    proTemplateId: str | None = Form(None),
):
    """
    프로 비교 분석 엔드포인트 (통합)
    
    두 가지 방식으로 요청을 처리합니다:
    1. 멀티파트(파일 업로드): 사용자가 올린 파일과 선택한 프로 템플릿으로 바로 진행
    2. JSON: 기존 방식(userVideoKey, proTemplateId)으로 S3에서 다운로드 후 진행
    
    Args:
        request: FastAPI Request 객체
        userVideo: 사용자 영상 파일 (멀티파트 요청 시)
        proTemplateId: 프로 템플릿 ID
        
    Returns:
        dict: 분석 ID
        
    Raises:
        HTTPException: 필수 파라미터 누락 또는 처리 오류 시
    """
    try:
        # 고유한 분석 ID 생성
        an_id = str(uuid.uuid4())
        temp_dir = os.path.join(settings.TMP_DIR, an_id)
        os.makedirs(temp_dir, exist_ok=True)

        content_type = request.headers.get("content-type", "")

        # 1) 멀티파트 업로드 경로 (권장)
        if content_type.startswith("multipart/form-data"):
            if not userVideo or not proTemplateId:
                raise HTTPException(status_code=422, detail="userVideo 파일과 proTemplateId가 필요합니다")

            # 사용자 영상 파일 저장
            user_video_path = os.path.join(temp_dir, f"user_{an_id}.mp4")
            with open(user_video_path, "wb") as f:
                shutil.copyfileobj(userVideo.file, f)

            # 프로 템플릿 다운로드
            templates = get_pro_templates()
            pro_key = templates.get(proTemplateId)
            if not pro_key:
                raise HTTPException(status_code=404, detail="프로 템플릿을 찾을 수 없습니다")

            pro_video_path = os.path.join(temp_dir, f"pro_{an_id}.mp4")
            ok_pro = download_s3_object_to_path(pro_key, pro_video_path)
            if not ok_pro or not os.path.exists(pro_video_path):
                raise HTTPException(status_code=502, detail="프로 템플릿 S3 다운로드에 실패했습니다")

            # 메타데이터 저장 (템플릿 키 기록)
            try:
                import json
                with open(os.path.join(temp_dir, "meta.json"), "w", encoding="utf-8") as mf:
                    json.dump({"userKey": None, "proKey": pro_key}, mf, ensure_ascii=False)
            except Exception:
                pass

            # Celery 비동기 태스크 실행
            run_user_comparison.delay(user_video_path, pro_video_path, PROJECT_ROOT)
            return {"analysisId": an_id}

        # 2) JSON 경로 (하위호환)
        try:
            body = await request.json()
        except Exception:
            body = {}

        # JSON에서 필요한 파라미터 추출
        user_key = body.get("userVideoKey")
        pro_template_id = body.get("proTemplateId")
        if not user_key or not pro_template_id:
            raise HTTPException(status_code=422, detail="userVideoKey/proTemplateId가 필요합니다")

        # 사용자 영상 S3에서 다운로드
        user_video_path = os.path.join(temp_dir, f"user_{an_id}.mp4")
        ok_user = download_s3_object_to_path(user_key, user_video_path)
        if not ok_user or not os.path.exists(user_video_path):
            raise HTTPException(status_code=502, detail="사용자 영상 S3 다운로드에 실패했습니다")

        # 프로 템플릿 S3에서 다운로드
        templates = get_pro_templates()
        pro_key = templates.get(pro_template_id)
        if not pro_key:
            raise HTTPException(status_code=404, detail="프로 템플릿을 찾을 수 없습니다")

        pro_video_path = os.path.join(temp_dir, f"pro_{an_id}.mp4")
        ok_pro = download_s3_object_to_path(pro_key, pro_video_path)
        if not ok_pro or not os.path.exists(pro_video_path):
            raise HTTPException(status_code=502, detail="프로 템플릿 S3 다운로드에 실패했습니다")

        # 메타데이터 저장
        try:
            import json
            with open(os.path.join(temp_dir, "meta.json"), "w", encoding="utf-8") as mf:
                json.dump({"userKey": user_key, "proKey": pro_key}, mf, ensure_ascii=False)
        except Exception:
            pass

        # Celery 비동기 태스크 실행
        run_user_comparison.delay(user_video_path, pro_video_path, PROJECT_ROOT)
        return {"analysisId": an_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/combined/{analysis_id}")
def download_combined(analysis_id: str):
    """
    합성 영상 다운로드 엔드포인트 (MP4)
    
    Args:
        analysis_id: 분석 ID
        
    Returns:
        FileResponse: 합성 영상 파일
        
    Raises:
        HTTPException: 파일이 존재하지 않는 경우
    """
    out_dir = os.path.join(PROJECT_ROOT, "ssswing", "mp4")
    p = os.path.join(out_dir, f"combined_swing_{analysis_id}.mp4")
    
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="합성 영상이 없습니다.")
    
    # inline으로 전송하여 <video> 태그에서 재생되도록 설정 + 캐시 방지
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
    return FileResponse(
        p,
        media_type="video/mp4",
        filename="combined_swing.mp4",
        content_disposition_type="inline",
        headers=headers,
    )


@app.get("/download/combined-webm/{analysis_id}")
def download_combined_webm(analysis_id: str):
    """
    합성 영상 다운로드 엔드포인트 (WebM)
    
    Args:
        analysis_id: 분석 ID
        
    Returns:
        FileResponse: 합성 WebM 영상 파일
        
    Raises:
        HTTPException: 파일이 존재하지 않는 경우
    """
    out_dir = os.path.join(PROJECT_ROOT, "ssswing", "mp4")
    p = os.path.join(out_dir, f"combined_swing_{analysis_id}.webm")
    
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="합성 WebM 영상이 없습니다.")
    
    # 캐시 방지 헤더 설정
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
    return FileResponse(
        p,
        media_type="video/webm",
        filename="combined_swing.webm",
        content_disposition_type="inline",
        headers=headers,
    )


@app.get("/download/input/{analysis_id}/user")
def download_input_user(analysis_id: str):
    """
    사용자 입력 영상 다운로드 엔드포인트
    
    Args:
        analysis_id: 분석 ID
        
    Returns:
        FileResponse: 사용자 입력 영상 파일
        
    Raises:
        HTTPException: 파일이 존재하지 않는 경우
    """
    base = os.path.join(settings.TMP_DIR, analysis_id)
    # 가능한 파일명 후보들
    candidates = [
        os.path.join(base, f"user_{analysis_id}.mp4"),
        os.path.join(base, f"video1_{analysis_id}.mp4"),
    ]
    
    # 존재하는 파일 찾기
    p = next((c for c in candidates if os.path.exists(c)), None)
    if not p:
        raise HTTPException(status_code=404, detail="사용자 입력 영상을 찾을 수 없습니다.")
    
    # 캐시 방지 헤더 설정
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
    return FileResponse(
        p, 
        media_type="video/mp4", 
        filename=f"user_{analysis_id}.mp4", 
        content_disposition_type="inline", 
        headers=headers
    )


@app.get("/download/input/{analysis_id}/pro")
def download_input_pro(analysis_id: str):
    """
    프로 입력 영상 다운로드 엔드포인트
    
    Args:
        analysis_id: 분석 ID
        
    Returns:
        FileResponse: 프로 입력 영상 파일
        
    Raises:
        HTTPException: 파일이 존재하지 않는 경우
    """
    base = os.path.join(settings.TMP_DIR, analysis_id)
    # 가능한 파일명 후보들
    candidates = [
        os.path.join(base, f"pro_{analysis_id}.mp4"),
        os.path.join(base, f"video2_{analysis_id}.mp4"),
    ]
    
    # 존재하는 파일 찾기
    p = next((c for c in candidates if os.path.exists(c)), None)
    if not p:
        raise HTTPException(status_code=404, detail="프로 입력 영상을 찾을 수 없습니다.")
    
    # 캐시 방지 헤더 설정
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
    return FileResponse(
        p, 
        media_type="video/mp4", 
        filename=f"pro_{analysis_id}.mp4", 
        content_disposition_type="inline", 
        headers=headers
    )


@app.get("/analysis/user-comparison/{analysis_id}")
def get_user_comparison(analysis_id: str):
    """
    사용자 비교 분석 결과 조회 엔드포인트 (프런트엔드 폴링용)
    
    temp_uploads/{id}/result.json을 읽어 분석 결과를 반환합니다.
    
    Args:
        analysis_id: 분석 ID
        
    Returns:
        dict: 분석 상태 및 결과
            - status: "processing", "completed", "failed" 중 하나
            - result: 완료된 경우 분석 결과 데이터
            - error: 실패한 경우 오류 메시지
    """
    import json
    temp_dir = os.path.join(PROJECT_ROOT, "temp_uploads", analysis_id)
    result_json_path = os.path.join(temp_dir, "result.json")
    
    # 분석 디렉토리가 존재하지 않는 경우
    if not os.path.exists(temp_dir):
        return {"status": "processing", "message": "분석 준비 중"}
    
    # 결과 파일이 아직 생성되지 않은 경우
    if not os.path.exists(result_json_path):
        # 캐시 방지를 위해 파일시스템 타임스탬프와 디렉터리 존재만 반환
        return {"status": "processing", "dir": temp_dir}
    
    try:
        # 결과 파일 읽기
        with open(result_json_path, "r", encoding="utf-8") as rf:
            data = json.load(rf)
        return {"status": "completed", "result": data}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@app.get("/analysis/{analysis_id}")
def get_analysis(analysis_id: str, request: Request):
    """
    프로 비교 분석 상태/결과 조회 엔드포인트 (프런트엔드 공통 포맷으로 반환)
    
    Args:
        analysis_id: 분석 ID
        request: FastAPI Request 객체
        
    Returns:
        dict: 분석 상태 및 결과 (프런트엔드 공통 스키마)
    """
    import json
    temp_dir = os.path.join(settings.TMP_DIR, analysis_id)
    result_json_path = os.path.join(temp_dir, "result.json")
    
    # 분석 디렉토리가 존재하지 않는 경우
    if not os.path.exists(temp_dir):
        return {"status": "processing", "analysisId": analysis_id}
    
    # 결과 파일이 아직 생성되지 않은 경우 (processing 단계)
    if not os.path.exists(result_json_path):
        # processing 단계에서는 S3 공개 URL을 우선 제공하여 404를 방지
        assets = None
        try:
            import json
            # 메타데이터 파일에서 S3 키 정보 읽기
            with open(os.path.join(settings.TMP_DIR, analysis_id, "meta.json"), "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            user_key = meta.get("userKey")
            pro_key = meta.get("proKey")
            
            # S3 공개 URL 생성
            user_url = public_url_for_key(user_key) if user_key else None
            pro_url = public_url_for_key(pro_key) if pro_key else None
            if user_url or pro_url:
                assets = {"userUrl": user_url, "proUrl": pro_url}
        except Exception:
            # 메타가 없으면 로컬 파일이 존재하는 경우에만 로컬 URL 제공
            user_in = os.path.join(settings.TMP_DIR, analysis_id, f"user_{analysis_id}.mp4")
            pro_in = os.path.join(settings.TMP_DIR, analysis_id, f"pro_{analysis_id}.mp4")
            if os.path.exists(user_in) or os.path.exists(pro_in):
                base = str(request.base_url).rstrip('/')
                assets = {
                    "userUrl": f"{base}/download/input/{analysis_id}/user" if os.path.exists(user_in) else None,
                    "proUrl": f"{base}/download/input/{analysis_id}/pro" if os.path.exists(pro_in) else None,
                }
        
        # 자산 URL을 전혀 만들 수 없으면 에러 반환
        if not assets or (not assets.get("userUrl") and not assets.get("proUrl")):
            return {"status": "error", "analysisId": analysis_id, "error": "입력 영상 URL을 준비하지 못했습니다. 템플릿/업로드 키를 확인하세요."}
        
        return {"status": "processing", "analysisId": analysis_id, "assets": assets}
    
    try:
        # 결과 파일 읽기
        with open(result_json_path, "r", encoding="utf-8") as rf:
            data = json.load(rf)
        
        # ssswing3 태스크 스키마 → 프런트엔드 공통 스키마로 매핑
        raw_scores = data.get("scores", {})
        speed = raw_scores.get("speed_score") or raw_scores.get("speed") or 0
        angles = raw_scores.get("angle_score") or raw_scores.get("angles") or 0
        final = raw_scores.get("final_score") or raw_scores.get("final") or 0
        
        # 입력 영상 URL (로컬 파일 서빙 엔드포인트 제공)
        base = str(request.base_url).rstrip('/')
        assets = {
            "userUrl": f"{base}/download/input/{analysis_id}/user",
            "proUrl": f"{base}/download/input/{analysis_id}/pro",
        }

        # 프런트엔드 공통 스키마로 매핑
        mapped = {
            "status": "done",
            "analysisId": analysis_id,
            "scores": {
                "speed": speed,
                "angles": angles,
                "final": final,
                "overall": final,
            },
            "assets": assets,
            "userPoses": data.get("userPoses"),
            "proPoses": data.get("comparisonPoses") or data.get("proPoses"),
            "comparisonVideoUrl": (data.get("comparisonVideoUrl") if (isinstance(data.get("comparisonVideoUrl"), str) and data.get("comparisonVideoUrl").startswith("http")) else f"{base}{data.get('comparisonVideoUrl', '')}"),
            "speedMeta": data.get("speedMeta"),
        }
        return mapped
    except Exception as e:
        return {"status": "error", "analysisId": analysis_id, "error": str(e)}


@app.get("/analysis/{analysis_id}/progress")
def get_analysis_progress(analysis_id: str):
    """
    분석 진행 상황 조회 엔드포인트
    
    Args:
        analysis_id: 분석 ID
        
    Returns:
        dict: 분석 진행 상황 정보
    """
    import json
    temp_dir = os.path.join(settings.TMP_DIR, analysis_id)
    progress_file = os.path.join(temp_dir, "progress.json")
    
    # 분석 디렉토리가 존재하지 않는 경우
    if not os.path.exists(temp_dir):
        return {"status": "not_found", "analysisId": analysis_id}
    
    # 진행 상황 파일이 존재하지 않는 경우
    if not os.path.exists(progress_file):
        return {"status": "processing", "analysisId": analysis_id, "progress": 0, "current_step": "분석 시작"}
    
    try:
        # 진행 상황 파일 읽기
        with open(progress_file, "r", encoding="utf-8") as pf:
            progress_data = json.load(pf)
        
        # 경과 시간 계산
        if "start_time" in progress_data:
            from datetime import datetime
            start_time = datetime.fromisoformat(progress_data["start_time"])
            elapsed = (datetime.now() - start_time).total_seconds()
            progress_data["elapsed_seconds"] = round(elapsed, 1)
        
        return progress_data
        
    except Exception as e:
        return {"status": "error", "analysisId": analysis_id, "error": str(e)}

