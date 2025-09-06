"""
SSSwing3 Celery 애플리케이션 설정 모듈

이 모듈은 Celery 워커를 위한 설정을 제공합니다.
Redis를 브로커와 백엔드로 사용하여 비동기 작업 큐를 관리합니다.

주요 설정:
- Redis 연결 설정
- 워커 동시성 및 성능 최적화
- 태스크 라우팅 및 큐 설정
- 한국 시간대 설정
- 자동 태스크 발견
"""

from celery import Celery
import os
import sys

# Windows 멀티프로세싱 이슈 회피
# Windows에서 Celery 워커 실행 시 발생하는 멀티프로세싱 관련 문제를 해결
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')

# Windows 전용 추가 설정
import platform
if platform.system() == 'Windows':
    os.environ.setdefault('OBJC_DISABLE_INITIALIZE_FORK_SAFETY', 'YES')
    os.environ.setdefault('PYTHONPATH', os.pathsep.join(sys.path))
    # Windows 멀티프로세싱 문제 해결을 위한 추가 설정
    os.environ.setdefault('PYTHONHASHSEED', '0')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    # Windows에서 세마포어 권한 문제 해결
    os.environ.setdefault('PYTHONLEGACYWINDOWSSTDIO', '1')

# Redis 연결 URL 설정 (환경 변수에서 로드, 기본값: localhost:6381)
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6381/0')

# Celery 애플리케이션 인스턴스 생성
celery_app = Celery(
    'ssswing3',                    # 애플리케이션 이름
    broker=REDIS_URL,              # 메시지 브로커 (작업 큐)
    backend=REDIS_URL,             # 결과 백엔드 (작업 결과 저장)
)

# Celery 설정 업데이트
celery_app.conf.update(
    # Windows 호환성을 위한 워커 설정
    worker_pool='solo',            # Windows에서 안전한 워커 풀 사용
    worker_concurrency=1,          # 워커 동시성 (1개 프로세스)
    worker_prefetch_multiplier=1,  # 프리페치 배수 (메모리 사용량 제한)
    task_acks_late=True,           # 작업 완료 후 승인 (안정성 향상)
    worker_disable_rate_limits=True,  # 속도 제한 비활성화
    
    # Windows 멀티프로세싱 문제 해결
    worker_direct=True,            # 직접 모드로 워커 실행
    task_always_eager=False,       # 비동기 실행 유지
    
    # 기본 큐 및 라우팅 설정
    task_default_queue='default',      # 기본 큐 이름
    task_default_exchange='default',   # 기본 익스체인지
    task_default_routing_key='default',  # 기본 라우팅 키
    
    # 결과 백엔드 설정
    result_backend=REDIS_URL,      # 결과 저장소
    result_expires=3600,           # 결과 만료 시간 (1시간)
    
    # 직렬화 설정
    task_serializer='json',        # 태스크 직렬화 형식
    accept_content=['json'],       # 허용되는 콘텐츠 형식
    result_serializer='json',      # 결과 직렬화 형식
    
    # 시간대 및 지역 설정
    timezone='Asia/Seoul',         # 한국 시간대
    enable_utc=True,               # UTC 활성화
    
    # 워커 안정성 설정
    worker_pool_restarts=True,     # 워커 풀 재시작 허용
    worker_max_tasks_per_child=1,  # 워커당 최대 태스크 수 (메모리 누수 방지)
    
    # Windows 전용 추가 설정
    broker_connection_retry_on_startup=True,  # 시작 시 브로커 연결 재시도
    task_ignore_result=False,      # 결과 무시하지 않음
)

# 태스크 라우팅 설정
# 모든 app.tasks 모듈의 태스크를 'default' 큐로 라우팅
celery_app.conf.task_routes = {
    'app.tasks.*': {'queue': 'default'},
}

# 자동 태스크 발견
# app.tasks 패키지에서 Celery 태스크를 자동으로 발견하여 등록
celery_app.autodiscover_tasks(['app.tasks'])


