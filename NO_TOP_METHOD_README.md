# 백스윙 탑 무시 스윙 속도 비교 방법

## 개요

기존의 백스윙 탑(Top) 감지 방식의 불안정성을 해결하기 위해 **"임팩트 중심 2구간 비교"** 방법을 구현했습니다.

### 기존 방법의 문제점
- 백스윙 탑 감지가 뷰(정면/측면)에 따라 불안정
- 샤프트 각도나 CV 의존으로 인한 오감지
- 복잡한 후보 점수화 로직으로 인한 불확실성

### 새로운 방법의 장점
- **안정적인 임팩트 감지**: 손 속도 최대 + 손 높이 최소로 신뢰성 높음
- **뷰-불변**: 정면/측면 무관하게 동작
- **스윙 리듬의 핵심 포착**: 전체 스윙의 70-80%를 커버하는 구간1이 핵심

## 구간 정의

### 기존 방법 (3구간)
1. **구간1**: Address → Top (백스윙)
2. **구간2**: Top → Impact (다운스윙)  
3. **구간3**: Impact → Finish (팔로우스루)

### 새로운 방법 (2구간)
1. **구간1**: Address → Impact (전체 백스윙 + 다운스윙)
2. **구간2**: Impact → Finish (팔로우스루)

## 구현된 함수들

### 1. `detect_swing_phases_no_top()`
```python
# ssswing/swing_phase_detector.py에 추가됨
def detect_swing_phases_no_top(landmarks_data, total_frames=None, video_aspect_ratio=None, fps=30, video_path: str | None = None):
    """
    백스윙 탑을 무시한 스윙 단계 감지 함수
    
    Returns:
        dict: address/impact/finish 및 구간 정보
    """
```

### 2. `calc_speed_score_no_top()`
```python
# api/app/scorer.py에 추가됨
def calc_speed_score_no_top(b_address_s: float, b_impact_s: float, b_finish_s: float,
                           t_address_s: float, t_impact_s: float, t_finish_s: float,
                           frames_meta: dict | None = None,
                           fps_meta: dict | None = None) -> tuple[float, dict]:
    """
    백스윙 탑을 무시한 임팩트 중심 2구간 속도 점수 계산 함수
    """
```

### 3. `score_all()` (수정됨)
```python
# api/app/scorer.py에 수정됨
def score_all(baseline_path: str, target_path: str, use_no_top_method: bool = False) -> dict:
    """
    모든 점수를 계산하는 메인 함수
    
    Args:
        use_no_top_method: 백스윙 탑 무시 방법 사용 여부 (기본값: False)
    """
```

### 4. `score_all_no_top()` (편의 함수)
```python
# api/app/scorer.py에 추가됨
def score_all_no_top(baseline_path: str, target_path: str) -> dict:
    """
    백스윙 탑을 무시한 스윙 분석을 수행하는 편의 함수
    """
```

## 사용법

### 1. 기본 사용법 (기존 방법)
```python
from api.app.scorer import score_all

# 기존 방법 사용 (백스윙 탑 포함)
result = score_all("pro_swing.mp4", "user_swing.mp4")
# 또는 명시적으로
result = score_all("pro_swing.mp4", "user_swing.mp4", use_no_top_method=False)
```

### 2. 새로운 방법 사용
```python
from api.app.scorer import score_all, score_all_no_top

# 방법 1: score_all 함수에 옵션 전달
result = score_all("pro_swing.mp4", "user_swing.mp4", use_no_top_method=True)

# 방법 2: 편의 함수 사용
result = score_all_no_top("pro_swing.mp4", "user_swing.mp4")
```

### 3. 테스트 스크립트 사용
```bash
# 두 방법의 결과를 비교
python test_no_top_method.py pro_swing.mp4 user_swing.mp4
```

## 결과 해석

### 점수 범위
- **80+ 점**: 타이밍 우수 (프로 수준)
- **60-80 점**: 보통 수준
- **<60 점**: 속도 불균형 (예: 다운스윙 너무 빠름)

### 메타데이터 구조
```python
{
    "swing_speed_score": 85.2,      # 스윙 속도 점수
    "joint_angle_score": 78.5,      # 관절 각도 점수  
    "total_score": 81.85,           # 종합 점수
    "method_used": "no_top_impact_based",  # 사용된 방법
    "speed_meta": {
        "method": "no_top_impact_based",
        "segment_scores": {
            "s1": 88.1,  # 구간1: Address → Impact
            "s2": 82.3   # 구간2: Impact → Finish
        },
        "segment_descriptions": {
            "s1": "Address → Impact (전체 백스윙 + 다운스윙)",
            "s2": "Impact → Finish (팔로우스루)"
        },
        "times": {
            "baseline": {
                "address_s": 0.50,
                "impact_s": 1.20,
                "finish_s": 1.80,
                "d1_s": 0.70,  # Address → Impact 시간
                "d2_s": 0.60   # Impact → Finish 시간
            },
            "target": {
                "address_s": 0.45,
                "impact_s": 1.15,
                "finish_s": 1.75,
                "d1_s": 0.70,  # Address → Impact 시간
                "d2_s": 0.60   # Impact → Finish 시간
            }
        }
    }
}
```

## 기술적 특징

### 1. 안정적인 임팩트 감지
- 손 중심 속도 최대점 + 손 높이 최소점 결합
- 기존 `detect_impact()` 함수 재사용
- 폴백 로직: Address + 0.3초

### 2. 최소 간격 보장
- 구간 간 최소 0.2초 간격 보장
- 논리적 순서 검증 (Address < Impact < Finish)

### 3. 뷰-불변 설계
- MediaPipe 정규화 좌표 사용
- 손 중심 궤적 기반 분석
- 정면/측면 뷰 무관하게 동작

### 4. 점수 계산 공식
```python
def time_diff_score(d_base: float, d_target: float) -> float:
    base = max(0.2, float(d_base))  # 최소 0.2초 보장
    diff = abs(float(d_target) - base)
    return float(np.clip(100.0 * (1.0 - diff / base), 0.0, 100.0))

# 최종 점수 = (구간1 점수 + 구간2 점수) / 2
```

## 성능 비교

### 기존 방법 vs 새로운 방법
- **안정성**: 새로운 방법이 백스윙 탑 오감지 문제 해결
- **정확도**: 임팩트 중심으로 스윙 리듬의 핵심 포착
- **처리 속도**: 백스윙 탑 복잡한 로직 제거로 약간 향상
- **호환성**: 기존 코드와 완전 호환 (옵션으로 전환 가능)

## 주의사항

1. **최소 스윙 길이**: 0.5초 이상의 스윙이 필요
2. **임팩트 감지**: 손목이 가려지면 정확도 저하 가능
3. **FPS 의존성**: 낮은 FPS에서는 정확도 제한
4. **폴백 로직**: 임팩트 감지 실패 시 기본값 사용

## 향후 개선 방향

1. **구간1 세분화**: 손 속도 곡선의 최대 가속 지점 추가
2. **평균 속도 지표**: 변위/시간 기반 실제 속도 비교
3. **적응적 임계값**: 스윙 스타일에 따른 동적 조정
4. **머신러닝 보강**: 임팩트 감지 정확도 향상
