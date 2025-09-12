#!/usr/bin/env python3
"""
백스윙 탑 무시 방법 테스트 스크립트

이 스크립트는 새로 구현된 "임팩트 중심 2구간 비교" 방법을 테스트합니다.
기존 방법과 새로운 방법의 결과를 비교할 수 있습니다.

사용법:
    python test_no_top_method.py baseline_video.mp4 target_video.mp4

특징:
    - 백스윙 탑의 불안정성 제거
    - 임팩트 중심의 안정적인 구간 비교
    - Address → Impact → Finish 2구간 분석
"""

import sys
import os
import logging

# 프로젝트 루트를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_no_top_method(baseline_path: str, target_path: str):
    """
    백스윙 탑 무시 방법을 테스트하는 함수
    
    Args:
        baseline_path: 기준 영상 경로
        target_path: 타겟 영상 경로
    """
    try:
        # API 모듈에서 함수들 import
        from api.app.scorer import score_all, score_all_no_top
        
        logger.info("=" * 60)
        logger.info("백스윙 탑 무시 방법 테스트 시작")
        logger.info("=" * 60)
        
        # 파일 존재 확인
        if not os.path.exists(baseline_path):
            logger.error(f"기준 영상을 찾을 수 없습니다: {baseline_path}")
            return
            
        if not os.path.exists(target_path):
            logger.error(f"타겟 영상을 찾을 수 없습니다: {target_path}")
            return
        
        logger.info(f"기준 영상: {baseline_path}")
        logger.info(f"타겟 영상: {target_path}")
        logger.info("")
        
        # 1. 기존 방법 (백스윙 탑 포함) 테스트
        logger.info("1. 기존 방법 (백스윙 탑 포함) 분석 중...")
        try:
            result_traditional = score_all(baseline_path, target_path, use_no_top_method=False)
            logger.info("✓ 기존 방법 분석 완료")
            logger.info(f"  - 스윙 속도 점수: {result_traditional['swing_speed_score']}")
            logger.info(f"  - 관절 각도 점수: {result_traditional['joint_angle_score']}")
            logger.info(f"  - 종합 점수: {result_traditional['total_score']}")
            logger.info(f"  - 사용된 방법: {result_traditional['method_used']}")
        except Exception as e:
            logger.error(f"✗ 기존 방법 분석 실패: {e}")
            result_traditional = None
        
        logger.info("")
        
        # 2. 새로운 방법 (백스윙 탑 무시) 테스트
        logger.info("2. 새로운 방법 (백스윙 탑 무시) 분석 중...")
        try:
            # 이미지 저장까지 수행하려면 score_all에 옵션을 명시적으로 전달
            # (score_all_no_top은 기본적으로 이미지를 저장하지 않음)
            result_no_top = score_all(
                baseline_path,
                target_path,
                use_no_top_method=True,
                save_key_images=True,
            )
            logger.info("✓ 새로운 방법 분석 완료")
            logger.info(f"  - 스윙 속도 점수: {result_no_top['swing_speed_score']}")
            logger.info(f"  - 관절 각도 점수: {result_no_top['joint_angle_score']}")
            logger.info(f"  - 종합 점수: {result_no_top['total_score']}")
            logger.info(f"  - 사용된 방법: {result_no_top['method_used']}")
        except Exception as e:
            logger.error(f"✗ 새로운 방법 분석 실패: {e}")
            result_no_top = None
        
        logger.info("")
        
        # 3. 결과 비교
        if result_traditional and result_no_top:
            logger.info("3. 결과 비교:")
            logger.info("-" * 40)
            
            speed_diff = result_no_top['swing_speed_score'] - result_traditional['swing_speed_score']
            angle_diff = result_no_top['joint_angle_score'] - result_traditional['joint_angle_score']
            total_diff = result_no_top['total_score'] - result_traditional['total_score']
            
            logger.info(f"스윙 속도 점수 차이: {speed_diff:+.1f} ({result_traditional['swing_speed_score']} → {result_no_top['swing_speed_score']})")
            logger.info(f"관절 각도 점수 차이: {angle_diff:+.1f} ({result_traditional['joint_angle_score']} → {result_no_top['joint_angle_score']})")
            logger.info(f"종합 점수 차이: {total_diff:+.2f} ({result_traditional['total_score']} → {result_no_top['total_score']})")
            
            # 새로운 방법의 상세 정보 출력
            if 'speed_meta' in result_no_top and 'segment_scores' in result_no_top['speed_meta']:
                logger.info("")
                logger.info("새로운 방법 상세 정보:")
                logger.info(f"  - 구간1 점수 (Address → Impact): {result_no_top['speed_meta']['segment_scores']['s1']:.1f}")
                logger.info(f"  - 구간2 점수 (Impact → Finish): {result_no_top['speed_meta']['segment_scores']['s2']:.1f}")
                
                if 'times' in result_no_top['speed_meta']:
                    baseline_times = result_no_top['speed_meta']['times']['baseline']
                    target_times = result_no_top['speed_meta']['times']['target']
                    logger.info("")
                    logger.info("시간 분석 (초 단위):")
                    logger.info(f"  기준 영상: Address={baseline_times['address_s']:.2f}s, Impact={baseline_times['impact_s']:.2f}s, Finish={baseline_times['finish_s']:.2f}s")
                    logger.info(f"  타겟 영상: Address={target_times['address_s']:.2f}s, Impact={target_times['impact_s']:.2f}s, Finish={target_times['finish_s']:.2f}s")
                    logger.info(f"  구간1 시간: 기준={baseline_times['d1_s']:.2f}s, 타겟={target_times['d1_s']:.2f}s")
                    logger.info(f"  구간2 시간: 기준={baseline_times['d2_s']:.2f}s, 타겟={target_times['d2_s']:.2f}s")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("테스트 완료")
        logger.info("=" * 60)
        
    except ImportError as e:
        logger.error(f"모듈 import 실패: {e}")
        logger.error("프로젝트 구조를 확인하고 필요한 모듈이 설치되어 있는지 확인하세요.")
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 함수"""
    if len(sys.argv) != 3:
        print("사용법: python test_no_top_method.py <기준영상경로> <타겟영상경로>")
        print("예시: python test_no_top_method.py pro_swing.mp4 user_swing.mp4")
        sys.exit(1)
    
    baseline_path = sys.argv[1]
    target_path = sys.argv[2]
    
    test_no_top_method(baseline_path, target_path)


if __name__ == "__main__":
    main()
