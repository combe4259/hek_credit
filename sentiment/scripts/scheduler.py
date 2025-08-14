# ==================================================================================
# 자동화 스케줄러
# 매월 1일에 뉴스 수집 + 모델 재훈련 자동 실행
# ==================================================================================

import schedule
import time
import subprocess
import logging
from datetime import datetime
import os
import sys

# 로깅 설정
# logs 디렉토리 생성
os.makedirs('./logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/scheduler.log'),
    ]
)

logger = logging.getLogger(__name__)

def run_monthly_update():
    """매월 자동 업데이트 실행"""
    logger.info("매월 자동 업데이트 시작...")
    
    try:
        # 1. 뉴스 데이터 수집
        logger.info("뉴스 데이터 수집 중")
        news_result = subprocess.run([
            'node', './monthly_data_update.js'
        ], capture_output=True, text=True, timeout=3600)  # 1시간 타임아웃
        
        if news_result.returncode != 0:
            logger.error(f"뉴스 수집 실패: {news_result.stderr}")
            return False
            
        logger.info("뉴스 데이터 수집 완료")
        
        # 2. 모델 재훈련
        logger.info("모델 재훈련 중...")
        training_script = os.path.join(os.path.dirname(__file__), 'monthly_model_update.py')
        model_result = subprocess.run([
            sys.executable, training_script
        ], capture_output=True, text=True, timeout=7200)  # 2시간 타임아웃
        
        if model_result.returncode != 0:
            logger.error(f"모델 훈련 실패: {model_result.stderr}")
            return False
            
        logger.info("모델 재훈련 완료")
        
        logger.info("매월 자동 업데이트 성공!")
        return True
        
    except subprocess.TimeoutExpired as e:
        logger.error(f"타임아웃 발생: {e}")
        return False
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        return False

def main():
    """메인 스케줄러"""
    logger.info("자동화 스케줄러 시작")
    
    # 매월 1일 오전 2시에 실행
    schedule.every().month.do(run_monthly_update)
    
    # 무한 루프로 스케줄 실행
    while True:
        try:
            schedule.run_pending()
            time.sleep(3600)  # 1시간마다 체크
        except KeyboardInterrupt:
            logger.info("⏹스케줄러 종료")
            break
        except Exception as e:
            logger.error(f"스케줄러 오류: {e}")
            time.sleep(3600)  # 오류 발생 시에도 1시간 후 재시도

if __name__ == "__main__":
    main()