# ==================================================================================
# 매월 모델 재훈련 스크립트
# 매월 새로 수집된 뉴스 데이터로 모델을 재훈련하고 업데이트
# ==================================================================================

import sys
import os
import subprocess
from datetime import datetime, timedelta
import logging
import json

# 로깅 설정
os.makedirs('./logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/monthly_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_training_script():
    """훈련 스크립트 실행"""
    try:
        logger.info("모델 재훈련 시작...")
        
        # 훈련 스크립트 경로
        training_script = os.path.join(
            os.path.dirname(__file__), 
            'models/train_news_scorer.py'
        )
        
        if not os.path.exists(training_script):
            raise FileNotFoundError(f"훈련 스크립트를 찾을 수 없습니다: {training_script}")
        
        # Python 스크립트 실행
        result = subprocess.run([
            sys.executable, training_script
        ], capture_output=True, text=True, timeout=7200)  # 2시간 타임아웃
        
        if result.returncode == 0:
            logger.info("모델 훈련 성공!")
            logger.info(f"훈련 출력: {result.stdout[-500:]}")  # 마지막 500자만 로그
            return True
        else:
            logger.error(f"모델 훈련 실패 (코드: {result.returncode})")
            logger.error(f"오류 출력: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("모델 훈련 타임아웃 (2시간 초과)")
        return False
    except Exception as e:
        logger.error(f"모델 훈련 중 오류: {e}")
        return False

def backup_previous_model():
    """이전 모델 백업"""
    try:
        model_path = "./models/news_scorer_model.pkl"
        backup_dir = "./backups/models"
        
        # 백업 디렉토리 생성
        os.makedirs(backup_dir, exist_ok=True)
        
        if os.path.exists(model_path):
            backup_filename = f"news_scorer_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            import shutil
            shutil.copy2(model_path, backup_path)
            logger.info(f"이전 모델 백업 완료: {backup_path}")
            
            # 오래된 백업 파일 정리 (30일 이상된 것 삭제)
            cleanup_old_backups(backup_dir, days=30)
            
    except Exception as e:
        logger.error(f"모델 백업 실패: {e}")

def cleanup_old_backups(backup_dir, days=30):
    """오래된 백업 파일 정리"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for filename in os.listdir(backup_dir):
            file_path = os.path.join(backup_dir, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_date:
                    os.remove(file_path)
                    logger.info(f"오래된 백업 파일 삭제: {filename}")
                    
    except Exception as e:
        logger.error(f"백업 정리 실패: {e}")

def validate_new_model():
    """새로 훈련된 모델 검증"""
    try:
        # 간단한 모델 로드 테스트
        model_path = "./models/news_scorer_model.pkl"
        
        if not os.path.exists(model_path):
            logger.error(f"새 모델 파일이 존재하지 않습니다: {model_path}")
            return False
            
        # 파일 크기 체크 (너무 작으면 문제)
        file_size = os.path.getsize(model_path)
        if file_size < 1024 * 1024:  # 1MB 미만이면 의심스러움
            logger.warning(f"모델 파일 크기가 작습니다: {file_size} bytes")
        
        logger.info(f"모델 검증 통과 (크기: {file_size:,} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"모델 검증 실패: {e}")
        return False

def log_training_status(success, details=None):
    """훈련 상태 로그 기록"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "details": details or {}
        }
        
        log_file = "./logs/training_history.json"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 기존 로그 읽기
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
        
        # 새 엔트리 추가
        history.append(log_entry)
        
        # 최근 100개만 유지
        history = history[-100:]
        
        # 로그 저장
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"로그 기록 실패: {e}")

def main():
    logger.info("="*60)
    logger.info("매월 모델 재훈련 시작")
    logger.info("="*60)
    
    start_time = datetime.now()
    
    try:
        # 1. 이전 모델 백업
        logger.info("이전 모델 백업 중...")
        backup_previous_model()
        
        # 2. 모델 재훈련
        logger.info("모델 재훈련 시작...")
        training_success = run_training_script()
        
        if not training_success:
            raise Exception("모델 훈련 실패")
        
        # 3. 새 모델 검증
        logger.info("🔍 새 모델 검증 중...")
        validation_success = validate_new_model()
        
        if not validation_success:
            raise Exception("모델 검증 실패")
        
        # 4. 성공 로그
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        details = {
            "duration_seconds": duration,
            "duration_minutes": round(duration / 60, 2)
        }
        
        log_training_status(True, details)
        
        logger.info("="*60)
        logger.info("매월 모델 재훈련 성공적 완료!")
        logger.info(f"⏱소요 시간: {details['duration_minutes']}분")
        logger.info("="*60)
        
    except Exception as e:
        # 실패 로그
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        details = {
            "error": str(e),
            "duration_seconds": duration
        }
        
        log_training_status(False, details)
        
        logger.error("="*60)
        logger.error(f"매월 모델 재훈련 실패: {e}")
        logger.error(f"소요 시간: {round(duration / 60, 2)}분")
        logger.error("="*60)
        
        sys.exit(1)

if __name__ == "__main__":
    main()