# 뉴스 호재/악재 판별 시스템

월간 자동 뉴스 크롤링 및 AI 모델 재훈련을 통한 실시간 뉴스 분석 시스템

## 디렉토리 구조

```
sentiment/scripts/
├── data/                          # 데이터 파일
│   ├── news_full_features_robust.csv    # 뉴스 학습 데이터
│   └── sp500_korean_stocks_with_symbols.xlsx # 종목 매핑 파일
├── models/                        # AI 모델 파일
│   └── news_scorer_model.pkl      # 훈련된 뉴스 분석 모델
├── logs/                          # 로그 파일
├── backups/                       # 백업 파일
├── monthly_data_update.js         # 월간 뉴스 크롤링 스크립트
├── monthly_model_training.py      # 월간 모델 재훈련 스크립트
├── scheduler.py                   # 자동화 스케줄러
└── run_prediction.py             # 뉴스 예측 실행 스크립트 (백엔드용)
```

## 주요 기능

### 1. 월간 뉴스 데이터 수집
- 네이버 금융 뉴스에서 최신 데이터 크롤링
- MongoDB에 자동 저장
- 중복 데이터 제거 및 품질 관리

### 2. AI 모델 자동 재훈련
- 새로 수집된 데이터로 모델 성능 개선
- 백업 및 검증 시스템
- Optuna 기반 하이퍼파라미터 최적화

### 3. 실시간 뉴스 분석
- 뉴스 호재/악재 자동 판별
- 영향도 점수 (0-10점) 제공
- 주가 변동 확률 예측

## 사용 방법

### 설치
```bash
cd sentiment/scripts
npm install  # Node.js 의존성 설치
pip install -r requirements.txt  # Python 의존성 설치
```

### 개별 실행
```bash
# 뉴스 크롤링
node monthly_data_update.js

# 모델 훈련
python monthly_model_training.py

# 뉴스 예측 (테스트)
python run_prediction.py --test

# 뉴스 예측 (파일)
python run_prediction.py news_data.json
```

### 자동화 스케줄러
```bash
# 백그라운드 실행 (매월 자동 업데이트)
python scheduler.py &
```

## 백엔드 연동

### 뉴스 예측 API 사용 예시
```python
from run_prediction import NewsPredictor

# 모델 로드 (서버 시작시 1회)
predictor = NewsPredictor()

# 뉴스 분석
news_data = {
    'original_stock': '삼성전자',
    'news_date': '2025-01-15 09:30:00',
    'content': '삼성전자가 신규 반도체 공장 건설을 발표했다.',
    'positive': 0.7, 'negative': 0.1, 'neutral': 0.2,
    'sentiment_score': 0.6
}

result = predictor.predict_single_news(news_data)
print(f"영향도: {result['impact_score']}/10")
print(f"예측: {result['prediction']}")
```

### 결과 형식
```json
{
  "impact_score": 7.2,
  "direction_probability": 0.735,
  "expected_magnitude": 0.0234,
  "prediction": "POSITIVE",
  "confidence": "HIGH"
}
```

## 환경 변수

### MongoDB 연결
```bash
# .env 파일 또는 시스템 환경변수
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net
```

## 로그 모니터링

```bash
# 크롤링 로그
tail -f logs/monthly_crawling.log

# 훈련 로그  
tail -f logs/monthly_training.log

# 스케줄러 로그
tail -f logs/scheduler.log
```

## 주의사항

1. **데이터 품질**: 크롤링된 데이터의 품질을 정기적으로 확인
2. **모델 성능**: 월간 재훈련 후 성능 지표 모니터링
3. **저장 공간**: 백업 파일이 누적되므로 정기적으로 정리
4. **네트워크**: 크롤링 시 사이트 차단 방지를 위한 지연 시간 준수

## 문제 해결

### 크롤링 실패시
- 네트워크 연결 확인
- 사이트 접근 차단 여부 확인
- User-Agent 및 지연 시간 조정

### 모델 훈련 실패시
- 데이터 파일 존재 여부 확인
- 메모리 사용량 점검
- 로그 파일에서 상세 오류 확인

### 예측 오류시
- 모델 파일 존재 확인
- 입력 데이터 형식 검증
- 의존성 모듈 설치 상태 확인