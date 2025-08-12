# ==================================================================================
# 뉴스 호재/악재 예측 실행 스크립트 (백엔드용)
# 훈련된 모델로 실시간 뉴스 예측 수행
# ==================================================================================

import sys
import os
import json
from datetime import datetime

from .models.predict_news_scorer import NewsScorer

class NewsPredictor:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        try:
            self.model = NewsScorer.load_model()
            print("뉴스 예측 모델 로드 완료")
        except FileNotFoundError as e:
            print(f"모델 파일을 찾을 수 없습니다: {e}")
            raise
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise
    
    def predict_single_news(self, news_data):
        """단일 뉴스 예측"""
        if not self.model:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        try:
            result = self.model.predict_news_impact(news_data, verbose=False)
            return result
        except Exception as e:
            print(f"예측 실행 실패: {e}")
            raise
    
    def predict_batch_news(self, news_list):
        """배치 뉴스 예측"""
        results = []
        for i, news_data in enumerate(news_list):
            try:
                result = self.predict_single_news(news_data)
                result['news_index'] = i
                result['processed_at'] = datetime.now().isoformat()
                results.append(result)
            except Exception as e:
                error_result = {
                    'news_index': i,
                    'error': str(e),
                    'processed_at': datetime.now().isoformat(),
                    'impact_score': 5.0,  # 기본값
                    'direction_probability': 0.5,  # 중립
                    'prediction': 'NEUTRAL',
                    'confidence': 'LOW'
                }
                results.append(error_result)
        
        return results

def main():
    """메인 실행 함수"""
    if len(sys.argv) < 2:
        print("사용법: python run_prediction.py <news_json_file>")
        print("또는: python run_prediction.py --test")
        sys.exit(1)
    
    try:
        predictor = NewsPredictor()
        
        if sys.argv[1] == '--test':
            # 테스트 모드
            print("\n테스트 모드로 예측 실행")
            print("=" * 50)
            
            test_news = {
                'original_stock': '엔비디아',
                'news_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'content': 'NVIDIA가 새로운 AI 칩을 발표하며 시장 예상을 뛰어넘는 실적을 기록했다.',
                'positive': 0.8, 
                'negative': 0.1, 
                'neutral': 0.1, 
                'sentiment_score': 0.7
            }
            
            # FinBERT 임베딩 더미 데이터 추가
            for i in range(768):
                test_news[f'finbert_{i}'] = 0.001 * i
            
            result = predictor.predict_single_news(test_news)
            
            print(f"뉴스: {test_news['content'][:50]}...")
            print(f"영향도 점수: {result['impact_score']}/10")
            print(f"상승 확률: {result['direction_probability']:.1%}")
            print(f"예측: {result['prediction']} ({result['confidence']} 신뢰도)")
            print(f"예상 변동폭: {result['expected_magnitude']:.2%}")
            
        else:
            # 파일에서 뉴스 데이터 로드
            news_file = sys.argv[1]
            
            if not os.path.exists(news_file):
                print(f"파일을 찾을 수 없습니다: {news_file}")
                sys.exit(1)
            
            print(f"뉴스 데이터 파일 처리: {news_file}")
            
            with open(news_file, 'r', encoding='utf-8') as f:
                if news_file.endswith('.json'):
                    news_data = json.load(f)
                else:
                    print("JSON 파일만 지원됩니다.")
                    sys.exit(1)
            
            if isinstance(news_data, list):
                # 배치 처리
                print(f"{len(news_data)}개 뉴스 배치 처리 시작...")
                results = predictor.predict_batch_news(news_data)
                
                # 결과 저장
                output_file = news_file.replace('.json', '_predictions.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                print(f"예측 결과 저장 완료: {output_file}")
                
                # 요약 통계
                positive_count = sum(1 for r in results if r.get('prediction') == 'POSITIVE')
                negative_count = sum(1 for r in results if r.get('prediction') == 'NEGATIVE')
                neutral_count = sum(1 for r in results if r.get('prediction') == 'NEUTRAL')
                
                print(f"\n예측 요약:")
                print(f"  긍정적: {positive_count}개")
                print(f"  부정적: {negative_count}개")
                print(f"  중립적: {neutral_count}개")
                
            else:
                # 단일 처리
                result = predictor.predict_single_news(news_data)
                print(json.dumps(result, ensure_ascii=False, indent=2))
    
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()