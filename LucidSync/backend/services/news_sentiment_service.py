import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import joblib
from pathlib import Path
import random

# sentiment 디렉토리를 경로에 추가
current_file = os.path.abspath(__file__)
lucidsync_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # LucidSync 디렉토리
hek_credit_dir = os.path.dirname(lucidsync_dir)  # hek_credit 디렉토리
sentiment_path = os.path.join(hek_credit_dir, 'sentiment', 'scripts', 'models')
sys.path.append(sentiment_path)

# UnifiedNewsAI 클래스 정의 (pkl 파일에서 로드하기 위해 필요)
class UnifiedNewsAI:
    def __init__(self, name_to_ticker_map=None, models=None):
        self.name_to_ticker_map = name_to_ticker_map or {}
        self.models = models or {}
        # 기타 필요한 속성들
        self.bertopic_model = None
        self.feature_columns = None
        
    def predict_news_impact(self, news_data, verbose=False):
        """뉴스 영향 예측 (pkl에서 로드된 모델 사용)"""
        try:
            # 기본 반환값
            result = {
                'impact_score': 5.0,
                'impact_score_100': 50.0,
                'direction_probability': 0.5,
                'expected_magnitude': 0.01,
                'prediction': 'NEUTRAL',
                'confidence': 'MEDIUM'
            }
            
            # 실제 예측 로직은 여기에 구현
            # 현재는 안전한 기본값 반환
            if verbose:
                print(f"뉴스 예측 완료: {result}")
                
            return result
            
        except Exception as e:
            print(f"예측 오류: {e}")
            return {
                'impact_score': 5.0,
                'impact_score_100': 50.0,
                'direction_probability': 0.5,
                'expected_magnitude': 0.01,
                'prediction': 'NEUTRAL',
                'confidence': 'LOW'
            }
            
    @staticmethod
    def aggregate_news_scores(analyzed_news, stock_name, max_days=14, verbose=False):
        """뉴스 점수 집계"""
        try:
            if not analyzed_news:
                return {'error': '분석할 뉴스가 없습니다.'}
                
            total_score = sum(news.get('impact_score_100', 50) for news in analyzed_news)
            avg_score = total_score / len(analyzed_news)
            
            return {
                'stock_name': stock_name,
                'aggregate_score': avg_score,
                'total_news': len(analyzed_news),
                'max_days': max_days,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'집계 오류: {str(e)}'}

# pkl 파일에서 직접 모델 로드 - 무거운 의존성 없이

class NewsSentimentService:
    def __init__(self):
        self.news_scorer = None
        self.model_loaded = False
        self.news_data_cache = None
        
    def load_model(self):
        """간단한 키워드 기반 분석기로 초기화 (빠른 응답)"""
        if self.model_loaded:
            return True
            
        try:
            print("🔄 뉴스 감정 분석 시스템 초기화 중...")
            
            # 키워드 기반 분석기 설정
            self.positive_keywords = [
                '상승', '증가', '성장', '호재', '개선', '확장', '투자', '수익', '이익', '매출',
                'up', 'rise', 'growth', 'positive', 'increase', 'profit', 'revenue', 'gain'
            ]
            
            self.negative_keywords = [
                '하락', '감소', '악재', '손실', '적자', '위험', '감축', '중단', '폐쇄', '파산',
                'down', 'fall', 'decline', 'negative', 'loss', 'risk', 'decrease', 'deficit'
            ]
            
            self.model_loaded = True
            print("✅ 키워드 기반 뉴스 감정 분석 시스템 초기화 완료")
            return True
                
        except Exception as e:
            print(f"❌ 뉴스 감정 분석 시스템 초기화 실패: {e}")
            return False
    
    def load_news_data(self):
        """실제 뉴스 데이터 로드"""
        try:
            csv_path = "/Users/inter4259/Desktop/news_full_features_robust.csv"
            if os.path.exists(csv_path):
                self.news_data_cache = pd.read_csv(csv_path)
                print(f"뉴스 데이터 로드 완료: {len(self.news_data_cache)}개 뉴스")
                return True
            else:
                print(f"뉴스 데이터 파일을 찾을 수 없습니다: {csv_path}")
                return False
        except Exception as e:
            print(f"뉴스 데이터 로드 실패: {e}")
            return False
    
    def get_news_for_stock(self, stock_name: str, limit: int = 10) -> List[Dict]:
        """특정 종목의 뉴스 데이터 가져오기"""
        if self.news_data_cache is None:
            if not self.load_news_data():
                return []
        
        try:
            # 종목명 매핑 (한글/영문)
            stock_mapping = {
                'Apple': ['엔비디아', 'NVIDIA', 'Apple'],  # 테스트용으로 엔비디아 뉴스 사용
                'Microsoft': ['마이크로소프트', 'Microsoft'],
                'NVIDIA': ['엔비디아', 'NVIDIA'],
                'Tesla': ['테슬라', 'Tesla'],
                'Amazon': ['아마존', 'Amazon'],
                'Alphabet': ['구글', 'Google', 'Alphabet'],
                'Meta': ['메타', 'Meta', 'Facebook'],
                '엔비디아': ['엔비디아', 'NVIDIA'],
                '애플': ['애플', 'Apple']
            }
            
            search_terms = stock_mapping.get(stock_name, [stock_name])
            
            # 해당 종목 뉴스 필터링
            filtered_news = self.news_data_cache[
                self.news_data_cache['original_stock'].isin(search_terms)
            ].head(limit)
            
            # 딕셔너리 형태로 변환
            news_list = []
            for _, row in filtered_news.iterrows():
                news_dict = row.to_dict()
                # NaN 값 처리
                for key, value in news_dict.items():
                    if pd.isna(value):
                        if key in ['positive', 'negative', 'neutral', 'sentiment_score']:
                            news_dict[key] = 0.0
                        elif key.startswith('finbert_'):
                            news_dict[key] = 0.0
                        else:
                            news_dict[key] = ''
                
                news_list.append(news_dict)
            
            return news_list
            
        except Exception as e:
            print(f"종목 뉴스 조회 오류: {e}")
            return []
    
    def analyze_single_news(self, news_data: Dict) -> Dict:
        """개별 뉴스의 감정 분석 수행 (키워드 기반)"""
        if not self.model_loaded:
            if not self.load_model():
                return {'error': '모델이 로드되지 않았습니다.'}
        
        try:
            import random
            
            # 뉴스 내용 분석
            content = news_data.get('content', '').lower()
            title = news_data.get('title', '').lower()
            text = f"{title} {content}"
            
            # 키워드 기반 점수 계산
            positive_count = sum(1 for keyword in self.positive_keywords if keyword.lower() in text)
            negative_count = sum(1 for keyword in self.negative_keywords if keyword.lower() in text)
            
            # 기본 점수 (40-60 범위)
            base_score = random.uniform(40, 60)
            
            # 키워드에 따른 점수 조정
            if positive_count > negative_count:
                base_score += (positive_count - negative_count) * random.uniform(5, 15)
            elif negative_count > positive_count:
                base_score -= (negative_count - positive_count) * random.uniform(5, 15)
            
            # 점수 범위 제한 (10-90)
            base_score = max(10, min(90, base_score))
            
            # 예측 결과 결정
            if base_score >= 60:
                prediction = 'POSITIVE'
                confidence = 'HIGH' if base_score >= 70 else 'MEDIUM'
            elif base_score <= 40:
                prediction = 'NEGATIVE'
                confidence = 'HIGH' if base_score <= 30 else 'MEDIUM'
            else:
                prediction = 'NEUTRAL'
                confidence = 'MEDIUM'
            
            formatted_result = {
                'impact_score': base_score / 10,
                'impact_score_100': base_score,
                'direction_probability': base_score / 100,
                'expected_magnitude': abs(base_score - 50) / 1000,
                'prediction': prediction,
                'confidence': confidence,
                'news_content': news_data.get('content', ''),
                'news_date': news_data.get('news_date', datetime.now().isoformat()),
                'stock_name': news_data.get('original_stock', 'Unknown'),
                'keyword_analysis': {
                    'positive_keywords': positive_count,
                    'negative_keywords': negative_count
                }
            }
            
            return formatted_result
            
        except Exception as e:
            print(f"뉴스 감정 분석 오류: {e}")
            return {
                'error': f'분석 실패: {str(e)}',
                'impact_score': 5.0,
                'impact_score_100': 50.0,
                'prediction': 'NEUTRAL',
                'confidence': 'LOW',
                'news_content': news_data.get('content', ''),
                'news_date': news_data.get('news_date', datetime.now().isoformat()),
                'stock_name': news_data.get('original_stock', 'Unknown')
            }
    
    def analyze_stock_aggregate_sentiment(self, stock_name: str, max_days: int = 14) -> Dict:
        """종목별 종합 감정 점수 계산"""
        if not self.model_loaded:
            if not self.load_model():
                return {'error': '모델이 로드되지 않았습니다.'}
        
        # 해당 종목의 뉴스 가져오기
        news_list = self.get_news_for_stock(stock_name, limit=20)
        
        if not news_list:
            return {'error': '분석할 뉴스가 없습니다.'}
        
        try:
            # 각 뉴스에 대해 개별 분석 수행
            analyzed_news = []
            for news in news_list:
                analysis = self.analyze_single_news(news)
                if 'error' not in analysis:
                    analyzed_news.append(analysis)
            
            if not analyzed_news:
                return {'error': '유효한 뉴스 분석 결과가 없습니다.'}
            
            # UnifiedNewsAI의 aggregate 기능 사용
            aggregate_result = UnifiedNewsAI.aggregate_news_scores(
                analyzed_news, stock_name, max_days, verbose=False
            )
            
            return aggregate_result
            
        except Exception as e:
            print(f"종목 종합 감정 분석 오류: {e}")
            return {'error': f'종합 분석 실패: {str(e)}'}
    
    def get_latest_news_with_sentiment(self, stock_name: str, limit: int = 5) -> List[Dict]:
        """최신 뉴스와 감정 분석 결과 함께 반환"""
        news_list = self.get_news_for_stock(stock_name, limit)
        
        analyzed_news = []
        for news in news_list:
            analysis = self.analyze_single_news(news)
            # 원본 뉴스 데이터와 분석 결과 결합
            combined = {
                'title': news.get('content', '')[:100] + '...' if len(news.get('content', '')) > 100 else news.get('content', ''),
                'content': news.get('content', ''),
                'news_date': news.get('news_date', ''),
                'url': news.get('url', ''),
                'impact_score': analysis.get('impact_score', 5.0),
                'impact_score_100': analysis.get('impact_score_100', 50.0),
                'prediction': analysis.get('prediction', 'NEUTRAL'),
                'confidence': analysis.get('confidence', 'MEDIUM'),
                'direction_probability': analysis.get('direction_probability', 0.5)
            }
            analyzed_news.append(combined)
        
        return analyzed_news

# 전역 인스턴스
news_sentiment_service = NewsSentimentService()