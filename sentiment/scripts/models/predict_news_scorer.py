# ==================================================================================
# Optuna 최적화, 토픽 모델링, 앙상블 및 고급 피처 기반 뉴스 호재/악재 평가 모델 (예측용)
# ==================================================================================


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import warnings
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Optional
import joblib
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import os

warnings.filterwarnings('ignore')

class NewsScorer:
    def __init__(self, name_to_ticker_map: Dict, name_to_sector_map: Dict):
        self.name_to_ticker_map = name_to_ticker_map
        self.name_to_sector_map = name_to_sector_map
        self.base_models = {}
        self.meta_model = LogisticRegression()
        self.magnitude_model = lgb.LGBMRegressor(objective='regression_l1', random_state=42, verbose=-1)
        self.scaler = RobustScaler()
        self.feature_names = None
        self.stock_cache = {}
        self.bert_pca = None
        self.topic_model = None
        self.best_params = {}

    def create_features(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        feature_parts = []
        df_copy = df.copy()
        
        # 1. 토픽 피처
        if 'content' in df_copy.columns and not df_copy['content'].isnull().all():
            if self.topic_model is None:
                print("  - BERTopic 모델 훈련")
                embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                self.topic_model = BERTopic(embedding_model=embedding_model, verbose=False, min_topic_size=15)
                topics, _ = self.topic_model.fit_transform(df_copy['content'].fillna(""))
            else:
                topics, _ = self.topic_model.transform(df_copy['content'].fillna(""))
            df_copy['topic'] = topics
            topic_dummies = pd.get_dummies(df_copy['topic'], prefix='topic')
            feature_parts.append(topic_dummies)

        # 2. 감성 피처
        sentiment_cols = ['positive', 'negative', 'neutral', 'sentiment_score']
        sentiment_df = df_copy[sentiment_cols].fillna(0)
        sentiment_df['sentiment_intensity'] = sentiment_df['positive'] + sentiment_df['negative']
        sentiment_df['sentiment_balance'] = (sentiment_df['positive'] - sentiment_df['negative']) / (sentiment_df['sentiment_intensity'] + 1e-8)
        feature_parts.append(sentiment_df)

        # 3. 시간 피처
        news_dates = pd.to_datetime(df_copy['news_date'])
        time_df = pd.DataFrame(index=df_copy.index)
        time_df['hour'] = news_dates.dt.hour
        time_df['day_of_week'] = news_dates.dt.dayofweek
        feature_parts.append(time_df)

        # 4. 시장 환경 피처
        market_df = pd.DataFrame(index=df_copy.index)
        market_df['vix_close'] = df_copy.get('vix_close', 20)
        market_df['vix_high'] = (market_df['vix_close'] > 25).astype(int)
        feature_parts.append(market_df)

        # 5. 종목/섹터 피처
        df_copy['sector'] = df_copy['original_stock'].str.strip('$').map(self.name_to_sector_map).fillna('Unknown')
        sector_dummies = pd.get_dummies(df_copy['sector'], prefix='sector', dummy_na=True)
        feature_parts.append(sector_dummies)

        # 6. 뉴스 빈도 피처 (Expanding Window 방식)
        df_sorted = df_copy.sort_values('news_date').copy()
        df_sorted['days_since_last_news'] = df_sorted.groupby('original_stock')['news_date'].diff().dt.days.fillna(0)
        df_sorted['news_frequency_cumulative'] = df_sorted.groupby('original_stock').cumcount() + 1
        feature_parts.append(df_sorted[['days_since_last_news', 'news_frequency_cumulative']].reindex(df_copy.index))

        # 7. 과거 주가 정보 (다중 기간 모멘텀) - 데이터 누수 방지
        if 'price_at_news_time' in df_copy.columns:
            momentum_df = pd.DataFrame(index=df_copy.index)
            
            # 각 뉴스별로 개별적으로 모멘텀 계산 (훈련 시에만 사용, 예측 시에는 실시간 계산)
            for idx, row in df_copy.iterrows():
                stock_name = row['original_stock'].strip('$')
                ticker = self.name_to_ticker_map.get(stock_name)
                news_date = pd.to_datetime(row['news_date'])
                
                if ticker and hasattr(self, 'stock_cache'):  # 훈련된 모델에서만
                    try:
                        # 뉴스 발생 전 30일간의 주가 데이터 확보
                        stock_data = self._get_stock_data(ticker, news_date - timedelta(days=30), news_date)
                        if stock_data is not None and len(stock_data) >= 20:
                            # 뉴스 발생 시점 기준으로 과거 모멘텀 계산
                            prices = stock_data['Close']
                            news_price = self._get_price_at_time(stock_data, news_date)
                            
                            if news_price:
                                # 1일 전 가격과 비교
                                price_1d = self._get_price_at_time(stock_data, news_date - timedelta(days=1))
                                momentum_df.loc[idx, 'return_1d_before_news'] = (news_price - price_1d) / price_1d if price_1d else 0
                                
                                # 5일 전 가격과 비교  
                                price_5d = self._get_price_at_time(stock_data, news_date - timedelta(days=5))
                                momentum_df.loc[idx, 'return_5d_before_news'] = (news_price - price_5d) / price_5d if price_5d else 0
                                
                                # 20일 전 가격과 비교
                                price_20d = self._get_price_at_time(stock_data, news_date - timedelta(days=20))
                                momentum_df.loc[idx, 'return_20d_before_news'] = (news_price - price_20d) / price_20d if price_20d else 0
                    except Exception:
                        pass
            
            # 결측값을 0으로 채움
            momentum_df = momentum_df.fillna(0)
            feature_parts.append(momentum_df)

        # 8. 상호작용 피처
        interaction_df = pd.DataFrame(index=df_copy.index)
        interaction_df['vix_x_sentiment'] = market_df['vix_close'] * sentiment_df['sentiment_balance']
        feature_parts.append(interaction_df)

        # 9. FinBERT 피처
        bert_cols = [f'finbert_{i}' for i in range(768) if f'finbert_{i}' in df_copy.columns]
        if bert_cols:
            bert_data = df_copy[bert_cols].fillna(0)
            if self.bert_pca is None:
                self.bert_pca = PCA(n_components=30, random_state=42)
                bert_reduced = self.bert_pca.fit_transform(bert_data)
            else:
                bert_reduced = self.bert_pca.transform(bert_data)
            bert_df = pd.DataFrame(bert_reduced, index=df_copy.index, columns=[f'bert_pc_{i}' for i in range(30)])
            feature_parts.append(bert_df)
        
        X_final = pd.concat(feature_parts, axis=1)
        
        if self.feature_names is None:
            self.feature_names = X_final.columns.tolist()
        else:
            for col in self.feature_names:
                if col not in X_final.columns: X_final[col] = 0
            X_final = X_final[self.feature_names]
            
        return X_final.fillna(0)

    def predict_news_impact(self, news_data: Dict, verbose: bool = True) -> Dict:
        if not self.base_models: raise ValueError("모델이 훈련되지 않았습니다!")
        
        df_input = pd.DataFrame([news_data])
        df_input['news_date'] = pd.to_datetime(df_input['news_date'])
        
        vix_data = self._get_stock_data('^VIX', df_input['news_date'].min() - timedelta(days=5), df_input['news_date'].max() + timedelta(days=1))
        if vix_data is not None:
             df_input['vix_close'] = vix_data['Close'].asof(df_input['news_date'].iloc[0])
        else:
            df_input['vix_close'] = 20
        
        ticker = self.name_to_ticker_map.get(news_data['original_stock'].strip('$'))
        if ticker:
            price_data = self._get_stock_data(ticker, df_input['news_date'].min() - timedelta(days=10), df_input['news_date'].max() + timedelta(days=1))
            if price_data is not None:
                 df_input['price_at_news_time'] = self._get_price_at_time(price_data, df_input['news_date'].iloc[0])
        if 'price_at_news_time' not in df_input.columns:
             df_input['price_at_news_time'] = 100

        X = self.create_features(df_input, verbose=False)
        X_scaled = self.scaler.transform(X)
        
        base_predictions = np.array([model.predict_proba(X_scaled)[:, 1] for model in self.base_models.values()]).T
        final_prob = self.meta_model.predict_proba(base_predictions)[0, 1]
        magnitude = self.magnitude_model.predict(X_scaled)[0]
        
        impact_score = 5 + (final_prob - 0.5) * 10 * min(magnitude * 50, 1)
        impact_score = np.clip(impact_score, 0, 10)
        
        # 0~100 범위 점수 추가 계산
        impact_score_100 = impact_score * 10
        
        prediction = 'POSITIVE' if final_prob > 0.7 else 'NEGATIVE' if final_prob < 0.3 else 'NEUTRAL'
        confidence = 'HIGH' if abs(final_prob - 0.5) > 0.2 else 'MEDIUM'
        
        result = {'impact_score': round(float(impact_score), 2), 'impact_score_100': round(float(impact_score_100), 1), 'direction_probability': round(float(final_prob), 3), 'expected_magnitude': round(float(magnitude), 4), 'prediction': prediction, 'confidence': confidence}
        
        if verbose:
            print(f"뉴스: \"{news_data.get('content', 'N/A')[:40]}...\"")
            print(f"  - 최종 영향도 점수: {result['impact_score']}/10 ({result["impact_score_100"]}/100)")
            print(f"  - 최종 상승 확률: {result['direction_probability']:.1%}")
            print(f"  - 예상 변동폭: {result['expected_magnitude']:.2%}")
            print(f"  - 예측: {result['prediction']} ({result['confidence']} 신뢰도)")
        
        return result

    @staticmethod
    def aggregate_news_scores(predictions_list: list, stock_name: str, max_days: int = 14, verbose: bool = True) -> Dict:
        """
        시간 가중치와 기간 제한을 적용한 종목별 종합 점수 계산
        
        Args:
            predictions_list: predict_news_impact 결과들의 리스트 (news_date 필드 포함)
            stock_name: 종목명
            max_days: 분석 대상 기간 (기본 14일)
            verbose: 출력 여부
        
        Returns:
            Dict: 시간 가중치가 적용된 종합 점수, 가장 영향력 큰 뉴스 content 포함
        """
        if not predictions_list:
            return {'error': '분석할 예측 결과가 없습니다.'}
        
        df_preds = pd.DataFrame(predictions_list)
        if 'news_date' not in df_preds.columns:
            return {'error': '뉴스 날짜 정보가 없습니다.'}
        df_preds['news_date'] = pd.to_datetime(df_preds['news_date'])
        
        analysis_date = datetime.now()
        filtered_data = []
        for _, pred in df_preds.iterrows():
            if pd.notna(pred['news_date']):
                days_ago = (analysis_date - pred['news_date']).days
                if 0 <= days_ago <= max_days:
                    filtered_data.append((pred.to_dict(), pred['news_date'], days_ago))

        if not filtered_data:
            return {'error': f'최근 {max_days}일 이내 뉴스가 없습니다.'}
        
        total_weight, weighted_score, weighted_prob = 0, 0, 0
        positive_count, negative_count, neutral_count = 0, 0, 0
        
        for pred, news_date, days_ago in filtered_data:
            time_weight = np.exp(-days_ago / 7.0)  # 7일 반감기
            confidence_weight = 1.5 if pred['confidence'] == 'HIGH' else 1.0
            final_weight = time_weight * confidence_weight
            
            # impact_score_100이 있으면 사용, 없으면 impact_score * 10
            score_100 = pred.get('impact_score_100', pred.get('impact_score', 5) * 10)
            weighted_score += score_100 * final_weight
            weighted_prob += pred['direction_probability'] * final_weight
            total_weight += final_weight
            
            if pred['prediction'] == 'POSITIVE':
                positive_count += 1
            elif pred['prediction'] == 'NEGATIVE':
                negative_count += 1
            else:
                neutral_count += 1
                
        if total_weight == 0:
            return {'error': '유효 가중치 없음'}
        
        agg_score = round(weighted_score / total_weight, 1)
        agg_prob = round(weighted_prob / total_weight, 3)
        
        overall_pred = 'POSITIVE' if agg_prob > 0.6 else 'NEGATIVE' if agg_prob < 0.4 else 'NEUTRAL'
        overall_conf = 'HIGH' if abs(agg_prob - 0.5) > 0.2 else 'MEDIUM'
        
        # 대표 뉴스 선정 (가장 최근이면서 가장 극단적인 영향력)
        representative_news_data = max(filtered_data, key=lambda item: np.exp(-item[2]/7.0) * abs(item[0].get('impact_score_100', item[0].get('impact_score', 5) * 10) - 50))[0]

        result = {
            'stock_name': stock_name,
            'aggregate_score_100': agg_score,
            'aggregate_probability': agg_prob,
            'overall_prediction': overall_pred,
            'overall_confidence': overall_conf,
            'news_breakdown': {'positive': positive_count, 'negative': negative_count, 'neutral': neutral_count},
            'representative_news_content': representative_news_data.get('content', ''),  # Gemini 요약용 content
            'total_news_count': len(filtered_data)
        }
        
        if verbose:
            print(f"\n{stock_name} 종합 분석 결과 (최근 {max_days}일):")
            print(f"시간가중 종합점수: {agg_score}/100 | 시간가중 상승 확률: {agg_prob:.1%}")
            print(f"호재 {positive_count}개, 악재 {negative_count}개, 중립 {neutral_count}개")
            print(f"가장 영향력 큰 뉴스: {representative_news_data.get('content', '')[:80]}...")
        return result

    def _get_stock_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        cache_key = f"{ticker}_{start_date.date()}_{end_date.date()}"
        if cache_key in self.stock_cache: return self.stock_cache[cache_key]
        try:
            data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
            if not data.empty:
                data.index = data.index.tz_localize(None)
                self.stock_cache[cache_key] = data
                return data
        except Exception: pass
        return None

    def _get_price_at_time(self, stock_data: Optional[pd.DataFrame], target_time: datetime) -> Optional[float]:
        if stock_data is None: return None
        try:
            target_time_naive = pd.to_datetime(target_time).tz_localize(None)
            return stock_data['Close'].asof(target_time_naive)
        except: return None

    @classmethod
    def load_model(cls, filepath: str = None):
        if filepath is None:
            # 기본 모델 경로 설정
            current_dir = os.path.dirname(os.path.abspath(__file__))
            scripts_dir = os.path.join(current_dir, '../../../scripts')
            scripts_dir = os.path.abspath(scripts_dir)
            filepath = os.path.join(scripts_dir, "models/news_scorer_model.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")
            
        instance = joblib.load(filepath)
        print(f"모델 로드 완료: {filepath}")
        return instance

# ==================================================================================
# 실행부 (예측용)
# ==================================================================================
if __name__ == "__main__":
    import sys
    
    # 운영 환경에서는 직접 실행하지 않고 모듈로만 사용
    print("사용 예시:")
    print("  from predict_news_scorer import NewsScorer")
    print("  model = NewsScorer.load_model('/path/to/model.pkl')")
    print("  result = model.predict_news_impact(news_data)")
    
    # 테스트 모드 실행 (개발용)
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("\n테스트 모드로 실행합니다...")
        
        # 1. 훈련된 모델 로드 (서버 시작 시 1회)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scripts_dir = os.path.join(current_dir, '../../../scripts')
        scripts_dir = os.path.abspath(scripts_dir)
        model_path = os.path.join(scripts_dir, "models/news_scorer_model.pkl")
        try:
            ai_system = NewsScorer.load_model(model_path)
            
            # 2. 테스트용 뉴스 데이터
            sample_news = {
                'original_stock': '엔비디아',
                'news_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'content': 'NVIDIA, 시장 예상을 뛰어넘는 혁신적인 AI 칩 발표로 모두를 놀라게 하다.',
                'positive': 0.85, 'negative': 0.05, 'neutral': 0.1, 'sentiment_score': 0.8,
            }
            
            # finbert 임베딩 전처리 파이프라인을 거쳐야 함
            for i in range(768):
                sample_news[f'finbert_{i}'] = np.random.rand()

            # 3. 예측 실행
            print("\n" + "="*60)
            print("새로운 뉴스에 대한 실시간 예측 시연")
            print("="*60)
            prediction_result = ai_system.predict_news_impact(sample_news, verbose=True)
            print("\n- 예측 결과 (JSON):")
            print(prediction_result)
        except FileNotFoundError:
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        except Exception as e:
            print(f"테스트 실행 중 오류: {e}")