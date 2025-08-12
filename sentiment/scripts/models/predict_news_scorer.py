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

        # 7. 과거 주가 정보
        if 'price_at_news_time' in df_copy.columns:
            df_sorted['return_5d_before_news'] = df_sorted.groupby('original_stock')['price_at_news_time'].pct_change(periods=5).fillna(0)
            feature_parts.append(df_sorted[['return_5d_before_news']].reindex(df_copy.index))

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
        
        prediction = 'POSITIVE' if final_prob > 0.7 else 'NEGATIVE' if final_prob < 0.3 else 'NEUTRAL'
        confidence = 'HIGH' if abs(final_prob - 0.5) > 0.2 else 'MEDIUM'
        
        result = {'impact_score': round(float(impact_score), 2), 'direction_probability': round(float(final_prob), 3), 'expected_magnitude': round(float(magnitude), 4), 'prediction': prediction, 'confidence': confidence}
        
        if verbose:
            print(f"뉴스: \"{news_data.get('content', 'N/A')[:40]}...\"")
            print(f"  - 최종 영향도 점수: {result['impact_score']}/10")
            print(f"  - 최종 상승 확률: {result['direction_probability']:.1%}")
            print(f"  - 예상 변동폭: {result['expected_magnitude']:.2%}")
            print(f"  - 예측: {result['prediction']} ({result['confidence']} 신뢰도)")
        
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