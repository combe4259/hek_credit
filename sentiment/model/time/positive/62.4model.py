# ==================================================================================
# 앙상블 및 고급 피처 기반 뉴스 AI 시스템
# ==================================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import warnings
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Optional
import joblib
from google.colab import drive

warnings.filterwarnings('ignore')

# 헬퍼 함수: 종목별 섹터 정보 가져오기
def get_sector_map(ticker_map: Dict) -> Dict:
    print("🌍 종목별 섹터 정보 수집 중... (시간이 소요될 수 있습니다)")
    sector_map = {}
    processed = 0
    total = len(ticker_map)
    for name, ticker in ticker_map.items():
        processed += 1
        if processed % 50 == 0:
            print(f"  - {processed}/{total}개 종목 처리 중")
        try:
            stock_info = yf.Ticker(ticker).info
            sector = stock_info.get('sector', 'Unknown')
            sector_map[name] = sector
        except Exception:
            sector_map[name] = 'Unknown'
    print("  ✅ 섹터 정보 수집 완료!")
    return sector_map

class AdvancedNewsAI:
    """앙상블 및 고급 피처 기반 투자 AI 시스템"""
    
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
        print("🚀 앙상블 기반 AI 시스템 초기화")

    def create_improved_targets(self, df_news: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print("\n📊 개선된 타겟 변수 생성 중...")
            print("=" * 50)
        
        results = []
        unique_stocks = df_news['original_stock'].unique()
        for i, stock_name in enumerate(unique_stocks):
            if verbose and (i + 1) % 50 == 0:
                print(f"  - 타겟 생성 진행: {i+1}/{len(unique_stocks)} 종목")

            stock_name_clean = stock_name.strip('$')
            ticker = self.name_to_ticker_map.get(stock_name_clean)
            if not ticker:
                continue

            stock_news = df_news[df_news['original_stock'] == stock_name].sort_values('news_date')
            min_date = stock_news['news_date'].min()
            max_date = stock_news['news_date'].max()
            stock_data = self._get_stock_data(ticker, min_date - timedelta(days=30), max_date + timedelta(days=10))

            if stock_data is None:
                continue
            
            stock_news_with_price = pd.merge_asof(
                stock_news,
                stock_data[['Close']].rename(columns={'Close': 'price_at_news_time'}),
                left_on='news_date',
                right_index=True,
                direction='nearest',
                tolerance=pd.Timedelta('1 day')
            )

            for _, news_row in stock_news_with_price.iterrows():
                targets = self._calculate_advanced_targets(news_row, stock_data)
                result_row = news_row.to_dict()
                result_row.update(targets)
                results.append(result_row)
        
        result_df = pd.DataFrame(results).dropna(subset=['direction_24h'])
        if verbose and not result_df.empty:
            print(f"✅ {len(result_df)}개 뉴스에 대한 타겟 생성 완료")
            self._print_target_stats(result_df)
        return result_df

    def _calculate_advanced_targets(self, news_row: pd.Series, stock_data: pd.DataFrame) -> Dict:
        news_date = pd.to_datetime(news_row['news_date'])
        base_price = self._get_price_at_time(stock_data, news_date)
        if base_price is None:
            return self._get_default_targets()

        prices = {f'{h}h': self._get_price_at_time(stock_data, news_date + timedelta(hours=h)) for h in [24, 72, 120]}
        targets = {}

        for timeframe, price in prices.items():
            if price and base_price and base_price > 0:
                return_rate = (price - base_price) / base_price
                targets[f'direction_{timeframe}'] = 1 if return_rate > 0.001 else 0
                targets[f'return_{timeframe}'] = return_rate
            else:
                targets.update({f'direction_{timeframe}': np.nan, f'return_{timeframe}': 0})
        
        return targets

    def create_advanced_features(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print("\n🛠️ 고급 피처 엔지니어링...")
        
        feature_parts = []
        df_copy = df.copy()

        # 1. 감성 피처
        sentiment_cols = ['positive', 'negative', 'neutral', 'sentiment_score']
        sentiment_df = df_copy[sentiment_cols].fillna(0)
        sentiment_df['sentiment_intensity'] = sentiment_df['positive'] + sentiment_df['negative']
        sentiment_df['sentiment_balance'] = (sentiment_df['positive'] - sentiment_df['negative']) / (sentiment_df['sentiment_intensity'] + 1e-8)
        feature_parts.append(sentiment_df)

        # 2. 시간 피처
        news_dates = pd.to_datetime(df_copy['news_date'])
        time_df = pd.DataFrame(index=df_copy.index)
        time_df['hour'] = news_dates.dt.hour
        time_df['day_of_week'] = news_dates.dt.dayofweek
        feature_parts.append(time_df)

        # 3. 시장 환경 피처
        market_df = pd.DataFrame(index=df_copy.index)
        if 'vix_close' in df_copy.columns:
            market_df['vix_close'] = df_copy['vix_close'].fillna(20)
            market_df['vix_high'] = (market_df['vix_close'] > 25).astype(int)
        else:
            market_df['vix_close'] = 20
            market_df['vix_high'] = 0
        feature_parts.append(market_df)

        # 4. 종목/섹터 피처
        df_copy['sector'] = df_copy['original_stock'].str.strip('$').map(self.name_to_sector_map).fillna('Unknown')
        sector_dummies = pd.get_dummies(df_copy['sector'], prefix='sector', dummy_na=True)
        feature_parts.append(sector_dummies)

        # 5. 뉴스 빈도 피처 (Expanding Window)
        df_sorted = df_copy.sort_values('news_date').copy()
        df_sorted['days_since_last_news'] = df_sorted.groupby('original_stock')['news_date'].diff().dt.days.fillna(0)
        df_sorted['news_frequency_cumulative'] = df_sorted.groupby('original_stock').cumcount() + 1
        feature_parts.append(df_sorted[['days_since_last_news', 'news_frequency_cumulative']])

        # 6. 과거 주가 정보 (Lagged Features)
        if 'price_at_news_time' in df_copy.columns:
            lagged_df = pd.DataFrame(index=df_copy.index)
            lagged_df['return_5d_before_news'] = df_sorted.groupby('original_stock')['price_at_news_time'].pct_change(periods=5).fillna(0)
            feature_parts.append(lagged_df)

        # 7. 상호작용 피처
        interaction_df = pd.DataFrame(index=df_copy.index)
        interaction_df['vix_x_sentiment'] = market_df['vix_close'] * sentiment_df['sentiment_balance']
        feature_parts.append(interaction_df)

        # 8. FinBERT 피처
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
                if col not in X_final.columns:
                    X_final[col] = 0
            X_final = X_final[self.feature_names]
            
        return X_final.fillna(0)

    def train_models(self, df_news: pd.DataFrame, verbose: bool = True):
        if verbose:
            print("\n🚀 앙상블 모델 훈련 시작")
        
        df_with_targets = self.create_improved_targets(df_news, verbose=verbose)
        if df_with_targets.empty:
            return None
        
        min_date = df_with_targets['news_date'].min() - timedelta(days=5)
        max_date = df_with_targets['news_date'].max() + timedelta(days=5)
        vix_data = self._get_stock_data('^VIX', min_date, max_date)
        if vix_data is not None:
            df_with_targets = pd.merge_asof(
                df_with_targets.sort_values('news_date'),
                vix_data[['Close']].rename(columns={'Close':'vix_close'}),
                left_on='news_date',
                right_index=True,
                direction='backward'
            ).sort_index()
        else:
            df_with_targets['vix_close'] = 20

        X = self.create_advanced_features(df_with_targets, verbose=verbose)
        y = df_with_targets['direction_24h'].astype(int)
        
        tscv = TimeSeriesSplit(n_splits=3)
        oof_preds = np.zeros((len(X), 3))
        
        if verbose:
            print("\n📈 스태킹 앙상블 교차 검증...")
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"  - Fold {fold+1}/3 훈련 중...")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler_cv = RobustScaler()
            X_train_scaled = scaler_cv.fit_transform(X_train)
            X_test_scaled = scaler_cv.transform(X_test)

            models_cv = {
                'lgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'xgb': xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
                'cat': cb.CatBoostClassifier(random_state=42, verbose=0)
            }

            for i, (name, model) in enumerate(models_cv.items()):
                model.fit(X_train_scaled, y_train)
                oof_preds[test_idx, i] = model.predict_proba(X_test_scaled)[:, 1]
        
        last_fold_test_idx = test_idx
        meta_X = oof_preds[last_fold_test_idx]
        meta_y = y.iloc[last_fold_test_idx]
        
        print("  - 메타 모델 훈련 중...")
        self.meta_model.fit(meta_X, meta_y)
        meta_preds = self.meta_model.predict(meta_X)
        accuracy = accuracy_score(meta_y, meta_preds)
        print(f"  - 최종 Fold 메타 모델 정확도: {accuracy:.1%}")

        print("  - 전체 데이터로 최종 기본 모델들 훈련...")
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.base_models['lgbm'] = lgb.LGBMClassifier(random_state=42, verbose=-1).fit(X_scaled, y)
        self.base_models['xgb'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False).fit(X_scaled, y)
        self.base_models['cat'] = cb.CatBoostClassifier(random_state=42, verbose=0).fit(X_scaled, y)
        
        base_preds_final = np.array([model.predict_proba(X_scaled)[:, 1] for model in self.base_models.values()]).T
        self.meta_model.fit(base_preds_final, y)
        
        print("  - 최종 변동폭 예측 모델 훈련...")
        y_magnitude = np.abs(df_with_targets['return_24h']).fillna(0)
        self.magnitude_model.fit(X_scaled, y_magnitude)

        if verbose:
            print(f"\n✅ 앙상블 모델 훈련 완료!")
        return {'ensemble_accuracy': accuracy}

    def predict_news_impact(self, news_data: Dict, verbose: bool = True) -> Dict:
        if not self.base_models:
            raise ValueError("모델이 훈련되지 않았습니다!")
        
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

        X = self.create_advanced_features(df_input, verbose=False)
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
            print(f"📰 뉴스 영향도 분석 (앙상블 + 점수):")
            print(f"  🎯 최종 영향도 점수: {result['impact_score']}/10")
            print(f"  📈 최종 상승 확률: {result['direction_probability']:.1%}")
            print(f"  📊 예상 변동폭: {result['expected_magnitude']:.2%}")
            print(f"  🔍 예측: {result['prediction']} ({result['confidence']} 신뢰도)")
        
        return result

    def _get_stock_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        cache_key = f"{ticker}_{start_date.date()}_{end_date.date()}"
        if cache_key in self.stock_cache:
            return self.stock_cache[cache_key]
        try:
            data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
            if not data.empty:
                data.index = data.index.tz_localize(None)
                self.stock_cache[cache_key] = data
                return data
        except Exception:
            pass
        return None

    def _get_price_at_time(self, stock_data: Optional[pd.DataFrame], target_time: datetime) -> Optional[float]:
        if stock_data is None:
            return None
        try:
            target_time_naive = pd.to_datetime(target_time).tz_localize(None)
            return stock_data['Close'].asof(target_time_naive)
        except:
            return None

    def _get_default_targets(self) -> Dict:
        return {'direction_24h': np.nan, 'return_24h': 0, 'direction_72h': np.nan, 'return_72h': 0, 'direction_120h': np.nan, 'return_120h': 0}

    def _print_target_stats(self, df: pd.DataFrame):
        print("\n📊 타겟 변수 통계:")
        for timeframe in ['24h', '72h', '120h']:
            direction_col = f'direction_{timeframe}'
            return_col = f'return_{timeframe}'
            if direction_col in df.columns and return_col in df.columns:
                up_ratio = df[direction_col].mean()
                avg_return = df[return_col].mean()
                print(f"  {timeframe}: 상승비율 {up_ratio:.1%}, 평균수익률 {avg_return:.2%}")
        
    def _print_feature_importance(self):
        if self.base_models.get('lgbm'):
            print("\n📊 LGBM 모델의 주요 피처 중요도 Top 10:")
            importance_df = pd.DataFrame({'feature': self.feature_names, 'importance': self.base_models['lgbm'].feature_importances_}).sort_values('importance', ascending=False)
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"  {i+1:2d}. {row['feature']:<30}: {row['importance']}")

    def save_model(self, filepath: str):
        joblib.dump(self, filepath)
        print(f"💾 모델 저장 완료: {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        instance = joblib.load(filepath)
        print(f"📂 모델 로드 완료: {filepath}")
        return instance

# ==================================================================================
# 실행부
# ==================================================================================
if __name__ == "__main__":
    try:
        drive.mount('/content/drive', force_remount=True)
        print("🚀 앙상블 AI 시스템 훈련 및 시연 시작")
        print("=" * 60)

        mapping_xlsx_path = "/content/drive/MyDrive/sp500_korean_stocks_with_symbols.xlsx"
        news_csv_path = "/content/drive/MyDrive/news_full_features_robust.csv"
        model_save_path = "/content/drive/MyDrive/advanced_news_ai_model.pkl"

        df_mapping = pd.read_excel(mapping_xlsx_path, header=1)
        df_mapping.dropna(subset=['Symbol'], inplace=True)
        name_ticker_map = pd.Series(df_mapping.Symbol.values, index=df_mapping['Korean Name'].str.strip()).to_dict()

        name_sector_map = {name: 'Unknown' for name in name_ticker_map.keys()}
        
        df_news = pd.read_csv(news_csv_path)
        df_news['news_date'] = pd.to_datetime(df_news['news_date'])

        ai_system = AdvancedNewsAI(name_ticker_map, name_sector_map)
        training_results = ai_system.train_models(df_news, verbose=True)

        if training_results:
            ai_system.save_model(model_save_path)
            
            print("\n" + "="*60)
            print("🎬 저장된 모델 로드 및 예측 시연")
            print("="*60)
            
            loaded_ai = AdvancedNewsAI.load_model(model_save_path)
            
            sample_stock_name = '엔비디아'
            if sample_stock_name in loaded_ai.name_to_ticker_map:
                sample_news = df_news[df_news['original_stock'] == sample_stock_name].iloc[-1].to_dict()
                loaded_ai.predict_news_impact(sample_news)
            else:
                print(f"'{sample_stock_name}'이 뉴스 데이터에 없어 시연을 건너뜁니다.")

    except Exception as e:
        print(f"❌ 최종 실행 오류 발생: {e}")
        import traceback
        traceback.print_exc()