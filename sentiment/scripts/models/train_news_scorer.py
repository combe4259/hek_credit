# ==================================================================================
# Optuna 최적화, 토픽 모델링, 앙상블 및 고급 피처 기반 뉴스 호재/악재 평가 모델 (훈련용)
# ==================================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import warnings
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Optional
import joblib
import optuna
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import os
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 종목별 섹터 정보 가져오기
def get_sector_map(ticker_map: Dict) -> Dict:
    print("종목별 섹터 정보 수집 중...")
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
    print(" 섹터 정보 수집 완료")
    return sector_map

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

    def create_targets(self, df_news: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        if verbose: print("\n개선된 타겟 변수 생성 중...")
        results = []
        unique_stocks = df_news['original_stock'].unique()

        for i, stock_name in enumerate(unique_stocks):
            if verbose and (i + 1) % 50 == 0:
                print(f"  - 타겟 생성 진행: {i+1}/{len(unique_stocks)} 종목")
            stock_name_clean = stock_name.strip('$')
            ticker = self.name_to_ticker_map.get(stock_name_clean)

            if not ticker: continue
            stock_news = df_news[df_news['original_stock'] == stock_name].sort_values('news_date')
            min_date, max_date = stock_news['news_date'].min(), stock_news['news_date'].max()
            stock_data = self._get_stock_data(ticker, min_date - timedelta(days=30), max_date + timedelta(days=10))
            
            if stock_data is None: continue
            stock_news_with_price = pd.merge_asof(stock_news, stock_data[['Close']].rename(columns={'Close': 'price_at_news_time'}), left_on='news_date', right_index=True, direction='nearest', tolerance=pd.Timedelta('1 day'))
            
            for _, news_row in stock_news_with_price.iterrows():
                targets = self._calculate_advanced_targets(news_row, stock_data)
                result_row = news_row.to_dict()
                result_row.update(targets)
                results.append(result_row)
        
        result_df = pd.DataFrame(results).dropna(subset=['direction_24h'])
        
        if verbose and not result_df.empty:
            print(f"{len(result_df)}개 뉴스에 대한 타겟 생성 완료")
        return result_df

    def _calculate_advanced_targets(self, news_row: pd.Series, stock_data: pd.DataFrame) -> Dict:
        news_date = pd.to_datetime(news_row['news_date'])
        base_price = self._get_price_at_time(stock_data, news_date)
        
        if base_price is None: return self._get_default_targets()
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
            
            # 각 뉴스별로 개별적으로 모멘텀 계산 (데이터 누수 방지)
            for idx, row in df_copy.iterrows():
                stock_name = row['original_stock'].strip('$')
                ticker = self.name_to_ticker_map.get(stock_name)
                news_date = pd.to_datetime(row['news_date'])
                
                if ticker:
                    try:
                        # 뉴스 발생 전 30일간의 주가 데이터 확보
                        stock_data = self._get_stock_data(ticker, news_date - timedelta(days=30), news_date)
                        if stock_data is not None and len(stock_data) >= 20:
                            # 뉴스 발생 시점 기준으로 과거 모멘텀 계산
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

    def train(self, df_news: pd.DataFrame, n_trials: int = 30, verbose: bool = True):
        if verbose: print("\n앙상블 모델 최적화 및 훈련 시작")
        
        df_with_targets = self.create_targets(df_news, verbose=verbose)
        if df_with_targets.empty: return None
        
        vix_data = self._get_stock_data('^VIX', df_with_targets['news_date'].min() - timedelta(days=5), df_with_targets['news_date'].max() + timedelta(days=5))
        if vix_data is not None:
            df_with_targets = pd.merge_asof(df_with_targets.sort_values('news_date'), vix_data[['Close']].rename(columns={'Close':'vix_close'}), left_on='news_date', right_index=True, direction='backward').sort_index()
        else:
            df_with_targets['vix_close'] = 20

        X = self.create_features(df_with_targets, verbose=verbose)
        y = df_with_targets['direction_24h'].astype(int)
        
        def objective(trial):
            # 훨씬 더 많은 하이퍼파라미터 조합 탐색
            lgbm_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),  # 범위 축소
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),  # 10~1000 → 20~200
                'max_depth': trial.suggest_int('max_depth', 5, 12),  # 3~20 → 5~12
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),  # 1~200 → 10~50
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),  # 0.3~1.0 → 0.7~1.0
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),  # 축소
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),  # 범위 축소
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),  # 범위 축소
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),  # 0~2 → 0~0.5
                'random_state': 42,
                'verbose': -1
            }
            tscv = TimeSeriesSplit(n_splits=5)  # 교차검증도 더 엄격하게
            scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                scaler = RobustScaler()
                X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
                model = lgb.LGBMClassifier(**lgbm_params)
                model.fit(X_train_scaled, y_train)
                scores.append(accuracy_score(y_test, model.predict(X_test_scaled)))
            return np.mean(scores)

        print(f"\nOptuna 최적화 시작 ({n_trials} 시도)")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        print(f"최적화 완료. 최고 교차검증 정확도: {study.best_value:.1%}")
        print(f"최적 하이퍼파라미터:")
        for param, value in self.best_params.items():
            print(f"  - {param}: {value}")
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        final_lgbm_params = {key: val for key, val in self.best_params.items()}
        self.base_models['lgbm'] = lgb.LGBMClassifier(objective='binary', random_state=42, verbose=-1, **final_lgbm_params).fit(X_scaled, y)
        self.base_models['xgb'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False).fit(X_scaled, y)
        self.base_models['cat'] = cb.CatBoostClassifier(random_state=42, verbose=0).fit(X_scaled, y)
        
        base_preds_final = np.array([model.predict_proba(X_scaled)[:, 1] for model in self.base_models.values()]).T
        self.meta_model.fit(base_preds_final, y)
        
        y_magnitude = np.abs(df_with_targets['return_24h']).fillna(0)
        self.magnitude_model.fit(X_scaled, y_magnitude)

        # R^2 점수 계산 및 출력 추가
        magnitude_preds = self.magnitude_model.predict(X_scaled)
        magnitude_r2 = r2_score(y_magnitude, magnitude_preds)
        print(f"수익률 크기 예측 모델 R²: {magnitude_r2:.4f}")

        if verbose: print(f"\n앙상블 모델 훈련 완료")
            
        print("\n백테스팅 실행")
        tscv_backtest = TimeSeriesSplit(n_splits=3)
        oof_preds_backtest = np.zeros(len(X))
        
        # 교차 검증을 통해 Out-of-Fold 예측 생성
        for train_idx, test_idx in tscv_backtest.split(X):
             X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
             scaler_bt = RobustScaler()
             X_train_scaled = scaler_bt.fit_transform(X_train)
             X_test_scaled = scaler_bt.transform(X_test)
             
             base_models_bt = { 'lgbm': lgb.LGBMClassifier(random_state=42, verbose=-1, **final_lgbm_params).fit(X_train_scaled, y_train), 'xgb': xgb.XGBClassifier(random_state=42, use_label_encoder=False).fit(X_train_scaled, y_train), 'cat': cb.CatBoostClassifier(random_state=42, verbose=0).fit(X_train_scaled, y_train)}
             base_preds_bt = np.array([model.predict_proba(X_test_scaled)[:, 1] for model in base_models_bt.values()]).T
             
             meta_model_bt = LogisticRegression().fit(base_preds_bt, y_test)
             oof_preds_backtest[test_idx] = meta_model_bt.predict_proba(base_preds_bt)[:, 1]

        backtest_results = self._run_backtest(pd.Series(oof_preds_backtest, index=X.index), df_with_targets)
        self._print_backtest_results(backtest_results)
            
        return {'best_accuracy': study.best_value, 'backtest': backtest_results}

    def _run_backtest(self, predictions: pd.Series, actuals: pd.DataFrame, threshold=0.7) -> Dict:
        backtest_df = actuals.loc[predictions.index].copy()
        backtest_df['pred_prob'] = predictions
        
        positions = backtest_df[backtest_df['pred_prob'] > threshold]
        
        if positions.empty:
            return {'cagr': 0, 'sharpe': 0, 'max_drawdown': 0, 'trades': 0}
            
        daily_returns = positions['return_24h'].dropna()
        
        if len(daily_returns) < 2:
             return {'cagr': 0, 'sharpe': 0, 'max_drawdown': 0, 'trades': len(daily_returns)}

        cumulative_returns = (1 + daily_returns).cumprod()
        
        start_date = pd.to_datetime(positions.loc[daily_returns.index[0], 'news_date'])
        end_date = pd.to_datetime(positions.loc[daily_returns.index[-1], 'news_date'])
        days = (end_date - start_date).days
        
        cagr = (cumulative_returns.iloc[-1])**(365.0/days) - 1 if days > 0 else 0
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {'cagr': cagr, 'sharpe': sharpe, 'max_drawdown': max_drawdown, 'trades': len(positions)}

    def _print_backtest_results(self, results: Dict):
        print("\n백테스팅 결과:")
        print(f"  - 연평균 복리 수익률 (CAGR): {results['cagr']:.2%}")
        print(f"  - 샤프 지수 (위험 대비 수익): {results['sharpe']:.2f}")
        print(f"  - 최대 낙폭 (MDD): {results['max_drawdown']:.2%}")
        print(f"  - 총 거래 횟수: {results['trades']}회")

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

    def _get_default_targets(self) -> Dict:
        return {'direction_24h': np.nan, 'return_24h': 0}
    
    def save_model(self, filepath: str):
        self.stock_cache = {}
        joblib.dump(self, filepath)
        print(f"모델 저장 완료: {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        instance = joblib.load(filepath)
        print(f"모델 로드 완료: {filepath}")
        return instance

# ==================================================================================
# 실행부 (훈련용)
# ==================================================================================
if __name__ == "__main__":
    try:
        print("모델 학습 시작")
        print("=" * 60)

        # 설정: 운영 모드(True) 또는 개발 모드(False)
        PRODUCTION_MODE = True
        
        # scripts 폴더 기준으로 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scripts_dir = os.path.join(current_dir, '../../../scripts')
        scripts_dir = os.path.abspath(scripts_dir)
        
        mapping_xlsx_path = os.path.join(scripts_dir, "sp500_korean_stocks_with_symbols.xlsx")
        news_csv_path = os.path.join(scripts_dir, "data/news_full_features_robust.csv")
        model_save_path = os.path.join(scripts_dir, "models/news_scorer_model.pkl")
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(news_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        df_mapping = pd.read_excel(mapping_xlsx_path, header=1)
        df_mapping.dropna(subset=['Symbol'], inplace=True)
        name_ticker_map = pd.Series(df_mapping.Symbol.values, index=df_mapping['Korean Name'].str.strip()).to_dict()

        # 섹터 맵핑: 운영 모드에서는 실제 데이터 조회, 개발 모드에서는 Unknown 사용
        if PRODUCTION_MODE:
            print("운영 모드: 실제 섹터 데이터 조회 중...")
            name_sector_map = get_sector_map(name_ticker_map)
        else:
            print("개발 모드: 섹터 정보를 Unknown으로 설정")
            name_sector_map = {name: 'Unknown' for name in name_ticker_map.keys()}
        
        df_news = pd.read_csv(news_csv_path)
        df_news['news_date'] = pd.to_datetime(df_news['news_date'])

        # 데이터 사용량: 운영 모드와 개발 모드 모두 전체 데이터 사용
        print(f"{'운영' if PRODUCTION_MODE else '개발'} 모드: 전체 데이터 사용 ({len(df_news):,}개 뉴스)")
        df_news_final = df_news

        ai_system = NewsScorer(name_ticker_map, name_sector_map)
        training_results = ai_system.train(df_news_final, n_trials=30, verbose=True)

        if training_results:
            ai_system.save_model(model_save_path)
            print("\n모든 훈련 완료.")
            print(f"모델이 '{model_save_path}' 경로에 저장되었습니다.")
            print(f"사용된 데이터: {len(df_news_final):,}개 뉴스")
            print(f"모드: {'운영' if PRODUCTION_MODE else '개발'}")

    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()