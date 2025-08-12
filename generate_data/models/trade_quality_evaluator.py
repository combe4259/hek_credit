import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
import xgboost as xgb
import joblib
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TradeQualityEvaluator:
    """거래 품질 평가 모델완료된 매수-매도 거래의 품질을 평가"""

    def __init__(self, train_months=36, val_months=6, test_months=6, step_months=3):

        self.model = None
        self.trade_quality_scalers = {}
        self.features = None
        self.is_trained = False
        

        self.train_months = train_months
        self.val_months = val_months  
        self.test_months = test_months
        self.step_months = step_months
        

        self.fold_results = []
        self.best_params = None
        
    def create_quality_score(self, df, risk_scaler=None, eff_scaler=None, verbose=False):
        """ 거래 품질 점수 생성 """
        if verbose:
            print("거래 품질 점수 생성 중")
        
        df = df.copy()
        
        # 필수 컬럼 확인 및 NaN 처리
        required_columns = ['return_pct', 'entry_volatility_20d', 'entry_ratio_52w_high', 'holding_period_days']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
        
        df['return_pct'] = df['return_pct'].fillna(0)
        df['entry_volatility_20d'] = df['entry_volatility_20d'].fillna(0)
        df['entry_ratio_52w_high'] = df['entry_ratio_52w_high'].fillna(0)
        df['holding_period_days'] = df['holding_period_days'].fillna(0)

        # 1. 진입 품질 (30%) - Buy Signal 기반
        from buy_signal_predictor import BuySignalPredictor
        buy_predictor = BuySignalPredictor()
        
        # Buy Signal 점수 생성
        df_with_buy = buy_predictor.create_entry_signal_score(df, verbose=False)
        df['entry_quality'] = df_with_buy['buy_signal_score']
        
        # 2. 청산 타이밍 품질 (30%)
        df['exit_timing_quality'] = self._calculate_exit_timing_without_return(df)
        
        # 3. 최종 성과 (40%) - 수익률 기반
        df['result_quality'] = self._score_return(df['return_pct'])
        
        # 4. 최종 거래 품질 점수 (0-100점 스케일 유지)
        df['trade_quality_score'] = (
            df['entry_quality'] * 0.3 + 
            df['exit_timing_quality'] * 0.3 +
            df['result_quality'] * 0.4
        )
        
        # 0-100 범위 보장
        df['trade_quality_score'] = np.clip(df['trade_quality_score'], 0, 100)
        
        # 스케일러는 호환성을 위해 유지
        if risk_scaler is None or eff_scaler is None:
            self.trade_quality_scalers['risk_scaler'] = None
            self.trade_quality_scalers['efficiency_scaler'] = None
        
        if verbose:
            print(f"   Trade Quality 점수 생성 완료")
            print(f"  범위: {df['trade_quality_score'].min():.4f} ~ {df['trade_quality_score'].max():.4f}")
            print(f"  평균: {df['trade_quality_score'].mean():.4f}")

        
        return df
    
    def _calculate_exit_timing_without_return(self, df):
        """청산 타이밍 품질 계산 (데이터 기반)"""
        
        # 1. 보유 기간 적절성 (분위수 기반)
        holding_percentiles = np.percentile(df['holding_period_days'], [25, 50, 75])
        p25, p50, p75 = holding_percentiles
        
        # 25%-75% 구간이 적절한 보유기간으로 평가
        holding_score = np.where(
            (df['holding_period_days'] >= p25) & (df['holding_period_days'] <= p75), 80,  # 적절한 구간
            np.where(df['holding_period_days'] < p25, 60,  # 너무 짧음
                np.where(df['holding_period_days'] <= p50 * 2, 70, 50))  # 적당히 김 vs 너무 김
        )
        
        # 2. VIX 변화 대응 (분위수 기반)
        if 'change_vix' in df.columns:
            vix_percentiles = np.percentile(df['change_vix'], [25, 75])
            vix_p25, vix_p75 = vix_percentiles
            
            # VIX 상위 25% 상승 시 청산 = 좋은 판단
            vix_response = np.where(
                df['change_vix'] >= vix_p75, 80,  # 상위 25% VIX 급등
                np.where(df['change_vix'] >= 0, 60,  # 일반적 VIX 상승
                    np.where(df['change_vix'] >= vix_p25, 50, 40))  # VIX 하락
            )
        else:
            vix_response = 50
        
        # 3. 모멘텀 대응 (분위수 기반)
        if 'exit_momentum_20d' in df.columns:
            momentum_percentiles = np.percentile(df['exit_momentum_20d'], [25, 75])
            momentum_p25, momentum_p75 = momentum_percentiles
            
            # 하위 25% 모멘텀에서 청산 = 좋은 타이밍
            momentum_response = np.where(
                df['exit_momentum_20d'] <= momentum_p25, 80,  # 하위 25% 약한 모멘텀
                np.where(df['exit_momentum_20d'] <= 0, 70,  # 음수 모멘텀
                    np.where(df['exit_momentum_20d'] <= momentum_p75, 60, 40))  # 강한 모멘텀
            )
        else:
            momentum_response = 50
        
        # 종합 청산 타이밍 점수 (수익률과 무관)
        exit_timing_score = (
            holding_score * 0.4 +
            vix_response * 0.3 +
            momentum_response * 0.3
        )
        
        return exit_timing_score
    
    def _score_return(self, return_pct):
        """
        수익률을 0-100 점수로 변환 (데이터 분포 기반)
        """
        # 데이터 기반 분위수 계산
        percentiles = np.percentile(return_pct, [10, 25, 50, 75, 90])
        p10, p25, p50, p75, p90 = percentiles
        
        # 분위수 기반 점수 할당
        return np.where(
            return_pct >= p90, 100,     # 상위 10% 
            np.where(return_pct >= p75, 85,      # 상위 25%
            np.where(return_pct >= p50, 70,      # 상위 50% (중앙값 이상)
            np.where(return_pct >= p25, 55,      # 상위 75%
            np.where(return_pct >= p10, 40,      # 상위 90%
            np.where(return_pct >= 0, 25,        # 수익은 내지만 하위 10%
            10))))))  # 손실

    def prepare_features(self, df, verbose=False):
        """거래 품질 예측용 피처 준비"""
        if verbose:
            print("품질 평가용 피처 준비")

        excluded_features = {
            'return_pct', 'entry_volatility_20d', 'entry_ratio_52w_high', 'holding_period_days',
            'risk_adj_return', 'price_safety', 'risk_management_score',
            'time_efficiency', 'efficiency_score', 'quality_score', 'trade_quality_score'
        }
        

        available_features = []
        
        # 1. 기본 거래 정보
        basic_features = ['position_size_pct']
        available_features.extend([col for col in basic_features if col in df.columns])
        
        # 2. 진입 시점 기술적 지표
        entry_features = [
            'entry_momentum_5d', 'entry_momentum_20d', 'entry_momentum_60d',
            'entry_ma_dev_5d', 'entry_ma_dev_20d', 'entry_ma_dev_60d',
            'entry_volatility_5d', 'entry_volatility_60d',
            'entry_vol_change_5d', 'entry_vol_change_20d', 'entry_vol_change_60d',
            'entry_vix', 'entry_tnx_yield'
        ]
        available_features.extend([col for col in entry_features if col in df.columns])
        
        #  3. 종료 시점 지표
        exit_features = [
            'exit_momentum_5d', 'exit_momentum_20d', 'exit_momentum_60d',
            'exit_ma_dev_5d', 'exit_ma_dev_20d', 'exit_ma_dev_60d',
            'exit_volatility_5d', 'exit_volatility_20d', 'exit_volatility_60d',
            'exit_vix', 'exit_tnx_yield', 'exit_ratio_52w_high'
        ]
        available_features.extend([col for col in exit_features if col in df.columns])
        
        # 4. 변화량 지표
        change_features = [
            'change_momentum_5d', 'change_momentum_20d', 'change_momentum_60d',
            'change_ma_dev_5d', 'change_ma_dev_20d', 'change_ma_dev_60d',
            'change_volatility_5d', 'change_volatility_20d', 'change_volatility_60d',
            'change_vix', 'change_tnx_yield', 'change_ratio_52w_high'
        ]
        available_features.extend([col for col in change_features if col in df.columns])
        
        # 5. 보유 기간 중 시장 정보
        market_features = [
            'market_return_during_holding',
            'excess_return'
        ]
        available_features.extend([col for col in market_features if col in df.columns])
        

        self.features = [col for col in available_features 
                        if col in df.columns and col not in excluded_features]
        
        if verbose:
            print(f"  사용 피처: {len(self.features)}개")

        feature_data = df[self.features].select_dtypes(include=[np.number])
        
        if verbose and len(feature_data.columns) != len(self.features):
            print(f"  비숫자형 컬럼 제외: {len(self.features) - len(feature_data.columns)}개")
        
        return feature_data

    def train_model(self, df, hyperparameter_search=False, verbose=False):
        """거래 품질 모델 훈련"""
        if verbose:
            print("거래 품질 모델 훈련 시작")
        
        # 품질 점수 생성
        df_with_score = self.create_quality_score(df, verbose=verbose)
        
        # 피처 준비
        X = self.prepare_features(df_with_score, verbose=verbose)
        y = df_with_score['trade_quality_score']
        
        # 하이퍼파라미터 최적화
        if hyperparameter_search:
            best_params = self._optimize_hyperparameters(X, y, verbose=verbose)
        else:
            # 기본 파라미터
            best_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42
            }
        
        # 최종 모델 훈련
        self.model = xgb.XGBRegressor(**best_params)
        self.model.fit(X, y)
        
        # 성능 평가
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        self.is_trained = True
        
        if verbose:
            print(f"  거래 품질 모델 훈련 완료")
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
        
        return {
            'model': self.model,
            'r2_score': r2,
            'rmse': rmse,
            'best_params': best_params,
            'feature_count': len(self.features)
        }

    def predict_quality(self, df, verbose=False):
        """
        거래 품질 예측
        """
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않음.")
        
        if verbose:
            print("거래 품질 예측")
        
        # 피처 준비
        X = self.prepare_features(df, verbose=False)
        
        # 예측
        predictions = self.model.predict(X)
        
        if verbose:
            print(f"  {len(predictions)}개 거래의 품질 예측 완료")
            print(f"  예측 범위: {predictions.min():.4f} ~ {predictions.max():.4f}")
        
        return predictions

    def _optimize_hyperparameters(self, X, y, verbose=False):
        """하이퍼파라미터 최적화"""
        if verbose:
            print("  하이퍼파라미터 최적화 중")
        
        param_grid = {
            'max_depth': [4, 5, 6, 7, 8],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'n_estimators': [200, 300, 400, 500],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0]
        }
        

        base_model = xgb.XGBRegressor(
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0
        )
        tscv = TimeSeriesSplit(n_splits=3)
        

        search = GridSearchCV(
            base_model, param_grid, 
            cv=tscv, scoring='r2',
            verbose=1
        )
        search.fit(X, y)
        
        if verbose:
            print(f"  최적 R² Score: {search.best_score_:.4f}")
        
        return search.best_params_

    def save_model(self, filename=None):
        """모델 저장"""
        if not self.is_trained:
            raise ValueError("훈련된 모델이 없음")
        
        if filename is None:
            filename = f"trade_quality_evaluator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        save_data = {
            'model': self.model,
            'scalers': self.trade_quality_scalers,
            'features': self.features,
            'model_type': 'TRADE_QUALITY_EVALUATOR',
            'created_at': datetime.now().isoformat()
        }
        
        joblib.dump(save_data, filename)
        print(f" Trade Quality 모델 저장: {filename}")
        return filename

    def load_model(self, filename):
        """모델 로드"""
        save_data = joblib.load(filename)
        
        self.model = save_data['model']
        self.trade_quality_scalers = save_data['scalers'] 
        self.features = save_data['features']
        self.is_trained = True
        
        print(f" Trade Quality 모델 로드: {filename}")
        return True

    # ================================
    # Walk-Forward 학습 파이프라인 
    # ================================
    
    def create_time_folds(self, df, verbose=False):
        if verbose:
            print(" Trade Quality Walk-Forward 시간 폴드 생성")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['entry_datetime'])
        df = df.sort_values('date').reset_index(drop=True)
        
        start_date = df['date'].min()
        end_date = df['date'].max()
        
        folds = []
        current_start = start_date
        fold_id = 1
        
        while current_start + pd.DateOffset(months=self.train_months + self.val_months + self.test_months) <= end_date:
            train_end = current_start + pd.DateOffset(months=self.train_months)
            val_end = train_end + pd.DateOffset(months=self.val_months)
            test_end = val_end + pd.DateOffset(months=self.test_months)
            
            train_mask = (df['date'] >= current_start) & (df['date'] < train_end)
            val_mask = (df['date'] >= train_end) & (df['date'] < val_end)
            test_mask = (df['date'] >= val_end) & (df['date'] < test_end)
            
            if train_mask.sum() > 1000 and val_mask.sum() > 100 and test_mask.sum() > 100:
                fold_info = {
                    'fold_id': fold_id,
                    'train_start': current_start.strftime('%Y-%m-%d'),
                    'train_end': train_end.strftime('%Y-%m-%d'),
                    'val_start': train_end.strftime('%Y-%m-%d'),
                    'val_end': val_end.strftime('%Y-%m-%d'),
                    'test_start': val_end.strftime('%Y-%m-%d'),
                    'test_end': test_end.strftime('%Y-%m-%d'),
                    'train_indices': df[train_mask].index.tolist(),
                    'val_indices': df[val_mask].index.tolist(),
                    'test_indices': df[test_mask].index.tolist()
                }
                folds.append(fold_info)
                fold_id += 1
            
            current_start += pd.DateOffset(months=self.step_months)
        
        if verbose:
            print(f"  생성된 폴드 수: {len(folds)}개")
            for i, fold in enumerate(folds):
                print(f"  폴드 {i+1}: {fold['train_start']} ~ {fold['test_end']}")
                print(f"    Train: {len(fold['train_indices']):,}개, Val: {len(fold['val_indices']):,}개, Test: {len(fold['test_indices']):,}개")
        
        return folds
    
    def run_walk_forward_training(self, data_path, hyperparameter_search=True, verbose=True):
        """Trade Quality Walk-Forward 학습 및 평가"""
        if verbose:
            print(" Trade Quality Walk-Forward 학습 시작")
            print("="*60)
        
        # 데이터 로드
        df = pd.read_csv(data_path)
        if verbose:
            print(f" 데이터 로드: {len(df):,}개 거래")
        
        # Trade Quality 점수 생성
        df = self.create_quality_score(df, verbose=verbose)
        
        # 시간 폴드 생성
        folds = self.create_time_folds(df, verbose=verbose)
        
        fold_results = []
        
        for fold_info in tqdm(folds, desc="폴드별 학습"):
            if verbose:
                print(f"\n 폴드 {fold_info['fold_id']} 학습 중...")
            
            # 폴드별 데이터 분할
            train_data = df.loc[fold_info['train_indices']]
            val_data = df.loc[fold_info['val_indices']]
            test_data = df.loc[fold_info['test_indices']]
            
            # 피처 준비
            X_train = self.prepare_features(train_data, verbose=False)
            X_val = self.prepare_features(val_data, verbose=False)
            X_test = self.prepare_features(test_data, verbose=False)
            
            y_train = train_data['trade_quality_score']
            y_val = val_data['trade_quality_score']
            y_test = test_data['trade_quality_score']
            
            # 하이퍼파라미터 최적화
            if hyperparameter_search:
                search_result = self._optimize_hyperparameters(X_train, y_train, verbose=False)
                best_params = search_result
            else:
                # 기본 파라미터 (Trade Quality 특화)
                best_params = {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 300,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42
                }
            
            # 최적 파라미터로 모델 학습
            best_model = xgb.XGBRegressor(**best_params)
            best_model.fit(X_train, y_train)
            
            # 검증 및 테스트 평가
            val_pred = best_model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            
            test_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, test_pred)

            # 테스트셋 점수-수익률 상관관계 분석
            if verbose:
                self.calculate_ranking_performance(test_pred, y_test, verbose=True)

            # 결과 저장
            fold_result = {
                'fold_id': fold_info['fold_id'],
                'fold_info': fold_info,
                'best_params': best_params,
                'val_r2': val_r2,
                'test_r2': test_r2,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'features_used': len(X_train.columns),
                'best_model': best_model
            }
            
            fold_results.append(fold_result)
            
            if verbose:
                print(f"  폴드 {fold_info['fold_id']} 완료: Val R² = {val_r2:.4f}, Test R² = {test_r2:.4f}")
        
        self.fold_results = fold_results
        
        # 최고 성능 모델을 최종 모델로 선택
        best_fold = max(fold_results, key=lambda x: x['test_r2'])
        self.model = best_fold['best_model']
        self.best_params = best_fold['best_params']
        self.is_trained = True
        
        # 전체 결과 요약
        if verbose:
            self._print_fold_summary()
        
        return fold_results
    
    def _print_fold_summary(self):
        """폴드별 결과 요약 출력"""
        if not self.fold_results:
            print(" 폴드 결과가 없습니다.")
            return
        
        print("\n" + "="*70)
        print(" Trade Quality Walk-Forward 결과 요약")
        print("="*70)
        
        val_r2_scores = [result['val_r2'] for result in self.fold_results]
        test_r2_scores = [result['test_r2'] for result in self.fold_results]
        
        print(f" 폴드별 성능:")
        for result in self.fold_results:
            print(f"  폴드 {result['fold_id']}: Val R² = {result['val_r2']:.4f}, Test R² = {result['test_r2']:.4f}")
        
        print(f"\n 전체 통계:")
        print(f"  Validation R²: {np.mean(val_r2_scores):.4f} ± {np.std(val_r2_scores):.4f}")
        print(f"  Test R²:       {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")
        print(f"  최고 성능:     {np.max(test_r2_scores):.4f} (폴드 {np.argmax(test_r2_scores) + 1})")
        print(f"  평균 피처 수:  {np.mean([r['features_used'] for r in self.fold_results]):.0f}개")
        
        print("="*70)
    
    def calculate_ranking_performance(self, predictions, actuals, verbose=True):
        if len(predictions) != len(actuals):
            raise ValueError("예측값과 실제값의 길이가 다름")
        
        # 1. 상관관계 분석
        spearman_corr, spearman_p = spearmanr(predictions, actuals)
        pearson_corr = np.corrcoef(predictions, actuals)[0, 1]
        
        # 2. 구간별 분석
        quintiles = np.quantile(predictions, [0.2, 0.4, 0.6, 0.8])
        
        results = {
            'correlations': {
                'spearman': spearman_corr,
                'spearman_pvalue': spearman_p,
                'pearson': pearson_corr
            },
            'quintile_analysis': []
        }
        
        if verbose:
            print("\n" + "="*60)
            print(" 점수-수익률 상관관계 분석")
            print("="*60)
            print(f" Spearman 상관계수: {spearman_corr:.4f} (p={spearman_p:.4f})")
            print(f" Pearson 상관계수:  {pearson_corr:.4f}")
            
            print(f"\n 점수 구간별 분석 (총 {len(predictions):,}개 샘플):")
            print("-" * 60)
        
        # 각 분위별 분석
        quintile_names = ['하위 20%', '하위중 20%', '중위 20%', '상위중 20%', '상위 20%']
        
        for i in range(5):
            if i == 0:
                mask = predictions <= quintiles[0]
            elif i == 4:
                mask = predictions > quintiles[3]
            else:
                mask = (predictions > quintiles[i-1]) & (predictions <= quintiles[i])
            
            if np.sum(mask) > 0:
                quintile_actuals = actuals[mask]
                quintile_preds = predictions[mask]
                
                quintile_result = {
                    'quintile': i + 1,
                    'name': quintile_names[i],
                    'count': int(np.sum(mask)),
                    'pred_range': [float(quintile_preds.min()), float(quintile_preds.max())],
                    'actual_mean': float(quintile_actuals.mean()),
                    'actual_std': float(quintile_actuals.std()),
                    'actual_median': float(np.median(quintile_actuals))
                }
                
                results['quintile_analysis'].append(quintile_result)
                
                if verbose:
                    print(f"{quintile_names[i]:>8} | "
                          f"샘플: {np.sum(mask):>5,}개 | "
                          f"예측범위: [{quintile_preds.min():>6.2f}, {quintile_preds.max():>6.2f}] | "
                          f"실제평균: {quintile_actuals.mean():>7.3f} ± {quintile_actuals.std():>6.3f}")
        
        # 3. 단조성 검사
        quintile_means = [q['actual_mean'] for q in results['quintile_analysis']]
        is_monotonic = all(quintile_means[i] <= quintile_means[i+1] for i in range(len(quintile_means)-1))
        
        results['monotonicity'] = {
            'is_monotonic': is_monotonic,
            'mean_difference': quintile_means[-1] - quintile_means[0] if len(quintile_means) >= 2 else 0
        }
        
        # 4. Top/Bottom 분석
        top20_mask = predictions >= np.percentile(predictions, 80)
        bottom20_mask = predictions <= np.percentile(predictions, 20)
        
        results['top_bottom_analysis'] = {
            'top20_mean': float(actuals[top20_mask].mean()),
            'bottom20_mean': float(actuals[bottom20_mask].mean()),
            'spread': float(actuals[top20_mask].mean() - actuals[bottom20_mask].mean())
        }
        
        if verbose:
            print("-" * 60)
            print(f" 단조성 검사: {' 통과' if is_monotonic else ' 실패'}")
            print(f" 상하위 스프레드: {results['top_bottom_analysis']['spread']:.3f}")
            print(f"   - 상위 20% 평균: {results['top_bottom_analysis']['top20_mean']:.3f}")
            print(f"   - 하위 20% 평균: {results['top_bottom_analysis']['bottom20_mean']:.3f}")
            print("="*60)
        
        return results
    
    def save_training_results(self, filename=None):
        """학습 결과 저장 """
        if filename is None:
            filename = f"trade_quality_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        save_data = {
            'model_type': 'TRADE_QUALITY_EVALUATOR',
            'model_name': 'Trade Quality Evaluator',
            'created_at': datetime.now().isoformat(),
            'total_folds': len(self.fold_results),
            'best_params': self.best_params,
            'fold_results': []
        }
        
        for result in self.fold_results:
            fold_data = {key: value for key, value in result.items() 
                        if key != 'best_model'}
            save_data['fold_results'].append(fold_data)
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f" Trade Quality 학습 결과 저장: {filename}")
        return filename

def main():
    """Trade Quality Evaluator 학습 파이프라인 실행"""
    print(" Trade Quality Evaluator - 거래 품질 평가 모델 학습")
    print("="*70)

    
    # 데이터 경로 설정
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../results/final/trading_episodes_with_rebuilt_market_component.csv')
    
    # 파일 존재 확인
    if not os.path.exists(data_path):
        print(f" 데이터 파일을 찾을 수 없습니다: {data_path}")
        return
    
    # 모델 초기화
    evaluator = TradeQualityEvaluator()
    
    # 분할 학습 실행
    try:
        # 데이터 로드
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        import numpy as np
        
        df = pd.read_csv(data_path)
        print(f"\n데이터 로드: {len(df):,}개 거래")
        

        df_filtered = df[
            df['entry_pe_ratio'].notna() & 
            df['entry_roe'].notna() & 
            df['entry_earnings_growth'].notna() &
            df['return_pct'].notna() &
            df['holding_period_days'].notna() &
            df['entry_volatility_20d'].notna() &
            df['entry_ratio_52w_high'].notna()
        ].copy()
        
        print(f"📊 펀더멘털 데이터 필터링: {len(df_filtered):,}개 ({len(df_filtered)/len(df)*100:.1f}%)")
        
        # Train/Val/Test 분할 (60/20/20)
        train_val_df, test_df = train_test_split(df_filtered, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
        
        print(f"\n📊 데이터 분할:")
        print(f"  Train: {len(train_df):,}개 ({len(train_df)/len(df_filtered)*100:.1f}%)")
        print(f"  Val:   {len(val_df):,}개 ({len(val_df)/len(df_filtered)*100:.1f}%)")
        print(f"  Test:  {len(test_df):,}개 ({len(test_df)/len(df_filtered)*100:.1f}%)")
        
        # 모델 학습
        print(f"\n 모델 학습 시작...")
        result = evaluator.train_model(train_df, hyperparameter_search=False, verbose=True)
        
        # 평가 함수
        def evaluate_model(evaluator, data, name):
            data_with_score = evaluator.create_quality_score(data, verbose=False)
            X = evaluator.prepare_features(data_with_score, verbose=False)
            y = data_with_score['trade_quality_score']
            y_pred = evaluator.model.predict(X)
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            return {
                'name': name,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'y_mean': y.mean(),
                'y_std': y.std(),
                'pred_mean': y_pred.mean(),
                'pred_std': y_pred.std()
            }
        
        # 각 세트 평가
        train_metrics = evaluate_model(evaluator, train_df, 'Train')
        val_metrics = evaluate_model(evaluator, val_df, 'Val')
        test_metrics = evaluate_model(evaluator, test_df, 'Test')
        
        # 성과 출력
        print(f"\n 성과 지표:")
        print("="*60)
        print(f"{'Dataset':<10} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Mean':>8} {'Std':>8}")
        print("-"*60)
        for metrics in [train_metrics, val_metrics, test_metrics]:
            print(f"{metrics['name']:<10} {metrics['r2']:>8.4f} {metrics['rmse']:>8.4f} {metrics['mae']:>8.4f} {metrics['y_mean']:>8.4f} {metrics['y_std']:>8.4f}")
        
        # 오버피팅 체크
        overfit_score = train_metrics['r2'] - val_metrics['r2']
        print(f"\n 오버피팅 분석:")
        if overfit_score > 0.05:
            print(f"  ️  오버피팅 가능성: Train-Val R² 차이 = {overfit_score:.4f}")
        else:
            print(f"   오버피팅 없음: Train-Val R² 차이 = {overfit_score:.4f}")
        
        # Val-Test 성능 안정성
        stability_score = abs(val_metrics['r2'] - test_metrics['r2'])
        print(f"\n📏 성능 안정성:")
        if stability_score < 0.05:
            print(f"   안정적: Val-Test R² 차이 = {stability_score:.4f}")
        else:
            print(f"    불안정: Val-Test R² 차이 = {stability_score:.4f}")
        
        # 모델 저장
        model_filename = evaluator.save_model()
        
        print(f"\n Trade Quality 모델 학습 완료!")
        print(f" 저장된 모델: {model_filename}")
        
        # 사용법 안내
        print(f"\n 모델 사용법:")
        print(f"evaluator = TradeQualityEvaluator()")
        print(f"evaluator.load_model('{model_filename}')")
        print(f"quality_scores = evaluator.predict_quality(completed_trades_df)")
        
        return evaluator
        
    except Exception as e:
        print(f" 학습 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()