import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SellSignalPredictor:
    """매도  신호 예측 모델
    - 타이밍 적절성 (40%): 보유 기간과 수익률의 효율성
    - 수익 실현 품질 (35%): 손익 관리의 적절성  
    - 시장 대응 (25%): 시장 상황 변화에 대한 대응력
    """

    def __init__(self, train_months=36, val_months=6, test_months=6, step_months=3):
        self.model = None
        self.sell_signal_scalers = {}
        self.features = None
        self.is_trained = False

        self.train_months = train_months
        self.val_months = val_months  
        self.test_months = test_months
        self.step_months = step_months

        self.fold_results = []
        self.best_params = None
        
    def create_exit_signal_score(self, df, timing_scaler=None, profit_scaler=None, market_scaler=None, verbose=False):
        """매도 신호 점수 생성"""
        if verbose:
            print("매도 신호 점수 생성 중")

        df = df.copy()

        required_columns = ['return_pct', 'holding_period_days', 'exit_volatility_20d', 
                          'exit_momentum_20d', 'change_volatility_5d', 'change_vix']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"컬럼이 없음: {missing_columns}")
        
        df['return_pct'] = df['return_pct'].fillna(0)
        df['holding_period_days'] = df['holding_period_days'].fillna(1)
        df['exit_volatility_20d'] = df['exit_volatility_20d'].fillna(20)
        df['exit_momentum_20d'] = df['exit_momentum_20d'].fillna(0)
        df['change_volatility_5d'] = df['change_volatility_5d'].fillna(0)
        df['change_vix'] = df['change_vix'].fillna(0)

        # 1. 타이밍 적절성 점수 (40%)
        # 변동성 고려한 기간별 위험 조정
        annual_vol = df['exit_volatility_20d']
        period_vol = annual_vol * np.sqrt(df['holding_period_days'] / 365)
        period_vol_safe = np.maximum(period_vol, 1)
        
        # 변동성 대비 수익률
        df['vol_adjusted_efficiency'] = df['return_pct'] / period_vol_safe
        
        # 로그 변환
        efficiency_scaled = df['vol_adjusted_efficiency'] * 5
        df['timing_score_raw'] = np.sign(efficiency_scaled) * np.log1p(np.abs(efficiency_scaled))

        #2. 수익 실현 품질 점수 (35%)
        # 데이터 분포를 기반
        return_std = df['return_pct'].std()
        return_median = df['return_pct'].median()
        
        # 수익률 점수
        df['return_score'] = np.tanh((df['return_pct'] - return_median) / return_std) * 3
        
        # 리스크 조정 수익률
        volatility_safe = np.maximum(df['exit_volatility_20d'], 1)
        risk_ratio = df['return_pct'] / volatility_safe
        risk_std = risk_ratio.std()
        risk_median = risk_ratio.median()
        df['risk_adjusted_score'] = np.tanh((risk_ratio - risk_median) / risk_std) * 2
        
        # 보유기간 효율성
        period_efficiency = df['return_pct'] / np.log1p(df['holding_period_days'])
        efficiency_std = period_efficiency.std()
        efficiency_median = period_efficiency.median()
        df['period_efficiency_score'] = np.tanh((period_efficiency - efficiency_median) / efficiency_std) * 1
        
        # 최종 품질 점수 조합
        df['profit_quality_raw'] = (df['return_score'] * 0.5 + 
                                   df['risk_adjusted_score'] * 0.3 + 
                                   df['period_efficiency_score'] * 0.2)

        # 3. 시장 대응 점수 (25%) - 데이터 기반 상호작용
        # 각 시장 지표를 데이터 기반으로 정규화
        momentum_std = df['exit_momentum_20d'].std()
        momentum_median = df['exit_momentum_20d'].median()
        df['momentum_normalized'] = (df['exit_momentum_20d'] - momentum_median) / momentum_std
        
        vix_change_std = df['change_vix'].std()
        vix_change_median = df['change_vix'].median()
        df['vix_change_normalized'] = (df['change_vix'] - vix_change_median) / vix_change_std
        
        vol_change_std = df['change_volatility_5d'].std()
        vol_change_median = df['change_volatility_5d'].median()
        df['vol_change_normalized'] = (df['change_volatility_5d'] - vol_change_median) / vol_change_std
        
        # 수익률과 시장 지표의 상호작용을 연속함수로
        df['momentum_interaction'] = np.tanh(df['momentum_normalized'] * np.sign(df['return_pct'])) * 1.5
        df['vix_interaction'] = np.tanh(df['vix_change_normalized'] * np.tanh(df['return_pct'] / 5)) * 1.0
        df['vol_interaction'] = np.tanh(df['vol_change_normalized'] * np.tanh(df['return_pct'] / 8)) * 0.8
        
        # 시장 대응 점수 조합 (데이터가 알아서 패턴 찾게)
        df['market_response_raw'] = (df['momentum_interaction'] * 0.5 + 
                                    df['vix_interaction'] * 0.3 + 
                                    df['vol_interaction'] * 0.2)

        # 최종 점수 계산
        # 각 구성 요소별 스케일링
        if timing_scaler is None or profit_scaler is None or market_scaler is None:
            timing_scaler = RobustScaler()
            profit_scaler = RobustScaler() 
            market_scaler = RobustScaler()
            
            timing_scaled = timing_scaler.fit_transform(df[['timing_score_raw']])
            profit_scaled = profit_scaler.fit_transform(df[['profit_quality_raw']])
            market_scaled = market_scaler.fit_transform(df[['market_response_raw']])
            
            self.sell_signal_scalers['timing_scaler'] = timing_scaler
            self.sell_signal_scalers['profit_scaler'] = profit_scaler
            self.sell_signal_scalers['market_scaler'] = market_scaler
        else:
            timing_scaled = timing_scaler.transform(df[['timing_score_raw']])
            profit_scaled = profit_scaler.transform(df[['profit_quality_raw']])
            market_scaled = market_scaler.transform(df[['market_response_raw']])
        
        # 가중 평균으로 최종 점수 계산
        df['sell_signal_score'] = (timing_scaled.flatten() * 0.4 + 
                                  profit_scaled.flatten() * 0.35 + 
                                  market_scaled.flatten() * 0.25)
        
        if verbose:
            print(f"  매도 점수 생성 완료")
            print(f"  범위: {df['sell_signal_score'].min():.4f} ~ {df['sell_signal_score'].max():.4f}")
            print(f"  평균: {df['sell_signal_score'].mean():.4f}")

        
        return df

    def prepare_features(self, df, verbose=False):
        """ 피처 준비"""
        if verbose:
            print("매도 피처 준비")
        

        excluded_features = {
            'return_pct', 'holding_period_days', 'exit_volatility_20d', 'exit_momentum_20d',
            'change_volatility_5d', 'change_vix',
            # 중간 계산 변수들 (데이터 기반 변수명 반영)
            'vol_adjusted_efficiency', 'timing_score_raw', 'return_score', 
            'risk_adjusted_score', 'period_efficiency_score', 'profit_quality_raw',
            'momentum_normalized', 'vix_change_normalized', 'vol_change_normalized',
            'momentum_interaction', 'vix_interaction', 'vol_interaction',
            'market_response_raw', 'sell_signal_score'
        }
        

        available_features = []
        
        # 1. 기본 거래 정보
        basic_features = ['position_size_pct']
        available_features.extend([col for col in basic_features if col in df.columns])
        

        entry_features = [
            'entry_momentum_5d', 'entry_momentum_20d', 'entry_momentum_60d',
            'entry_ma_dev_5d', 'entry_ma_dev_20d', 'entry_ma_dev_60d',
            'entry_volatility_5d', 'entry_volatility_60d',
            'entry_vol_change_5d', 'entry_vol_change_20d', 'entry_vol_change_60d',
            'entry_vix', 'entry_tnx_yield', 'entry_ratio_52w_high'
        ]
        available_features.extend([col for col in entry_features if col in df.columns])
        
        # 3.  시점 정보
        exit_features = [
            'exit_momentum_5d', 'exit_momentum_60d',
            'exit_ma_dev_5d', 'exit_ma_dev_20d', 'exit_ma_dev_60d',
            'exit_volatility_5d', 'exit_volatility_60d',
            'exit_vix', 'exit_tnx_yield', 'exit_ratio_52w_high'
        ]
        available_features.extend([col for col in exit_features if col in df.columns])
        
        # 4. 보유 기간 중 변화
        change_features = [
            'change_momentum_5d', 'change_momentum_20d', 'change_momentum_60d',
            'change_ma_dev_5d', 'change_ma_dev_20d', 'change_ma_dev_60d',
            'change_volatility_20d', 'change_volatility_60d',  # change_volatility_5d 제외
            'change_tnx_yield', 'change_ratio_52w_high'

        ]
        available_features.extend([col for col in change_features if col in df.columns])
        
        # 5. 시장 환경 정보
        market_features = [
            'market_return_during_holding',
            'excess_return'
        ]
        available_features.extend([col for col in market_features if col in df.columns])
        

        self.features = [col for col in available_features 
                        if col in df.columns and col not in excluded_features]
        
        if verbose:
            print(f"  매도 사용 피처: {len(self.features)}개")

        feature_data = df[self.features].select_dtypes(include=[np.number])
        
        if verbose and len(feature_data.columns) != len(self.features):
            print(f"  비숫자형 컬럼 제외: {len(self.features) - len(feature_data.columns)}개")
        
        return feature_data

    def train_model(self, df, hyperparameter_search=False, verbose=False):
        """매도  신호 예측 모델 훈련"""
        if verbose:
            print("매도 신호 모델 훈련 시작")
        
        #  신호 점수 생성
        df_with_score = self.create_exit_signal_score(df, verbose=verbose)
        
        # 피처 준비
        X = self.prepare_features(df_with_score, verbose=verbose)
        y = df_with_score['sell_signal_score']
        
        # 하이퍼파라미터 최적화
        if hyperparameter_search:
            best_params = self._optimize_hyperparameters(X, y, verbose=verbose)
        else:
            # 기본 파라미터
            best_params = {
                'max_depth': 7,
                'learning_rate': 0.05,
                'n_estimators': 600,
                'subsample': 0.8,
                'colsample_bytree': 0.9,
                'reg_alpha': 1.0,
                'reg_lambda': 5.0,
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
            print(f"  매도 신호 예측 모델 훈련 완료")
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
        
        return {
            'model': self.model,
            'r2_score': r2,
            'rmse': rmse,
            'best_params': best_params,
            'feature_count': len(self.features)
        }

    def predict_exit_signal(self, df, verbose=False):
        """신호 강도 예측 """
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않음.")
        
        if verbose:
            print("매도 신호 예측")
        
        # 피처 준비
        X = self.prepare_features(df, verbose=False)
        
        # 예측
        predictions = self.model.predict(X)
        
        if verbose:
            print(f"  {len(predictions)}개 포지션의  신호 예측 완료")
            print(f"  신호 강도 범위: {predictions.min():.4f} ~ {predictions.max():.4f}")
            print(f"  평균 신호 강도: {predictions.mean():.4f}")
        
        return predictions

    def get_signal_interpretation(self, score):
        """신호 점수 해석"""
        if score > 2:
            return "즉시 매도 권장"
        elif score > 1:
            return "강한 매도 신호"
        elif score > 0:
            return "중간 매도 신호"
        elif score > -1:
            return "약한 매도 신호"
        else:
            return "보유 유지"

    def _optimize_hyperparameters(self, X, y, verbose=False):
        """하이퍼파라미터 최적화"""
        if verbose:
            print("  하이퍼파라미터 최적화 중...")
        
        param_grid = {
            'max_depth': [5, 6, 7, 8, 9],
            'learning_rate': [0.02, 0.05, 0.07, 0.1],
            'n_estimators': [400, 500, 600, 800],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.8, 0.9, 0.95],
            'reg_alpha': [0.5, 1.0, 2.0],
            'reg_lambda': [3.0, 5.0, 8.0]
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
            raise ValueError("훈련된 모델이 없습니다.")
        
        if filename is None:
            filename = f"sell_signal_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        save_data = {
            'model': self.model,
            'scalers': self.sell_signal_scalers,
            'features': self.features,
            'model_type': 'SELL_SIGNAL_PREDICTOR',
            'created_at': datetime.now().isoformat()
        }
        
        joblib.dump(save_data, filename)
        print(f" Sell Signal 모델 저장: {filename}")
        return filename

    def load_model(self, filename):
        """모델 로드"""
        save_data = joblib.load(filename)
        
        self.model = save_data['model']
        self.sell_signal_scalers = save_data['scalers']
        self.features = save_data['features']
        self.is_trained = True
        
        print(f"📂 Sell Signal 모델 로드: {filename}")
        return True

    # ================================
    # Walk-Forward 학습 파이프라인 
    # ================================
    
    def create_time_folds(self, df, verbose=False):
        if verbose:
            print("Sell Signal Walk-Forward 시간 폴드 생성")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['exit_date'])
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

        if verbose:
            print("Sell Signal Walk-Forward 학습 시작")
            print("="*60)
        
        # 데이터 로드
        df = pd.read_csv(data_path)
        if verbose:
            print(f"데이터 로드: {len(df):,}개 거래")
        
        # Sell Signal 점수 생성
        df = self.create_exit_signal_score(df, verbose=verbose)
        
        # 시간 폴드 생성
        folds = self.create_time_folds(df, verbose=verbose)
        
        fold_results = []
        
        for fold_info in tqdm(folds, desc="폴드별 학습"):
            if verbose:
                print(f"\n 폴드 {fold_info['fold_id']} 학습 중")
            
            # 폴드별 데이터 분할
            train_data = df.loc[fold_info['train_indices']]
            val_data = df.loc[fold_info['val_indices']]
            test_data = df.loc[fold_info['test_indices']]
            
            # 피처 준비
            X_train = self.prepare_features(train_data, verbose=False)
            X_val = self.prepare_features(val_data, verbose=False)
            X_test = self.prepare_features(test_data, verbose=False)
            
            y_train = train_data['sell_signal_score']
            y_val = val_data['sell_signal_score']
            y_test = test_data['sell_signal_score']

            if hyperparameter_search:
                search_result = self._optimize_hyperparameters(X_train, y_train, verbose=False)
                best_params = search_result
            else:

                best_params = {
                    'tree_method': 'hist',
                    'subsample': 0.75, 
                    'scale_pos_weight': 10, 
                    'reg_lambda': 20.0, 
                    'reg_alpha': 1.0, 
                    'objective': 'reg:squarederror', 
                    'n_estimators': 800, 
                    'min_child_weight': 2, 
                    'max_leaves': 255, 
                    'max_depth': 9, 
                    'max_delta_step': 0, 
                    'max_bin': 128, 
                    'learning_rate': 0.01, 
                    'grow_policy': 'depthwise', 
                    'gamma': 0.5, 
                    'eval_metric': 'rmse', 
                    'colsample_bytree': 0.9, 
                    'colsample_bynode': 0.8, 
                    'colsample_bylevel': 0.8, 
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
            print("폴드 결과가 없습니다.")
            return
        
        print("\n" + "="*70)
        print("Sell Signal Walk-Forward 결과 요약")
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
    
    def save_training_results(self, filename=None):
        """학습 결과 저장 """
        if filename is None:
            filename = f"sell_signal_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        save_data = {
            'model_type': 'SELL_SIGNAL_PREDICTOR',
            'model_name': 'Sell Signal Predictor',
            'created_at': datetime.now().isoformat(),
            'total_folds': len(self.fold_results),
            'best_params': self.best_params,
            'fold_results': []
        }
        
        for result in self.fold_results:
            # 모델 객체 제외하고 저장
            fold_data = {key: value for key, value in result.items() 
                        if key != 'best_model'}
            save_data['fold_results'].append(fold_data)
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"💾 Sell Signal 학습 결과 저장: {filename}")
        return filename

def main():
    """Sell Signal Predictor 학습 파이프라인 실행"""
    print(" Sell Signal Predictor - 매도  신호 예측 모델 학습")
    print("="*70)
    print("="*70)
    
    # 데이터 경로 설정
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../results/final/trading_episodes_with_rebuilt_market_component.csv')
    
    # 파일 존재 확인
    if not os.path.exists(data_path):
        print(f" 데이터 파일을 찾을 수 없음: {data_path}")
        return
    
    # 모델 초기화
    predictor = SellSignalPredictor()
    
    # 랜덤 분할 학습 실행
    try:
        # 데이터 로드
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        import numpy as np
        
        df = pd.read_csv(data_path)
        print(f"\n📊 데이터 로드: {len(df):,}개 거래")
        
        # Train/Val/Test 분할 (60/20/20)
        train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
        
        print(f"\n📊 데이터 분할:")
        print(f"  Train: {len(train_df):,}개 ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val_df):,}개 ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_df):,}개 ({len(test_df)/len(df)*100:.1f}%)")
        
        # 모델 학습
        print(f"\n 모델 학습 시작...")
        result = predictor.train_model(train_df, hyperparameter_search=False, verbose=True)
        
        # 평가 함수
        def evaluate_model(predictor, data, name):
            data_with_score = predictor.create_exit_signal_score(data, verbose=False)
            X = predictor.prepare_features(data_with_score, verbose=False)
            y = data_with_score['sell_signal_score']
            y_pred = predictor.model.predict(X)
            
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
        train_metrics = evaluate_model(predictor, train_df, 'Train')
        val_metrics = evaluate_model(predictor, val_df, 'Val')
        test_metrics = evaluate_model(predictor, test_df, 'Test')
        
        # 성과 출력
        print(f"\n 성과 지표:")
        print("="*60)
        print(f"{'Dataset':<10} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Mean':>8} {'Std':>8}")
        print("-"*60)
        for metrics in [train_metrics, val_metrics, test_metrics]:
            print(f"{metrics['name']:<10} {metrics['r2']:>8.4f} {metrics['rmse']:>8.4f} {metrics['mae']:>8.4f} {metrics['y_mean']:>8.4f} {metrics['y_std']:>8.4f}")
        
        # 오버피팅 체크
        overfit_score = train_metrics['r2'] - val_metrics['r2']
        print(f"\n🔍 오버피팅 분석:")
        if overfit_score > 0.05:
            print(f"   오버피팅 가능성: Train-Val R² 차이 = {overfit_score:.4f}")
        else:
            print(f"   오버피팅 없음: Train-Val R² 차이 = {overfit_score:.4f}")
        
        # Val-Test 성능 안정성
        stability_score = abs(val_metrics['r2'] - test_metrics['r2'])
        print(f"\n📏 성능 안정성:")
        if stability_score < 0.05:
            print(f"   안정적: Val-Test R² 차이 = {stability_score:.4f}")
        else:
            print(f"  ️  불안정: Val-Test R² 차이 = {stability_score:.4f}")
        
        # 모델 저장
        model_filename = predictor.save_model()
        
        print(f"\n Sell Signal 모델 학습 완료!")
        print(f" 저장된 모델: {model_filename}")
        

        return predictor
        
    except Exception as e:
        print(f" 학습 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()