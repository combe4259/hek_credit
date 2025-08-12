import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import ParameterGrid
from scipy.stats import uniform, randint
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
import torch
warnings.filterwarnings('ignore')

class CTypeExitSignalAI:
    """
    C-type 매도 청산 신호 AI
    
    목표: 매도 시점에서 청산 조건의 적절성을 평가
    - 타이밍 적절성: 얼마나 적절한 시점에 팔았는가?
    - 수익 실현 품질: 손익을 얼마나 잘 관리했는가?
    - 시장 대응: 시장 상황에 얼마나 잘 대응했는가?
    """

    def __init__(self, train_months=36, val_months=6, test_months=6, step_months=3, use_global_split=True):
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.step_months = step_months
        self.use_global_split = use_global_split
        
        # ===== C-type: 청산 신호 평가 =====
        self.c_type_exit_model = None         
        self.c_type_exit_scalers = {}         
        self.c_type_exit_features = None      
        
        self.fold_results = []
        self.global_results = None
        self.best_params_c = None
    
    # ================================
    # C-type: 매도 청산 점수화
    # ================================
    
    def create_c_type_exit_score(self, df, timing_scaler=None, profit_scaler=None, market_scaler=None, verbose=False):
        """
        C-type: 청산 시점의 적절성을 평가하는 점수 생성
        
        3가지 핵심 평가 지표:
        1. 타이밍 적절성 (40%): 보유 기간과 수익률의 효율성
        2. 수익 실현 품질 (35%): 손익 관리의 적절성
        3. 시장 대응 (25%): 시장 상황 변화에 대한 대응력
        """
        if verbose:
            print("🛑 C-type: 청산 점수 생성 중...")

        df = df.copy()
        
        # NaN 처리
        df['return_pct'] = df['return_pct'].fillna(0)
        df['holding_period_days'] = df['holding_period_days'].fillna(1)
        df['exit_volatility_20d'] = df['exit_volatility_20d'].fillna(20)
        df['exit_momentum_20d'] = df['exit_momentum_20d'].fillna(0)
        df['change_volatility_5d'] = df['change_volatility_5d'].fillna(0)
        df['change_vix'] = df['change_vix'].fillna(0)

        # ===== 1. 타이밍 적절성 점수 (40%) =====
        # 보유 기간 대비 수익률 효율성
        holding_safe = np.maximum(df['holding_period_days'], 1)
        df['daily_return_efficiency'] = df['return_pct'] / holding_safe
        
        # 보유 기간별 적절성 평가
        df['holding_timing_base'] = np.where(
            df['holding_period_days'] < 3, -2,     # 너무 빠른 청산: 매우 나쁨
            np.where(df['holding_period_days'] < 7, 1,      # 단기 청산: 보통
                    np.where(df['holding_period_days'] < 21, 3,     # 적정 보유: 좋음
                            np.where(df['holding_period_days'] < 60, 2,     # 중장기: 보통
                                    np.where(df['holding_period_days'] < 120, 0, -1)))))  # 장기: 감점
        
        # 수익률에 따른 타이밍 보정
        df['return_timing_adjustment'] = np.where(
            df['return_pct'] > 10, 1.5,    # 큰 수익: 타이밍 보너스
            np.where(df['return_pct'] > 5, 1.2,     # 중간 수익: 약간 보너스
                    np.where(df['return_pct'] > 0, 1.0,     # 소수익: 그대로
                            np.where(df['return_pct'] > -5, 0.8,    # 소손실: 약간 감점
                                    np.where(df['return_pct'] > -15, 0.6, 0.3)))))  # 큰 손실: 큰 감점
        
        df['timing_score_raw'] = df['holding_timing_base'] * df['return_timing_adjustment']

        # ===== 2. 수익 실현 품질 점수 (35%) =====
        # 절대 수익률 평가
        df['absolute_return_score'] = np.where(
            df['return_pct'] > 15, 5,      # 큰 수익: 매우 좋음
            np.where(df['return_pct'] > 8, 4,       # 좋은 수익: 좋음
                    np.where(df['return_pct'] > 3, 3,       # 적당한 수익: 보통
                            np.where(df['return_pct'] > 0, 1,       # 소수익: 약간 좋음
                                    np.where(df['return_pct'] > -3, -1,     # 소손실: 약간 나쁨
                                            np.where(df['return_pct'] > -8, -2,     # 손실: 나쁨
                                                    np.where(df['return_pct'] > -15, -3, -4)))))))  # 큰 손실: 매우 나쁨
        
        # 리스크 대비 수익률 (샤프 비율 개념)
        volatility_safe = np.maximum(df['exit_volatility_20d'], 1)
        df['risk_adjusted_return'] = df['return_pct'] / volatility_safe
        df['risk_adjusted_score'] = np.clip(df['risk_adjusted_return'] * 2, -3, 3)
        
        # 손절/익절 적절성
        df['cutloss_profit_score'] = np.where(
            (df['return_pct'] > 0) & (df['holding_period_days'] < 30), 2,    # 빠른 익절: 좋음
            np.where((df['return_pct'] < -5) & (df['holding_period_days'] < 10), 1,  # 빠른 손절: 보통
                    np.where((df['return_pct'] < -10) & (df['holding_period_days'] > 30), -2, 0))  # 늦은 손절: 나쁨
        )
        
        df['profit_quality_raw'] = (df['absolute_return_score'] * 0.5 + 
                                   df['risk_adjusted_score'] * 0.3 + 
                                   df['cutloss_profit_score'] * 0.2)

        # ===== 3. 시장 대응 점수 (25%) =====
        # 청산 시점의 모멘텀 대응
        df['exit_momentum_response'] = np.where(
            df['return_pct'] > 0,  # 수익 실현 시
            np.where(df['exit_momentum_20d'] < -5, 3,    # 하락장에서 수익실현: 매우 좋음
                    np.where(df['exit_momentum_20d'] > 5, -1, 1)),   # 상승장에서 수익실현: 아쉬움
            # 손실 청산 시
            np.where(df['exit_momentum_20d'] < -10, 2,   # 급락장에서 손절: 좋은 판단
                    np.where(df['exit_momentum_20d'] > 0, -2, 0))    # 상승장에서 손절: 나쁜 판단
        )
        
        # VIX 변화 대응 (공포지수 변화에 따른 대응)
        df['vix_change_response'] = np.where(
            df['change_vix'] > 5,  # VIX 급등 (공포 증가) 시
            np.where(df['return_pct'] > 0, 2,   # 수익 실현: 좋은 대응
                    np.where(df['return_pct'] > -5, 1, 0)),  # 소손실도 나쁘지 않음
            np.where(df['change_vix'] < -3,  # VIX 하락 (안정) 시
                    np.where(df['return_pct'] < 0, -1, 0), 0)    # 안정기 손실: 아쉬움
        )
        
        # 변동성 변화 대응
        df['volatility_change_response'] = np.where(
            df['change_volatility_5d'] > 15,  # 변동성 급증 시
            np.where(df['return_pct'] > 0, 2, 1),     # 수익실현 좋음, 손절도 나쁘지 않음
            np.where(df['change_volatility_5d'] < -10,  # 변동성 감소 시
                    np.where(df['return_pct'] < -5, -1, 0), 0)  # 안정기 손실: 아쉬움
        )
        
        df['market_response_raw'] = (df['exit_momentum_response'] * 0.5 + 
                                    df['vix_change_response'] * 0.3 + 
                                    df['volatility_change_response'] * 0.2)

        # ===== 최종 점수 계산 (스케일링 적용) =====
        # 각 구성 요소별 스케일링
        if timing_scaler is None or profit_scaler is None or market_scaler is None:
            timing_scaler = RobustScaler()
            profit_scaler = RobustScaler() 
            market_scaler = RobustScaler()
            
            timing_scaled = timing_scaler.fit_transform(df[['timing_score_raw']])
            profit_scaled = profit_scaler.fit_transform(df[['profit_quality_raw']])
            market_scaled = market_scaler.fit_transform(df[['market_response_raw']])
            
            self.c_type_exit_scalers['timing_scaler'] = timing_scaler
            self.c_type_exit_scalers['profit_scaler'] = profit_scaler
            self.c_type_exit_scalers['market_scaler'] = market_scaler
        else:
            timing_scaled = timing_scaler.transform(df[['timing_score_raw']])
            profit_scaled = profit_scaler.transform(df[['profit_quality_raw']])
            market_scaled = market_scaler.transform(df[['market_response_raw']])
        
        # 가중 평균으로 최종 점수 계산
        df['c_type_exit_score'] = (timing_scaled.flatten() * 0.4 + 
                                  profit_scaled.flatten() * 0.35 + 
                                  market_scaled.flatten() * 0.25)
        
        if verbose:
            print(f"  ✅ C-type Exit Score 생성 완료")
            print(f"  범위: {df['c_type_exit_score'].min():.4f} ~ {df['c_type_exit_score'].max():.4f}")
            print(f"  평균: {df['c_type_exit_score'].mean():.4f}")
            print(f"  구성: 타이밍 적절성(40%) + 수익 실현 품질(35%) + 시장 대응(25%)")
        
        return df

    def prepare_c_type_features(self, df, verbose=False):
        """
        C-type: 청산 시점에서 사용 가능한 피처 준비
        
        청산 시점에서 알 수 있는 정보:
        - 진입 시점 정보 (entry_*): 참고 정보
        - 현재(청산) 시점 정보 (exit_*): 핵심 정보
        - 보유 기간 중 변화 (change_*): 핵심 정보
        - 시장 정보 (market_*): 환경 정보
        """
        if verbose:
            print("🛑 C-type: 청산 판단용 피처 준비")
        
        # 라벨링에 사용된 피처들 제외
        excluded_features = {
            'return_pct', 'holding_period_days', 'exit_volatility_20d', 'exit_momentum_20d',
            'change_volatility_5d', 'change_vix',
            # 중간 계산 변수들
            'daily_return_efficiency', 'holding_timing_base', 'return_timing_adjustment',
            'timing_score_raw', 'absolute_return_score', 'risk_adjusted_return', 'risk_adjusted_score',
            'cutloss_profit_score', 'profit_quality_raw', 'exit_momentum_response',
            'vix_change_response', 'volatility_change_response', 'market_response_raw',
            'c_type_exit_score'
        }
        
        # C유형에서 사용 가능한 피처들
        c_type_features = []
        
        # ===== 1. 기본 거래 정보 =====
        basic_features = ['position_size_pct']
        c_type_features.extend([col for col in basic_features if col in df.columns])
        
        # ===== 2. 진입 시점 정보 (참고용) =====
        entry_features = [
            'entry_momentum_5d', 'entry_momentum_20d', 'entry_momentum_60d',
            'entry_ma_dev_5d', 'entry_ma_dev_20d', 'entry_ma_dev_60d',
            'entry_volatility_5d', 'entry_volatility_60d',  # entry_volatility_20d 제외
            'entry_vol_change_5d', 'entry_vol_change_20d', 'entry_vol_change_60d',
            'entry_vix', 'entry_tnx_yield', 'entry_ratio_52w_high'
        ]
        c_type_features.extend([col for col in entry_features if col in df.columns])
        
        # ===== 3. 청산 시점 정보 (핵심) =====
        exit_features = [
            'exit_momentum_5d', 'exit_momentum_60d',  # exit_momentum_20d 제외 (라벨링 사용)
            'exit_ma_dev_5d', 'exit_ma_dev_20d', 'exit_ma_dev_60d',
            'exit_volatility_5d', 'exit_volatility_60d',  # exit_volatility_20d 제외
            'exit_vix', 'exit_tnx_yield', 'exit_ratio_52w_high'
        ]
        c_type_features.extend([col for col in exit_features if col in df.columns])
        
        # ===== 4. 보유 기간 중 변화 (매우 중요) =====
        change_features = [
            'change_momentum_5d', 'change_momentum_20d', 'change_momentum_60d',
            'change_ma_dev_5d', 'change_ma_dev_20d', 'change_ma_dev_60d',
            'change_volatility_20d', 'change_volatility_60d',  # change_volatility_5d 제외
            'change_tnx_yield', 'change_ratio_52w_high'
            # change_vix는 라벨링에 사용되므로 제외
        ]
        c_type_features.extend([col for col in change_features if col in df.columns])
        
        # ===== 5. 시장 환경 정보 =====
        market_features = [
            'market_return_during_holding',
            'excess_return'
        ]
        c_type_features.extend([col for col in market_features if col in df.columns])
        
        # 실제 존재하고 제외되지 않은 피처만 선택
        self.c_type_exit_features = [col for col in c_type_features 
                                    if col in df.columns and col not in excluded_features]
        
        if verbose:
            print(f"  C유형 사용 피처: {len(self.c_type_exit_features)}개")
            print(f"  구성: 진입 정보(참고) + 청산 정보(핵심) + 변화 정보(중요) + 시장 정보")
            print(f"  제외된 피처: 라벨링에 사용된 변수들 및 중간 계산 변수들")
        
        # 숫자형 데이터만 선택
        feature_data = df[self.c_type_exit_features].select_dtypes(include=[np.number])
        
        if verbose and len(feature_data.columns) != len(self.c_type_exit_features):
            print(f"  비숫자형 컬럼 제외: {len(self.c_type_exit_features) - len(feature_data.columns)}개")
        
        return feature_data

    # ================================
    # 하이퍼파라미터 최적화
    # ================================
    
    def get_c_type_hyperparameter_grid(self):
        """
        C-type 모델에 특화된 하이퍼파라미터 그리드
        청산 신호 예측에 최적화된 파라미터 범위
        """
        
        # C-type 특화 파라미터 (청산 신호 예측)
        c_param_grid = {
            'objective': ['reg:squarederror'],
            'eval_metric': ['rmse'],
            
            # 청산 신호 예측에 특화된 트리 구조
            'max_depth': [4, 5, 6, 7, 8, 9],  # 중간 깊이 (과적합 방지)
            'min_child_weight': [1, 2, 3, 4, 5, 8],
            'subsample': [0.7, 0.75, 0.8, 0.85, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.85, 0.9, 0.95],
            'colsample_bylevel': [0.7, 0.8, 0.9, 1.0],
            
            # 청산 신호에 적합한 학습률 (신호 예측이므로 중간값)
            'learning_rate': [0.02, 0.03, 0.05, 0.07, 0.1, 0.15],
            'n_estimators': [200, 300, 400, 500, 600, 800],
            
            # 청산 신호 특화 정규화 (노이즈 제거 중시)
            'reg_alpha': [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            'reg_lambda': [1.0, 2.0, 3.0, 5.0, 8.0, 10.0],
            
            # 청산 시점 예측 특화
            'gamma': [0, 0.01, 0.05, 0.1, 0.2, 0.5],
            'max_delta_step': [0, 1, 2, 3],
            'scale_pos_weight': [1, 2, 3],
            
            # 트리 생성
            'tree_method': ['hist'],
            'grow_policy': ['depthwise', 'lossguide'],
            'max_leaves': [0, 31, 63, 127],
            'max_bin': [128, 256]
        }
        
        return c_param_grid

    def smart_c_type_hyperparameter_search(self, X_train, y_train, X_val, y_val, n_iter=150):
        """
        C-type 모델을 위한 스마트 하이퍼파라미터 검색
        1단계: RandomizedSearchCV로 넓은 범위 탐색
        2단계: 최적 파라미터 주변 세밀 탐색
        """
        print("🛑 C-type 스마트 하이퍼파라미터 검색 시작")
        
        param_grid = self.get_c_type_hyperparameter_grid()
        base_model = xgb.XGBRegressor(random_state=42, n_jobs=1)
        
        # TimeSeriesSplit 사용 (시계열 데이터)
        tscv = TimeSeriesSplit(n_splits=3)
        
        # 1단계: RandomizedSearchCV
        print("  1단계: 광범위 파라미터 탐색...")
        random_search = RandomizedSearchCV(
            base_model, param_grid, 
            n_iter=n_iter, cv=tscv, scoring='r2',
            random_state=42, n_jobs=1
        )
        random_search.fit(X_train, y_train)
        
        best_params_stage1 = random_search.best_params_
        stage1_score = random_search.best_score_
        
        print(f"  1단계 완료: CV R² = {stage1_score:.4f}")
        
        # 2단계: 최적 파라미터 주변 세밀 탐색
        print("  2단계: 세밀 파라미터 탐색...")
        refined_grid = self._create_c_type_refined_grid(best_params_stage1)
        
        grid_search = GridSearchCV(
            base_model, refined_grid, 
            cv=tscv, scoring='r2', n_jobs=1
        )
        grid_search.fit(X_train, y_train)
        
        final_best_params = grid_search.best_params_
        stage2_score = grid_search.best_score_
        
        print(f"  2단계 완료: CV R² = {stage2_score:.4f}")
        
        # 최종 모델로 검증 세트 평가
        final_model = xgb.XGBRegressor(**final_best_params, random_state=42)
        final_model.fit(X_train, y_train)
        val_pred = final_model.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        
        print(f"  검증 세트 R²: {val_r2:.4f}")
        
        return {
            'best_params': final_best_params,
            'best_model': final_model,
            'cv_score_stage1': stage1_score,
            'cv_score_stage2': stage2_score,
            'val_score': val_r2,
            'search_iterations': n_iter
        }

    def _create_c_type_refined_grid(self, best_params):
        """최적 파라미터 주변의 세밀한 그리드 생성"""
        
        refined_grid = {}
        
        # 각 파라미터별 세밀 조정
        if 'max_depth' in best_params:
            depth = best_params['max_depth']
            refined_grid['max_depth'] = [max(3, depth-1), depth, min(10, depth+1)]
        
        if 'learning_rate' in best_params:
            lr = best_params['learning_rate']
            refined_grid['learning_rate'] = [max(0.01, lr-0.02), lr, min(0.3, lr+0.02)]
            
        if 'n_estimators' in best_params:
            n_est = best_params['n_estimators']
            refined_grid['n_estimators'] = [max(100, n_est-100), n_est, min(1000, n_est+100)]
            
        if 'reg_alpha' in best_params:
            alpha = best_params['reg_alpha']
            refined_grid['reg_alpha'] = [max(0, alpha-0.1), alpha, alpha+0.1]
            
        if 'reg_lambda' in best_params:
            lambda_val = best_params['reg_lambda']
            refined_grid['reg_lambda'] = [max(0.5, lambda_val-1), lambda_val, lambda_val+1]
        
        # 다른 파라미터들은 최적값 고정
        for key, value in best_params.items():
            if key not in refined_grid:
                refined_grid[key] = [value]
        
        return refined_grid

    # ================================
    # Walk-Forward Validation
    # ================================
    
    def create_time_folds(self, df, verbose=False):
        """시계열 데이터를 위한 Walk-Forward 폴드 생성"""
        if verbose:
            print("🛑 C-type Walk-Forward 시간 폴드 생성")
        
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

    def run_c_type_walk_forward_training(self, data_path, verbose=True):
        """C-type 모델의 Walk-Forward 학습 및 평가"""
        if verbose:
            print("🛑 C-type Walk-Forward 학습 시작")
            print("="*60)
        
        # 데이터 로드
        df = pd.read_csv(data_path)
        if verbose:
            print(f"📊 데이터 로드: {len(df):,}개 거래")
        
        # C-type 점수 생성
        df = self.create_c_type_exit_score(df, verbose=verbose)
        
        # 시간 폴드 생성
        folds = self.create_time_folds(df, verbose=verbose)
        
        fold_results = []
        
        for fold_info in tqdm(folds, desc="폴드별 학습"):
            if verbose:
                print(f"\n🛑 폴드 {fold_info['fold_id']} 학습 중...")
            
            # 폴드별 데이터 분할
            train_data = df.loc[fold_info['train_indices']]
            val_data = df.loc[fold_info['val_indices']]
            test_data = df.loc[fold_info['test_indices']]
            
            # 피처 준비
            X_train = self.prepare_c_type_features(train_data, verbose=False)
            X_val = self.prepare_c_type_features(val_data, verbose=False)
            X_test = self.prepare_c_type_features(test_data, verbose=False)
            
            y_train = train_data['c_type_exit_score']
            y_val = val_data['c_type_exit_score']
            y_test = test_data['c_type_exit_score']
            
            # TODO: 하이퍼파라미터 최적화 (현재 주석처리 - 빠른 테스트용)
            # search_result = self.smart_c_type_hyperparameter_search(
            #     X_train, y_train, X_val, y_val, n_iter=100
            # )
            
            # 미리 최적화된 파라미터 사용
            best_params = {
                'tree_method': 'approx', 'subsample': 0.75, 'scale_pos_weight': 10, 
                'reg_lambda': 20.0, 'reg_alpha': 1.0, 'objective': 'reg:squarederror', 
                'n_estimators': 800, 'min_child_weight': 2, 'max_leaves': 255, 
                'max_depth': 9, 'max_delta_step': 0, 'max_bin': 128, 
                'learning_rate': 0.01, 'grow_policy': 'depthwise', 'gamma': 0.5, 
                'eval_metric': 'rmse', 'colsample_bytree': 0.9, 'colsample_bynode': 0.8, 
                'colsample_bylevel': 0.8, 'random_state': 42
            }
            
            # 최적 파라미터로 모델 학습
            best_model = xgb.XGBRegressor(**best_params)
            best_model.fit(X_train, y_train)
            val_pred = best_model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            
            search_result = {
                'best_params': best_params,
                'best_model': best_model,
                'val_score': val_r2
            }
            
            # 테스트 세트 평가
            test_pred = search_result['best_model'].predict(X_test)
            test_r2 = r2_score(y_test, test_pred)
            
            # 결과 저장
            fold_result = {
                'fold_id': fold_info['fold_id'],
                'fold_info': fold_info,
                'best_params': search_result['best_params'],
                'val_r2': search_result['val_score'],
                'test_r2': test_r2,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'features_used': len(X_train.columns)
            }
            
            fold_results.append(fold_result)
            
            if verbose:
                print(f"  폴드 {fold_info['fold_id']} 완료: Val R² = {search_result['val_score']:.4f}, Test R² = {test_r2:.4f}")
        
        self.fold_results = fold_results
        
        # 전체 결과 요약
        if verbose:
            self.print_c_type_fold_summary()
        
        return fold_results

    def print_c_type_fold_summary(self):
        """폴드별 결과 요약 출력"""
        if not self.fold_results:
            print("❌ 폴드 결과가 없습니다.")
            return
        
        print("\n" + "="*70)
        print("🏆 C-type Walk-Forward 결과 요약")
        print("="*70)
        
        val_r2_scores = [result['val_r2'] for result in self.fold_results]
        test_r2_scores = [result['test_r2'] for result in self.fold_results]
        
        print(f"📊 폴드별 성능:")
        for result in self.fold_results:
            print(f"  폴드 {result['fold_id']}: Val R² = {result['val_r2']:.4f}, Test R² = {result['test_r2']:.4f}")
        
        print(f"\n📈 전체 통계:")
        print(f"  Validation R²: {np.mean(val_r2_scores):.4f} ± {np.std(val_r2_scores):.4f}")
        print(f"  Test R²:       {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")
        print(f"  최고 성능:     {np.max(test_r2_scores):.4f} (폴드 {np.argmax(test_r2_scores) + 1})")
        print(f"  평균 피처 수:  {np.mean([r['features_used'] for r in self.fold_results]):.0f}개")
        
        print("="*70)

    def save_c_type_results(self, filename=None):
        """C-type 결과 저장"""
        if filename is None:
            filename = f"c_type_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        save_data = {
            'model_version': 'c_type_v6',
            'model_name': 'C-type Exit Signal AI',
            'created_at': datetime.now().isoformat(),
            'total_folds': len(self.fold_results),
            'fold_results': []
        }
        
        for result in self.fold_results:
            # 모델 객체 제외하고 저장
            fold_data = {key: value for key, value in result.items() 
                        if key != 'best_model'}
            save_data['fold_results'].append(fold_data)
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"💾 C-type 결과 저장: {filename}")
        return filename

def main():
    """메인 실행 함수"""
    print("🛑 C-type Exit Signal AI - 매도 청산 신호 예측")
    print("="*60)
    print("📋 모델 목표:")
    print("  - 매도 시점에서 청산 조건의 적절성 평가")
    print("  - 타이밍 적절성 + 수익 실현 품질 + 시장 대응 종합 점수")
    print("  - Walk-Forward Validation으로 시계열 안정성 검증")
    print("="*60)
    
    # 데이터 경로
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../results/final/trading_episodes_with_rebuilt_market_component.csv')
    
    # 파일 존재 확인
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return
    
    # 모델 초기화 및 학습
    model = CTypeExitSignalAI()
    
    # Walk-Forward 학습 실행
    fold_results = model.run_c_type_walk_forward_training(data_path, verbose=True)
    
    # 결과 저장
    model.save_c_type_results()
    
    print("\n✅ C-type Exit Signal AI 학습 완료!")
    return model

if __name__ == "__main__":
    main()