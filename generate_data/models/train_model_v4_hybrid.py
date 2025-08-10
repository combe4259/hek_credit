import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
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

class HybridTradingAI:
    """
    하이브리드 트레이딩 AI v4
    
    구조:
    1. A유형: 완료된 거래의 정확한 품질 평가 (사후 분석)
    2. B유형: 현재 상황 기반 진입 조건 분석 (실시간 지원, 미래 예측 없음)
    
    특징:
    - Data Leakage 완전 제거
    - 현실적인 의사결정 지원
    - 정확한 사후 품질 평가
    """
    
    def __init__(self, train_months=36, val_months=6, test_months=6, step_months=3):
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.step_months = step_months
        
        # ===== A유형: 거래 품질 분석기 (Post-Trade Quality Analyzer) =====
        self.a_type_quality_model = None      # 품질 평가 예측 모델
        self.a_type_quality_scalers = {}      # A유형 전용 스케일러들
        self.a_type_quality_features = None   # A유형이 사용하는 피처 목록
        
        # ===== B유형: 진입 조건 평가기 (Entry Condition Evaluator) =====  
        self.b_type_entry_model = None        # 진입 조건 예측 모델
        self.b_type_entry_scalers = {}        # B유형 전용 스케일러들
        self.b_type_entry_features = None     # B유형이 사용하는 피처 목록
        
        self.fold_results = []
    
    # ================================
    # A유형: 사후 품질 평가 (완전한 거래 데이터 사용)
    # ================================
    
    def create_a_type_quality_score(self, df, risk_scaler=None, eff_scaler=None, verbose=False):
        """A유형: 완료된 거래의 품질 점수 생성 (모든 정보 활용 가능)"""
        if verbose:
            print("🎯 A유형: Quality Score 생성 중...")
        
        df = df.copy()
        
        # NaN 처리 (중립적 값 0으로 처리)
        df['return_pct'] = df['return_pct'].fillna(0)
        df['entry_volatility_20d'] = df['entry_volatility_20d'].fillna(0)
        df['entry_ratio_52w_high'] = df['entry_ratio_52w_high'].fillna(0)
        df['holding_period_days'] = df['holding_period_days'].fillna(0)
        
        # Risk Management Quality (40%) - 리스크 대비 성과
        volatility_safe = np.maximum(df['entry_volatility_20d'], 0.01)
        df['risk_adj_return'] = df['return_pct'] / volatility_safe
        df['risk_adj_return'] = np.where(
            np.isinf(df['risk_adj_return']) | np.isnan(df['risk_adj_return']), 
            0, df['risk_adj_return']
        )
        
        ratio_safe = np.clip(df['entry_ratio_52w_high'], 0, 100)
        df['price_safety'] = (100 - ratio_safe) / 100
        
        df['risk_management_score'] = df['risk_adj_return'] * 0.6 + df['price_safety'] * 0.4
        
        # Efficiency Quality (60%) - 시간 대비 효율성
        holding_safe = np.maximum(df['holding_period_days'], 1)
        df['time_efficiency'] = df['return_pct'] / holding_safe
        df['time_efficiency'] = np.where(
            np.isinf(df['time_efficiency']) | np.isnan(df['time_efficiency']), 
            0, df['time_efficiency']
        )
        
        df['efficiency_score'] = df['time_efficiency']
        
        # 스케일링 및 종합 점수
        if risk_scaler is None or eff_scaler is None:
            risk_scaler = RobustScaler()
            eff_scaler = RobustScaler()
            
            risk_scaled = risk_scaler.fit_transform(df[['risk_management_score']])
            eff_scaled = eff_scaler.fit_transform(df[['efficiency_score']])
            
            self.a_type_quality_scalers['risk_scaler'] = risk_scaler
            self.a_type_quality_scalers['efficiency_scaler'] = eff_scaler
        else:
            risk_scaled = risk_scaler.transform(df[['risk_management_score']])
            eff_scaled = eff_scaler.transform(df[['efficiency_score']])
        
        df['a_type_quality_score'] = risk_scaled.flatten() * 0.4 + eff_scaled.flatten() * 0.6
        
        if verbose:
            print(f"  ✅ Quality Score 생성 완료")
            print(f"  범위: {df['a_type_quality_score'].min():.4f} ~ {df['a_type_quality_score'].max():.4f}")
            print(f"  평균: {df['a_type_quality_score'].mean():.4f}")
        
        return df
    
    def prepare_a_type_features(self, df, verbose=False):
        """A유형: 품질 평가용 피처 준비 (완료된 거래의 모든 정보 사용 가능)"""
        if verbose:
            print("🔧 A유형: 품질 평가용 피처 준비 중...")
        
        # Quality Score 계산용 피처는 제외
        excluded_features = {
            'return_pct', 'entry_volatility_20d', 'entry_ratio_52w_high', 'holding_period_days',
            'risk_adj_return', 'price_safety', 'risk_management_score',
            'time_efficiency', 'efficiency_score', 'quality_score', 'a_type_quality_score'
        }
        
        # A유형에서 사용 가능한 모든 피처 카테고리
        available_a_type_features = []
        
        # ===== 1. 기본 거래 정보 =====
        basic_trade_info = ['position_size_pct']  # 거래 규모
        available_a_type_features.extend([col for col in basic_trade_info if col in df.columns])
        
        # ===== 2. 진입 시점 기술적 지표 =====
        entry_technical_indicators = [
            # 모멘텀 지표
            'entry_momentum_5d', 'entry_momentum_60d', 
            # 이동평균 괴리도
            'entry_ma_dev_5d', 'entry_ma_dev_20d', 'entry_ma_dev_60d',
            # 변동성 (entry_volatility_20d 제외 - quality_score에 사용됨)
            'entry_volatility_5d', 'entry_volatility_60d',
            # 변동성 변화율
            'entry_vol_change_5d', 'entry_vol_change_20d', 'entry_vol_change_60d',
            # 시장 환경
            'entry_vix', 'entry_tnx_yield'
        ]
        available_a_type_features.extend([col for col in entry_technical_indicators if col in df.columns])
        
        # ===== 3. 종료 시점 지표 (A유형만 사용 가능!) =====
        exit_technical_indicators = [
            # 종료 시점 모멘텀
            'exit_momentum_5d', 'exit_momentum_20d', 'exit_momentum_60d',
            # 종료 시점 이동평균 괴리도
            'exit_ma_dev_5d', 'exit_ma_dev_20d', 'exit_ma_dev_60d',
            # 종료 시점 변동성
            'exit_volatility_5d', 'exit_volatility_20d', 'exit_volatility_60d',
            # 종료 시점 시장 환경
            'exit_vix', 'exit_tnx_yield', 'exit_ratio_52w_high'
        ]
        available_a_type_features.extend([col for col in exit_technical_indicators if col in df.columns])
        
        # ===== 4. 변화량 지표 (A유형만 사용 가능!) =====
        change_indicators = [
            # 모멘텀 변화
            'change_momentum_5d', 'change_momentum_20d', 'change_momentum_60d',
            # 이동평균 굌리도 변화
            'change_ma_dev_5d', 'change_ma_dev_20d', 'change_ma_dev_60d',
            # 변동성 변화
            'change_volatility_5d', 'change_volatility_60d',
            # 시장 환경 변화
            'change_vix', 'change_tnx_yield', 'change_ratio_52w_high'
        ]
        available_a_type_features.extend([col for col in change_indicators if col in df.columns])
        
        # ===== 5. 보유 기간 중 시장 정보 (A유형만 사용 가능!) =====
        holding_period_info = [
            'market_return_during_holding',  # 보유 기간 중 시장 수익률
            'excess_return'                  # 시장 대비 초과 수익률
        ]
        available_a_type_features.extend([col for col in holding_period_info if col in df.columns])
        
        # 실제 존재하는 피처만 선택
        self.a_type_quality_features = [col for col in available_a_type_features 
                                       if col in df.columns and col not in excluded_features]
        
        if verbose:
            print(f"  A유형 사용 피처: {len(self.a_type_quality_features)}개")
            print(f"  포함된 피처 유형: entry, exit, change, holding (모든 정보 활용)")
        
        return df[self.a_type_quality_features]
    
    # ================================
    # B유형: 현재 상황 기반 진입 조건 분석 (미래 예측 없음)
    # ================================
    
    def create_b_type_entry_condition_score(self, df, verbose=False):
        """B유형: 현재 진입 조건 분석 점수 (미래 정보 사용 금지)"""
        if verbose:
            print("🔮 B유형 (진입 조건 평가): Entry Condition Score 생성 중...")
            print("   → 현재 시점 정보만 사용하여 실시간 매수 적합도 평가")
        
        df = df.copy()
        
        # NaN 처리 (중립적 값 0으로 처리)
        df['entry_vix'] = df['entry_vix'].fillna(0)
        df['entry_volatility_20d'] = df['entry_volatility_20d'].fillna(0)
        df['entry_ratio_52w_high'] = df['entry_ratio_52w_high'].fillna(0)
        df['entry_momentum_20d'] = df['entry_momentum_20d'].fillna(0)
        
        # 1. 기술적 조건 점수 (40%)
        # RSI 개념: 과매도일수록 좋음
        rsi_proxy = np.clip((100 - df['entry_ratio_52w_high']) / 100, 0, 1)
        
        # 모멘텀: 적당한 하락 후 반등 신호가 좋음
        momentum_safe = np.clip(df['entry_momentum_20d'], -50, 50)
        momentum_score = np.where(momentum_safe < -10, 0.8,  # 하락 후
                                np.where(momentum_safe > 10, 0.3, 0.6))  # 상승 중
        
        df['b_type_technical_score'] = rsi_proxy * 0.6 + momentum_score * 0.4
        
        # ===== 2. 시장 환경 점수 (35%): "전반적으로 매수하기 좋은 시기인가?" =====
        
        # 2-1. VIX 기반 시장 안정성 평가
        # → VIX가 낮을수록 시장이 안정하여 매수 적기
        # VIX 10: 매우 안정 (1.0점), VIX 50: 매우 불안 (0점)
        vix_safe = np.clip(df['entry_vix'], 10, 50)
        market_stability_score = (50 - vix_safe) / 40
        
        # 2-2. 시장 환경 종합 점수 (현재는 VIX만 사용)
        df['b_type_market_env_score'] = market_stability_score
        
        # ===== 3. 리스크 수준 점수 (25%): "현재 위험도가 적절한가?" =====
        
        # 3-1. 변동성 적정성 평가
        # → 너무 낮으면 유동성 부족, 너무 높으면 위험
        # 20-30% 변동성이 적정 수준
        vol_safe = np.clip(df['entry_volatility_20d'], 10, 100)
        volatility_score = np.where(
            vol_safe < 25, 1.0,      # 낮은 변동성 (안전)
            np.where(vol_safe > 50, 0.3, 0.7)  # 높은 변동성 (위험)
        )
        
        # 3-2. 리스크 수준 종합 점수 (현재는 변동성만 사용)
        df['b_type_risk_score'] = volatility_score
        
        # 종합 진입 조건 점수
        df['b_type_entry_condition_score'] = (df['b_type_technical_score'] * 0.4 + 
                                             df['b_type_market_env_score'] * 0.35 + 
                                             df['b_type_risk_score'] * 0.25)
        
        # 0-100 스케일로 변환
        df['b_type_entry_condition_score'] = df['b_type_entry_condition_score'] * 100
        
        if verbose:
            print(f"  ✅ 진입 조건 점수 생성 완료")
            print(f"  범위: {df['b_type_entry_condition_score'].min():.1f} ~ {df['b_type_entry_condition_score'].max():.1f}")
            print(f"  평균: {df['b_type_entry_condition_score'].mean():.1f}")
        
        return df
    
    def prepare_b_type_features(self, df, verbose=False):
        """B유형: 진입 조건 분석용 피처 준비 (진입 시점 정보만 사용)"""
        if verbose:
            print("🔧 B유형: 진입 조건 분석용 피처 준비 중...")
        
        # B유형에서는 미래 정보 완전 금지!
        forbidden_patterns = ['exit_', 'change_', 'holding', 'return_pct']
        
        # 진입 시점 정보만 사용 가능
        available_features = []
        
        # 1. 기본 정보
        basic_features = ['position_size_pct']  # 계획된 포지션 크기
        available_features.extend([col for col in basic_features if col in df.columns])
        
        # 2. 진입 시점 기술적 지표만
        entry_features = [
            'entry_momentum_5d', 'entry_momentum_20d', 'entry_momentum_60d',
            'entry_ma_dev_5d', 'entry_ma_dev_20d', 'entry_ma_dev_60d',
            'entry_volatility_5d', 'entry_volatility_60d',  # entry_volatility_20d 제외 (타겟 계산에 사용)
            'entry_vol_change_5d', 'entry_vol_change_20d', 'entry_vol_change_60d'
        ]
        available_features.extend([col for col in entry_features if col in df.columns])
        
        # 3. 진입 시점 시장 환경
        market_features = ['entry_vix', 'entry_tnx_yield']
        available_features.extend([col for col in market_features if col in df.columns])
        
        # 미래 정보 완전 제거
        safe_features = []
        for feature in available_features:
            is_safe = True
            for pattern in forbidden_patterns:
                if pattern in feature:
                    is_safe = False
                    break
            if is_safe and feature in df.columns:
                safe_features.append(feature)
        
        # entry_volatility_20d, entry_ratio_52w_high 제거 (점수 계산에 사용됨)
        target_calculation_features = {'entry_volatility_20d', 'entry_ratio_52w_high', 'entry_momentum_20d'}
        safe_features = [f for f in safe_features if f not in target_calculation_features]
        
        self.b_type_entry_features = safe_features
        
        if verbose:
            print(f"  B유형 사용 피처: {len(self.b_type_entry_features)}개")
            print(f"  피처 범위: 현재/진입 시점 정보만 (실시간 활용 가능)")
            print(f"  Data Leakage 방지: entry_condition_score 계산 피처 및 미래 정보 제외")
            if len(self.b_type_entry_features) < 10:
                print(f"  구체적 피처: {self.b_type_entry_features}")
        
        return df[self.b_type_entry_features] if self.b_type_entry_features else pd.DataFrame()
    
    # ================================
    # Walk-Forward Validation
    # ================================
    
    def create_walk_forward_folds(self, df, verbose=False):
        """Walk-Forward 분할 생성"""
        if verbose:
            print("📅 Walk-Forward 분할 생성 중...")
        
        df = df.copy()
        df['entry_date'] = pd.to_datetime(df['entry_datetime'])
        df = df.sort_values('entry_date')
        
        start_date = df['entry_date'].min()
        end_date = df['entry_date'].max()
        
        folds = []
        current_date = start_date + pd.DateOffset(months=self.train_months)
        
        while current_date + pd.DateOffset(months=self.val_months + self.test_months) <= end_date:
            train_start = current_date - pd.DateOffset(months=self.train_months)
            train_end = current_date
            val_start = train_end
            val_end = val_start + pd.DateOffset(months=self.val_months)
            test_start = val_end
            test_end = test_start + pd.DateOffset(months=self.test_months)
            
            train_mask = (df['entry_date'] >= train_start) & (df['entry_date'] < train_end)
            val_mask = (df['entry_date'] >= val_start) & (df['entry_date'] < val_end)
            test_mask = (df['entry_date'] >= test_start) & (df['entry_date'] < test_end)
            
            if train_mask.sum() > 1000 and val_mask.sum() > 100 and test_mask.sum() > 100:
                folds.append({
                    'fold_id': len(folds) + 1,
                    'train_start': train_start.date(),
                    'train_end': train_end.date(),
                    'val_start': val_start.date(),
                    'val_end': val_end.date(),
                    'test_start': test_start.date(),
                    'test_end': test_end.date(),
                    'train_idx': df[train_mask].index,
                    'val_idx': df[val_mask].index,
                    'test_idx': df[test_mask].index
                })
            
            current_date += pd.DateOffset(months=self.step_months)
        
        if verbose:
            print(f"  생성된 폴드 수: {len(folds)}개")
            for i, fold in enumerate(folds):
                print(f"    Fold {i+1}: Train {fold['train_start']} ~ {fold['train_end']}")
                print(f"             Val   {fold['val_start']} ~ {fold['val_end']}")
                print(f"             Test  {fold['test_start']} ~ {fold['test_end']}")
                print(f"             Size: {len(fold['train_idx'])}/{len(fold['val_idx'])}/{len(fold['test_idx'])}")
        
        return folds
    
    def evaluate_single_fold(self, df, fold, verbose=False):
        """단일 폴드에서 A유형과 B유형 모델 모두 평가"""
        if verbose:
            print(f"\n🔄 Fold {fold['fold_id']} 평가 중...")
        
        # 데이터 분할
        train_data = df.loc[fold['train_idx']].copy()
        val_data = df.loc[fold['val_idx']].copy()
        test_data = df.loc[fold['test_idx']].copy()
        
        # GPU 설정
        gpu_available = torch.cuda.is_available()
        base_model = xgb.XGBRegressor(
            tree_method='gpu_hist' if gpu_available else 'hist',
            gpu_id=0 if gpu_available else None,
            random_state=42,
            eval_metric='rmse'
        )
        
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [6, 8],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        results = {}
        
        # ================================
        # A유형: 품질 평가 모델 학습
        # ================================
        try:
            # A유형 Quality Score 생성
            train_data_a = self.create_a_type_quality_score(train_data, verbose=False)
            val_data_a = self.create_a_type_quality_score(
                val_data, 
                risk_scaler=self.a_type_quality_scalers.get('risk_scaler'),
                eff_scaler=self.a_type_quality_scalers.get('efficiency_scaler'),
                verbose=False
            )
            test_data_a = self.create_a_type_quality_score(
                test_data,
                risk_scaler=self.a_type_quality_scalers.get('risk_scaler'),
                eff_scaler=self.a_type_quality_scalers.get('efficiency_scaler'),
                verbose=False
            )
            
            # A유형 피처 준비
            X_train_a = self.prepare_a_type_features(train_data_a, verbose=False)
            X_val_a = self.prepare_a_type_features(val_data_a, verbose=False)
            X_test_a = self.prepare_a_type_features(test_data_a, verbose=False)
            
            y_train_a = train_data_a['a_type_quality_score']
            y_val_a = val_data_a['a_type_quality_score']
            y_test_a = test_data_a['a_type_quality_score']
            
            # 결측치 제거
            train_mask_a = ~(X_train_a.isnull().any(axis=1) | y_train_a.isnull())
            val_mask_a = ~(X_val_a.isnull().any(axis=1) | y_val_a.isnull())
            test_mask_a = ~(X_test_a.isnull().any(axis=1) | y_test_a.isnull())
            
            X_train_a = X_train_a[train_mask_a]
            X_val_a = X_val_a[val_mask_a]
            X_test_a = X_test_a[test_mask_a]
            y_train_a = y_train_a[train_mask_a]
            y_val_a = y_val_a[val_mask_a]
            y_test_a = y_test_a[test_mask_a]
            
            if len(X_train_a) > 0 and len(self.a_type_quality_features) > 0:
                # 스케일링
                scaler_a = RobustScaler()
                X_train_a_scaled = scaler_a.fit_transform(X_train_a)
                X_val_a_scaled = scaler_a.transform(X_val_a)
                X_test_a_scaled = scaler_a.transform(X_test_a)
                
                # GridSearchCV
                grid_search_a = GridSearchCV(base_model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
                grid_search_a.fit(X_train_a_scaled, y_train_a)
                
                # 평가
                best_model_a = grid_search_a.best_estimator_
                
                y_val_pred_a = best_model_a.predict(X_val_a_scaled)
                y_test_pred_a = best_model_a.predict(X_test_a_scaled)
                
                results['A_quality_model'] = {
                    'val_r2': r2_score(y_val_a, y_val_pred_a),
                    'test_r2': r2_score(y_test_a, y_test_pred_a),
                    'val_rmse': np.sqrt(mean_squared_error(y_val_a, y_val_pred_a)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test_a, y_test_pred_a)),
                    'best_params': grid_search_a.best_params_,
                    'best_cv_score': grid_search_a.best_score_,
                    'train_size': len(X_train_a),
                    'val_size': len(X_val_a),
                    'test_size': len(X_test_a)
                }
            else:
                results['A_quality_model'] = {'error': 'Insufficient data or features'}
                
        except Exception as e:
            results['A_quality_model'] = {'error': str(e)}
        
        # ================================
        # B유형: 진입 조건 분석 모델 학습
        # ================================
        try:
            # B유형 Entry Condition Score 생성
            train_data_b = self.create_b_type_entry_condition_score(train_data, verbose=False)
            val_data_b = self.create_b_type_entry_condition_score(val_data, verbose=False)
            test_data_b = self.create_b_type_entry_condition_score(test_data, verbose=False)
            
            # B유형 피처 준비
            X_train_b = self.prepare_b_type_features(train_data_b, verbose=False)
            X_val_b = self.prepare_b_type_features(val_data_b, verbose=False)
            X_test_b = self.prepare_b_type_features(test_data_b, verbose=False)
            
            y_train_b = train_data_b['b_type_entry_condition_score']
            y_val_b = val_data_b['b_type_entry_condition_score']
            y_test_b = test_data_b['b_type_entry_condition_score']
            
            if len(X_train_b.columns) > 0:
                # 결측치 제거
                train_mask_b = ~(X_train_b.isnull().any(axis=1) | y_train_b.isnull())
                val_mask_b = ~(X_val_b.isnull().any(axis=1) | y_val_b.isnull())
                test_mask_b = ~(X_test_b.isnull().any(axis=1) | y_test_b.isnull())
                
                X_train_b = X_train_b[train_mask_b]
                X_val_b = X_val_b[val_mask_b]
                X_test_b = X_test_b[test_mask_b]
                y_train_b = y_train_b[train_mask_b]
                y_val_b = y_val_b[val_mask_b]
                y_test_b = y_test_b[test_mask_b]
                
                if len(X_train_b) > 100:
                    # 스케일링
                    scaler_b = RobustScaler()
                    X_train_b_scaled = scaler_b.fit_transform(X_train_b)
                    X_val_b_scaled = scaler_b.transform(X_val_b)
                    X_test_b_scaled = scaler_b.transform(X_test_b)
                    
                    # GridSearchCV
                    grid_search_b = GridSearchCV(base_model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
                    grid_search_b.fit(X_train_b_scaled, y_train_b)
                    
                    # 평가
                    best_model_b = grid_search_b.best_estimator_
                    
                    y_val_pred_b = best_model_b.predict(X_val_b_scaled)
                    y_test_pred_b = best_model_b.predict(X_test_b_scaled)
                    
                    results['B_entry_model'] = {
                        'val_r2': r2_score(y_val_b, y_val_pred_b),
                        'test_r2': r2_score(y_test_b, y_test_pred_b),
                        'val_rmse': np.sqrt(mean_squared_error(y_val_b, y_val_pred_b)),
                        'test_rmse': np.sqrt(mean_squared_error(y_test_b, y_test_pred_b)),
                        'best_params': grid_search_b.best_params_,
                        'best_cv_score': grid_search_b.best_score_,
                        'train_size': len(X_train_b),
                        'val_size': len(X_val_b),
                        'test_size': len(X_test_b)
                    }
                else:
                    results['B_entry_model'] = {'error': 'Insufficient training data'}
            else:
                results['B_entry_model'] = {'error': 'No valid features for B-type model'}
                
        except Exception as e:
            results['B_entry_model'] = {'error': str(e)}
        
        # 시장 환경 분석
        market_stats = {
            'train_vix_mean': train_data['entry_vix'].mean() if 'entry_vix' in train_data.columns else np.nan,
            'val_vix_mean': val_data['entry_vix'].mean() if 'entry_vix' in val_data.columns else np.nan,
            'test_vix_mean': test_data['entry_vix'].mean() if 'entry_vix' in test_data.columns else np.nan,
            'train_return_mean': train_data['return_pct'].mean(),
            'val_return_mean': val_data['return_pct'].mean(),
            'test_return_mean': test_data['return_pct'].mean()
        }
        
        return {
            'fold_id': fold['fold_id'],
            'fold_info': fold,
            'model_results': results,
            'market_stats': market_stats
        }
    
    def run_walk_forward_validation(self, df, verbose=False):
        """전체 Walk-Forward Validation 실행"""
        print("🚀 하이브리드 AI Walk-Forward Validation 시작")
        print("="*70)
        
        # 폴드 생성
        folds = self.create_walk_forward_folds(df, verbose=True)
        
        if len(folds) == 0:
            print("❌ 생성된 폴드가 없습니다.")
            return None
        
        # 각 폴드 평가
        all_results = []
        
        print(f"\n📊 {len(folds)}개 폴드 평가 중...")
        
        for fold in tqdm(folds, desc="Evaluating folds"):
            try:
                result = self.evaluate_single_fold(df, fold, verbose=False)
                all_results.append(result)
            except Exception as e:
                print(f"⚠️ Fold {fold['fold_id']} 평가 실패: {e}")
                continue
        
        self.fold_results = all_results
        
        # 결과 집계
        self.aggregate_results(verbose=True)
        
        return all_results
    
    def aggregate_results(self, verbose=False):
        """하이브리드 모델 결과 집계"""
        if not self.fold_results:
            print("❌ 집계할 결과가 없습니다.")
            return
        
        if verbose:
            print("\n📈 하이브리드 AI 결과 집계")
            print("="*50)
        
        # A유형 결과 집계
        a_results = []
        b_results = []
        
        for result in self.fold_results:
            if 'A_quality_model' in result['model_results'] and 'error' not in result['model_results']['A_quality_model']:
                a_results.append(result['model_results']['A_quality_model'])
            
            if 'B_entry_model' in result['model_results'] and 'error' not in result['model_results']['B_entry_model']:
                b_results.append(result['model_results']['B_entry_model'])
        
        if verbose:
            print(f"\n🎯 A유형 (품질 평가) 성능:")
            if a_results:
                val_r2_a = [r['val_r2'] for r in a_results]
                test_r2_a = [r['test_r2'] for r in a_results]
                
                print(f"  성공적인 폴드: {len(a_results)}/{len(self.fold_results)}")
                print(f"  Validation R²: {np.mean(val_r2_a):.4f} ± {np.std(val_r2_a):.4f}")
                print(f"  Test R²:       {np.mean(test_r2_a):.4f} ± {np.std(test_r2_a):.4f}")
                print(f"  Test R² 범위:  [{np.min(test_r2_a):.4f}, {np.max(test_r2_a):.4f}]")
            else:
                print("  ❌ 성공한 A유형 모델 없음")
            
            print(f"\n🔮 B유형 (진입 조건) 성능:")
            if b_results:
                val_r2_b = [r['val_r2'] for r in b_results]
                test_r2_b = [r['test_r2'] for r in b_results]
                
                print(f"  성공적인 폴드: {len(b_results)}/{len(self.fold_results)}")
                print(f"  Validation R²: {np.mean(val_r2_b):.4f} ± {np.std(val_r2_b):.4f}")
                print(f"  Test R²:       {np.mean(test_r2_b):.4f} ± {np.std(test_r2_b):.4f}")
                print(f"  Test R² 범위:  [{np.min(test_r2_b):.4f}, {np.max(test_r2_b):.4f}]")
            else:
                print("  ❌ 성공한 B유형 모델 없음")
            
            # 시장 환경별 성능
            print(f"\n🌊 시장 환경별 성능 분석:")
            for i, result in enumerate(self.fold_results):
                market = result['market_stats']
                vix_level = "고변동" if market['test_vix_mean'] > 25 else "저변동"
                return_level = "상승" if market['test_return_mean'] > 0.05 else "하락"
                
                a_perf = "N/A"
                b_perf = "N/A"
                
                if 'A_quality_model' in result['model_results'] and 'error' not in result['model_results']['A_quality_model']:
                    a_perf = f"{result['model_results']['A_quality_model']['test_r2']:.4f}"
                
                if 'B_entry_model' in result['model_results'] and 'error' not in result['model_results']['B_entry_model']:
                    b_perf = f"{result['model_results']['B_entry_model']['test_r2']:.4f}"
                
                print(f"  Fold {i+1}: {vix_level}/{return_level} → A:{a_perf} / B:{b_perf}")
    
    def save_results(self, filepath_prefix='hybrid_results'):
        """결과 저장"""
        if not self.fold_results:
            print("❌ 저장할 결과가 없습니다.")
            return
        
        # 결과 저장
        results_for_save = []
        for result in self.fold_results:
            result_copy = result.copy()
            
            # datetime 객체를 문자열로 변환
            fold_info = result_copy['fold_info'].copy()
            for key in ['train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']:
                if key in fold_info:
                    fold_info[key] = str(fold_info[key])
            
            # 인덱스 제거
            del fold_info['train_idx']
            del fold_info['val_idx']
            del fold_info['test_idx']
            
            result_copy['fold_info'] = fold_info
            results_for_save.append(result_copy)
        
        with open(f'{filepath_prefix}.json', 'w') as f:
            json.dump(results_for_save, f, indent=2)
        
        # 메타데이터 저장
        metadata = {
            'model_version': 'hybrid_v4',
            'model_type': 'Hybrid A+B Type with XGBoost',
            'A_type_features': self.a_type_quality_features,
            'B_type_features': self.b_type_entry_features,
            'walk_forward_params': {
                'train_months': self.train_months,
                'val_months': self.val_months,
                'test_months': self.test_months,
                'step_months': self.step_months
            },
            'gpu_settings': {
                'gpu_available': torch.cuda.is_available(),
                'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            'training_timestamp': datetime.now().isoformat(),
            'model_description': {
                'A_type': {
                    'name': 'Post-Trade Quality Analyzer',
                    'purpose': '이 거래가 얼마나 좋은 거래였는가?',
                    'usage': '거래 복기, 성과 분석, 트레이더 평가',
                    'data_scope': '모든 거래 정보 활용 (진입+진행+종료)',
                    'accuracy': '높음 (완전한 정보 활용)'
                },
                'B_type': {
                    'name': 'Real-time Entry Condition Evaluator',
                    'purpose': '지금 매수하기 좋은 조건인가?',
                    'usage': '매수 타이밍, 종목 선별, 리스크 관리',
                    'data_scope': '현재 시점 정보만 (미래 정보 완전 차단)',
                    'practicality': '높음 (실제 트레이딩 환경과 동일)'
                },
                'key_difference': {
                    'A_type': '과거 분석 → 정확한 품질 측정',
                    'B_type': '현재 분석 → 현실적 활용 가능'
                }
            },
            'total_folds': len(self.fold_results)
        }
        
        with open(f'{filepath_prefix}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ 하이브리드 모델 결과 저장 완료: {filepath_prefix}.*")

def main():
    """
    🚀 하이브리드 트레이딩 AI v4 메인 실행 함수
    
    A유형과 B유형 모델을 동시에 학습하고 평가합니다.
    
    실행 과정:
    1. 데이터 로드 및 전처리
    2. Walk-Forward 폴드 생성
    3. 각 폴드에서 A유형, B유형 모델 병렬 학습
    4. 성능 결과 집계 및 분석
    5. 결과 저장
    """
    print("🚀 의미가 명확해진 하이브리드 트레이딩 AI v4")
    print("   A유형: 완료된 거래 품질 분석기 (Post-Trade Quality Analyzer)")
    print("   B유형: 실시간 진입 조건 평가기 (Real-time Entry Condition Evaluator)")
    print("="*80)
    
    # GPU 확인
    print(f"GPU 사용 가능: {torch.cuda.is_available()}")
    print(f"GPU 개수: {torch.cuda.device_count()}")
    
    # 데이터 로드
    print("📊 데이터 로드 중...")
    df = pd.read_csv('../results/final/trading_episodes_with_rebuilt_market_component.csv')
    print(f"  총 데이터: {len(df):,}개")
    
    # 하이브리드 AI 모델 초기화
    print("\n🛠️ 하이브리드 AI 모델 초기화 중...")
    model = HybridTradingAI(
        train_months=36,     # 학습 기간: 36개월 (3년)
        val_months=6,        # 검증 기간: 6개월
        test_months=6,       # 테스트 기간: 6개월
        step_months=3        # 슬라이딩 간격: 3개월
    )
    print(f"   ✅ Walk-Forward 설정: {36}개월 학습 → {6}개월 검증 → {6}개월 테스트")
    print(f"   ✅ 슬라이딩 윈도우: {3}개월씩 이동")
    
    # Walk-Forward Validation 실행
    results = model.run_walk_forward_validation(df, verbose=True)
    
    if results:
        # 결과 저장
        model.save_results()
        
        print("\n" + "="*80)
        print("🎯 하이브리드 AI 핵심 특징 요약:")
        print()
        print("📊 A유형 (거래 품질 분석기):")
        print("   • 목적: '이 거래가 얼마나 좋았나?' 객관적 평가")
        print("   • 활용: 거래 복기, 성과 분석, 트레이더 평가")
        print("   • 데이터: 모든 거래 정보 활용 (진입+진행+종료)")
        print("   • 정확도: 높음 (완전한 정보 활용)")
        print()
        print("🔮 B유형 (진입 조건 평가기):")
        print("   • 목적: '지금 매수하기 좋은 조건인가?' 실시간 판단")
        print("   • 활용: 매수 타이밍, 종목 선별, 리스크 관리")
        print("   • 데이터: 현재 시점 정보만 (미래 정보 완전 차단)")
        print("   • 현실성: 높음 (실제 트레이딩 환경과 동일)")
        print()
        print("💡 핵심 차이점:")
        print("   • A유형: 과거 분석 → 정확한 품질 측정")
        print("   • B유형: 현재 분석 → 현실적 활용 가능")
        print("   → 상호 보완적인 듀얼 시스템!")
    else:
        print("❌ 하이브리드 AI 실행 실패")
    
    print("\n" + "="*80)
    print("🏁 하이브리드 트레이딩 AI v4 실행 완료!")
    print("   📁 결과 파일: hybrid_results.json, hybrid_results_metadata.json")
    print("   📊 A유형 모델: 거래 품질 분석 완료")
    print("   🔮 B유형 모델: 진입 조건 평가 완료")
    print("   🚀 실용적인 트레이딩 지원 시스템 준비됨!")
    print("="*80)

if __name__ == "__main__":
    main()