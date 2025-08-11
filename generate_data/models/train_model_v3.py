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

class WalkForwardQualityModel:
    """
    Walk-Forward Validation Quality Score 모델
    
    개선사항:
    1. 간소화된 Quality Score (Risk Management + Efficiency)
    2. Data Leakage 최소화 (3개 피처만 Quality Score에 사용)
    3. Walk-Forward Validation으로 시계열 견고성 확보
    4. 시장 레짐별 성능 분석
    """
    
    def __init__(self, train_months=36, val_months=6, test_months=6, step_months=3):
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.step_months = step_months
        
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.fold_results = []
        
    def create_quality_score(self, df, risk_scaler=None, eff_scaler=None, verbose=False):
        """간소화된 Quality Score 생성 (Data Leakage 방지)"""
        if verbose:
            print("🎯 Quality Score 생성 중...")
        
        df = df.copy()
        
        # --------------------------------
        # 1. NaN 처리 (중립적 값 0으로 처리)
        # --------------------------------
        # 필수 컬럼들의 NaN을 0으로 처리 (정보 없음을 의미)
        df['return_pct'] = df['return_pct'].fillna(0)  # 수익률 없으면 0
        df['entry_volatility_20d'] = df['entry_volatility_20d'].fillna(0)  # 변동성 정보 없으면 0
        df['entry_ratio_52w_high'] = df['entry_ratio_52w_high'].fillna(0)  # 52주 고점 비율 정보 없으면 0
        df['holding_period_days'] = df['holding_period_days'].fillna(0)  # 보유기간 정보 없으면 0
        
        # --------------------------------
        # 2. Risk Management Quality (40%)
        # --------------------------------
        # 리스크 조정 수익률 (NaN 안전 처리)
        volatility_safe = np.maximum(df['entry_volatility_20d'], 0.01)
        df['risk_adj_return'] = df['return_pct'] / volatility_safe
        
        # 무한값 처리
        df['risk_adj_return'] = np.where(
            np.isinf(df['risk_adj_return']) | np.isnan(df['risk_adj_return']), 
            0, 
            df['risk_adj_return']
        )
        
        # 가격 안전도 (NaN 안전 처리)
        ratio_safe = np.clip(df['entry_ratio_52w_high'], 0, 100)
        df['price_safety'] = (100 - ratio_safe) / 100
        
        # 종합 리스크 관리 점수
        df['risk_management_score'] = df['risk_adj_return'] * 0.6 + df['price_safety'] * 0.4
        
        # --------------------------------
        # 3. Efficiency Quality (60%)
        # --------------------------------
        # 시간 효율성 (NaN 안전 처리)
        holding_safe = np.maximum(df['holding_period_days'], 1)
        df['time_efficiency'] = df['return_pct'] / holding_safe
        
        # 무한값 처리
        df['time_efficiency'] = np.where(
            np.isinf(df['time_efficiency']) | np.isnan(df['time_efficiency']), 
            0, 
            df['time_efficiency']
        )
        
        # 효율성 점수
        df['efficiency_score'] = df['time_efficiency']
        
        # --------------------------------
        # 4. 종합 Quality Score (Data Leakage 방지)
        # --------------------------------
        if risk_scaler is None or eff_scaler is None:
            # 학습용: 새로운 스케일러 생성
            risk_scaler = RobustScaler()
            eff_scaler = RobustScaler()
            
            risk_scaled = risk_scaler.fit_transform(df[['risk_management_score']])
            eff_scaled = eff_scaler.fit_transform(df[['efficiency_score']])
            
            # 스케일러 저장
            self.scalers['risk_scaler'] = risk_scaler
            self.scalers['efficiency_scaler'] = eff_scaler
        else:
            # 검증/테스트용: 기존 스케일러 사용
            risk_scaled = risk_scaler.transform(df[['risk_management_score']])
            eff_scaled = eff_scaler.transform(df[['efficiency_score']])
        
        df['quality_score'] = risk_scaled.flatten() * 0.4 + eff_scaled.flatten() * 0.6
        
        if verbose:
            print(f"  ✅ Quality Score 생성 완료")
            print(f"  범위: {df['quality_score'].min():.4f} ~ {df['quality_score'].max():.4f}")
            print(f"  평균: {df['quality_score'].mean():.4f}")
            print(f"  NaN 개수: {df['quality_score'].isna().sum()}")
            
        return df
    
    def prepare_features(self, df, verbose=False):
        """ML 모델용 피처 준비 (Quality Score 계산에 사용된 피처 제외)"""
        if verbose:
            print("🔧 피처 준비 중...")
        
        # Quality Score 계산에 사용된 피처들 (제외해야 함)
        excluded_features = {
            'return_pct',  # 타겟과 직결
            'entry_volatility_20d',  # Risk Management에 사용
            'entry_ratio_52w_high',  # Risk Management에 사용
            'holding_period_days',  # Efficiency에 사용
            
            # Quality Score 계산 과정에서 생성된 컬럼들도 제외
            'risk_adj_return', 'price_safety', 'risk_management_score',
            'time_efficiency', 'efficiency_score', 'quality_score'
        }
        
        # 사용 가능한 피처들 선정
        available_features = []
        
        # 1. 기본 정보 (일부)
        basic_features = ['position_size_pct']  # return_pct, holding_period_days 제외
        available_features.extend([col for col in basic_features if col in df.columns])
        
        # 2. 모멘텀 지표들 (entry_momentum_20d 제외하고 나머지)
        momentum_features = ['entry_momentum_5d', 'entry_momentum_60d', 
                           'exit_momentum_5d', 'exit_momentum_20d', 'exit_momentum_60d',
                           'change_momentum_5d', 'change_momentum_20d', 'change_momentum_60d']
        available_features.extend([col for col in momentum_features if col in df.columns])
        
        # 3. 이동평균 괴리도
        ma_dev_features = ['entry_ma_dev_5d', 'entry_ma_dev_20d', 'entry_ma_dev_60d',
                          'exit_ma_dev_5d', 'exit_ma_dev_20d', 'exit_ma_dev_60d',
                          'change_ma_dev_5d', 'change_ma_dev_20d', 'change_ma_dev_60d']
        available_features.extend([col for col in ma_dev_features if col in df.columns])
        
        # 4. 변동성 지표들 (entry_volatility_20d 제외)
        vol_features = ['entry_volatility_5d', 'entry_volatility_60d',
                       'exit_volatility_5d', 'exit_volatility_20d', 'exit_volatility_60d',
                       'change_volatility_5d', 'change_volatility_60d',  # change_volatility_20d는 포함
                       'entry_vol_change_5d', 'entry_vol_change_20d', 'entry_vol_change_60d']
        available_features.extend([col for col in vol_features if col in df.columns])
        
        # 5. 시장 환경 (모두 사용 가능)
        market_features = ['entry_vix', 'exit_vix', 'change_vix',
                          'entry_tnx_yield', 'exit_tnx_yield', 'change_tnx_yield',
                          'market_return_during_holding', 'excess_return']
        available_features.extend([col for col in market_features if col in df.columns])
        
        # 6. 펀더멘털 (누락이 많아서 일단 제외)
        # fundamental_features = ['entry_pe_ratio', 'entry_pb_ratio', 'entry_roe', ...]
        # available_features.extend([col for col in fundamental_features if col in df.columns])
        
        # 7. 기타 비율 지표들 (entry_ratio_52w_high 제외)
        ratio_features = ['exit_ratio_52w_high', 'change_ratio_52w_high']
        available_features.extend([col for col in ratio_features if col in df.columns])
        
        # 실제 존재하는 피처만 선택
        self.feature_columns = [col for col in available_features 
                               if col in df.columns and col not in excluded_features]
        
        if verbose:
            print(f"  사용 가능한 피처: {len(self.feature_columns)}개")
            print(f"  제외된 피처: {len(excluded_features)}개")
            
        return df[self.feature_columns]
    
    def create_walk_forward_folds(self, df, verbose=False):
        """Walk-Forward 분할 생성"""
        if verbose:
            print("📅 Walk-Forward 분할 생성 중...")
        
        df = df.copy()
        df['entry_date'] = pd.to_datetime(df['entry_datetime'])
        df = df.sort_values('entry_date')
        
        # 시작일과 종료일
        start_date = df['entry_date'].min()
        end_date = df['entry_date'].max()
        
        folds = []
        
        # 첫 번째 폴드의 시작점 계산
        current_date = start_date + pd.DateOffset(months=self.train_months)
        
        while current_date + pd.DateOffset(months=self.val_months + self.test_months) <= end_date:
            # 각 구간 계산
            train_start = current_date - pd.DateOffset(months=self.train_months)
            train_end = current_date
            val_start = train_end
            val_end = val_start + pd.DateOffset(months=self.val_months)
            test_start = val_end
            test_end = test_start + pd.DateOffset(months=self.test_months)
            
            # 인덱스 생성
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
            
            # 다음 폴드로 이동
            current_date += pd.DateOffset(months=self.step_months)
        
        if verbose:
            print(f"  생성된 폴드 수: {len(folds)}개")
            print("  폴드 정보:")
            for i, fold in enumerate(folds):
                print(f"    Fold {i+1}: Train {fold['train_start']} ~ {fold['train_end']}")
                print(f"             Val   {fold['val_start']} ~ {fold['val_end']}")
                print(f"             Test  {fold['test_start']} ~ {fold['test_end']}")
                print(f"             Size: {len(fold['train_idx'])}/{len(fold['val_idx'])}/{len(fold['test_idx'])}")
        
        return folds
    
    def evaluate_single_fold(self, df, fold, verbose=False):
        """단일 폴드 평가"""
        if verbose:
            print(f"\n🔄 Fold {fold['fold_id']} 평가 중...")
        
        # 데이터 분할
        train_data = df.loc[fold['train_idx']].copy()
        val_data = df.loc[fold['val_idx']].copy()
        test_data = df.loc[fold['test_idx']].copy()
        
        # Quality Score 생성 (Train에서만 스케일러 학습)
        train_data = self.create_quality_score(train_data, verbose=False)
        val_data = self.create_quality_score(
            val_data, 
            risk_scaler=self.scalers['risk_scaler'],
            eff_scaler=self.scalers['efficiency_scaler'],
            verbose=False
        )
        test_data = self.create_quality_score(
            test_data,
            risk_scaler=self.scalers['risk_scaler'], 
            eff_scaler=self.scalers['efficiency_scaler'],
            verbose=False
        )
        
        # 피처 준비
        X_train = self.prepare_features(train_data, verbose=False)
        X_val = self.prepare_features(val_data, verbose=False)
        X_test = self.prepare_features(test_data, verbose=False)
        
        y_train = train_data['quality_score']
        y_val = val_data['quality_score'] 
        y_test = test_data['quality_score']
        
        # 결측치 제거
        train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
        val_mask = ~(X_val.isnull().any(axis=1) | y_val.isnull())
        test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
        
        X_train = X_train[train_mask]
        X_val = X_val[val_mask]
        X_test = X_test[test_mask]
        y_train = y_train[train_mask]
        y_val = y_val[val_mask]
        y_test = y_test[test_mask]
        
        # 피처 스케일링
        feature_scaler = RobustScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_val_scaled = feature_scaler.transform(X_val)
        X_test_scaled = feature_scaler.transform(X_test)
        
        # GPU 설정 확인
        gpu_available = torch.cuda.is_available()
        num_gpus = torch.cuda.device_count() if gpu_available else 0
        
        if verbose:
            print(f"  GPU 사용 가능: {gpu_available}")
            print(f"  GPU 개수: {num_gpus}")
        
        # XGBoost 베이스 모델 설정
        base_model = xgb.XGBRegressor(
            tree_method='gpu_hist' if gpu_available else 'hist',
            gpu_id=0 if gpu_available else None,
            random_state=42,
            eval_metric='rmse'
        )
        
        # GridSearchCV를 위한 하이퍼파라미터 그리드
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [6, 8, 10],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        # GridSearchCV 설정
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,  # 3-fold CV
            scoring='r2',
            n_jobs=min(num_gpus, 4) if num_gpus > 0 else -1,  # GPU 개수에 따라 조정
            verbose=1 if verbose else 0
        )
        
        if verbose:
            print(f"  GridSearchCV 시작: {len(param_grid['n_estimators']) * len(param_grid['learning_rate']) * len(param_grid['max_depth']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree'])} 조합 테스트")
        
        # 하이퍼파라미터 튜닝
        grid_search.fit(X_train_scaled, y_train)
        
        # 최적 모델로 평가
        best_model = grid_search.best_estimator_
        
        # 검증 세트 평가
        y_val_pred = best_model.predict(X_val_scaled)
        val_r2 = r2_score(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        # 테스트 세트 평가
        y_test_pred = best_model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        fold_results = {
            'xgboost': {
                'val_r2': val_r2,
                'val_rmse': val_rmse,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_
            }
        }
        
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
            'model_results': fold_results,
            'market_stats': market_stats
        }
    
    def run_walk_forward_validation(self, df, verbose=False):
        """전체 Walk-Forward Validation 실행"""
        print("🚀 Walk-Forward Validation 시작")
        print("="*70)
        
        # 폴드 생성
        folds = self.create_walk_forward_folds(df, verbose=True)
        
        if len(folds) == 0:
            print("❌ 생성된 폴드가 없습니다. 파라미터를 조정해주세요.")
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
    
    def get_best_models_from_folds(self, verbose=False):
        """폴드별 최고 성능 모델 선택"""
        if not self.fold_results:
            return None
            
        # XGBoost 성능 집계
        test_r2_scores = [r['model_results']['xgboost']['test_r2'] for r in self.fold_results]
        cv_scores = [r['model_results']['xgboost']['best_cv_score'] for r in self.fold_results]
        
        avg_test_r2 = np.mean(test_r2_scores)
        avg_cv_score = np.mean(cv_scores)
        
        if verbose:
            print(f"\n🏆 XGBoost Walk-Forward 성능 요약:")
            print(f"  평균 Test R²: {avg_test_r2:.4f} ± {np.std(test_r2_scores):.4f}")
            print(f"  평균 CV Score: {avg_cv_score:.4f} ± {np.std(cv_scores):.4f}")
            print(f"  Test R² 범위: [{np.min(test_r2_scores):.4f}, {np.max(test_r2_scores):.4f}]")
            
            # 각 폴드의 최적 파라미터 표시
            print(f"\n폴드별 최적 파라미터:")
            for i, result in enumerate(self.fold_results):
                best_params = result['model_results']['xgboost']['best_params']
                test_r2 = result['model_results']['xgboost']['test_r2']
                print(f"  Fold {i+1} (R²={test_r2:.4f}): {best_params}")
        
        return 'xgboost', {'xgboost': avg_test_r2}
    
    def aggregate_results(self, verbose=False):
        """폴드 결과 집계 및 통계"""
        if not self.fold_results:
            print("❌ 집계할 결과가 없습니다.")
            return
        
        if verbose:
            print("\n📈 Walk-Forward Validation 결과 집계")
            print("="*50)
        
        # XGBoost 성능 집계 디테일
        val_r2_scores = [r['model_results']['xgboost']['val_r2'] for r in self.fold_results]
        test_r2_scores = [r['model_results']['xgboost']['test_r2'] for r in self.fold_results]
        cv_scores = [r['model_results']['xgboost']['best_cv_score'] for r in self.fold_results]
        
        if verbose:
            print(f"\nXGBOOST 성능 상세:")
            print(f"  Validation R²: {np.mean(val_r2_scores):.4f} ± {np.std(val_r2_scores):.4f}")
            print(f"  Test R²:       {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")
            print(f"  CV Score:       {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            print(f"  Test R² 범위:  [{np.min(test_r2_scores):.4f}, {np.max(test_r2_scores):.4f}]")
            print(f"  일관성 (CV):   {np.std(test_r2_scores)/np.mean(test_r2_scores)*100:.1f}%")
        
        # 시장 환경별 성능
        if verbose:
            print(f"\n🌊 시장 환경별 성능 분석:")
            
            for i, result in enumerate(self.fold_results):
                market = result['market_stats']
                test_r2 = result['model_results']['xgboost']['test_r2']  # XGBoost 기준
                
                vix_level = "고변동" if market['test_vix_mean'] > 25 else "저변동"
                return_level = "상승" if market['test_return_mean'] > 0.05 else "하락"
                
                print(f"  Fold {i+1}: {vix_level}/{return_level} → Test R² = {test_r2:.4f}")
    
    def feature_importance_analysis(self, verbose=False):
        """마지막 폴드 XGBoost 모델로 피처 중요도 분석"""
        if verbose:
            print("\n🔍 피처 중요도 분석")
        
        if not self.fold_results:
            print("  분석할 폴드 결과가 없습니다.")
            return None
            
        # 마지막 폴드에서 XGBoost 모델 다시 학습 (피처 중요도 추출용)
        last_result = self.fold_results[-1]
        fold = last_result['fold_info']
        
        # 데이터 준비
        df = pd.read_csv('../results/final/trading_episodes_with_rebuilt_market_component.csv')
        train_data = df.loc[fold['train_idx']].copy()
        train_data = self.create_quality_score(train_data, verbose=False)
        
        X_train = self.prepare_features(train_data, verbose=False)
        y_train = train_data['quality_score']
        
        # 결측치 제거
        mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
        X_train = X_train[mask]
        y_train = y_train[mask]
        
        # 스케일링 및 학습
        feature_scaler = RobustScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        
        # GPU 설정 확인
        gpu_available = torch.cuda.is_available()
        
        # 마지막 폴드에서 사용된 최적 파라미터 사용
        last_best_params = self.fold_results[-1]['model_results']['xgboost']['best_params']
        
        xgb_model = xgb.XGBRegressor(
            tree_method='gpu_hist' if gpu_available else 'hist',
            gpu_id=0 if gpu_available else None,
            random_state=42,
            eval_metric='rmse',
            **last_best_params  # 최적 파라미터 적용
        )
        xgb_model.fit(X_train_scaled, y_train)
        
        # 피처 중요도 추출
        feature_importance = xgb_model.feature_importances_
        
        # 중요도 정렬
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        if verbose:
            print("\n상위 20개 피처:")
            print(importance_df.head(20).to_string(index=False))
            print(f"\n사용된 최적 파라미터: {last_best_params}")
        
        return importance_df
    
    def save_results(self, filepath_prefix='walk_forward_results'):
        """결과 저장"""
        if not self.fold_results:
            print("❌ 저장할 결과가 없습니다.")
            return
        
        # 결과를 JSON으로 저장 (datetime 처리)
        results_for_save = []
        for result in self.fold_results:
            result_copy = result.copy()
            
            # datetime 객체를 문자열로 변환
            fold_info = result_copy['fold_info'].copy()
            for key in ['train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']:
                if key in fold_info:
                    fold_info[key] = str(fold_info[key])
            
            # 인덱스 제거 (저장 용량 절약)
            del fold_info['train_idx']
            del fold_info['val_idx'] 
            del fold_info['test_idx']
            
            result_copy['fold_info'] = fold_info
            results_for_save.append(result_copy)
        
        with open(f'{filepath_prefix}.json', 'w') as f:
            json.dump(results_for_save, f, indent=2)
        
        # 메타데이터도 저장
        metadata = {
            'model_version': 'walk_forward_xgboost_v1',
            'model_type': 'XGBoost with GridSearchCV',
            'feature_columns': self.feature_columns,
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
            'quality_score_components': {
                'risk_management_weight': 0.4,
                'efficiency_weight': 0.6,
                'features_used_for_quality_score': [
                    'entry_volatility_20d',
                    'entry_ratio_52w_high', 
                    'holding_period_days'
                ]
            },
            'total_folds': len(self.fold_results)
        }
        
        with open(f'{filepath_prefix}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ 결과 저장 완료: {filepath_prefix}.*")

def main():
    """메인 실행 함수"""
    print("🚀 Walk-Forward XGBoost Quality Score 모델 검증")
    print("="*70)
    
    # GPU 확인
    print(f"GPU 사용 가능: {torch.cuda.is_available()}")
    print(f"GPU 개수: {torch.cuda.device_count()}")
    
    # 데이터 로드
    print("📊 데이터 로드 중...")
    df = pd.read_csv('../results/final/trading_episodes_with_rebuilt_market_component.csv')
    print(f"  총 데이터: {len(df):,}개")
    
    # Walk-Forward XGBoost 모델 초기화
    # 파라미터: train 36개월, val 6개월, test 6개월, step 3개월
    model = WalkForwardQualityModel(
        train_months=36,
        val_months=6, 
        test_months=6,
        step_months=3
    )
    
    # Walk-Forward Validation 실행
    results = model.run_walk_forward_validation(df, verbose=True)
    
    if results:
        # 최고 성능 모델 선택
        best_model_info = model.get_best_models_from_folds(verbose=True)
        
        # 피처 중요도 분석 (마지막 폴드 기준)
        importance_df = model.feature_importance_analysis(verbose=True)
        
        # 결과 저장
        model.save_results()
    else:
        print("❌ Walk-Forward Validation 실행 실패")
    
    print("\n" + "="*70)
    print("✅ Walk-Forward Validation 완료!")
    print("="*70)

if __name__ == "__main__":
    main()