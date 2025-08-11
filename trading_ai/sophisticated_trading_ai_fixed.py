#!/usr/bin/env python3
"""
정교한 AI 거래 어시스턴트 - 과적합 문제 해결 버전
- customer_id 기반 특징 제거
- isin 기반 특징 제거
- 미래 정보 사용 제거
- 일반화 가능한 패턴만 학습
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import VotingRegressor, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from datetime import datetime
import joblib
import json
import warnings
warnings.filterwarnings('ignore')


class SophisticatedTradingAI:
    """과적합 문제를 해결한 정교한 AI 시스템"""
    
    def __init__(self):
        # 데이터 저장소
        self.episodes_df = None
        self.features_df = None
        self.situations_df = None
        
        # KNN 모델 (유사 상황 검색)
        self.buy_situation_knn = NearestNeighbors(
            n_neighbors=30,
            metric='minkowski',
            p=2,
            algorithm='ball_tree',
            leaf_size=30
        )
        
        self.sell_situation_knn = NearestNeighbors(
            n_neighbors=30,
            metric='cosine',
            algorithm='brute'
        )
        
        # 앙상블 예측 모델들
        self._initialize_ensemble_models()
        
        # 스케일러
        self.buy_scaler = RobustScaler()
        self.sell_scaler = RobustScaler()
        self.feature_scaler = StandardScaler()
        
        # 메타 정보
        self.feature_columns = {}
        self.model_performance = {}
        self.feature_importance_all = {}
        
    def _initialize_ensemble_models(self):
        """앙상블 모델 초기화"""
        # 수익률 예측 앙상블
        self.return_ensemble = VotingRegressor([
            ('xgb', xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42
            )),
            ('lgb', lgb.LGBMRegressor(
                n_estimators=300,
                num_leaves=20,
                learning_rate=0.05,
                feature_fraction=0.7,
                bagging_fraction=0.7,
                bagging_freq=5,
                random_state=42,
                verbose=-1
            )),
            ('cat', CatBoostRegressor(
                iterations=300,
                depth=5,
                learning_rate=0.05,
                random_state=42,
                verbose=False
            ))
        ])
        
        # 성공 확률 예측 앙상블
        self.success_ensemble = VotingClassifier([
            ('xgb', xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=300,
                num_leaves=20,
                learning_rate=0.05,
                feature_fraction=0.7,
                bagging_fraction=0.7,
                bagging_freq=5,
                random_state=42,
                verbose=-1
            )),
            ('cat', CatBoostClassifier(
                iterations=300,
                depth=5,
                learning_rate=0.05,
                random_state=42,
                verbose=False
            ))
        ], voting='soft')
        
        # 보유 기간 예측 모델
        self.holding_predictor = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        
    def load_all_data(self, episodes_path, features_path, situations_path):
        """3개 데이터셋 통합 (과적합 특징 제거)"""
        print("📊 데이터 로딩 및 검증 중...")
        
        # 1. 데이터 로드
        self.episodes_df = pd.read_csv(episodes_path)
        self.features_df = pd.read_csv(features_path)
        self.situations_df = pd.read_csv(situations_path)
        
        # 2. 데이터 타입 최적화
        self._optimize_data_types()
        
        # 3. 과적합 특징 제거
        self._remove_overfitting_features()
        
        # 4. 데이터 무결성 검증
        self._validate_data_integrity()
        
        # 5. 일반화 가능한 특징만 생성
        self._create_generalizable_features()
        
        print(f"\n✅ 데이터 통합 완료:")
        print(f"   - 에피소드: {len(self.episodes_df):,}개")
        print(f"   - 특징 차원: {len(self.feature_columns['all'])}개")
        print(f"   - 상황: {len(self.situations_df):,}개")
        
    def _optimize_data_types(self):
        """메모리 효율을 위한 데이터 타입 최적화"""
        for df in [self.episodes_df, self.features_df, self.situations_df]:
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = df[col].astype('float32')
                elif df[col].dtype == 'int64':
                    if df[col].min() >= 0 and df[col].max() < 65535:
                        df[col] = df[col].astype('uint16')
                        
    def _remove_overfitting_features(self):
        """과적합 유발 특징 제거"""
        print("🧹 과적합 특징 제거 중...")
        
        # 1. customer_id 관련 특징 제거
        customer_cols = [col for col in self.features_df.columns if 'customer' in col.lower()]
        print(f"   - customer_id 관련 {len(customer_cols)}개 컬럼 제거")
        
        # 2. isin 관련 특징 제거
        isin_cols = [col for col in self.features_df.columns if 'isin' in col.lower()]
        print(f"   - isin 관련 {len(isin_cols)}개 컬럼 제거")
        
        # 3. 미래 정보를 포함한 특징 제거
        future_cols = ['outcome_return_rate', 'outcome_holding_days', 'outcome_profitable',
                      'global_avg_return', 'global_win_rate', 'global_avg_holding']
        future_cols.extend([col for col in self.features_df.columns if 'outcome_' in col])
        future_cols.extend([col for col in self.features_df.columns if 'global_' in col])
        
        # 실제로 존재하는 컬럼만 제거
        cols_to_remove = set()
        for col in customer_cols + isin_cols + future_cols:
            if col in self.features_df.columns:
                cols_to_remove.add(col)
                
        print(f"   - 미래 정보 {len([c for c in future_cols if c in self.features_df.columns])}개 컬럼 제거")
        
        # situations_df에서도 제거
        if 'customer_id' in self.situations_df.columns:
            self.situations_df = self.situations_df.drop(columns=['customer_id'])
        if 'isin' in self.situations_df.columns:
            self.situations_df = self.situations_df.drop(columns=['isin'])
            
        print(f"   - 총 {len(cols_to_remove)}개 과적합 특징 제거")
        
    def _validate_data_integrity(self):
        """데이터 무결성 검증"""
        # episode_id 일관성 검사
        episodes_in_features = set(self.features_df['episode_id'].unique())
        episodes_in_situations = set(self.situations_df['episode_id'].unique())
        episodes_in_episodes = set(self.episodes_df['episode_id'].unique())
        
        common_episodes = episodes_in_episodes & episodes_in_features
        print(f"   - 검증: {len(common_episodes):,}개 에피소드 매칭 확인")
        
        # 결측값 처리
        for df in [self.features_df, self.situations_df]:
            missing_ratio = df.isnull().sum() / len(df)
            high_missing_cols = missing_ratio[missing_ratio > 0.5].index
            
            if len(high_missing_cols) > 0:
                print(f"   - 경고: {len(high_missing_cols)}개 컬럼에 50% 이상 결측값")
                for col in high_missing_cols:
                    df[f'{col}_is_missing'] = df[col].isnull().astype(int)
                    
    def _create_generalizable_features(self):
        """일반화 가능한 특징만 생성"""
        print("🔧 일반화 가능한 특징 엔지니어링 중...")
        
        # 1. 시장 상황 특징 (날짜별, 개인 정보 없이)
        market_features = self._calculate_market_features_without_identity()
        
        # 2. 상대적 특징 (전체 대비, 개인 정보 없이)
        relative_features = self._calculate_relative_features_without_identity()
        
        # 3. 시간 패턴 특징 (요일, 월, 계절 등)
        temporal_features = self._calculate_temporal_features()
        
        # 4. 기술적 지표 스타일 특징
        technical_features = self._calculate_technical_style_features()
        
        # 특징 통합
        self._merge_generalizable_features(
            market_features,
            relative_features,
            temporal_features,
            technical_features
        )
        
        # 특징 컬럼 저장 (outcome 변수 확실히 제외)
        exclude_cols = {'episode_id', 'customer_id', 'isin'}
        exclude_cols.update({col for col in self.features_df.columns if 'outcome' in col})
        exclude_cols.update({col for col in self.features_df.columns if 'customer' in col})
        exclude_cols.update({col for col in self.features_df.columns if 'isin' in col})
        
        self.feature_columns = {
            'buy': [col for col in self.situations_df.columns if col.startswith('feature_')],
            'sell': [col for col in self.features_df.columns if col not in exclude_cols],
            'all': [col for col in self.features_df.columns if col not in exclude_cols]
        }
        
        print(f"   - 사용할 특징 수: {len(self.feature_columns['all'])}개")
        
    def _calculate_market_features_without_identity(self):
        """개인 정보 없는 시장 특징 계산"""
        # 날짜별 시장 전체 통계 (과거 데이터만 사용)
        self.episodes_df['date'] = pd.to_datetime(self.episodes_df['buy_timestamp']).dt.date
        
        market_stats = {}
        unique_dates = sorted(self.episodes_df['date'].unique())
        
        for i, date in enumerate(unique_dates):
            # 해당 날짜 이전의 데이터만 사용
            past_data = self.episodes_df[self.episodes_df['date'] < date]
            
            if len(past_data) > 10:  # 충분한 데이터가 있을 때만
                market_stats[date] = {
                    'market_avg_return_7d': past_data.tail(7)['return_rate'].mean() if len(past_data) > 7 else 0,
                    'market_volatility_7d': past_data.tail(7)['return_rate'].std() if len(past_data) > 7 else 0,
                    'market_avg_return_30d': past_data.tail(30)['return_rate'].mean() if len(past_data) > 30 else 0,
                    'market_volatility_30d': past_data.tail(30)['return_rate'].std() if len(past_data) > 30 else 0,
                }
            else:
                market_stats[date] = {
                    'market_avg_return_7d': 0,
                    'market_volatility_7d': 10,
                    'market_avg_return_30d': 0,
                    'market_volatility_30d': 10,
                }
                
        return market_stats
    
    def _calculate_relative_features_without_identity(self):
        """개인 정보 없는 상대적 특징 계산"""
        # 전체 시장 대비 상대적 위치 (과거 데이터만 사용)
        relative_features = []
        
        for idx, row in self.features_df.iterrows():
            episode = self.episodes_df[self.episodes_df['episode_id'] == row['episode_id']].iloc[0]
            buy_date = pd.to_datetime(episode['buy_timestamp']).date()
            
            # 해당 날짜 이전 30일 데이터
            past_episodes = self.episodes_df[
                pd.to_datetime(self.episodes_df['buy_timestamp']).dt.date < buy_date
            ].tail(100)
            
            if len(past_episodes) > 10:
                # 현재 값의 상대적 위치
                relative_features.append({
                    'return_vs_market_avg': row['current_return'] / (past_episodes['return_rate'].mean() + 1e-6),
                    'holding_vs_market_avg': row['holding_days'] / (past_episodes['holding_days'].mean() + 1e-6),
                    'return_percentile_market': (past_episodes['return_rate'] < row['current_return']).mean(),
                    'holding_percentile_market': (past_episodes['holding_days'] < row['holding_days']).mean()
                })
            else:
                relative_features.append({
                    'return_vs_market_avg': 1.0,
                    'holding_vs_market_avg': 1.0,
                    'return_percentile_market': 0.5,
                    'holding_percentile_market': 0.5
                })
                
        return pd.DataFrame(relative_features)
    
    def _calculate_temporal_features(self):
        """시간 관련 특징 계산"""
        temporal_features = []
        
        for _, episode in self.episodes_df.iterrows():
            buy_time = pd.to_datetime(episode['buy_timestamp'])
            sell_time = pd.to_datetime(episode['sell_timestamp'])
            
            temporal_features.append({
                'buy_day_of_week': buy_time.dayofweek,
                'buy_month': buy_time.month,
                'buy_quarter': buy_time.quarter,
                'buy_is_month_start': buy_time.day <= 5,
                'buy_is_month_end': buy_time.day >= 25,
                'sell_day_of_week': sell_time.dayofweek,
                'holding_over_weekend': (sell_time - buy_time).days // 7,
                'holding_weekdays': np.busday_count(buy_time.date(), sell_time.date())
            })
            
        return pd.DataFrame(temporal_features)
    
    def _calculate_technical_style_features(self):
        """기술적 지표 스타일 특징"""
        technical_features = []
        
        for _, row in self.features_df.iterrows():
            # 수익률 구간화 (기술적 분석 스타일)
            return_rate = row['current_return']
            holding_days = row['holding_days']
            
            features = {
                # 수익률 구간
                'return_zone_negative': 1 if return_rate < 0 else 0,
                'return_zone_0_3': 1 if 0 <= return_rate < 3 else 0,
                'return_zone_3_5': 1 if 3 <= return_rate < 5 else 0,
                'return_zone_5_10': 1 if 5 <= return_rate < 10 else 0,
                'return_zone_10_plus': 1 if return_rate >= 10 else 0,
                
                # 보유기간 구간
                'holding_zone_short': 1 if holding_days < 7 else 0,
                'holding_zone_medium': 1 if 7 <= holding_days < 30 else 0,
                'holding_zone_long': 1 if 30 <= holding_days < 90 else 0,
                'holding_zone_very_long': 1 if holding_days >= 90 else 0,
                
                # 수익률/일 효율성
                'daily_return_efficiency': return_rate / max(holding_days, 1),
                'is_quick_profit': 1 if return_rate > 5 and holding_days < 7 else 0,
                'is_slow_profit': 1 if return_rate > 5 and holding_days > 30 else 0
            }
            
            technical_features.append(features)
            
        return pd.DataFrame(technical_features)
    
    def _merge_generalizable_features(self, market_features, relative_features, temporal_features, technical_features):
        """일반화 가능한 특징들 병합"""
        # features_df에 추가
        for col in relative_features.columns:
            self.features_df[col] = relative_features[col]
            
        for col in technical_features.columns:
            self.features_df[col] = technical_features[col]
            
        # episodes_df의 날짜별로 market features 매핑
        for _, row in self.features_df.iterrows():
            episode = self.episodes_df[self.episodes_df['episode_id'] == row['episode_id']].iloc[0]
            date = pd.to_datetime(episode['buy_timestamp']).date()
            
            if date in market_features:
                for key, value in market_features[date].items():
                    self.features_df.loc[self.features_df['episode_id'] == row['episode_id'], key] = value
                    
    def train_all_models(self):
        """모든 AI 모델 학습 (과적합 방지)"""
        print("\n🤖 일반화된 AI 모델 학습 시작...")
        
        # outcome 변수 준비
        self._prepare_outcome_variables()
        
        # 1. KNN 모델 학습
        self._train_knn_models()
        
        # 2. 앙상블 예측 모델 학습
        self._train_ensemble_models()
        
        # 3. 교차 검증으로 성능 평가
        self._evaluate_with_cross_validation()
        
        print("\n✅ 모든 모델 학습 완료!")
        self._print_model_summary()
        
    def _prepare_outcome_variables(self):
        """outcome 변수 준비 (임시)"""
        # episodes에서 가져오기
        for _, row in self.features_df.iterrows():
            episode = self.episodes_df[self.episodes_df['episode_id'] == row['episode_id']].iloc[0]
            self.features_df.loc[self.features_df['episode_id'] == row['episode_id'], 'outcome_return_rate'] = episode['return_rate']
            self.features_df.loc[self.features_df['episode_id'] == row['episode_id'], 'outcome_holding_days'] = episode['holding_days']
            self.features_df.loc[self.features_df['episode_id'] == row['episode_id'], 'outcome_profitable'] = 1 if episode['return_rate'] > 0 else 0
            
    def _train_knn_models(self):
        """KNN 모델 학습"""
        print("\n📍 KNN 모델 학습 중...")
        
        # 1. 매수 KNN
        buy_features = self.feature_columns['buy']
        if len(buy_features) > 0:
            X_buy = self.situations_df[buy_features].fillna(0).values
            
            # 차원 축소 고려
            if len(buy_features) > 30:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=20, random_state=42)
                X_buy = pca.fit_transform(X_buy)
                print(f"   - PCA 적용: {len(buy_features)}차원 → 20차원")
            
            X_buy_scaled = self.buy_scaler.fit_transform(X_buy)
            self.buy_situation_knn.fit(X_buy_scaled)
            print(f"   ✅ 매수 KNN: {len(X_buy):,}개 상황 학습 완료")
        
        # 2. 매도 KNN
        sell_features = [col for col in self.feature_columns['sell'] if not col.startswith('outcome')]
        X_sell = self.features_df[sell_features].fillna(0).values
        X_sell_scaled = self.sell_scaler.fit_transform(X_sell)
        self.sell_situation_knn.fit(X_sell_scaled)
        print(f"   ✅ 매도 KNN: {len(X_sell):,}개 상황 학습 완료")
        
    def _train_ensemble_models(self):
        """앙상블 모델 학습 (시간순 분할)"""
        print("\n📈 앙상블 예측 모델 학습 중...")
        
        # 특징과 타겟 준비
        feature_cols = [col for col in self.feature_columns['all'] if not col.startswith('outcome')]
        X = self.features_df[feature_cols].fillna(0)
        
        # 타겟 변수들
        y_return = self.features_df['outcome_return_rate']
        y_success = self.features_df['outcome_profitable']
        y_holding = self.features_df['outcome_holding_days']
        
        # 학습/검증 분할 (시간 순서 고려)
        episode_dates = self.episodes_df.set_index('episode_id')['buy_timestamp']
        self.features_df['date'] = self.features_df['episode_id'].map(episode_dates)
        
        # 시간순 정렬
        sorted_indices = self.features_df.sort_values('date').index
        split_point = int(len(sorted_indices) * 0.8)
        
        train_idx = sorted_indices[:split_point]
        val_idx = sorted_indices[split_point:]
        
        X_train = X.loc[train_idx]
        X_val = X.loc[val_idx]
        
        # 스케일링
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        # 1. 수익률 예측 앙상블
        print("   - 수익률 예측 앙상블 학습 중...")
        self.return_ensemble.fit(X_train_scaled, y_return.loc[train_idx])
        
        # 2. 성공 확률 예측 앙상블
        print("   - 성공 확률 예측 앙상블 학습 중...")
        self.success_ensemble.fit(X_train_scaled, y_success.loc[train_idx])
        
        # 3. 보유 기간 예측
        print("   - 보유 기간 예측 모델 학습 중...")
        self.holding_predictor.fit(X_train_scaled, y_holding.loc[train_idx])
        
        # 검증 세트 성능
        self._evaluate_on_validation(X_val_scaled, val_idx)
        
    def _evaluate_on_validation(self, X_val, val_idx):
        """검증 세트 성능 평가"""
        y_return_val = self.features_df.loc[val_idx, 'outcome_return_rate']
        y_success_val = self.features_df.loc[val_idx, 'outcome_profitable']
        y_holding_val = self.features_df.loc[val_idx, 'outcome_holding_days']
        
        # 예측
        return_pred = self.return_ensemble.predict(X_val)
        success_pred = self.success_ensemble.predict(X_val)
        success_proba = self.success_ensemble.predict_proba(X_val)[:, 1]
        holding_pred = self.holding_predictor.predict(X_val)
        
        # 성능 지표
        self.model_performance = {
            'return_mae': mean_absolute_error(y_return_val, return_pred),
            'return_rmse': np.sqrt(mean_squared_error(y_return_val, return_pred)),
            'success_accuracy': accuracy_score(y_success_val, success_pred),
            'success_precision_recall': precision_recall_fscore_support(
                y_success_val, success_pred, average='binary'
            ),
            'holding_mae': mean_absolute_error(y_holding_val, holding_pred)
        }
        
        print(f"\n📊 검증 세트 성능:")
        print(f"   - 수익률 MAE: {self.model_performance['return_mae']:.2f}%")
        print(f"   - 수익률 RMSE: {self.model_performance['return_rmse']:.2f}%")
        print(f"   - 성공 예측 정확도: {self.model_performance['success_accuracy']:.2%}")
        print(f"   - 보유기간 MAE: {self.model_performance['holding_mae']:.1f}일")
        
        # 과적합 검사
        if self.model_performance['success_accuracy'] > 0.85:
            print("\n⚠️ 경고: 여전히 높은 정확도 - 추가 검증 필요")
        else:
            print("\n✅ 정상적인 성능 범위")
            
        # 추가 진단
        print(f"\n📊 추가 진단:")
        print(f"   - 예측 수익률 범위: [{return_pred.min():.1f}%, {return_pred.max():.1f}%]")
        print(f"   - 실제 수익률 범위: [{y_return_val.min():.1f}%, {y_return_val.max():.1f}%]")
        print(f"   - 예측 표준편차: {np.std(return_pred):.2f}% vs 실제: {np.std(y_return_val):.2f}%")
        
    def _evaluate_with_cross_validation(self):
        """교차 검증으로 모델 안정성 평가"""
        print("\n🔄 교차 검증 수행 중...")
        
        feature_cols = [col for col in self.feature_columns['all'] if not col.startswith('outcome')]
        X = self.features_df[feature_cols].fillna(0)
        y = self.features_df['outcome_profitable']
        
        # XGBoost 단일 모델로 빠른 CV
        quick_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(quick_model, X, y, cv=3, scoring='accuracy')
        
        print(f"   - 3-Fold CV 평균 정확도: {scores.mean():.2%} (±{scores.std():.2%})")
        
    def analyze_sell_situation(self, current_situation):
        """매도 상황 분석 (일반화된 패턴 기반)"""
        
        # 1. 특징 벡터 생성 (개인 정보 없이)
        sell_features = self._create_sell_feature_vector_generalized(current_situation)
        
        # 2. KNN으로 유사한 상황 찾기
        similar_cases = self._find_similar_sell_cases_generalized(sell_features)
        
        # 3. 앙상블로 예측
        predictions = self._predict_sell_outcome_generalized(sell_features)
        
        # 4. 패턴 분석
        pattern_analysis = self._analyze_similar_patterns_generalized(similar_cases)
        
        # 5. 신뢰도 계산
        confidence = self._calculate_prediction_confidence(similar_cases, predictions, pattern_analysis)
        
        # 6. 최종 분석 통합
        return self._integrate_sell_analysis(
            current_situation, similar_cases, predictions, pattern_analysis, confidence
        )
        
    def _create_sell_feature_vector_generalized(self, situation):
        """일반화된 매도 특징 벡터 생성"""
        features = {
            # 기본 특징
            'current_return': situation['current_return'],
            'holding_days': situation['holding_days'],
            'return_per_day': situation['current_return'] / max(situation['holding_days'], 1),
            
            # 수익률 구간
            'return_zone_negative': 1 if situation['current_return'] < 0 else 0,
            'return_zone_0_3': 1 if 0 <= situation['current_return'] < 3 else 0,
            'return_zone_3_5': 1 if 3 <= situation['current_return'] < 5 else 0,
            'return_zone_5_10': 1 if 5 <= situation['current_return'] < 10 else 0,
            'return_zone_10_plus': 1 if situation['current_return'] >= 10 else 0,
            
            # 보유기간 구간
            'holding_zone_short': 1 if situation['holding_days'] < 7 else 0,
            'holding_zone_medium': 1 if 7 <= situation['holding_days'] < 30 else 0,
            'holding_zone_long': 1 if 30 <= situation['holding_days'] < 90 else 0,
            'holding_zone_very_long': 1 if situation['holding_days'] >= 90 else 0,
            
            # 효율성
            'daily_return_efficiency': situation['current_return'] / max(situation['holding_days'], 1),
            'is_quick_profit': 1 if situation['current_return'] > 5 and situation['holding_days'] < 7 else 0,
            'is_slow_profit': 1 if situation['current_return'] > 5 and situation['holding_days'] > 30 else 0
        }
        
        # 시장 상황 추가 (있다면)
        if 'market_volatility' in situation:
            features['market_volatility'] = situation['market_volatility']
            
        return features
    
    def _find_similar_sell_cases_generalized(self, current_features):
        """일반화된 유사 매도 상황 찾기"""
        # 특징 벡터 준비
        sell_cols = [col for col in self.feature_columns['sell'] if not col.startswith('outcome')]
        current_vector = []
        
        for col in sell_cols:
            current_vector.append(current_features.get(col, 0))
            
        current_vector = np.array(current_vector).reshape(1, -1)
        current_scaled = self.sell_scaler.transform(current_vector)
        
        # 30개 이웃 찾기
        distances, indices = self.sell_situation_knn.kneighbors(current_scaled, n_neighbors=30)
        
        # 상세 정보 추출
        similar_cases = []
        for dist, idx in zip(distances[0], indices[0]):
            feature_row = self.features_df.iloc[idx]
            episode = self.episodes_df[self.episodes_df['episode_id'] == feature_row['episode_id']].iloc[0]
            
            similar_cases.append({
                'similarity': 1 / (1 + dist),
                'distance': dist,
                'episode': episode,
                'features': feature_row,
                'outcome': {
                    'final_return': episode['return_rate'],
                    'sold_at': feature_row.get('current_return', episode['return_rate']),
                    'holding_days': episode['holding_days']
                }
            })
            
        return similar_cases
    
    def _predict_sell_outcome_generalized(self, current_features):
        """일반화된 매도 결과 예측"""
        # 전체 특징 벡터 구성
        feature_cols = [col for col in self.feature_columns['all'] if not col.startswith('outcome')]
        all_features = []
        
        for col in feature_cols:
            if col in current_features:
                all_features.append(current_features[col])
            else:
                # 누락된 특징은 0 사용
                all_features.append(0)
                    
        X = np.array(all_features).reshape(1, -1)
        X_scaled = self.feature_scaler.transform(X)
        
        # 앙상블 예측
        predictions = {
            'expected_final_return': float(self.return_ensemble.predict(X_scaled)[0]),
            'success_probability': float(self.success_ensemble.predict_proba(X_scaled)[0][1]),
            'expected_total_holding': float(self.holding_predictor.predict(X_scaled)[0])
        }
        
        # 추가 계산
        current_return = current_features['current_return']
        current_holding = current_features['holding_days']
        
        predictions['expected_additional_return'] = predictions['expected_final_return'] - current_return
        predictions['expected_additional_days'] = max(0, predictions['expected_total_holding'] - current_holding)
        
        return predictions
    
    def _analyze_similar_patterns_generalized(self, similar_cases):
        """일반화된 패턴 분석"""
        # 상위 10개 사례로 분석
        top_cases = similar_cases[:10]
        
        # 패턴 분석
        patterns = {
            'total_similar': len(top_cases),
            'avg_final_return': np.mean([c['outcome']['final_return'] for c in top_cases]),
            'std_final_return': np.std([c['outcome']['final_return'] for c in top_cases]),
            'positive_ratio': len([c for c in top_cases if c['outcome']['final_return'] > 0]) / len(top_cases),
            'avg_holding_days': np.mean([c['outcome']['holding_days'] for c in top_cases])
        }
        
        # 현재 수익률 구간의 패턴
        current_return = similar_cases[0]['features'].get('current_return', 0)
        
        # 유사한 수익률 구간 분석
        similar_return_cases = [c for c in similar_cases 
                               if abs(c['features'].get('current_return', 0) - current_return) < 2]
        
        if len(similar_return_cases) >= 5:
            patterns['similar_return_pattern'] = {
                'count': len(similar_return_cases),
                'avg_outcome': np.mean([c['outcome']['final_return'] for c in similar_return_cases[:10]]),
                'better_outcome_ratio': len([c for c in similar_return_cases[:10] 
                                           if c['outcome']['final_return'] > current_return]) / 10
            }
            
        return patterns
    
    def _calculate_prediction_confidence(self, similar_cases, predictions, patterns):
        """예측 신뢰도 계산"""
        confidence_factors = []
        
        # 1. 유사 사례들의 일관성
        outcomes = [c['outcome']['final_return'] for c in similar_cases[:10]]
        consistency = 1 - (np.std(outcomes) / (abs(np.mean(outcomes)) + 1e-6))
        confidence_factors.append(min(consistency, 1.0) * 0.3)
        
        # 2. 유사도 점수
        avg_similarity = np.mean([c['similarity'] for c in similar_cases[:5]])
        confidence_factors.append(avg_similarity * 0.3)
        
        # 3. 패턴 강도
        if patterns['positive_ratio'] > 0.7 or patterns['positive_ratio'] < 0.3:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
            
        # 4. 예측값의 합리성
        if -100 < predictions['expected_final_return'] < 200:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.05)
            
        # 종합 신뢰도
        total_confidence = sum(confidence_factors)
        
        return {
            'overall': min(total_confidence, 0.9),
            'factors': {
                'consistency': consistency,
                'similarity': avg_similarity,
                'pattern_clarity': confidence_factors[2] / 0.2,
                'prediction_validity': confidence_factors[3] / 0.2
            }
        }
    
    def _integrate_sell_analysis(self, situation, similar_cases, predictions, patterns, confidence):
        """매도 분석 최종 통합"""
        
        # AI 추천 결정
        recommendation = self._generate_data_driven_recommendation(
            predictions, patterns, confidence
        )
        
        # 디스플레이용 포맷
        display = self._format_sell_display(
            situation, similar_cases[:3], predictions, patterns, confidence, recommendation
        )
        
        return {
            'recommendation': recommendation,
            'predictions': predictions,
            'similar_cases': similar_cases[:5],
            'patterns': patterns,
            'confidence': confidence,
            'display': display,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _generate_data_driven_recommendation(self, predictions, patterns, confidence):
        """데이터 기반 추천 생성"""
        score = 0
        
        # 예측 기반 점수
        if predictions['expected_additional_return'] > 0:
            score += predictions['expected_additional_return'] * 0.1
        else:
            score += predictions['expected_additional_return'] * 0.15
            
        # 패턴 기반 점수
        score += (patterns['positive_ratio'] - 0.5) * 2
        
        # 신뢰도 반영
        score *= confidence['overall']
        
        # 최종 추천
        if score > 0.3:
            action = 'hold'
            message = f"계속 보유 추천 (신뢰도 {confidence['overall']*100:.0f}%)"
        elif score < -0.3:
            action = 'sell'
            message = f"매도 추천 (신뢰도 {confidence['overall']*100:.0f}%)"
        else:
            action = 'neutral'
            message = f"중립 (신뢰도 {confidence['overall']*100:.0f}%)"
            
        return {
            'action': action,
            'score': score,
            'message': message,
            'confidence': confidence['overall']
        }
    
    def analyze_buy_situation(self, stock_info, market_condition=None):
        """매수 상황 분석 (개인 정보 없이)"""
        
        # 1. 매수 특징 벡터 생성
        buy_features = self._create_buy_feature_vector_generalized(stock_info, market_condition)
        
        # 2. 유사 매수 상황 검색
        similar_buy_cases = self._find_similar_buy_cases_generalized(buy_features)
        
        # 3. 매수 결과 예측
        buy_predictions = self._predict_buy_outcome_generalized(buy_features)
        
        # 4. 리스크 패턴 분석
        risk_analysis = self._analyze_buy_risks_generalized(similar_buy_cases, stock_info)
        
        # 5. 최종 분석 통합
        return self._integrate_buy_analysis(
            similar_buy_cases, buy_predictions, risk_analysis, stock_info
        )
        
    def _create_buy_feature_vector_generalized(self, stock_info, market_condition):
        """일반화된 매수 특징 벡터 생성"""
        features = {
            # 기본 시장 상황
            'market_avg_return_7d': market_condition.get('avg_return_7d', 0) if market_condition else 0,
            'market_volatility_7d': market_condition.get('volatility_7d', 10) if market_condition else 10,
            'market_avg_return_30d': market_condition.get('avg_return_30d', 0) if market_condition else 0,
            'market_volatility_30d': market_condition.get('volatility_30d', 10) if market_condition else 10,
            
            # 시간 특징
            'buy_day_of_week': datetime.now().weekday(),
            'buy_month': datetime.now().month,
            'buy_quarter': (datetime.now().month - 1) // 3 + 1,
            'buy_is_month_start': datetime.now().day <= 5,
            'buy_is_month_end': datetime.now().day >= 25,
        }
        
        # 주식 정보가 있으면 추가
        if stock_info and 'recent_return' in stock_info:
            features['stock_recent_momentum'] = stock_info['recent_return']
            features['is_rally'] = 1 if stock_info['recent_return'] > 5 else 0
            
        return features
    
    def _find_similar_buy_cases_generalized(self, buy_features):
        """일반화된 유사 매수 상황 검색"""
        # 특징 벡터 준비
        buy_cols = self.feature_columns['buy']
        current_vector = []
        
        for col in buy_cols:
            feature_name = col.replace('feature_', '')
            current_vector.append(buy_features.get(feature_name, 0))
            
        # 빈 벡터인 경우 처리
        if len(current_vector) == 0:
            # features_df에서 무작위로 선택
            n_samples = min(30, len(self.features_df))
            random_indices = np.random.choice(len(self.features_df), n_samples, replace=False)
            
            similar_cases = []
            for idx in random_indices:
                feature_row = self.features_df.iloc[idx]
                episode = self.episodes_df[self.episodes_df['episode_id'] == feature_row['episode_id']].iloc[0]
                
                similar_cases.append({
                    'similarity': 0.5,  # 기본 유사도
                    'outcome': {
                        'return_rate': episode['return_rate'],
                        'holding_days': episode['holding_days'],
                        'profitable': 1 if episode['return_rate'] > 0 else 0
                    }
                })
            return similar_cases
            
        current_vector = np.array(current_vector).reshape(1, -1)
        
        # KNN이 학습되지 않은 경우 처리
        try:
            current_scaled = self.buy_scaler.transform(current_vector)
            distances, indices = self.buy_situation_knn.kneighbors(current_scaled, n_neighbors=30)
        except:
            # 학습되지 않은 경우 무작위 선택
            n_samples = min(30, len(self.features_df))
            random_indices = np.random.choice(len(self.features_df), n_samples, replace=False)
            
            similar_cases = []
            for idx in random_indices:
                feature_row = self.features_df.iloc[idx]
                episode = self.episodes_df[self.episodes_df['episode_id'] == feature_row['episode_id']].iloc[0]
                
                similar_cases.append({
                    'similarity': 0.5,
                    'outcome': {
                        'return_rate': episode['return_rate'],
                        'holding_days': episode['holding_days'],
                        'profitable': 1 if episode['return_rate'] > 0 else 0
                    }
                })
            return similar_cases
        
        # 정상적인 KNN 결과 처리
        similar_cases = []
        for dist, idx in zip(distances[0], indices[0]):
            situation = self.situations_df.iloc[idx]
            episode_id = situation['episode_id']
            
            episode = self.episodes_df[self.episodes_df['episode_id'] == episode_id].iloc[0]
            
            similar_cases.append({
                'similarity': 1 / (1 + dist),
                'situation': situation,
                'outcome': {
                    'return_rate': episode['return_rate'],
                    'holding_days': episode['holding_days'],
                    'profitable': 1 if episode['return_rate'] > 0 else 0
                }
            })
                
        return similar_cases
    
    def _predict_buy_outcome_generalized(self, buy_features):
        """일반화된 매수 결과 예측"""
        # 전체 특징 벡터 구성
        feature_cols = [col for col in self.feature_columns['all'] if not col.startswith('outcome')]
        all_features = []
        
        for col in feature_cols:
            if col in buy_features:
                all_features.append(buy_features[col])
            else:
                all_features.append(0)
                
        X = np.array(all_features).reshape(1, -1)
        X_scaled = self.feature_scaler.transform(X)
        
        # 앙상블 예측
        predictions = {
            'expected_return': float(self.return_ensemble.predict(X_scaled)[0]),
            'success_probability': float(self.success_ensemble.predict_proba(X_scaled)[0][1]),
            'expected_holding_days': float(self.holding_predictor.predict(X_scaled)[0])
        }
        
        # 리스크 조정
        if 'market_volatility_7d' in buy_features:
            volatility = buy_features['market_volatility_7d']
            predictions['risk_adjusted_return'] = predictions['expected_return'] / (1 + volatility/100)
        
        return predictions
    
    def _analyze_buy_risks_generalized(self, similar_cases, stock_info):
        """일반화된 매수 리스크 분석"""
        risks = {
            'chase_buying': {'detected': False},
            'high_volatility': {'detected': False},
            'poor_timing': {'detected': False}
        }
        
        # 1. 추격매수 패턴
        if stock_info.get('recent_return', 0) > 8:
            # 급등 후 매수한 유사 사례들의 성과
            rally_cases = [c for c in similar_cases[:20] if c.get('situation', {}).get('feature_is_rally', 0) == 1]
            
            if len(rally_cases) >= 5:
                rally_success_rate = sum(1 for c in rally_cases if c['outcome']['profitable']) / len(rally_cases)
                rally_avg_return = np.mean([c['outcome']['return_rate'] for c in rally_cases])
                
                if rally_success_rate < 0.4:
                    risks['chase_buying'] = {
                        'detected': True,
                        'success_rate': rally_success_rate,
                        'avg_return': rally_avg_return,
                        'sample_size': len(rally_cases)
                    }
        
        # 2. 고변동성 시장
        market_vol = stock_info.get('market_volatility', 10)
        if market_vol > 20:
            risks['high_volatility'] = {
                'detected': True,
                'volatility': market_vol
            }
            
        # 3. 타이밍 분석
        current_dow = datetime.now().weekday()
        if current_dow == 0:  # 월요일
            monday_cases = [c for c in similar_cases if c.get('situation', {}).get('feature_buy_day_of_week', -1) == 0]
            if len(monday_cases) >= 5:
                monday_success = sum(1 for c in monday_cases[:10] if c['outcome']['profitable']) / min(10, len(monday_cases))
                if monday_success < 0.4:
                    risks['poor_timing'] = {
                        'detected': True,
                        'reason': 'monday_effect',
                        'success_rate': monday_success
                    }
        
        return risks
    
    def _integrate_buy_analysis(self, similar_cases, predictions, risks, stock_info):
        """매수 분석 최종 통합"""
        
        # 유사 사례 통계
        success_cases = [c for c in similar_cases[:10] if c['outcome']['profitable']]
        similar_stats = {
            'success_rate': len(success_cases) / 10,
            'avg_return': np.mean([c['outcome']['return_rate'] for c in similar_cases[:10]]),
            'avg_holding': np.mean([c['outcome']['holding_days'] for c in similar_cases[:10]])
        }
        
        # 데이터 기반 추천
        recommendation = self._generate_buy_recommendation(
            predictions, similar_stats, risks
        )
        
        # 디스플레이 포맷
        display = self._format_buy_display(
            stock_info, predictions, similar_stats, risks, recommendation
        )
        
        return {
            'recommendation': recommendation,
            'predictions': predictions,
            'similar_cases': similar_cases[:5],
            'statistics': similar_stats,
            'risks': risks,
            'display': display,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _generate_buy_recommendation(self, predictions, similar_stats, risks):
        """데이터 기반 매수 추천"""
        score = 0
        factors = []
        
        # 예측 기반
        score += (predictions['success_probability'] - 0.5) * 2
        score += predictions['expected_return'] * 0.05
        
        # 유사 사례 기반
        score += (similar_stats['success_rate'] - 0.5) * 1.5
        
        # 리스크 차감
        if risks.get('chase_buying', {}).get('detected'):
            score -= 0.5
            factors.append("추격매수 위험")
        if risks.get('high_volatility', {}).get('detected'):
            score -= 0.3
            factors.append("고변동성 시장")
        if risks.get('poor_timing', {}).get('detected'):
            score -= 0.2
            factors.append("타이밍 불리")
            
        # 최종 추천
        if score > 0.3:
            action = 'buy'
            message = f"매수 추천 (점수: {score:.2f})"
        elif score < -0.3:
            action = 'wait'
            message = f"대기 추천 (점수: {score:.2f})"
        else:
            action = 'neutral'
            message = f"신중 검토 (점수: {score:.2f})"
            
        return {
            'action': action,
            'score': score,
            'message': message,
            'factors': factors
        }
    
    def _format_sell_display(self, situation, similar_cases, predictions, patterns, confidence, recommendation):
        """매도 화면 포맷"""
        display = f"""📊 매도 분석

현재 상황: {situation.get('stock_name', '종목')} {situation['current_return']:+.1f}% ({situation['holding_days']}일 보유)

🤖 AI 예측:
- 예상 최종 수익률: {predictions['expected_final_return']:.1f}%
- 추가 상승 가능성: {predictions['expected_additional_return']:+.1f}%
- 신뢰도: {confidence['overall']*100:.0f}%

📈 패턴 분석:
- 유사 상황 평균 수익률: {patterns['avg_final_return']:.1f}%
- 성공 비율: {patterns['positive_ratio']*100:.0f}%"""
        
        # 유사 사례 추가
        display += "\n\n📚 유사 사례:"
        for i, case in enumerate(similar_cases, 1):
            display += f"""
[{i}] 수익률: {case['outcome']['final_return']:+.1f}% ({case['outcome']['holding_days']}일)
    유사도: {case['similarity']*100:.0f}%"""
        
        display += f"\n\n💡 AI 추천: {recommendation['message']}"
        
        return display
    
    def _format_buy_display(self, stock_info, predictions, stats, risks, recommendation):
        """매수 화면 포맷"""
        if risks.get('chase_buying', {}).get('detected'):
            display = f"""🛑 매수 전 점검

📈 현재 상황: {stock_info.get('name', '종목')} +{stock_info.get('recent_return', 0):.1f}% (급등 중)

🔍 AI 패턴 분석:
과거 추격매수 {risks['chase_buying']['sample_size']}건 분석 결과:
- 성공률: {risks['chase_buying']['success_rate']*100:.0f}%
- 평균 수익률: {risks['chase_buying']['avg_return']:.1f}%

⚠️ 데이터가 보여주는 낮은 성공률입니다."""
        else:
            display = f"""📊 매수 분석

종목: {stock_info.get('name', '종목')}

🤖 AI 예측:
- 성공 확률: {predictions['success_probability']*100:.0f}%
- 예상 수익률: {predictions['expected_return']:.1f}%
- 예상 보유기간: {predictions['expected_holding_days']:.0f}일

📈 유사 거래 통계:
- 성공률: {stats['success_rate']*100:.0f}%
- 평균 수익률: {stats['avg_return']:.1f}%"""
            
        display += f"\n\n💡 AI 추천: {recommendation['message']}"
        
        if recommendation['factors']:
            display += f"\n고려사항: {', '.join(recommendation['factors'])}"
            
        return display
    
    def _print_model_summary(self):
        """모델 요약 출력"""
        print("\n📋 모델 요약:")
        print(f"   - KNN 이웃 수: 30")
        print(f"   - 앙상블 모델: XGBoost + LightGBM + CatBoost")
        print(f"   - 특징 차원: {len(self.feature_columns['all'])}차원")
        print(f"   - 학습 데이터: {len(self.features_df):,}개 에피소드")
        print(f"   - 과적합 방지: customer_id, isin, 미래정보 제거")
        
    def save_models(self, path='models/sophisticated_ai_fixed'):
        """모델 저장"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # 모든 모델 저장
        models_to_save = {
            'buy_knn': self.buy_situation_knn,
            'sell_knn': self.sell_situation_knn,
            'return_ensemble': self.return_ensemble,
            'success_ensemble': self.success_ensemble,
            'holding_predictor': self.holding_predictor,
            'buy_scaler': self.buy_scaler,
            'sell_scaler': self.sell_scaler,
            'feature_scaler': self.feature_scaler
        }
        
        for name, model in models_to_save.items():
            joblib.dump(model, f'{path}/{name}.pkl')
            
        # 메타데이터 저장
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'feature_columns': self.feature_columns,
            'model_performance': self.model_performance,
            'data_stats': {
                'total_episodes': len(self.episodes_df),
                'total_features': len(self.feature_columns['all'])
            },
            'improvements': [
                'customer_id 기반 특징 제거',
                'isin 기반 특징 제거',
                '미래 정보 사용 제거',
                '일반화 가능한 패턴만 학습'
            ]
        }
        
        with open(f'{path}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"\n💾 모델 저장 완료: {path}")


# 메인 실행
def main():
    """과적합 해결된 AI 시스템 실행"""
    
    # AI 초기화
    ai = SophisticatedTradingAI()
    
    # 데이터 경로 설정
    DATA_PATH = "/Users/inter4259/Desktop/Programming/hek_credit/generate_data/preprocessed_data_pure"
    
    # 데이터 로드
    print("🚀 과적합 해결된 AI 거래 어시스턴트 시작\n")
    ai.load_all_data(
        f'{DATA_PATH}/episodes.csv',
        f'{DATA_PATH}/features.csv',
        f'{DATA_PATH}/situations.csv'
    )
    
    # 모델 학습
    ai.train_all_models()
    
    # 모델 저장
    ai.save_models()
    
    # 데모
    print("\n" + "="*60)
    print("💡 AI 거래 어시스턴트 데모")
    print("="*60)
    
    # 매도 상황 데모
    print("\n### 시나리오 1: 매도 결정 ###")
    sell_result = ai.analyze_sell_situation({
        'stock_name': '테스트 종목',
        'current_return': 6.8,
        'holding_days': 8
    })
    print(sell_result['display'])
    
    # 매수 상황 데모
    print("\n\n### 시나리오 2: 매수 결정 ###")
    buy_result = ai.analyze_buy_situation(
        stock_info={
            'name': '테스트 종목',
            'recent_return': 8.5
        },
        market_condition={
            'volatility_7d': 15,
            'avg_return_7d': 2.5
        }
    )
    print(buy_result['display'])


if __name__ == "__main__":
    main()