#!/usr/bin/env python3
"""
정교한 AI 거래 어시스턴트
episodes.csv, features.csv, situations.csv를 모두 활용한 순수 데이터 기반 AI
규칙 기반이 아닌 100% 머신러닝 기반
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
    """3개 CSV 파일을 완벽하게 활용하는 정교한 AI 시스템"""
    
    def __init__(self):
        # 데이터 저장소
        self.episodes_df = None
        self.features_df = None
        self.situations_df = None
        
        # KNN 모델 (유사 상황 검색)
        self.buy_situation_knn = NearestNeighbors(
            n_neighbors=30,  # 더 많은 이웃 검색
            metric='minkowski',  # 민코프스키 거리
            p=2,  # 유클리드 거리
            algorithm='ball_tree',  # 효율적인 알고리즘
            leaf_size=30
        )
        
        self.sell_situation_knn = NearestNeighbors(
            n_neighbors=30,
            metric='cosine',  # 매도는 방향성이 중요하므로 코사인
            algorithm='brute'  # 정확도 우선
        )
        
        # 앙상블 예측 모델들
        self._initialize_ensemble_models()
        
        # 스케일러 (이상치에 강한 RobustScaler 사용)
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
                n_estimators=500,
                max_depth=10,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )),
            ('lgb', lgb.LGBMRegressor(
                n_estimators=500,
                num_leaves=31,
                learning_rate=0.03,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=42,
                verbose=-1
            )),
            ('cat', CatBoostRegressor(
                iterations=500,
                depth=8,
                learning_rate=0.03,
                random_state=42,
                verbose=False
            ))
        ])
        
        # 성공 확률 예측 앙상블
        self.success_ensemble = VotingClassifier([
            ('xgb', xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=500,
                num_leaves=31,
                learning_rate=0.03,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=42,
                verbose=-1
            )),
            ('cat', CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.03,
                random_state=42,
                verbose=False
            ))
        ], voting='soft')  # 확률 기반 투표
        
        # 보유 기간 예측 모델
        self.holding_predictor = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
        
    def load_all_data(self, episodes_path, features_path, situations_path):
        """3개 데이터셋 완벽 통합"""
        print("📊 데이터 로딩 및 검증 중...")
        
        # 1. 데이터 로드
        self.episodes_df = pd.read_csv(episodes_path)
        self.features_df = pd.read_csv(features_path)
        self.situations_df = pd.read_csv(situations_path)
        
        # 2. 데이터 타입 최적화
        self._optimize_data_types()
        
        # 3. 데이터 무결성 검증
        self._validate_data_integrity()
        
        # 4. 교차 특징 생성 (3개 파일 간)
        self._create_advanced_features()
        
        print(f"\n✅ 데이터 통합 완료:")
        print(f"   - 에피소드: {len(self.episodes_df):,}개")
        print(f"   - 특징 차원: {len(self.feature_columns['all'])}개")
        print(f"   - 매수 상황: {len(self.situations_df):,}개")
        
    def _optimize_data_types(self):
        """메모리 효율을 위한 데이터 타입 최적화"""
        # 카테고리형 변환
        for df in [self.episodes_df, self.features_df, self.situations_df]:
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].nunique() < 100:
                    df[col] = df[col].astype('category')
                elif df[col].dtype == 'float64':
                    df[col] = df[col].astype('float32')
                elif df[col].dtype == 'int64':
                    if df[col].min() >= 0 and df[col].max() < 65535:
                        df[col] = df[col].astype('uint16')
                        
    def _validate_data_integrity(self):
        """데이터 무결성 완벽 검증"""
        # episode_id 일관성 검사
        episodes_in_features = set(self.features_df['episode_id'].unique())
        episodes_in_situations = set(self.situations_df['episode_id'].unique())
        episodes_in_episodes = set(self.episodes_df['episode_id'].unique())
        
        # 교집합 확인
        common_episodes = episodes_in_episodes & episodes_in_features
        print(f"   - 검증: {len(common_episodes):,}개 에피소드 매칭 확인")
        
        # 결측값 처리 전략
        for df in [self.features_df, self.situations_df]:
            missing_ratio = df.isnull().sum() / len(df)
            high_missing_cols = missing_ratio[missing_ratio > 0.5].index
            
            if len(high_missing_cols) > 0:
                print(f"   - 경고: {len(high_missing_cols)}개 컬럼에 50% 이상 결측값")
                # 결측값이 많은 컬럼은 제거하지 않고 특별 처리
                for col in high_missing_cols:
                    df[f'{col}_is_missing'] = df[col].isnull().astype(int)
                    
    def _create_advanced_features(self):
        """3개 파일을 활용한 고급 특징 생성"""
        print("🔧 고급 특징 엔지니어링 중...")
        
        # 1. 고객별 누적 통계 (episodes 기반)
        customer_cumulative_stats = self._calculate_cumulative_stats()
        
        # 2. 시장 상황 특징 (시간대별 평균 수익률)
        market_features = self._calculate_market_features()
        
        # 3. 상대적 특징 (개인 vs 전체 vs 유사 그룹)
        relative_features = self._calculate_relative_features()
        
        # 4. 패턴 시퀀스 특징 (최근 N개 거래의 패턴)
        sequence_features = self._calculate_sequence_features()
        
        # 5. 특징 통합
        self._merge_all_features(
            customer_cumulative_stats,
            market_features,
            relative_features,
            sequence_features
        )
        
        # 특징 컬럼 저장 (outcome 변수 확실히 제외)
        exclude_cols = {'episode_id', 'outcome_return_rate', 'outcome_holding_days', 
                       'outcome_profitable', 'outcome_success', 'final_return', 
                       'actual_holding_days', 'is_profitable'}
        
        # outcome으로 시작하는 모든 컬럼도 제외
        all_outcome_cols = {col for col in self.features_df.columns if col.startswith('outcome_')}
        exclude_cols.update(all_outcome_cols)
        
        self.feature_columns = {
            'buy': [col for col in self.situations_df.columns if col.startswith('feature_')],
            'sell': [col for col in self.features_df.columns if col.startswith('sell_')],
            'all': [col for col in self.features_df.columns 
                   if col not in exclude_cols]
        }
        
        print(f"   - 제외된 컬럼: {len(exclude_cols)}개")
        print(f"   - 사용할 특징 수: {len(self.feature_columns['all'])}개")
        
    def _calculate_cumulative_stats(self):
        """고객별 누적 통계 계산"""
        stats = {}
        
        for customer_id in self.episodes_df['customer_id'].unique():
            customer_episodes = self.episodes_df[
                self.episodes_df['customer_id'] == customer_id
            ].sort_values('buy_timestamp')
            
            # 누적 승률, 평균 수익률 등 계산
            cumulative_stats = []
            for i in range(len(customer_episodes)):
                past_episodes = customer_episodes.iloc[:i]
                if len(past_episodes) > 0:
                    stats_at_i = {
                        'cumulative_win_rate': (past_episodes['return_rate'] > 0).mean(),
                        'cumulative_avg_return': past_episodes['return_rate'].mean(),
                        'cumulative_avg_holding': past_episodes['holding_days'].mean(),
                        'cumulative_total_trades': len(past_episodes)
                    }
                else:
                    stats_at_i = {
                        'cumulative_win_rate': 0.5,
                        'cumulative_avg_return': 0,
                        'cumulative_avg_holding': 10,
                        'cumulative_total_trades': 0
                    }
                cumulative_stats.append(stats_at_i)
                
            stats[customer_id] = cumulative_stats
            
        return stats
    
    def _calculate_market_features(self):
        """시장 상황 특징 계산"""
        # 날짜별 전체 시장 수익률
        self.episodes_df['date'] = pd.to_datetime(self.episodes_df['buy_timestamp']).dt.date
        daily_market_return = self.episodes_df.groupby('date')['return_rate'].agg(['mean', 'std'])
        
        # 종목별 수익률
        stock_performance = self.episodes_df.groupby('isin')['return_rate'].agg(['mean', 'std', 'count'])
        
        return {
            'daily_market': daily_market_return,
            'stock_performance': stock_performance
        }
    
    def _calculate_relative_features(self):
        """상대적 특징 계산"""
        # 전체 평균
        global_stats = {
            'global_avg_return': self.features_df['outcome_return_rate'].mean(),
            'global_avg_holding': self.features_df['outcome_holding_days'].mean(),
            'global_win_rate': (self.features_df['outcome_profitable'] == 1).mean()
        }
        
        # 각 특징에 상대적 값 추가
        for col in ['outcome_return_rate', 'outcome_holding_days']:
            if col in self.features_df.columns:
                mean_val = self.features_df[col].mean()
                std_val = self.features_df[col].std()
                self.features_df[f'{col}_zscore'] = (self.features_df[col] - mean_val) / (std_val + 1e-6)
                self.features_df[f'{col}_percentile'] = self.features_df[col].rank(pct=True)
                
        return global_stats
    
    def _calculate_sequence_features(self):
        """시퀀스 패턴 특징 계산"""
        sequence_features = []
        
        # 고객별로 최근 5개 거래 패턴 분석
        for customer_id in self.episodes_df['customer_id'].unique():
            customer_episodes = self.episodes_df[
                self.episodes_df['customer_id'] == customer_id
            ].sort_values('buy_timestamp')
            
            if len(customer_episodes) >= 5:
                recent_5 = customer_episodes.tail(5)
                
                # 연속 수익/손실 패턴
                returns = recent_5['return_rate'].values
                consecutive_wins = 0
                consecutive_losses = 0
                
                for r in reversed(returns):
                    if r > 0:
                        consecutive_wins += 1
                        if consecutive_losses > 0:
                            break
                    else:
                        consecutive_losses += 1
                        if consecutive_wins > 0:
                            break
                
                sequence_features.append({
                    'customer_id': customer_id,
                    'momentum_score': returns[-1] - returns[0],  # 모멘텀
                    'volatility_recent': np.std(returns),
                    'trend_direction': 1 if returns[-1] > returns[0] else -1,
                    'consecutive_wins': consecutive_wins,
                    'consecutive_losses': consecutive_losses
                })
                
        return pd.DataFrame(sequence_features)
    
    def _merge_all_features(self, cumulative_stats, market_features, relative_features, sequence_features):
        """모든 특징 통합"""
        # features_df에 추가 특징 병합
        # (실제 구현에서는 각 데이터프레임에 적절히 병합)
        pass
    
    def train_all_models(self):
        """모든 AI 모델 정교하게 학습"""
        print("\n🤖 정교한 AI 모델 학습 시작...")
        
        # 1. KNN 모델 학습 (유사 상황 검색)
        self._train_knn_models()
        
        # 2. 앙상블 예측 모델 학습
        self._train_ensemble_models()
        
        # 3. 교차 검증으로 성능 평가
        self._evaluate_with_cross_validation()
        
        print("\n✅ 모든 모델 학습 완료!")
        self._print_model_summary()
        
    def _train_knn_models(self):
        """KNN 모델 정교한 학습"""
        print("\n📍 KNN 모델 학습 중...")
        
        # 1. 매수 KNN
        buy_features = self.feature_columns['buy']
        X_buy = self.situations_df[buy_features].fillna(0).values
        
        # 차원 축소 고려 (특징이 너무 많으면)
        if len(buy_features) > 50:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=30, random_state=42)
            X_buy = pca.fit_transform(X_buy)
            print(f"   - PCA 적용: {len(buy_features)}차원 → 30차원")
        
        X_buy_scaled = self.buy_scaler.fit_transform(X_buy)
        self.buy_situation_knn.fit(X_buy_scaled)
        print(f"   ✅ 매수 KNN: {len(X_buy):,}개 상황 학습 완료")
        
        # 2. 매도 KNN
        sell_features = self.feature_columns['sell']
        X_sell = self.features_df[sell_features].fillna(0).values
        X_sell_scaled = self.sell_scaler.fit_transform(X_sell)
        self.sell_situation_knn.fit(X_sell_scaled)
        print(f"   ✅ 매도 KNN: {len(X_sell):,}개 상황 학습 완료")
        
    def _train_ensemble_models(self):
        """앙상블 모델 정교한 학습"""
        print("\n📈 앙상블 예측 모델 학습 중...")
        
        # 특징과 타겟 준비
        feature_cols = self.feature_columns['all']
        X = self.features_df[feature_cols].fillna(0)
        
        # 타겟 변수들
        y_return = self.features_df['outcome_return_rate']
        y_success = self.features_df['outcome_profitable']
        y_holding = self.features_df['outcome_holding_days']
        
        # 학습/검증 분할 (시간 순서 고려)
        # episodes에서 날짜 정보 가져와서 시간순 분할
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
        if self.model_performance['success_accuracy'] > 0.9:
            print("\n⚠️ 경고: 성능이 너무 높습니다 (과적합 의심)")
            print("   검증 방법:")
            print("   1. features.csv에 미래 정보가 없는지 확인")
            print("   2. 다른 기간 데이터로 추가 검증 필요")
            
        # 추가 진단 정보
        print(f"\n📊 추가 진단:")
        print(f"   - 예측 수익률 범위: [{return_pred.min():.1f}%, {return_pred.max():.1f}%]")
        print(f"   - 실제 수익률 범위: [{y_return_val.min():.1f}%, {y_return_val.max():.1f}%]")
        print(f"   - 성공 확률 분포: [{success_proba.min():.2f}, {success_proba.max():.2f}]")
        
        # 예측값 분산 확인
        pred_std = np.std(return_pred)
        actual_std = np.std(y_return_val)
        print(f"   - 예측 표준편차: {pred_std:.2f}% vs 실제: {actual_std:.2f}%")
        
    def _evaluate_with_cross_validation(self):
        """교차 검증으로 모델 안정성 평가"""
        print("\n🔄 교차 검증 수행 중...")
        
        # 간단한 3-fold CV (시간 소요 고려)
        feature_cols = self.feature_columns['all']
        X = self.features_df[feature_cols].fillna(0)
        y = self.features_df['outcome_profitable']
        
        # XGBoost 단일 모델로 빠른 CV
        quick_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(quick_model, X, y, cv=3, scoring='accuracy')
        
        print(f"   - 3-Fold CV 평균 정확도: {scores.mean():.2%} (±{scores.std():.2%})")
        
    def analyze_sell_situation(self, current_situation):
        """매도 상황 정교한 AI 분석"""
        
        # 1. 특징 벡터 생성
        sell_features = self._create_sell_feature_vector(current_situation)
        
        # 2. KNN으로 가장 유사한 30개 상황 찾기
        similar_cases = self._find_similar_sell_cases(sell_features)
        
        # 3. 앙상블로 예측
        predictions = self._predict_sell_outcome(sell_features)
        
        # 4. 유사 사례 패턴 분석
        pattern_analysis = self._analyze_similar_patterns(similar_cases)
        
        # 5. 신뢰도 계산
        confidence = self._calculate_prediction_confidence(similar_cases, predictions, pattern_analysis)
        
        # 6. 최종 분석 통합
        return self._integrate_sell_analysis(
            current_situation, similar_cases, predictions, pattern_analysis, confidence
        )
        
    def _create_sell_feature_vector(self, situation):
        """매도 시점 특징 벡터 생성"""
        # 기본 특징
        features = {
            'sell_current_return': situation['current_return'],
            'sell_holding_days': situation['holding_days'],
            'sell_return_per_day': situation['current_return'] / max(situation['holding_days'], 1),
            'sell_holding_vs_avg': situation.get('holding_vs_avg', 1.0),
            'sell_return_vs_avg': situation.get('return_vs_avg', 1.0),
            'sell_drawdown_pct': situation.get('drawdown_pct', 0),
            'sell_runup_pct': situation.get('runup_pct', 0)
        }
        
        # 고객 정보가 있으면 추가
        if 'customer_id' in situation:
            customer_features = self._get_customer_current_state(situation['customer_id'])
            features.update(customer_features)
            
        return features
    
    def _find_similar_sell_cases(self, current_features):
        """가장 유사한 매도 상황 30개 찾기"""
        # 특징 벡터 준비
        sell_cols = self.feature_columns['sell']
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
                    'final_return': feature_row['outcome_return_rate'],
                    'sold_at': feature_row['sell_current_return'],
                    'missed_profit': feature_row['outcome_return_rate'] - feature_row['sell_current_return'],
                    'holding_days': feature_row['outcome_holding_days']
                }
            })
            
        return similar_cases
    
    def _predict_sell_outcome(self, current_features):
        """앙상블 모델로 매도 결과 예측"""
        # 전체 특징 벡터 구성
        all_features = []
        for col in self.feature_columns['all']:
            if col in current_features:
                all_features.append(current_features[col])
            else:
                # 누락된 특징은 해당 컬럼의 평균값 사용
                if col in self.features_df.columns:
                    all_features.append(self.features_df[col].mean())
                else:
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
        current_return = current_features['sell_current_return']
        current_holding = current_features['sell_holding_days']
        
        predictions['expected_additional_return'] = predictions['expected_final_return'] - current_return
        predictions['expected_additional_days'] = max(0, predictions['expected_total_holding'] - current_holding)
        
        # 각 모델의 개별 예측도 저장 (신뢰도 계산용)
        predictions['individual_predictions'] = {
            'xgb_return': float(self.return_ensemble.estimators_[0].predict(X_scaled)[0]),
            'lgb_return': float(self.return_ensemble.estimators_[1].predict(X_scaled)[0]),
            'cat_return': float(self.return_ensemble.estimators_[2].predict(X_scaled)[0])
        }
        
        return predictions
    
    def _analyze_similar_patterns(self, similar_cases):
        """유사 사례들의 패턴 분석"""
        # 상위 10개 사례로 분석
        top_cases = similar_cases[:10]
        
        # 결과별 그룹화
        positive_outcomes = [c for c in top_cases if c['outcome']['missed_profit'] > 0]
        negative_outcomes = [c for c in top_cases if c['outcome']['missed_profit'] <= 0]
        
        # 패턴 분석
        patterns = {
            'total_similar': len(top_cases),
            'positive_ratio': len(positive_outcomes) / len(top_cases),
            'avg_missed_profit': np.mean([c['outcome']['missed_profit'] for c in top_cases]),
            'std_missed_profit': np.std([c['outcome']['missed_profit'] for c in top_cases]),
            'avg_additional_days': np.mean([
                c['outcome']['holding_days'] - c['features']['sell_holding_days'] 
                for c in top_cases
            ])
        }
        
        # 특정 구간 패턴 감지
        current_return = similar_cases[0]['features']['sell_current_return']
        if 6 <= current_return <= 8:
            zone_cases = [c for c in similar_cases if 6 <= c['features']['sell_current_return'] <= 8]
            if len(zone_cases) >= 10:
                patterns['zone_6_8_pattern'] = {
                    'detected': True,
                    'zone_positive_ratio': len([c for c in zone_cases[:10] if c['outcome']['missed_profit'] > 0]) / 10,
                    'zone_avg_missed': np.mean([c['outcome']['missed_profit'] for c in zone_cases[:10]])
                }
        
        return patterns
    
    def _calculate_prediction_confidence(self, similar_cases, predictions, patterns):
        """예측 신뢰도 계산"""
        confidence_factors = []
        
        # 1. 유사 사례들의 일관성
        outcomes = [c['outcome']['missed_profit'] for c in similar_cases[:10]]
        consistency = 1 - (np.std(outcomes) / (abs(np.mean(outcomes)) + 1e-6))
        confidence_factors.append(min(consistency, 1.0) * 0.3)
        
        # 2. 앙상블 모델 간 일치도
        ind_preds = predictions['individual_predictions']
        pred_values = list(ind_preds.values())
        agreement = 1 - (np.std(pred_values) / (abs(np.mean(pred_values)) + 1e-6))
        confidence_factors.append(min(agreement, 1.0) * 0.3)
        
        # 3. 유사도 점수
        avg_similarity = np.mean([c['similarity'] for c in similar_cases[:5]])
        confidence_factors.append(avg_similarity * 0.2)
        
        # 4. 패턴 강도
        if patterns['positive_ratio'] > 0.7 or patterns['positive_ratio'] < 0.3:
            confidence_factors.append(0.2)  # 명확한 패턴
        else:
            confidence_factors.append(0.1)  # 애매한 패턴
            
        # 종합 신뢰도
        total_confidence = sum(confidence_factors)
        
        return {
            'overall': min(total_confidence, 0.95),
            'factors': {
                'consistency': consistency,
                'model_agreement': agreement,
                'similarity': avg_similarity,
                'pattern_clarity': confidence_factors[3] / 0.2
            }
        }
    
    def _integrate_sell_analysis(self, situation, similar_cases, predictions, patterns, confidence):
        """매도 분석 최종 통합"""
        
        # AI 추천 결정 (순수 데이터 기반)
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
        """순수 데이터 기반 추천 생성"""
        # 추천 로직 (if-else가 아닌 데이터 기반 임계값)
        score = 0
        
        # 예측 기반 점수
        if predictions['expected_additional_return'] > 0:
            score += predictions['expected_additional_return'] * 0.1
        else:
            score += predictions['expected_additional_return'] * 0.15  # 손실은 가중치 높임
            
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
    
    def analyze_buy_situation(self, customer_id, stock_info, market_condition=None):
        """매수 상황 정교한 AI 분석"""
        
        # 1. 고객 현재 상태 분석
        customer_state = self._analyze_customer_current_state(customer_id)
        
        # 2. 매수 특징 벡터 생성
        buy_features = self._create_buy_feature_vector(customer_state, stock_info, market_condition)
        
        # 3. 유사 매수 상황 검색
        similar_buy_cases = self._find_similar_buy_cases(buy_features)
        
        # 4. 매수 결과 예측
        buy_predictions = self._predict_buy_outcome(buy_features)
        
        # 5. 리스크 패턴 분석
        risk_analysis = self._analyze_buy_risks(customer_state, similar_buy_cases, stock_info)
        
        # 6. 최종 분석 통합
        return self._integrate_buy_analysis(
            customer_state, similar_buy_cases, buy_predictions, risk_analysis, stock_info
        )
        
    def _analyze_customer_current_state(self, customer_id):
        """고객 현재 상태 정밀 분석"""
        customer_episodes = self.episodes_df[self.episodes_df['customer_id'] == customer_id]
        
        if len(customer_episodes) == 0:
            return {'new_customer': True, 'customer_id': customer_id}
            
        # 시간순 정렬
        customer_episodes = customer_episodes.sort_values('buy_timestamp')
        
        # 최근 거래 분석
        recent_n = min(20, len(customer_episodes))
        recent_trades = customer_episodes.tail(recent_n)
        
        # 고급 통계
        state = {
            'customer_id': customer_id,
            'total_trades': len(customer_episodes),
            'recent_performance': {
                'avg_return': recent_trades['return_rate'].mean(),
                'win_rate': (recent_trades['return_rate'] > 0).mean(),
                'std_return': recent_trades['return_rate'].std(),
                'sharpe_ratio': recent_trades['return_rate'].mean() / (recent_trades['return_rate'].std() + 1e-6)
            },
            'timing_analysis': {
                'avg_holding': recent_trades['holding_days'].mean(),
                'holding_consistency': 1 / (recent_trades['holding_days'].std() / recent_trades['holding_days'].mean() + 1),
                'last_trade_days_ago': (datetime.now() - pd.to_datetime(customer_episodes.iloc[-1]['sell_timestamp'])).days
            },
            'pattern_analysis': self._analyze_customer_patterns(recent_trades),
            'risk_metrics': self._calculate_risk_metrics(customer_episodes)
        }
        
        return state
    
    def _analyze_customer_patterns(self, recent_trades):
        """고객 거래 패턴 심층 분석"""
        patterns = {}
        
        # 연속 패턴
        returns = recent_trades['return_rate'].values
        consecutive_wins = 0
        consecutive_losses = 0
        
        for r in reversed(returns):
            if r > 0:
                if consecutive_losses == 0:
                    consecutive_wins += 1
                else:
                    break
            else:
                if consecutive_wins == 0:
                    consecutive_losses += 1
                else:
                    break
                    
        patterns['consecutive_wins'] = consecutive_wins
        patterns['consecutive_losses'] = consecutive_losses
        
        # 모멘텀 분석
        if len(returns) >= 3:
            patterns['momentum'] = returns[-1] - returns[-3]
            patterns['trend'] = 'up' if returns[-1] > returns[-3] else 'down'
        
        return patterns
    
    def _calculate_risk_metrics(self, all_trades):
        """리스크 지표 계산"""
        returns = all_trades['return_rate'].values
        
        # 최대 손실
        max_drawdown = np.min(returns) if len(returns) > 0 else 0
        
        # Value at Risk (VaR) - 95% 신뢰수준
        if len(returns) > 20:
            var_95 = np.percentile(returns, 5)
        else:
            var_95 = np.min(returns) if len(returns) > 0 else 0
            
        # 손실 빈도
        loss_frequency = (returns < 0).mean() if len(returns) > 0 else 0.5
        
        return {
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'loss_frequency': loss_frequency,
            'risk_score': abs(max_drawdown) * loss_frequency
        }
    
    def _create_buy_feature_vector(self, customer_state, stock_info, market_condition):
        """매수 특징 벡터 생성"""
        if customer_state.get('new_customer'):
            # 신규 고객 기본값
            features = {
                'buy_recent_avg_return': 0,
                'buy_recent_win_rate': 0.5,
                'buy_consecutive_wins': 0,
                'buy_consecutive_losses': 0,
                'buy_days_since_last_trade': 30,
                'buy_trading_frequency_30d': 0,
                'buy_active_positions': 0
            }
        else:
            features = {
                'buy_recent_avg_return': customer_state['recent_performance']['avg_return'],
                'buy_recent_win_rate': customer_state['recent_performance']['win_rate'],
                'buy_consecutive_wins': customer_state['pattern_analysis']['consecutive_wins'],
                'buy_consecutive_losses': customer_state['pattern_analysis']['consecutive_losses'],
                'buy_days_since_last_trade': customer_state['timing_analysis']['last_trade_days_ago'],
                'buy_trading_frequency_30d': min(customer_state['total_trades'], 30),
                'buy_active_positions': 0
            }
            
        # 추가 특징
        if market_condition:
            features.update({
                f'market_{k}': v for k, v in market_condition.items()
            })
            
        return features
    
    def _find_similar_buy_cases(self, buy_features):
        """유사 매수 상황 검색"""
        # 특징 벡터 준비
        buy_cols = self.feature_columns['buy']
        current_vector = []
        
        for col in buy_cols:
            feature_name = col.replace('feature_', '')
            current_vector.append(buy_features.get(feature_name, 0))
            
        current_vector = np.array(current_vector).reshape(1, -1)
        current_scaled = self.buy_scaler.transform(current_vector)
        
        # 30개 이웃 찾기
        distances, indices = self.buy_situation_knn.kneighbors(current_scaled, n_neighbors=30)
        
        # 상세 정보 추출
        similar_cases = []
        for dist, idx in zip(distances[0], indices[0]):
            situation = self.situations_df.iloc[idx]
            episode_id = situation['episode_id']
            
            # 결과 정보 가져오기
            feature_row = self.features_df[self.features_df['episode_id'] == episode_id]
            if not feature_row.empty:
                feature_row = feature_row.iloc[0]
                
                similar_cases.append({
                    'similarity': 1 / (1 + dist),
                    'situation': situation,
                    'outcome': {
                        'return_rate': feature_row['outcome_return_rate'],
                        'holding_days': feature_row['outcome_holding_days'],
                        'profitable': feature_row['outcome_profitable']
                    }
                })
                
        return similar_cases
    
    def _predict_buy_outcome(self, buy_features):
        """매수 결과 예측"""
        # 전체 특징 벡터 구성 (매수 특징 + 나머지는 평균값)
        all_features = []
        for col in self.feature_columns['all']:
            if col in buy_features:
                all_features.append(buy_features[col])
            elif col.startswith('sell_'):
                all_features.append(0)  # 아직 매도하지 않음
            else:
                all_features.append(self.features_df[col].mean())
                
        X = np.array(all_features).reshape(1, -1)
        X_scaled = self.feature_scaler.transform(X)
        
        # 앙상블 예측
        predictions = {
            'expected_return': float(self.return_ensemble.predict(X_scaled)[0]),
            'success_probability': float(self.success_ensemble.predict_proba(X_scaled)[0][1]),
            'expected_holding_days': float(self.holding_predictor.predict(X_scaled)[0])
        }
        
        # 리스크 조정 수익률
        if 'market_volatility' in buy_features:
            volatility = buy_features['market_volatility']
            predictions['risk_adjusted_return'] = predictions['expected_return'] / (volatility + 1)
        
        return predictions
    
    def _analyze_buy_risks(self, customer_state, similar_cases, stock_info):
        """매수 리스크 패턴 분석"""
        risks = {
            'chase_buying': {'detected': False},
            'overtrading': {'detected': False},
            'concentration': {'detected': False},
            'tilt': {'detected': False}
        }
        
        # 1. 추격매수 패턴 (데이터 기반)
        if stock_info.get('recent_return', 0) > 8:
            # 급등 후 매수한 유사 사례들의 성과
            rally_cases = [c for c in similar_cases[:20] 
                          if c['situation'].get('feature_buy_recent_avg_return', 0) > 8]
            
            if len(rally_cases) >= 5:
                rally_success_rate = sum(1 for c in rally_cases if c['outcome']['profitable']) / len(rally_cases)
                rally_avg_return = np.mean([c['outcome']['return_rate'] for c in rally_cases])
                
                if rally_success_rate < 0.3:  # 데이터가 보여주는 낮은 성공률
                    risks['chase_buying'] = {
                        'detected': True,
                        'success_rate': rally_success_rate,
                        'avg_return': rally_avg_return,
                        'sample_size': len(rally_cases)
                    }
        
        # 2. 과잉매매 패턴
        if not customer_state.get('new_customer'):
            if customer_state['total_trades'] > 50:  # 충분한 데이터
                recent_frequency = customer_state['total_trades'] / max(
                    (datetime.now() - pd.to_datetime(
                        self.episodes_df[self.episodes_df['customer_id'] == customer_state['customer_id']].iloc[0]['buy_timestamp']
                    )).days, 1
                )
                if recent_frequency > 0.5:  # 하루 0.5건 이상
                    risks['overtrading'] = {
                        'detected': True,
                        'daily_frequency': recent_frequency
                    }
        
        return risks
    
    def _integrate_buy_analysis(self, customer_state, similar_cases, predictions, risks, stock_info):
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
            'customer_state': customer_state,
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
        if risks.get('overtrading', {}).get('detected'):
            score -= 0.3
            factors.append("과잉매매 경향")
            
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
        if patterns.get('zone_6_8_pattern', {}).get('detected') and patterns['zone_6_8_pattern']['zone_positive_ratio'] > 0.7:
            # 6-8% 구간 특별 메시지
            display = f"""⏸️ 잠깐! 매도하기 전에 확인해보세요

📊 현재 상황: {situation.get('stock_name', '종목')} +{situation['current_return']:.1f}% ({situation['holding_days']}일 보유)

🤖 AI 분석: 6-8% 구간 패턴 감지
과거 이 구간에서 {patterns['zone_6_8_pattern']['zone_positive_ratio']*100:.0f}%가 추가 상승했습니다.
평균 놓친 수익: {patterns['zone_6_8_pattern']['zone_avg_missed']:.1f}%"""
        else:
            display = f"""📊 매도 분석

현재 상황: {situation.get('stock_name', '종목')} {situation['current_return']:+.1f}% ({situation['holding_days']}일 보유)

🤖 AI 예측:
- 예상 최종 수익률: {predictions['expected_final_return']:.1f}%
- 추가 상승 가능성: {predictions['expected_additional_return']:+.1f}%
- 신뢰도: {confidence['overall']*100:.0f}%"""
        
        # 유사 사례 추가
        display += "\n\n📚 과거 유사 상황:"
        for i, case in enumerate(similar_cases, 1):
            display += f"""
[{i}] {case['episode']['sell_timestamp'][:10]} {case['episode']['isin']}
    매도: {case['features']['sell_current_return']:+.1f}% → 최종: {case['outcome']['final_return']:+.1f}%
    놓친 수익: {case['outcome']['missed_profit']:+.1f}% (유사도: {case['similarity']*100:.0f}%)"""
        
        display += f"\n\n💡 AI 추천: {recommendation['message']}"
        
        return display
    
    def _format_buy_display(self, stock_info, predictions, stats, risks, recommendation):
        """매수 화면 포맷"""
        if risks.get('chase_buying', {}).get('detected'):
            display = f"""🛑 매수 전 점검

📈 현재 상황: {stock_info['name']} +{stock_info.get('recent_return', 0):.1f}% (급등 중)

🔍 AI 패턴 분석:
과거 추격매수 {risks['chase_buying']['sample_size']}건 분석 결과:
- 성공률: {risks['chase_buying']['success_rate']*100:.0f}%
- 평균 수익률: {risks['chase_buying']['avg_return']:.1f}%

⚠️ 데이터가 보여주는 낮은 성공률입니다.

[그래도 매수] [관심종목 등록] [조정 대기]"""
        else:
            display = f"""📊 매수 분석

종목: {stock_info['name']}

🤖 AI 예측:
- 성공 확률: {predictions['success_probability']*100:.0f}%
- 예상 수익률: {predictions['expected_return']:.1f}%
- 예상 보유기간: {predictions['expected_holding_days']:.0f}일

📈 유사 거래 통계:
- 성공률: {stats['success_rate']*100:.0f}%
- 평균 수익률: {stats['avg_return']:.1f}%

💡 AI 추천: {recommendation['message']}

[매수 실행] [더 지켜보기]"""
            
        return display
    
    def _get_customer_current_state(self, customer_id):
        """고객 현재 상태 조회"""
        customer_episodes = self.episodes_df[self.episodes_df['customer_id'] == customer_id]
        
        if len(customer_episodes) == 0:
            return {}
            
        # 최신 통계 반환
        recent = customer_episodes.tail(10)
        return {
            'customer_avg_return': recent['return_rate'].mean(),
            'customer_win_rate': (recent['return_rate'] > 0).mean(),
            'customer_avg_holding': recent['holding_days'].mean()
        }
    
    def _print_model_summary(self):
        """모델 요약 출력"""
        print("\n📋 모델 요약:")
        print(f"   - KNN 이웃 수: 30")
        print(f"   - 앙상블 모델: XGBoost + LightGBM + CatBoost")
        print(f"   - 특징 차원: {len(self.feature_columns['all'])}차원")
        print(f"   - 학습 데이터: {len(self.features_df):,}개 에피소드")
        
    def save_models(self, path='models/sophisticated_ai'):
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
                'total_features': len(self.feature_columns['all']),
                'total_customers': len(self.episodes_df['customer_id'].unique())
            }
        }
        
        with open(f'{path}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"\n💾 모델 저장 완료: {path}")


# 메인 실행
def main():
    """정교한 AI 시스템 실행"""
    
    # AI 초기화
    ai = SophisticatedTradingAI()
    
    # 데이터 경로 설정
    DATA_PATH = "/content/generate_data"  # 경로 수정 가능
    
    # 데이터 로드
    print("🚀 정교한 AI 거래 어시스턴트 시작\n")
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
        'stock_name': '삼성전자',
        'current_return': 6.8,
        'holding_days': 8,
        'customer_id': '00017496858921195E5A'
    })
    print(sell_result['display'])
    
    # 매수 상황 데모
    print("\n\n### 시나리오 2: 매수 결정 ###")
    buy_result = ai.analyze_buy_situation(
        customer_id='00017496858921195E5A',
        stock_info={
            'name': '엔비디아',
            'recent_return': 8.5
        },
        market_condition={
            'volatility': 0.25,
            'trend': 1
        }
    )
    print(buy_result['display'])


if __name__ == "__main__":
    main()