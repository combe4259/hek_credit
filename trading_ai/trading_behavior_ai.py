# trading_behavior_ai.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class TradingBehaviorAI:
    """실제 매매 행동을 학습하는 AI 모델"""
    
    def __init__(self):
        self.sell_threshold_model = None  # 매도 임계값 예측 (회귀)
        self.sell_timing_model = None     # 매도 타이밍 예측 (분류)
        self.panic_sell_model = None      # 패닉셀 예측 (분류)
        self.feature_names = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generate_realistic_trading_data(self, n_users=1000, n_trades_per_user=100):
        """실제 매매 패턴을 반영한 학습 데이터 생성"""
        print(f"📊 {n_users}명의 실제 매매 패턴 데이터 생성 중...")
        
        data = []
        np.random.seed(42)
        
        for user_id in range(n_users):
            # 사용자별 고유 매매 특성
            user_traits = {
                'profit_sell_threshold': np.random.normal(0.07, 0.03),  # 평균 7% 수익에서 매도
                'loss_cut_threshold': np.random.normal(-0.05, 0.02),    # 평균 -5% 손실에서 손절
                'panic_volatility': np.random.uniform(0.03, 0.10),      # 패닉 반응 변동성
                'morning_trade_tendency': np.random.uniform(0, 1),      # 장초반 거래 성향
                'closing_trade_tendency': np.random.uniform(0, 1),      # 장마감 거래 성향
                'hold_loss_tendency': np.random.uniform(0, 1),          # 손실 보유 성향
                'quick_profit_tendency': np.random.uniform(0, 1)        # 빠른 수익실현 성향
            }
            
            for trade_idx in range(n_trades_per_user):
                # 거래 시작 (매수)
                buy_hour = np.random.choice([9, 10, 11, 13, 14, 15], 
                                          p=[0.3, 0.2, 0.1, 0.1, 0.2, 0.1])
                buy_minute = np.random.randint(0, 60)
                
                # 시장 상황
                market_trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                daily_volatility = np.random.uniform(0.005, 0.05)
                intraday_volatility = np.random.uniform(0.001, 0.03)
                
                # 보유 기간 중 가격 변동
                holding_days = np.random.randint(1, 30)  # 실제 보유 일수 (1~30일)
                current_profit_rate = 0
                max_profit_rate = 0
                min_profit_rate = 0
                
                # 매도 여부와 시점 결정
                sold = False
                sell_reason = None
                actual_sell_profit = 0
                
                # 보유 기간 동안 가격 변동 시뮬레이션
                for day in range(1, holding_days + 1):
                    # 일별 수익률 변화
                    daily_change = np.random.normal(
                        market_trend * 0.002,  # 시장 트렌드 반영
                        daily_volatility
                    )
                    current_profit_rate += daily_change
                    max_profit_rate = max(max_profit_rate, current_profit_rate)
                    min_profit_rate = min(min_profit_rate, current_profit_rate)
                
                # 최종 수익률에서 매도 결정
                # 1. 수익 실현 (수익률이 임계값 이상)
                if current_profit_rate > user_traits['profit_sell_threshold']:
                    if np.random.random() < user_traits['quick_profit_tendency'] * 0.8:
                        sold = True
                        sell_reason = 'profit_taking'
                
                # 2. 손절 (손실이 임계값 이하)
                elif current_profit_rate < user_traits['loss_cut_threshold']:
                    if np.random.random() > user_traits['hold_loss_tendency'] * 0.7:
                        sold = True
                        sell_reason = 'stop_loss'
                
                # 3. 패닉셀 (큰 손실시)
                elif current_profit_rate < -0.1:
                    if np.random.random() < 0.2:
                        sold = True
                        sell_reason = 'panic_sell'
                
                # 4. 시간 기반 매도
                elif holding_days > 20:
                    if np.random.random() < 0.4:
                        sold = True
                        sell_reason = 'time_based'
                
                # 5. 아직 보유 중 (매도하지 않음)
                else:
                    sold = False
                    sell_reason = 'holding'
                
                actual_sell_profit = current_profit_rate
                
                # 특징 데이터 생성
                trade_data = {
                    'user_id': f'user_{user_id}',
                    
                    # 현재 상태
                    'current_profit_rate': round(current_profit_rate, 4),
                    'holding_days': holding_days,
                    'max_profit_during_hold': round(max_profit_rate, 4),
                    'min_profit_during_hold': round(min_profit_rate, 4),
                    'profit_drawdown': round(max_profit_rate - current_profit_rate, 4),
                    
                    # 시장 상황
                    'market_trend': market_trend,
                    'daily_volatility': round(daily_volatility, 4),
                    'intraday_volatility': round(intraday_volatility, 4),
                    
                    # 시간 특성
                    'buy_hour': buy_hour,
                    'is_morning_trade': 1 if buy_hour <= 10 else 0,
                    'is_closing_trade': 1 if buy_hour >= 15 else 0,
                    
                    # 사용자 특성
                    'user_avg_holding_days': np.random.randint(5, 30),
                    'user_win_rate': np.random.uniform(0.3, 0.7),
                    'user_avg_profit': np.random.uniform(-0.02, 0.05),
                    
                    # 타겟 변수들
                    'did_sell': 1 if sold else 0,
                    'sell_reason': sell_reason,
                    'actual_sell_profit': round(actual_sell_profit, 4),
                    'is_profit_taking': 1 if sell_reason == 'profit_taking' else 0,
                    'is_stop_loss': 1 if sell_reason == 'stop_loss' else 0,
                    'is_panic_sell': 1 if sell_reason == 'panic_sell' else 0,
                    
                    # 실제 매도 임계값 (회귀 타겟)
                    'actual_profit_threshold': user_traits['profit_sell_threshold'] if sold and current_profit_rate > 0 else None,
                    'actual_loss_threshold': user_traits['loss_cut_threshold'] if sold and current_profit_rate < 0 else None,
                }
                
                data.append(trade_data)
        
        df = pd.DataFrame(data)
        print(f"✅ 총 {len(df):,}개 실제 매매 데이터 생성 완료")
        print(f"📊 매도 비율: {df['did_sell'].mean():.1%}")
        print(f"📊 평균 보유일수: {df['holding_days'].mean():.1f}일")
        
        return df
    
    def create_advanced_features(self, df):
        """고급 특징 엔지니어링"""
        df = df.copy()
        
        # 1. 수익률 관련 파생 특징
        df['profit_to_max_ratio'] = df['current_profit_rate'] / (df['max_profit_during_hold'] + 0.0001)
        df['loss_to_min_ratio'] = df['current_profit_rate'] / (df['min_profit_during_hold'] - 0.0001)
        df['profit_momentum'] = df['current_profit_rate'] / (df['holding_days'] + 1)
        
        # 2. 리스크 지표
        df['volatility_to_profit_ratio'] = df['daily_volatility'] / (abs(df['current_profit_rate']) + 0.0001)
        df['panic_risk_score'] = df['intraday_volatility'] * df['daily_volatility'] * 100
        
        # 3. 시간 기반 특징
        df['holding_days_squared'] = df['holding_days'] ** 2
        df['is_long_term'] = (df['holding_days'] > 20).astype(int)
        df['is_short_term'] = (df['holding_days'] < 5).astype(int)
        
        # 4. 심리적 지표
        df['greed_score'] = np.where(df['current_profit_rate'] > 0, 
                                     df['current_profit_rate'] * df['holding_days'], 0)
        df['fear_score'] = np.where(df['current_profit_rate'] < 0, 
                                    abs(df['current_profit_rate']) * df['holding_days'], 0)
        
        return df
    
    def train_models(self, test_size=0.2):
        """여러 모델 동시 훈련"""
        print("🤖 매매 행동 예측 모델 훈련 시작...")
        
        # 1. 데이터 생성 및 전처리
        df = self.generate_realistic_trading_data()
        df = self.create_advanced_features(df)
        
        # 2. 특징 선택
        feature_cols = [
            'current_profit_rate', 'holding_days', 'max_profit_during_hold',
            'min_profit_during_hold', 'profit_drawdown', 'market_trend',
            'daily_volatility', 'intraday_volatility', 'buy_hour',
            'is_morning_trade', 'is_closing_trade', 'user_avg_holding_days',
            'user_win_rate', 'user_avg_profit', 'profit_to_max_ratio',
            'loss_to_min_ratio', 'profit_momentum', 'volatility_to_profit_ratio',
            'panic_risk_score', 'holding_days_squared', 'is_long_term',
            'is_short_term', 'greed_score', 'fear_score'
        ]
        
        self.feature_names = feature_cols
        X = df[feature_cols]
        
        # 3. 타겟 변수들
        y_sell = df['did_sell']  # 매도 여부 (이진 분류)
        y_panic = df['is_panic_sell']  # 패닉셀 여부 (이진 분류)
        
        # 수익 실현 임계값 (회귀 - 수익인 경우만)
        profit_trades = df[df['current_profit_rate'] > 0]
        X_profit = profit_trades[feature_cols]
        y_profit_threshold = profit_trades['current_profit_rate']  # 실제 매도한 수익률
        
        # 4. 데이터 분할
        X_train, X_test, y_sell_train, y_sell_test = train_test_split(
            X, y_sell, test_size=test_size, random_state=42
        )
        
        # 5. 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 6. Model 1: 매도 타이밍 예측 (분류)
        print("⏳ 매도 타이밍 모델 훈련 중...")
        self.sell_timing_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_sell_train) / sum(y_sell_train),  # 불균형 처리
            random_state=42
        )
        self.sell_timing_model.fit(X_train_scaled, y_sell_train)
        
        # 성능 평가
        y_pred = self.sell_timing_model.predict(X_test_scaled)
        y_pred_proba = self.sell_timing_model.predict_proba(X_test_scaled)[:, 1]
        
        print(f"✅ 매도 타이밍 모델 AUC: {roc_auc_score(y_sell_test, y_pred_proba):.3f}")
        
        # 7. Model 2: 패닉셀 예측 (분류)
        print("⏳ 패닉셀 예측 모델 훈련 중...")
        _, _, y_panic_train, y_panic_test = train_test_split(
            X, y_panic, test_size=test_size, random_state=42
        )
        
        self.panic_sell_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
        self.panic_sell_model.fit(X_train_scaled, y_panic_train)
        
        # 8. Model 3: 수익 실현 임계값 예측 (회귀)
        if len(X_profit) > 100:
            print("⏳ 수익 실현 임계값 모델 훈련 중...")
            X_profit_train, X_profit_test, y_threshold_train, y_threshold_test = train_test_split(
                X_profit, y_profit_threshold, test_size=test_size, random_state=42
            )
            
            X_profit_scaled = self.scaler.transform(X_profit_train)
            
            self.sell_threshold_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            )
            self.sell_threshold_model.fit(X_profit_scaled, y_threshold_train)
        
        self.is_trained = True
        print("✅ 모든 모델 훈련 완료!")
        
        # 특징 중요도 출력
        self._print_feature_importance()
        
        return True
    
    def _print_feature_importance(self):
        """특징 중요도 출력"""
        if self.sell_timing_model:
            print("\n📊 매도 타이밍 예측 주요 특징 Top 5:")
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.sell_timing_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for _, row in importance_df.head().iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
    
    def predict_sell_decision(self, 
                            current_profit_rate: float,
                            holding_days: int,
                            max_profit_during_hold: float,
                            market_volatility: float = 0.02,
                            user_history: Optional[Dict] = None) -> Dict:
        """
        실시간 매도 의사결정 예측
        
        Returns:
            - sell_probability: 매도 확률 (0~1)
            - recommended_action: 추천 행동
            - risk_factors: 주요 리스크 요인
            - expected_profit_threshold: 예상 수익 실현 지점
        """
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 기본 사용자 이력 (없으면 평균값 사용)
        if user_history is None:
            user_history = {
                'user_avg_holding_days': 15,
                'user_win_rate': 0.5,
                'user_avg_profit': 0.02
            }
        
        # 특징 생성
        features = {
            'current_profit_rate': current_profit_rate,
            'holding_days': holding_days,
            'max_profit_during_hold': max_profit_during_hold,
            'min_profit_during_hold': min(0, current_profit_rate),
            'profit_drawdown': max_profit_during_hold - current_profit_rate,
            'market_trend': 0,  # 중립 가정
            'daily_volatility': market_volatility,
            'intraday_volatility': market_volatility * 0.5,
            'buy_hour': 10,  # 기본값
            'is_morning_trade': 0,
            'is_closing_trade': 0,
            'user_avg_holding_days': user_history['user_avg_holding_days'],
            'user_win_rate': user_history['user_win_rate'],
            'user_avg_profit': user_history['user_avg_profit']
        }
        
        # 데이터프레임 생성 및 고급 특징 추가
        df = pd.DataFrame([features])
        df = self.create_advanced_features(df)
        
        # 예측용 특징 추출
        X = df[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # 1. 매도 확률 예측
        sell_probability = self.sell_timing_model.predict_proba(X_scaled)[0, 1]
        
        # 2. 패닉셀 위험도
        panic_probability = self.panic_sell_model.predict_proba(X_scaled)[0, 1]
        
        # 3. 추천 행동 결정
        if sell_probability > 0.8:
            action = "즉시 매도 권장"
        elif sell_probability > 0.6:
            action = "매도 고려"
        elif sell_probability > 0.4:
            action = "신중한 관찰 필요"
        else:
            action = "보유 권장"
        
        # 4. 리스크 요인 분석
        risk_factors = []
        if panic_probability > 0.3:
            risk_factors.append("패닉셀 위험 높음")
        if df['profit_drawdown'].iloc[0] > 0.05:
            risk_factors.append("고점 대비 5% 이상 하락")
        if holding_days > 30:
            risk_factors.append("장기 보유 중")
        if current_profit_rate < -0.05:
            risk_factors.append("5% 이상 손실 중")
        
        # 5. 예상 수익 실현 지점
        expected_threshold = None
        if self.sell_threshold_model and current_profit_rate > 0:
            expected_threshold = self.sell_threshold_model.predict(X_scaled)[0]
        
        return {
            'sell_probability': round(float(sell_probability), 3),
            'panic_sell_risk': round(float(panic_probability), 3),
            'recommended_action': action,
            'risk_factors': risk_factors,
            'expected_profit_threshold': round(float(expected_threshold), 3) if expected_threshold else None,
            'analysis': {
                'holding_period_impact': 'high' if holding_days > 20 else 'medium',
                'profit_status': 'profit' if current_profit_rate > 0 else 'loss',
                'volatility_level': 'high' if market_volatility > 0.03 else 'normal'
            }
        }
    
    def save_models(self, filepath_prefix='trading_behavior'):
        """모델 저장"""
        if not self.is_trained:
            raise ValueError("훈련된 모델이 없습니다.")
        
        models_data = {
            'sell_timing_model': self.sell_timing_model,
            'panic_sell_model': self.panic_sell_model,
            'sell_threshold_model': self.sell_threshold_model,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }
        
        with open(f'{filepath_prefix}_models.pkl', 'wb') as f:
            pickle.dump(models_data, f)
        
        print(f"💾 모델이 {filepath_prefix}_models.pkl에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    # AI 모델 생성 및 훈련
    ai = TradingBehaviorAI()
    ai.train_models()
    
    # 실전 예측 테스트
    print("\n🎯 실전 매도 결정 예측 테스트")
    
    # 시나리오 1: 5% 수익 중, 7일 보유
    result1 = ai.predict_sell_decision(
        current_profit_rate=0.05,
        holding_days=7,
        max_profit_during_hold=0.06,
        market_volatility=0.02
    )
    print(f"\n시나리오 1 (5% 수익, 7일 보유):")
    print(f"매도 확률: {result1['sell_probability']:.1%}")
    print(f"추천: {result1['recommended_action']}")
    
    # 시나리오 2: -3% 손실 중, 3일 보유
    result2 = ai.predict_sell_decision(
        current_profit_rate=-0.03,
        holding_days=3,
        max_profit_during_hold=0.01,
        market_volatility=0.04
    )
    print(f"\n시나리오 2 (-3% 손실, 3일 보유):")
    print(f"매도 확률: {result2['sell_probability']:.1%}")
    print(f"패닉셀 위험: {result2['panic_sell_risk']:.1%}")
    print(f"추천: {result2['recommended_action']}")