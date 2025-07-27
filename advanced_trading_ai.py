# advanced_trading_ai.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict

class AdvancedTradingAI:
    """실전 투자 환경을 위한 고급 매매 패턴 학습 AI"""
    
    def __init__(self):
        # 모델들
        self.sell_probability_model = None      # 매도 확률 예측
        self.profit_zone_model = None          # 수익률 구간별 행동 예측
        self.time_pattern_model = None         # 시간대별 패턴 예측
        self.loss_pattern_model = None         # 손실 패턴 감지
        
        # 데이터 처리
        self.scaler = StandardScaler()
        self.stock_encoder = LabelEncoder()
        self.sector_encoder = LabelEncoder()
        
        # 개인 매매 이력
        self.trading_history = []
        self.loss_patterns = []
        self.profit_patterns = defaultdict(list)  # 수익률 구간별 행동 기록
        
        # 설정
        self.is_trained = False
        self.feature_names = None
        
    def generate_realistic_market_data(self, n_users=500, n_stocks=50, n_days=365):
        """실제 시장 환경을 반영한 데이터 생성"""
        print("📊 실제 시장 환경 데이터 생성 중...")
        
        # 종목 정보 생성
        stocks = self._generate_stock_info(n_stocks)
        
        # 사용자별 거래 데이터 생성
        all_trades = []
        
        for user_id in range(n_users):
            # 사용자 고유 특성
            user_profile = {
                'user_id': f'user_{user_id}',
                'profit_targets': {  # 수익률 구간별 매도 확률
                    '0-5%': np.random.uniform(0.1, 0.4),
                    '5-10%': np.random.uniform(0.3, 0.7),
                    '10-20%': np.random.uniform(0.5, 0.9),
                    '20%+': np.random.uniform(0.8, 1.0)
                },
                'loss_thresholds': {  # 손실률별 손절 확률
                    '0--3%': np.random.uniform(0.1, 0.3),
                    '-3--5%': np.random.uniform(0.2, 0.5),
                    '-5--10%': np.random.uniform(0.4, 0.8),
                    '-10%+': np.random.uniform(0.7, 1.0)
                },
                'time_preferences': {  # 시간대별 거래 성향
                    'morning': np.random.uniform(0, 1),      # 09:00-10:00
                    'mid_morning': np.random.uniform(0, 1),  # 10:00-11:00
                    'lunch': np.random.uniform(0, 1),        # 11:00-13:00
                    'afternoon': np.random.uniform(0, 1),    # 13:00-14:00
                    'closing': np.random.uniform(0, 1)       # 14:00-15:30
                },
                'panic_threshold': np.random.uniform(0.03, 0.08),  # 패닉 반응 변동성
                'fomo_tendency': np.random.uniform(0, 1),          # FOMO 성향
                'loss_aversion': np.random.uniform(0.3, 0.9)       # 손실 회피 성향
            }
            
            # 사용자별 거래 생성
            user_trades = self._generate_user_trades(
                user_profile, stocks, n_days
            )
            all_trades.extend(user_trades)
        
        df = pd.DataFrame(all_trades)
        print(f"✅ 총 {len(df):,}개 거래 데이터 생성 완료")
        
        # 손실 패턴 사례 생성
        self._generate_loss_pattern_cases(df)
        
        return df
    
    def _generate_stock_info(self, n_stocks):
        """종목 정보 생성"""
        sectors = ['전자', '화학', '금융', '바이오', '자동차', '건설', '유통', '엔터']
        
        stocks = []
        for i in range(n_stocks):
            stock = {
                'ticker': f'STOCK_{i:03d}',
                'name': f'종목{i}',
                'sector': np.random.choice(sectors),
                'market_cap': np.random.choice(['대형주', '중형주', '소형주'], 
                                             p=[0.3, 0.4, 0.3]),
                'avg_volatility': np.random.uniform(0.01, 0.05),
                'beta': np.random.uniform(0.5, 1.5)
            }
            
            # 실제 종목명 예시 추가
            if i == 0:
                stock.update({'ticker': '005930', 'name': '삼성전자', 
                            'sector': '전자', 'market_cap': '대형주'})
            elif i == 1:
                stock.update({'ticker': '051910', 'name': 'LG화학', 
                            'sector': '화학', 'market_cap': '대형주'})
            
            stocks.append(stock)
        
        return stocks
    
    def _generate_user_trades(self, user_profile, stocks, n_days):
        """사용자별 거래 데이터 생성"""
        trades = []
        n_trades = np.random.randint(50, 200)  # 사용자별 거래 수
        
        for _ in range(n_trades):
            # 종목 선택
            stock = np.random.choice(stocks)
            
            # 거래 시작일
            buy_date = np.random.randint(0, max(1, n_days - 30))
            
            # 매수 시간 (사용자 선호도 반영)
            buy_hour, buy_minute = self._get_trading_time(user_profile['time_preferences'])
            
            # 시장 상황
            market_condition = np.random.choice(['상승장', '하락장', '횡보장'], 
                                              p=[0.3, 0.3, 0.4])
            
            # 보유 기간 및 수익률 시뮬레이션
            trade_result = self._simulate_trade(
                user_profile, stock, market_condition
            )
            
            # 거래 데이터 구성
            trade = {
                'user_id': user_profile['user_id'],
                'ticker': stock['ticker'],
                'stock_name': stock['name'],
                'sector': stock['sector'],
                'market_cap': stock['market_cap'],
                'buy_date': buy_date,
                'buy_hour': buy_hour,
                'buy_minute': buy_minute,
                'market_condition': market_condition,
                **trade_result
            }
            
            trades.append(trade)
        
        return trades
    
    def _get_trading_time(self, time_preferences):
        """시간대별 선호도에 따른 거래 시간 생성"""
        time_slots = [
            (9, 0, 10, 0, 'morning'),
            (10, 0, 11, 0, 'mid_morning'),
            (11, 0, 13, 0, 'lunch'),
            (13, 0, 14, 0, 'afternoon'),
            (14, 0, 15, 30, 'closing')
        ]
        
        # 선호도에 따른 가중치 적용
        weights = [time_preferences[slot[4]] for slot in time_slots]
        weights = np.array(weights) / sum(weights)
        
        # 시간대 선택
        chosen_slot = np.random.choice(len(time_slots), p=weights)
        start_h, start_m, end_h, end_m, _ = time_slots[chosen_slot]
        
        # 구체적 시간 생성
        total_minutes = (end_h - start_h) * 60 + (end_m - start_m)
        random_minutes = np.random.randint(0, total_minutes)
        
        hour = start_h + random_minutes // 60
        minute = start_m + random_minutes % 60
        
        return hour, minute
    
    def _simulate_trade(self, user_profile, stock, market_condition):
        """거래 시뮬레이션"""
        # 초기 설정
        holding_days = np.random.randint(1, 60)  # 실제 보유기간 (1~60일)
        current_profit = 0
        max_profit = 0
        min_profit = 0
        daily_profits = []
        
        # 매도 여부
        sold = False
        sell_reason = None
        sell_hour = None
        sell_minute = None
        
        # 보유 기간 동안 가격 변동 시뮬레이션
        for day in range(1, holding_days + 1):
            # 일별 수익률 변화
            daily_change = self._calculate_daily_return(
                stock, market_condition, day
            )
            current_profit += daily_change
            daily_profits.append(current_profit)
            
            max_profit = max(max_profit, current_profit)
            min_profit = min(min_profit, current_profit)
        
        # 최종일에 매도 결정 (한 번만)
        intraday_volatility = stock['avg_volatility'] * np.random.uniform(0.5, 2)
        
        sell_decision = self._decide_sell(
            user_profile, current_profit, holding_days, 
            max_profit, intraday_volatility
        )
        
        # 추가로 30% 확률로는 무조건 보유 중 (매도하지 않음)
        if np.random.random() < 0.3:
            sold = False
            sell_reason = 'holding'
        elif sell_decision['sell']:
            sold = True
            sell_reason = sell_decision['reason']
            sell_hour, sell_minute = self._get_trading_time(
                user_profile['time_preferences']
            )
        else:
            sold = False
            sell_reason = 'holding'
        
        # buy_hour 생성
        buy_hour, buy_minute = self._get_trading_time(user_profile['time_preferences'])
        
        # 수익률 구간 계산
        profit_zone = self._get_profit_zone(current_profit)
        
        return {
            'holding_days': holding_days,
            'final_profit_rate': round(current_profit, 4),
            'max_profit_rate': round(max_profit, 4),
            'min_profit_rate': round(min_profit, 4),
            'profit_volatility': round(np.std(daily_profits), 4),
            'profit_zone': profit_zone,
            'sold': 1 if sold else 0,  # 명시적으로 0/1로 변환
            'sell_reason': sell_reason or 'holding',
            'sell_hour': sell_hour if sell_hour else buy_hour,
            'sell_minute': sell_minute if sell_minute else buy_minute,
            'is_loss_pattern': 1 if (max_profit > 0.05 and current_profit < -0.05) else 0
        }
    
    def _calculate_daily_return(self, stock, market_condition, day):
        """일별 수익률 계산"""
        base_return = 0
        
        if market_condition == '상승장':
            base_return = np.random.normal(0.002, stock['avg_volatility'])
        elif market_condition == '하락장':
            base_return = np.random.normal(-0.002, stock['avg_volatility'])
        else:  # 횡보장
            base_return = np.random.normal(0, stock['avg_volatility'] * 0.8)
        
        # 베타 적용
        base_return *= stock['beta']
        
        # 특별 이벤트 (5% 확률)
        if np.random.random() < 0.05:
            event_return = np.random.choice([-0.1, -0.05, 0.05, 0.1])
            base_return += event_return
        
        return base_return
    
    def _decide_sell(self, user_profile, current_profit, holding_days, 
                    max_profit, volatility):
        """매도 결정 로직"""
        # 수익률 구간 확인
        profit_zone = self._get_profit_zone(current_profit)
        
        # 수익 상황에서의 매도
        if current_profit > 0:
            if profit_zone in user_profile['profit_targets']:
                sell_prob = user_profile['profit_targets'][profit_zone] * 0.7  # 확률 낮춤
                if np.random.random() < sell_prob:
                    return {'sell': True, 'reason': f'profit_taking_{profit_zone}'}
        
        # 손실 상황에서의 매도
        else:
            loss_zone = self._get_loss_zone(current_profit)
            if loss_zone in user_profile['loss_thresholds']:
                sell_prob = user_profile['loss_thresholds'][loss_zone] * 0.6  # 확률 낮춤
                if np.random.random() < sell_prob:
                    return {'sell': True, 'reason': f'stop_loss_{loss_zone}'}
        
        # 패닉셀 체크
        if volatility > user_profile['panic_threshold']:
            if np.random.random() < 0.2:  # 30% -> 20%
                return {'sell': True, 'reason': 'panic_sell'}
        
        # 고점 대비 하락 (그리드)
        if max_profit > 0.1 and (max_profit - current_profit) > 0.05:
            if np.random.random() < 0.3:  # 40% -> 30%
                return {'sell': True, 'reason': 'drawdown_sell'}
        
        # 장기 보유
        if holding_days > 60:
            if np.random.random() < 0.03 * (holding_days - 60):  # 5% -> 3%
                return {'sell': True, 'reason': 'time_based'}
        
        return {'sell': False, 'reason': None}
    
    def _get_profit_zone(self, profit_rate):
        """수익률 구간 분류"""
        if profit_rate < 0:
            return 'loss'
        elif profit_rate < 0.05:
            return '0-5%'
        elif profit_rate < 0.10:
            return '5-10%'
        elif profit_rate < 0.20:
            return '10-20%'
        else:
            return '20%+'
    
    def _get_loss_zone(self, profit_rate):
        """손실률 구간 분류"""
        if profit_rate > -0.03:
            return '0--3%'
        elif profit_rate > -0.05:
            return '-3--5%'
        elif profit_rate > -0.10:
            return '-5--10%'
        else:
            return '-10%+'
    
    def _generate_loss_pattern_cases(self, df):
        """과거 손실 패턴 사례 생성"""
        # 실제 손실 사례 추가 (하드코딩된 예시)
        self.loss_patterns = [
            {
                'case_id': 'LOSS_001',
                'date': '2024-03-15',
                'stock': 'LG화학',
                'initial_loss': -0.042,
                'final_loss': -0.128,
                'holding_days': 15,
                'pattern_description': '손실 상황에서 홀딩 → 추가 하락',
                'market_condition': '하락장',
                'similar_cases': ['LOSS_005', 'LOSS_012']
            },
            {
                'case_id': 'LOSS_002',
                'date': '2024-02-20',
                'stock': '카카오',
                'initial_loss': -0.03,
                'final_loss': -0.15,
                'holding_days': 25,
                'pattern_description': '실적 발표 후 급락 미대응',
                'market_condition': '횡보장',
                'similar_cases': ['LOSS_008']
            }
        ]
        
        # 시뮬레이션 데이터에서 손실 패턴 추출
        loss_trades = df[df['is_loss_pattern'] == 1].head(20)
        for idx, trade in loss_trades.iterrows():
            self.loss_patterns.append({
                'case_id': f'LOSS_{idx:03d}',
                'date': f'2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}',
                'stock': trade['stock_name'],
                'initial_loss': trade['min_profit_rate'],
                'final_loss': trade['final_profit_rate'],
                'holding_days': trade['holding_days'],
                'pattern_description': '고점 대비 큰 폭 하락',
                'market_condition': trade['market_condition'],
                'similar_cases': []
            })
    
    def create_features(self, df):
        """고급 특징 엔지니어링"""
        df = df.copy()
        
        # 1. 시간대 특징
        df['time_slot'] = pd.cut(df['buy_hour'], 
                                 bins=[9, 10, 11, 13, 14, 16],
                                 labels=['morning', 'mid_morning', 'lunch', 
                                        'afternoon', 'closing'])
        df['is_closing_hour'] = (df['buy_hour'] >= 14).astype(int)
        df['is_morning_hour'] = (df['buy_hour'] <= 10).astype(int)
        
        # 2. 수익률 특징
        df['profit_to_max_ratio'] = df['final_profit_rate'] / (df['max_profit_rate'] + 0.001)
        df['drawdown'] = df['max_profit_rate'] - df['final_profit_rate']
        df['profit_per_day'] = df['final_profit_rate'] / (df['holding_days'] + 1)
        df['is_profitable'] = (df['final_profit_rate'] > 0).astype(int)
        
        # 3. 변동성 특징
        df['volatility_ratio'] = df['profit_volatility'] / (abs(df['final_profit_rate']) + 0.001)
        df['extreme_move'] = (abs(df['final_profit_rate']) > 0.1).astype(int)
        
        # 4. 종목 특징 인코딩
        df['sector_encoded'] = self.sector_encoder.fit_transform(df['sector'])
        df['market_cap_score'] = df['market_cap'].map({
            '대형주': 3, '중형주': 2, '소형주': 1
        })
        
        # 5. 시장 상황
        df['market_condition_encoded'] = df['market_condition'].map({
            '상승장': 1, '횡보장': 0, '하락장': -1
        })
        
        # 6. 보유기간 특징
        df['is_short_term'] = (df['holding_days'] < 5).astype(int)
        df['is_mid_term'] = ((df['holding_days'] >= 5) & (df['holding_days'] < 20)).astype(int)
        df['is_long_term'] = (df['holding_days'] >= 20).astype(int)
        
        return df
    
    def train_models(self, test_size=0.2):
        """모든 모델 훈련"""
        print("🤖 고급 매매 패턴 AI 모델 훈련 시작...")
        
        # 데이터 생성 및 전처리
        df = self.generate_realistic_market_data()
        df = self.create_features(df)
        
        # 특징 선택 (time_slot 제외 - 원핫인코딩으로 처리)
        feature_cols = [
            'sector_encoded', 'market_cap_score', 'buy_hour', 'buy_minute',
            'is_closing_hour', 'is_morning_hour',
            'holding_days', 'final_profit_rate', 'max_profit_rate', 
            'min_profit_rate', 'profit_volatility', 'profit_to_max_ratio',
            'drawdown', 'profit_per_day', 'is_profitable',
            'volatility_ratio', 'extreme_move', 'market_condition_encoded',
            'is_short_term', 'is_mid_term', 'is_long_term'
        ]
        
        # 원-핫 인코딩 추가
        time_slot_dummies = pd.get_dummies(df['time_slot'], prefix='time')
        profit_zone_dummies = pd.get_dummies(df['profit_zone'], prefix='zone')
        
        X = pd.concat([df[feature_cols], time_slot_dummies, profit_zone_dummies], axis=1)
        self.feature_names = X.columns.tolist()
        
        # 1. 매도 확률 예측 모델
        print("⏳ 매도 확률 예측 모델 훈련 중...")
        y_sell = df['sold'].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_sell, test_size=test_size, random_state=42, stratify=y_sell
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.sell_probability_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.sell_probability_model.fit(X_train_scaled, y_train)
        
        # 성능 평가
        y_pred_proba = self.sell_probability_model.predict_proba(X_test_scaled)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"✅ 매도 확률 모델 AUC: {auc_score:.3f}")
        
        # 2. 수익률 구간별 행동 예측 모델
        print("⏳ 수익률 구간별 행동 모델 훈련 중...")
        profit_df = df[df['is_profitable'] == 1]
        if len(profit_df) > 100:
            X_profit = pd.concat([
                profit_df[feature_cols], 
                pd.get_dummies(profit_df['time_slot'], prefix='time'),
                pd.get_dummies(profit_df['profit_zone'], prefix='zone')
            ], axis=1)
            
            y_profit_zone = profit_df['profit_zone']
            zone_encoder = LabelEncoder()
            y_profit_encoded = zone_encoder.fit_transform(y_profit_zone)
            
            X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(
                X_profit, y_profit_encoded, test_size=test_size, random_state=42
            )
            
            self.profit_zone_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42
            )
            
            # 특징 이름 정렬을 위해 동일한 순서 보장
            X_p_train = X_p_train.reindex(columns=X.columns, fill_value=0)
            X_p_test = X_p_test.reindex(columns=X.columns, fill_value=0)
            
            X_p_train_scaled = self.scaler.transform(X_p_train)
            self.profit_zone_model.fit(X_p_train_scaled, y_p_train)
        
        # 3. 손실 패턴 감지 모델
        print("⏳ 손실 패턴 감지 모델 훈련 중...")
        y_loss_pattern = df['is_loss_pattern']
        
        if sum(y_loss_pattern) > 50:
            X_l_train, X_l_test, y_l_train, y_l_test = train_test_split(
                X, y_loss_pattern, test_size=test_size, random_state=42
            )
            
            self.loss_pattern_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=len(y_l_train) / sum(y_l_train),
                random_state=42
            )
            
            X_l_train_scaled = self.scaler.transform(X_l_train)
            self.loss_pattern_model.fit(X_l_train_scaled, y_l_train)
        
        self.is_trained = True
        print("✅ 모든 모델 훈련 완료!")
        
        # 특징 중요도 출력
        self._print_feature_importance()
        
        return True
    
    def _print_feature_importance(self):
        """주요 특징 중요도 출력"""
        if self.sell_probability_model:
            print("\n📊 매도 결정 주요 요인 Top 10:")
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.sell_probability_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for _, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
    
    def predict_realtime(self, 
                        ticker: str,
                        stock_name: str,
                        current_profit_rate: float,
                        holding_days: int,
                        current_time: str,  # "14:30" 형식
                        market_data: Dict,
                        user_history: Optional[Dict] = None) -> Dict:
        """
        실시간 매매 의사결정 예측
        
        예시:
        predict_realtime(
            ticker="005930",
            stock_name="삼성전자", 
            current_profit_rate=0.068,
            holding_days=8,
            current_time="14:30",
            market_data={
                'sector': '전자',
                'market_cap': '대형주',
                'daily_volatility': 0.021,
                'market_condition': '상승장'
            }
        )
        """
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 시간 파싱
        hour, minute = map(int, current_time.split(':'))
        
        # 기본 특징 생성
        features = {
            'sector': market_data['sector'],
            'market_cap': market_data['market_cap'],
            'buy_hour': hour,
            'buy_minute': minute,
            'holding_days': holding_days,
            'final_profit_rate': current_profit_rate,
            'max_profit_rate': current_profit_rate * 1.1,  # 추정값
            'min_profit_rate': min(0, current_profit_rate * 0.9),
            'profit_volatility': market_data.get('daily_volatility', 0.02),
            'market_condition': market_data['market_condition']
        }
        
        # 데이터프레임 생성 및 특징 엔지니어링
        df = pd.DataFrame([features])
        # profit_zone 먼저 생성
        df['profit_zone'] = df['final_profit_rate'].apply(self._get_profit_zone)
        df = self.create_features(df)
        
        # 원-핫 인코딩 추가
        time_slot_dummies = pd.get_dummies(df['time_slot'], prefix='time')
        profit_zone_dummies = pd.get_dummies(df['profit_zone'], prefix='zone')
        
        # 필요한 컬럼만 선택
        numeric_cols = [col for col in self.feature_names if col in df.columns]
        X = pd.concat([df[numeric_cols], time_slot_dummies, profit_zone_dummies], axis=1)
        
        # 모든 특징이 있는지 확인하고 없는 것은 0으로 채움
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        
        # 예측
        X_scaled = self.scaler.transform(X)
        
        # 1. 매도 확률
        sell_probability = self.sell_probability_model.predict_proba(X_scaled)[0, 1]
        
        # 2. 과거 유사 패턴 검색
        similar_loss_pattern = self._find_similar_loss_pattern(
            stock_name, current_profit_rate, holding_days
        )
        
        # 3. 수익률 구간별 행동 분석
        profit_zone = self._get_profit_zone(current_profit_rate)
        zone_behavior = self._analyze_profit_zone_behavior(profit_zone, user_history)
        
        # 4. 시간대별 특성 반영
        time_factor = self._analyze_time_factor(hour, minute)
        
        # 5. 종합 분석 및 추천
        recommendation = self._generate_recommendation(
            sell_probability, similar_loss_pattern, zone_behavior, time_factor
        )
        
        return {
            'ticker': ticker,
            'stock_name': stock_name,
            'current_status': {
                'profit_rate': f"{current_profit_rate:.1%}",
                'holding_days': f"{holding_days}일",
                'time': current_time,
                'volatility': f"{market_data.get('daily_volatility', 0):.1%}"
            },
            'analysis': {
                'sell_probability': f"{sell_probability:.0%}",
                'profit_zone': profit_zone,
                'time_impact': time_factor,
                'similar_loss_pattern': similar_loss_pattern
            },
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
    
    def _find_similar_loss_pattern(self, stock_name, current_profit, holding_days):
        """과거 유사 손실 패턴 검색"""
        if current_profit >= 0:
            return None
        
        similar_patterns = []
        for pattern in self.loss_patterns:
            # 유사도 계산
            similarity = 0
            
            # 종목 일치
            if pattern['stock'] == stock_name:
                similarity += 0.3
            
            # 손실률 유사도
            if abs(pattern['initial_loss'] - current_profit) < 0.02:
                similarity += 0.4
            
            # 보유기간 유사도
            if abs(pattern['holding_days'] - holding_days) < 5:
                similarity += 0.3
            
            if similarity > 0.6:
                similar_patterns.append({
                    'pattern': pattern,
                    'similarity': similarity
                })
        
        if similar_patterns:
            best_match = max(similar_patterns, key=lambda x: x['similarity'])
            return {
                'warning': f"⚠️ 위험 패턴 감지",
                'message': f"현재 상황은 과거 손실 패턴과 {best_match['similarity']*100:.0f}% 유사합니다",
                'case': best_match['pattern'],
                'recommendation': "지금 손절하거나 스탑로스 설정"
            }
        
        return None
    
    def _analyze_profit_zone_behavior(self, profit_zone, user_history):
        """수익률 구간별 행동 분석"""
        if not user_history or profit_zone not in self.profit_patterns:
            return {
                'zone': profit_zone,
                'historical_sell_rate': 'N/A',
                'recommendation': '과거 데이터 부족'
            }
        
        zone_history = self.profit_patterns[profit_zone]
        sell_rate = sum(1 for h in zone_history if h['action'] == 'sell') / len(zone_history)
        
        return {
            'zone': profit_zone,
            'historical_sell_rate': f"{sell_rate:.0%}",
            'past_actions': len(zone_history),
            'recommendation': f"과거 {profit_zone} 구간에서 {sell_rate:.0%} 매도"
        }
    
    def _analyze_time_factor(self, hour, minute):
        """시간대별 특성 분석"""
        time_str = f"{hour:02d}:{minute:02d}"
        
        if hour < 10:
            return {
                'period': '장 초반',
                'impact': '충동적 거래 주의',
                'factor': '+10%'
            }
        elif hour >= 14:
            return {
                'period': '장 마감 전',
                'impact': '차익실현 성향 강함',
                'factor': '+15%'
            }
        else:
            return {
                'period': '장 중',
                'impact': '일반적 거래 시간',
                'factor': '+0%'
            }
    
    def _generate_recommendation(self, sell_prob, loss_pattern, zone_behavior, time_factor):
        """종합 추천 생성"""
        reasons = []
        action = "보유"
        urgency = "낮음"
        
        # 매도 확률 기반
        if sell_prob > 0.7:
            action = "매도"
            urgency = "높음"
            reasons.append(f"높은 매도 확률 ({sell_prob:.0%})")
        elif sell_prob > 0.5:
            action = "매도 고려"
            urgency = "중간"
            reasons.append(f"중간 매도 확률 ({sell_prob:.0%})")
        
        # 손실 패턴
        if loss_pattern:
            action = "즉시 손절"
            urgency = "매우 높음"
            reasons.append(loss_pattern['message'])
        
        # 수익률 구간
        if zone_behavior.get('historical_sell_rate', 0) != 'N/A':
            reasons.append(zone_behavior['recommendation'])
        
        # 시간대
        if time_factor['factor'] != '+0%':
            reasons.append(f"{time_factor['period']} - {time_factor['impact']}")
        
        return {
            'action': action,
            'urgency': urgency,
            'reasons': reasons,
            'summary': f"🔴 {action} 권장" if urgency in ["높음", "매우 높음"] else f"🟡 {action}"
        }
    
    def update_with_result(self, trade_id: str, actual_action: str, final_profit: float):
        """실제 거래 결과로 모델 업데이트 (온라인 학습)"""
        # 거래 결과 저장
        self.trading_history.append({
            'trade_id': trade_id,
            'actual_action': actual_action,
            'final_profit': final_profit,
            'timestamp': datetime.now()
        })
        
        # 수익률 구간별 행동 기록
        profit_zone = self._get_profit_zone(final_profit)
        self.profit_patterns[profit_zone].append({
            'action': actual_action,
            'profit': final_profit
        })
        
        # 일정 거래 수 이상 축적시 재학습
        if len(self.trading_history) % 100 == 0:
            print(f"📊 {len(self.trading_history)}개 거래 완료 - 모델 재학습 예정")
            # self.retrain_models()  # 실제 구현시 백그라운드에서 실행
        
        return True
    
    def get_performance_report(self):
        """AI 성능 리포트 생성"""
        if not self.trading_history:
            return "거래 이력이 없습니다."
        
        # 성능 분석
        total_trades = len(self.trading_history)
        profitable_trades = sum(1 for t in self.trading_history if t['final_profit'] > 0)
        
        report = {
            'total_trades': total_trades,
            'win_rate': profitable_trades / total_trades,
            'avg_profit': np.mean([t['final_profit'] for t in self.trading_history]),
            'profit_zones': dict(self.profit_patterns),
            'model_accuracy': {
                'sell_prediction': 'N/A',  # 실제 정확도 계산 필요
                'loss_pattern_detection': 'N/A'
            }
        }
        
        return report

# 사용 예시
if __name__ == "__main__":
    # AI 모델 생성 및 훈련
    ai = AdvancedTradingAI()
    ai.train_models()
    
    # 실시간 예측 테스트
    print("\n🎯 실시간 매매 의사결정 예측")
    
    # 시나리오: 삼성전자 +6.8% (보유 8일차), 14:30
    result = ai.predict_realtime(
        ticker="005930",
        stock_name="삼성전자",
        current_profit_rate=0.068,
        holding_days=8,
        current_time="14:30",
        market_data={
            'sector': '전자',
            'market_cap': '대형주',
            'daily_volatility': 0.021,
            'market_condition': '상승장'
        }
    )
    
    print(f"\n📊 분석 결과:")
    print(f"종목: {result['stock_name']} ({result['ticker']})")
    print(f"현재 상태: {result['current_status']['profit_rate']} ({result['current_status']['holding_days']})")
    print(f"시간: {result['current_status']['time']}")
    print(f"\n매도 확률: {result['analysis']['sell_probability']}")
    print(f"추천: {result['recommendation']['summary']}")
    print(f"근거:")
    for reason in result['recommendation']['reasons']:
        print(f"  - {reason}")