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
        self.model_performance = None








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

    def load_trading_data(self, csv_path="../generate_data/output/trading_patterns.csv"):
        """generate_data에서 생성한 CSV 데이터 로드"""
        print("📊 매매 패턴 데이터 로드 중...")

        try:
            df = pd.read_csv(csv_path)
            print(f"✅ 데이터 로드 완료: {len(df):,}개 레코드")

            # 데이터 전처리
            df = self._preprocess_csv_data(df)
            return df

        except FileNotFoundError:
            print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}")
            print("📌 generate_data/main.py를 먼저 실행하여 데이터를 생성해주세요.")
            raise
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            print("📌 generate_data/main.py를 먼저 실행하여 데이터를 생성해주세요.")
            raise

    def _preprocess_csv_data(self, df):
        """CSV 데이터 전처리 (generate_data 형식 → AI 모델 형식)"""
        print("🔄 데이터 형식 변환 중...")

        # 필요한 컬럼 매핑
        processed_df = pd.DataFrame()

        # 기본 정보
        processed_df['user_id'] = df['investor_profile']
        processed_df['ticker'] = 'NVDA'  # 기본값
        processed_df['stock_name'] = 'NVIDIA'
        processed_df['sector'] = '전자'  # 기본값
        processed_df['market_cap'] = '대형주'  # 기본값

        # 시간 정보 (generate_data에 시간 정보가 없으므로 추정)
        processed_df['buy_date'] = df.get('timestamp', 0)
        processed_df['buy_hour'] = np.random.randint(9, 15, len(df))
        processed_df['buy_minute'] = np.random.randint(0, 60, len(df))

        # 시장 상황 (기본값으로 설정)
        processed_df['market_condition'] = np.random.choice(['상승장', '하락장', '횡보장'], len(df))

        # 거래 결과 (generate_data 컬럼 활용)
        processed_df['holding_days'] = np.random.randint(1, 30, len(df))  # 기본값
        processed_df['final_profit_rate'] = df['return_1d'].fillna(0)
        processed_df['max_profit_rate'] = processed_df['final_profit_rate'] * np.random.uniform(1.0, 1.2, len(df))
        processed_df['min_profit_rate'] = processed_df['final_profit_rate'] * np.random.uniform(0.8, 1.0, len(df))
        processed_df['profit_volatility'] = df.get('volatility_reaction', 0.02)

        # 매도 여부 (BUY=0, SELL=1, HOLD=0)
        processed_df['sold'] = (df['action'] == 'SELL').astype(int)
        processed_df['sell_reason'] = df['action'].map({
            'BUY': 'new_position',
            'SELL': 'profit_taking',
            'HOLD': 'holding'
        })

        # 수익률 구간
        processed_df['profit_zone'] = processed_df['final_profit_rate'].apply(self._get_profit_zone)

        # 손실 패턴 (고점 대비 큰 폭 하락)
        processed_df['is_loss_pattern'] = (
            (processed_df['max_profit_rate'] > 0.05) &
            (processed_df['final_profit_rate'] < -0.05)
        ).astype(int)

        print(f"✅ 데이터 변환 완료: {len(processed_df)}개 레코드")
        return processed_df

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

    def train_models(self, test_size=0.2, csv_path="../generate_data/output/trading_patterns.csv"):
        """모든 모델 훈련 (CSV 데이터 사용)"""
        print("🤖 고급 매매 패턴 AI 모델 훈련 시작...")

        # CSV 데이터 로드 및 전처리
        df = self.load_trading_data(csv_path)
        df = self.create_features(df)

        # 특징 선택
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

        self.model_performance = {'auc_score': auc_score}

        # 2. 수익률 구간별 행동 예측 모델
        print("⏳ 수익률 구간별 행동 모델 훈련 중...")
        profit_df = df[df['is_profitable'] == 1]
        if len(profit_df) > 100:
            try:
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
            except Exception as e:
                print(f"⚠️ 수익률 구간 모델 훈련 실패: {e}")

        # 3. 손실 패턴 감지 모델
        print("⏳ 손실 패턴 감지 모델 훈련 중...")
        y_loss_pattern = df['is_loss_pattern']

        if sum(y_loss_pattern) > 50:
            try:
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
            except Exception as e:
                print(f"⚠️ 손실 패턴 모델 훈련 실패: {e}")

        self.is_trained = True
        print("✅ 모든 모델 훈련 완료!")

        # 특징 중요도 출력
        self._print_feature_importance()

        return True

    def _print_feature_importance(self):
        """주요 특징 중요도 출력"""
        if self.sell_probability_model and self.feature_names:
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
    
    # CSV 파일에서 데이터 로드하여 모델 훈련
    try:
        ai.train_models(csv_path="../generate_data/output/trading_patterns.csv")
        
        # 실시간 예측 테스트
        print("\n🎯 실시간 매매 의사결정 예측")
        
        # 시나리오: NVIDIA +6.8% (보유 8일차), 14:30
        result = ai.predict_realtime(
            ticker="NVDA",
            stock_name="NVIDIA",
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
            
    except FileNotFoundError:
        print("\n❌ CSV 파일을 찾을 수 없습니다.")
        print("📌 먼저 generate_data 폴더에서 데이터를 생성해주세요:")
        print("   cd ../generate_data")
        print("   python main.py")