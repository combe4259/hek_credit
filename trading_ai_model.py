# trading_ai_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List, Any
import pickle
from datetime import datetime

class TradingPatternAI:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.model_performance = {}

    def generate_sample_data(self, n_users=1000, n_trades_per_user=50):
        """
        가상의 매매 데이터 생성 (실제 서비스에서는 DB에서 가져옴)
        """
        print(f"📊 {n_users}명 사용자, 각 {n_trades_per_user}개 거래 데이터 생성 중...")

        data = []
        np.random.seed(42)  # 재현 가능한 결과를 위해

        for user_id in range(n_users):
            # 사용자별 고유한 투자 성향 설정
            profit_threshold = max(0.03, np.random.normal(0.08, 0.03))  # 3~15% 수익실현
            loss_threshold = min(-0.02, np.random.normal(-0.05, 0.02))   # -2~-8% 손절
            risk_tolerance = np.random.uniform(0.1, 0.9)  # 위험 허용도

            for trade in range(n_trades_per_user):
                # 시장 상황 시뮬레이션
                market_volatility = np.random.uniform(0.005, 0.15)  # 0.5~15% 변동성
                market_trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])  # 하락/횡보/상승

                # 사용자 성향에 따른 수익률 생성
                if np.random.random() < (0.6 + risk_tolerance * 0.2):  # 수익 거래
                    profit_rate = np.random.uniform(0.005, profit_threshold * 1.5)
                    if market_trend == 1:  # 상승장에서 더 높은 수익
                        profit_rate *= 1.3
                else:  # 손실 거래
                    profit_rate = np.random.uniform(loss_threshold * 1.5, -0.005)
                    if market_trend == -1:  # 하락장에서 더 큰 손실
                        profit_rate *= 1.4

                # 보유일수 (수익률과 사용자 성향에 따라)
                if profit_rate > 0:
                    # 수익시 빨리 매도하는 경향
                    holding_days = int(np.random.exponential(5 + (1-risk_tolerance) * 10))
                else:
                    # 손실시 오래 보유하는 경향 (손실 회피)
                    holding_days = int(np.random.exponential(10 + (1-risk_tolerance) * 20))

                holding_days = min(max(holding_days, 1), 200)  # 1~200일 제한

                # 매매 의사결정 패턴
                is_profit_taking = 1 if profit_rate > profit_threshold else 0
                is_loss_cutting = 1 if profit_rate < loss_threshold else 0
                is_panic_sell = 1 if (profit_rate < -0.1 and holding_days < 3) else 0
                is_diamond_hands = 1 if (profit_rate < -0.05 and holding_days > 30) else 0

                data.append({
                    'user_id': f"user_{user_id}",
                    'profit_rate': round(profit_rate, 4),
                    'holding_days': holding_days,
                    'market_volatility': round(market_volatility, 4),
                    'market_trend': market_trend,
                    'is_profit_taking': is_profit_taking,
                    'is_loss_cutting': is_loss_cutting,
                    'is_panic_sell': is_panic_sell,
                    'is_diamond_hands': is_diamond_hands,
                    'risk_tolerance': round(risk_tolerance, 2)
                })

        df = pd.DataFrame(data)
        print(f"✅ 총 {len(df):,}개 거래 데이터 생성 완료")
        return df

    def create_user_features(self, df):
        """
        사용자별 투자 패턴 특성 추출
        """
        print("🔍 사용자별 투자 패턴 분석 중...")

        user_stats = df.groupby('user_id').agg({
            'profit_rate': ['mean', 'std', 'min', 'max', 'count'],
            'holding_days': ['mean', 'std', 'median'],
            'market_volatility': 'mean',
            'is_profit_taking': 'mean',      # 수익실현 비율
            'is_loss_cutting': 'mean',       # 손절 비율
            'is_panic_sell': 'mean',         # 패닉 매도 비율
            'is_diamond_hands': 'mean'       # 묻지마 홀딩 비율
        }).round(4)

        # 컬럼명 정리
        user_stats.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0]
                              for col in user_stats.columns]
        
        # risk_tolerance는 별도로 처리 (사용자별로 동일한 값)
        risk_tolerance_df = df.groupby('user_id')['risk_tolerance'].first()
        user_stats['risk_tolerance'] = risk_tolerance_df

        # 추가 파생 특성 생성
        user_stats['win_rate'] = df.groupby('user_id')['profit_rate'].apply(lambda x: (x > 0).mean())
        user_stats['avg_win'] = df[df['profit_rate'] > 0].groupby('user_id')['profit_rate'].mean().fillna(0)
        user_stats['avg_loss'] = df[df['profit_rate'] < 0].groupby('user_id')['profit_rate'].mean().fillna(0)
        user_stats['profit_factor'] = abs(user_stats['avg_win'] / user_stats['avg_loss']).fillna(1)

        # 매매 스타일 라벨링 (0: 보수적, 1: 공격적, 2: 단타형)
        user_stats['trading_style'] = 0  # 기본값: 보수적

        # 공격적 투자자 (높은 수익률, 높은 위험 허용도)
        aggressive_mask = (
                (user_stats['profit_rate_mean'] > 0.02) &
                (user_stats['risk_tolerance'] > 0.6) &
                (user_stats['profit_rate_std'] > 0.05)
        )
        user_stats.loc[aggressive_mask, 'trading_style'] = 1

        # 단타형 투자자 (짧은 보유기간, 높은 거래빈도)
        day_trader_mask = (
                (user_stats['holding_days_mean'] < 7) &
                (user_stats['is_profit_taking_mean'] > 0.3) |
                (user_stats['is_panic_sell_mean'] > 0.1)
        )
        user_stats.loc[day_trader_mask, 'trading_style'] = 2

        user_stats = user_stats.reset_index()

        print(f"✅ {len(user_stats)}명 사용자 특성 추출 완료")
        print(f"📊 투자 스타일 분포:")
        print(f"   보수적: {(user_stats['trading_style'] == 0).sum()}명")
        print(f"   공격적: {(user_stats['trading_style'] == 1).sum()}명")
        print(f"   단타형: {(user_stats['trading_style'] == 2).sum()}명")

        return user_stats

    def train_model(self, test_size=0.2):
        """
        XGBoost 모델 훈련
        """
        print("🤖 AI 모델 훈련 시작...")

        # 1. 데이터 생성
        df = self.generate_sample_data()
        user_features = self.create_user_features(df)

        # 2. 특성과 타겟 분리
        exclude_cols = ['user_id', 'trading_style']
        self.feature_names = [col for col in user_features.columns if col not in exclude_cols]

        X = user_features[self.feature_names]
        y = user_features['trading_style']

        print(f"📊 특성 개수: {len(self.feature_names)}")
        print(f"📊 학습 데이터: {len(X)}개")

        # 3. 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # 4. XGBoost 모델 훈련
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )

        print("⏳ 모델 훈련 중...")
        self.model.fit(X_train, y_train)

        # 5. 성능 평가
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.model_performance = {
            'accuracy': round(accuracy, 3),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(self.feature_names),
            'trained_at': datetime.now().isoformat()
        }

        self.is_trained = True

        print(f"✅ 모델 훈련 완료!")
        print(f"📈 정확도: {accuracy:.1%}")
        print(f"📊 특성 중요도 Top 5:")

        # 특성 중요도 출력
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        for i, row in feature_importance.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")

        return self.model_performance

    def predict_trading_style(self, user_trades_data):
        """
        사용자 매매 스타일 예측
        """
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다. train_model()을 먼저 호출하세요.")

        # 데이터프레임으로 변환
        if isinstance(user_trades_data, list):
            df = pd.DataFrame(user_trades_data)
        else:
            df = user_trades_data.copy()

        # 사용자 특성 계산
        user_stats = self.create_user_features(df)

        if len(user_stats) == 0:
            raise ValueError("분석할 거래 데이터가 충분하지 않습니다.")

        # 예측을 위한 특성 추출
        X = user_stats[self.feature_names].iloc[0:1]

        # 예측 실행
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # 결과 해석
        styles = {0: "보수적", 1: "공격적", 2: "단타형"}
        style_descriptions = {
            0: "안정적인 수익을 추구하며 리스크를 회피하는 투자자",
            1: "높은 수익을 위해 위험을 감수하는 적극적 투자자",
            2: "단기간에 빠른 매매를 반복하는 투자자"
        }

        # 개인 특성 분석
        user_profile = user_stats.iloc[0]
        analysis = {
            "평균_수익률": f"{user_profile['profit_rate_mean']:.1%}",
            "승률": f"{user_profile['win_rate']:.1%}",
            "평균_보유기간": f"{user_profile['holding_days_mean']:.0f}일",
            "수익실현_비율": f"{user_profile['is_profit_taking_mean']:.1%}",
            "손절_비율": f"{user_profile['is_loss_cutting_mean']:.1%}"
        }

        return {
            "predicted_style": styles[prediction],
            "style_description": style_descriptions[prediction],
            "confidence": float(max(probabilities)),
            "style_probabilities": {
                "보수적": float(probabilities[0]),
                "공격적": float(probabilities[1]),
                "단타형": float(probabilities[2])
            },
            "user_analysis": analysis,
            "analyzed_trades": len(df)
        }

    def predict_sell_probability(self, current_profit_rate, holding_days, user_style_probs, market_volatility=0.02):
        """
        현재 상황에서 매도 확률 예측 (규칙 기반)
        """
        sell_probability = 0.3  # 기본 확률

        # 수익률에 따른 조정
        if current_profit_rate > 0.1:  # 10% 이상 수익
            sell_probability += 0.4
        elif current_profit_rate > 0.05:  # 5% 이상 수익
            sell_probability += 0.2
        elif current_profit_rate < -0.1:  # 10% 이상 손실
            sell_probability += 0.3
        elif current_profit_rate < -0.05:  # 5% 이상 손실
            sell_probability += 0.1

        # 보유기간에 따른 조정
        if holding_days > 60:  # 2개월 이상
            sell_probability += 0.2
        elif holding_days > 30:  # 1개월 이상
            sell_probability += 0.1
        elif holding_days < 3:  # 3일 미만 (충동적 매도 위험)
            if current_profit_rate < 0:
                sell_probability += 0.2

        # 투자 스타일에 따른 조정
        if user_style_probs:
            if user_style_probs.get("단타형", 0) > 0.5:
                sell_probability += 0.2  # 단타형은 빨리 매도
            elif user_style_probs.get("보수적", 0) > 0.5:
                if current_profit_rate > 0.03:  # 보수적은 작은 수익에도 매도
                    sell_probability += 0.1

        # 시장 변동성에 따른 조정
        if market_volatility > 0.05:  # 높은 변동성
            sell_probability += 0.1

        # 0~1 사이로 제한
        sell_probability = min(max(sell_probability, 0.0), 1.0)

        # 추천 메시지 생성
        if sell_probability > 0.8:
            recommendation = "강력한 매도 신호"
        elif sell_probability > 0.6:
            recommendation = "매도 고려"
        elif sell_probability > 0.4:
            recommendation = "신중한 판단 필요"
        else:
            recommendation = "보유 권장"

        return {
            "sell_probability": round(sell_probability, 2),
            "recommendation": recommendation,
            "factors": {
                "profit_rate_impact": "높음" if abs(current_profit_rate) > 0.05 else "보통",
                "holding_period_impact": "높음" if holding_days > 30 or holding_days < 3 else "보통",
                "market_volatility_impact": "높음" if market_volatility > 0.05 else "낮음"
            }
        }

    def save_model(self, filepath):
        """모델 저장"""
        if not self.is_trained:
            raise ValueError("훈련된 모델이 없습니다.")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'performance': self.model_performance
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"💾 모델이 {filepath}에 저장되었습니다.")

    def load_model(self, filepath):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_performance = model_data['performance']
        self.is_trained = True

        print(f"📂 모델이 {filepath}에서 로드되었습니다.")
        print(f"📊 모델 정확도: {self.model_performance['accuracy']}")

# 사용 예시
if __name__ == "__main__":
    # AI 모델 생성 및 훈련
    ai = TradingPatternAI()
    performance = ai.train_model()

    # 테스트 데이터로 예측
    test_trades = [
        {
            'user_id': 'test_user',
            'profit_rate': 0.05,
            'holding_days': 7,
            'market_volatility': 0.02,
            'market_trend': 1,
            'is_profit_taking': 1,
            'is_loss_cutting': 0,
            'is_panic_sell': 0,
            'is_diamond_hands': 0,
            'risk_tolerance': 0.6
        },
        {
            'user_id': 'test_user',
            'profit_rate': -0.03,
            'holding_days': 15,
            'market_volatility': 0.04,
            'market_trend': -1,
            'is_profit_taking': 0,
            'is_loss_cutting': 1,
            'is_panic_sell': 0,
            'is_diamond_hands': 0,
            'risk_tolerance': 0.6
        }
    ]

    result = ai.predict_trading_style(test_trades)
    print("\n🎯 투자 스타일 예측 결과:")
    for key, value in result.items():
        print(f"{key}: {value}")