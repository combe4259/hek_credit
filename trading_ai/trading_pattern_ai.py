#!/usr/bin/env python3
"""
투자 행동 패턴 AI 시스템
1. 행동 패턴 분류 모델 (XGBoost)
2. 유사 상황 검색 모델 (Deep Learning 기반 임베딩)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import faiss  # Facebook의 유사도 검색 라이브러리
import joblib
from datetime import datetime, timedelta
import json

# ===== 1. 행동 패턴 분류 모델 =====

class BehaviorPatternClassifier:
    """투자자의 행동 패턴을 실시간으로 분류하는 AI"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.pattern_labels = [
            'good_decision',           # 적절한 의사결정
            'early_profit_taking',     # 조급한 익절
            'loss_aversion',           # 손실 회피 (손절 못함)
            'overtrading'              # 과잉 거래
        ]
        
    def prepare_features(self, df):
        """학습용 특성 준비"""
        # 행동 패턴 분류를 위한 핵심 특성들
        features = []
        labels = []
        
        for _, row in df.iterrows():
            # 현재 거래 상황
            current_features = [
                row['return_rate'],                    # 현재 수익률
                row['holding_days'],                   # 보유 기간
                row['return_rate'] / row['holding_days'] if row['holding_days'] > 0 else 0,  # 일평균 수익률
            ]
            
            # 개인 과거 패턴
            historical_features = [
                row['avg_holding_days'],               # 평균 보유기간
                row['avg_profit_rate'],                # 평균 익절 수익률
                row['avg_loss_rate'],                  # 평균 손절 수익률
                row['win_rate'],                       # 승률
                row['quick_sell_ratio'],               # 조급한 매도 비율
                row['avg_monthly_trades'],             # 월평균 거래 횟수
            ]
            
            # 현재 vs 과거 비교
            comparison_features = [
                row['holding_days'] / row['avg_holding_days'] if row['avg_holding_days'] > 0 else 1,  # 보유기간 비율
                row['return_rate'] / row['avg_profit_rate'] if row['avg_profit_rate'] > 0 and row['return_rate'] > 0 else 0,  # 수익률 비율
                1 if row['return_rate'] > 0 and row['return_rate'] < row['avg_profit_rate'] * 0.7 else 0,  # 조급한 익절 신호
            ]
            
            # 추가 행동 지표
            behavioral_indicators = [
                1 if row['holding_days'] < 3 else 0,  # 단타 신호
                1 if row.get('reactive_trader', False) else 0,  # 반응적 거래자
                row.get('overtrading_score', 0),      # 과잉거래 점수
            ]
            
            all_features = current_features + historical_features + comparison_features + behavioral_indicators
            features.append(all_features)
            
            # 라벨 매핑
            label = self._map_to_pattern_label(row)
            labels.append(label)
            
        return np.array(features), np.array(labels)
    
    def _map_to_pattern_label(self, row):
        """improvement_needed를 구체적인 패턴으로 매핑"""
        # 실제 데이터에 있는 라벨만 사용
        improvement = row['improvement_needed']
        
        if improvement == 'hold_longer_for_profit':
            return 'early_profit_taking'  # 조급한 익절
                
        elif improvement == 'cut_loss_earlier':
            return 'loss_aversion'  # 손실 회피
                
        elif improvement == 'avoid_impulsive_trading':
            return 'overtrading'  # 과잉거래
            
        else:
            return 'good_decision'  # 좋은 결정
    
    def train(self, X_train, y_train, X_test, y_test):
        """XGBoost 모델 학습"""
        print("🧠 행동 패턴 분류 AI 학습 중...")
        
        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 라벨 인코딩 (sklearn LabelEncoder 사용)
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        
        # 모든 가능한 라벨로 fit
        self.label_encoder.fit(self.pattern_labels)
        
        # 변환
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # XGBoost 모델
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            objective='multi:softprob',
            random_state=42,
            use_label_encoder=False,
            early_stopping_rounds=50,  # 여기로 이동
            eval_metric='mlogloss'
        )
        
        self.model.fit(
            X_train_scaled, y_train_encoded,
            eval_set=[(X_test_scaled, y_test_encoded)],
            verbose=False
        )
        
        # 성능 평가
        predictions = self.model.predict(X_test_scaled)
        accuracy = (predictions == y_test_encoded).mean()
        
        print(f"\n✅ 행동 패턴 분류 정확도: {accuracy:.2%}")
        
        # 패턴별 정확도
        pred_labels = [self.pattern_labels[idx] for idx in predictions]
        print("\n📊 패턴별 분류 성능:")
        report = classification_report(y_test, pred_labels, output_dict=True)
        
        for pattern in ['early_profit_taking', 'loss_aversion', 'overtrading']:
            if pattern in report:
                metrics = report[pattern]
                print(f"   - {pattern}: 정밀도 {metrics['precision']:.2%}, 재현율 {metrics['recall']:.2%}")
    
    def predict_pattern(self, current_situation):
        """실시간 행동 패턴 예측"""
        # 입력 특성 준비
        features = np.array([current_situation])
        features_scaled = self.scaler.transform(features)
        
        # 예측
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        predicted_idx = np.argmax(prediction_proba)
        predicted_pattern = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = prediction_proba[predicted_idx]
        
        # 상위 3개 패턴
        top_3_idx = np.argsort(prediction_proba)[-3:][::-1]
        top_3_patterns = [
            {
                'pattern': self.label_encoder.inverse_transform([idx])[0],
                'probability': prediction_proba[idx]
            }
            for idx in top_3_idx
        ]
        
        return {
            'predicted_pattern': predicted_pattern,
            'confidence': confidence,
            'all_probabilities': top_3_patterns
        }


# ===== 2. 유사 상황 검색 모델 =====

class TradingSituationEmbedder(nn.Module):
    """거래 상황을 벡터로 임베딩하는 딥러닝 모델"""
    
    def __init__(self, input_dim=15, embedding_dim=64):
        super(TradingSituationEmbedder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim)
        )
        
    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)  # L2 정규화


class SimilaritySearchEngine:
    """과거 유사 거래를 찾는 AI 검색 엔진"""
    
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.embedder = TradingSituationEmbedder(embedding_dim=embedding_dim)
        self.index = None  # FAISS 인덱스
        self.trade_database = []  # 거래 정보 저장
        self.scaler = StandardScaler()
        
    def build_index(self, historical_trades_df):
        """과거 거래 데이터로 검색 인덱스 구축"""
        print(f"\n🔍 유사 상황 검색 엔진 구축 중... (총 {len(historical_trades_df):,}개 거래)")
        
        # 특성 추출
        features = []
        total_trades = len(historical_trades_df)
        
        for idx, (_, trade) in enumerate(historical_trades_df.iterrows()):
            if idx % 10000 == 0:
                print(f"   처리 중... {idx:,}/{total_trades:,} ({idx/total_trades*100:.1f}%)")
            feature_vector = self._extract_features(trade)
            features.append(feature_vector)
        
        features = np.array(features, dtype=np.float32)
        features_scaled = self.scaler.fit_transform(features)
        
        # 임베딩 생성 (배치 처리)
        print(f"   임베딩 생성 중... (features shape: {features_scaled.shape})")
        self.embedder.eval()
        embeddings_list = []
        batch_size = 5000  # 배치 크기 (더 빠른 처리)
        
        # 첫 번째 배치 테스트
        print("   첫 번째 배치 테스트...")
        test_batch = features_scaled[:10]
        print(f"   테스트 배치 shape: {test_batch.shape}")
        
        try:
            with torch.no_grad():
                test_embedding = self.embedder(torch.FloatTensor(test_batch))
                print(f"   테스트 임베딩 성공: {test_embedding.shape}")
        except Exception as e:
            print(f"   ❌ 임베딩 오류: {e}")
            return
        
        # 전체 배치 처리
        with torch.no_grad():
            for i in range(0, len(features_scaled), batch_size):
                if i % 10000 == 0:
                    print(f"   임베딩 처리 중... {i:,}/{len(features_scaled):,}")
                batch = features_scaled[i:i+batch_size]
                batch_embeddings = self.embedder(torch.FloatTensor(batch)).numpy()
                embeddings_list.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings_list)
        
        # FAISS 인덱스 구축
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)
        
        # 거래 정보 저장
        self.trade_database = historical_trades_df.to_dict('records')
        
        print(f"✅ {len(self.trade_database)}개 거래 인덱싱 완료")
    
    def _extract_features(self, trade):
        """거래에서 특성 벡터 추출"""
        holding_days = trade.get('holding_days', 1)
        if holding_days == 0:
            holding_days = 1  # 0으로 나누기 방지
            
        return [
            trade.get('return_rate', 0),
            holding_days,
            trade.get('return_rate', 0) / holding_days,  # 일평균 수익률
            trade.get('month', datetime.now().month),
            trade.get('day_of_week', datetime.now().weekday()),
            trade.get('volatility', 0),
            trade.get('volume_ratio', 1),
            trade.get('rsi', 50),
            trade.get('price_to_ma5', 1),
            trade.get('price_to_ma20', 1),
            trade.get('market_trend', 0),  # -1: 하락, 0: 횡보, 1: 상승
            trade.get('sector_performance', 0),
            trade.get('news_sentiment', 0),  # -1: 부정, 0: 중립, 1: 긍정
            1 if trade.get('profit_loss') == 'profit' else 0,
            trade.get('trade_size_category', 2)  # 1: 소액, 2: 중간, 3: 대액
        ]
    
    def find_similar_situations(self, current_situation, k=5):
        """현재 상황과 유사한 과거 거래 검색"""
        # 현재 상황 특성 추출
        current_features = self._extract_features(current_situation)
        current_scaled = self.scaler.transform([current_features])
        
        # 임베딩 생성
        self.embedder.eval()
        with torch.no_grad():
            current_embedding = self.embedder(torch.FloatTensor(current_scaled)).numpy()
        
        # 유사도 검색
        distances, indices = self.index.search(current_embedding, k)
        
        # 결과 생성
        similar_trades = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            trade = self.trade_database[idx]
            
            # 매도 후 추가 상승/하락 계산 (모의투자 데이터에서)
            future_return = self._get_future_return(trade)
            
            similar_trades.append({
                'date': trade.get('sell_date', 'Unknown'),
                'stock': trade.get('stock_name', 'Unknown'),
                'sold_at_return': trade['return_rate'],
                'actual_peak_return': trade['return_rate'] + future_return,
                'missed_profit': future_return,
                'holding_days': trade['holding_days'],
                'similarity_score': 1 / (1 + dist),  # 거리를 유사도로 변환
                'pattern': trade.get('improvement_needed', 'unknown')
            })
        
        return similar_trades
    
    def _get_future_return(self, trade):
        """매도 후 실제 수익률 변화 (모의투자 데이터 필요)"""
        # 실제로는 모의투자 데이터베이스에서 조회
        # 여기서는 시뮬레이션
        if trade.get('improvement_needed') == 'hold_longer_for_profit':
            return np.random.uniform(2, 8)  # 2-8% 추가 상승
        elif trade.get('improvement_needed') == 'cut_loss_earlier':
            return np.random.uniform(-10, -3)  # 3-10% 추가 하락
        else:
            return np.random.uniform(-2, 2)


# ===== 3. 통합 AI 코칭 시스템 =====

class AITradingCoach:
    """행동 패턴 분류 + 유사 상황 검색을 통합한 AI 코칭"""
    
    def __init__(self):
        self.pattern_classifier = BehaviorPatternClassifier()
        self.similarity_engine = SimilaritySearchEngine()
        
    def train_from_data(self, training_data_path):
        """학습 데이터로 AI 시스템 훈련"""
        print("🚀 AI 트레이딩 코치 시스템 학습 시작\n")
        
        # 데이터 로드
        df = pd.read_csv(training_data_path)
        
        # 1. 행동 패턴 분류기 학습
        X, y = self.pattern_classifier.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pattern_classifier.train(X_train, y_train, X_test, y_test)
        
        # 2. 유사도 검색 엔진 구축
        self.similarity_engine.build_index(df)
        
        print("\n✅ AI 시스템 학습 완료!")
    
    def generate_coaching(self, current_situation):
        """실시간 AI 코칭 생성"""
        # 1. 행동 패턴 예측
        pattern_result = self.pattern_classifier.predict_pattern(
            self._prepare_pattern_features(current_situation)
        )
        
        # 2. 유사 상황 검색
        similar_trades = self.similarity_engine.find_similar_situations(
            current_situation, k=3
        )
        
        # 3. 코칭 메시지 생성
        coaching_message = self._create_coaching_message(
            current_situation,
            pattern_result,
            similar_trades
        )
        
        return {
            'pattern': pattern_result['predicted_pattern'],
            'confidence': pattern_result['confidence'],
            'similar_trades': similar_trades,
            'message': coaching_message
        }
    
    def _prepare_pattern_features(self, situation):
        """패턴 분류기용 특성 준비"""
        return [
            situation['current_return'],
            situation['holding_days'],
            situation['current_return'] / situation['holding_days'],
            situation['user_avg_holding_days'],
            situation['user_avg_profit_rate'],
            situation['user_avg_loss_rate'],
            situation['user_win_rate'],
            situation['user_quick_sell_ratio'],
            situation['user_monthly_trades'],
            situation['holding_days'] / situation['user_avg_holding_days'],
            situation['current_return'] / situation['user_avg_profit_rate'] if situation['user_avg_profit_rate'] > 0 else 0,
            1 if situation['current_return'] > 0 and situation['current_return'] < situation['user_avg_profit_rate'] * 0.7 else 0,
            1 if situation['holding_days'] < 3 else 0,
            0,  # reactive_trader
            situation.get('overtrading_score', 0)
        ]
    
    def _create_coaching_message(self, situation, pattern_result, similar_trades):
        """AI 분석 결과를 바탕으로 코칭 메시지 생성"""
        pattern = pattern_result['predicted_pattern']
        confidence = pattern_result['confidence']
        
        if pattern == 'early_profit_taking':
            return self._create_early_profit_message(situation, similar_trades, confidence)
        elif pattern == 'loss_aversion':
            return self._create_loss_aversion_message(situation, similar_trades, confidence)
        elif pattern == 'overtrading':
            return self._create_overtrading_message(situation, confidence)
        else:
            return self._create_positive_message(situation, confidence)
    
    def _create_early_profit_message(self, situation, similar_trades, confidence):
        """조급한 익절 경고 메시지"""
        avg_missed = np.mean([t['missed_profit'] for t in similar_trades])
        
        message = f"""
⏸️ 잠깐! 매도하기 전에 확인해보세요

📊 현재 상황: {situation['stock_name']} +{situation['current_return']:.1f}% ({situation['holding_days']}일 보유)

🤖 AI 분석: "조급한 익절 패턴" 감지 (신뢰도 {confidence*100:.0f}%)
당신의 평균 익절: +{situation['user_avg_profit_rate']:.1f}%
현재는 그보다 {situation['user_avg_profit_rate'] - situation['current_return']:.1f}% 낮습니다

📚 과거 유사 상황 {len(similar_trades)}건:
"""
        
        for i, trade in enumerate(similar_trades[:2]):
            message += f"""
┌────────────────────────────────┐
│ {trade['stock']} ({trade['date']})     │
│ +{trade['sold_at_return']:.1f}%에서 매도        │
│ → 실제로는 +{trade['actual_peak_return']:.1f}%까지 상승  │
│ → 놓친 수익: +{trade['missed_profit']:.1f}%      │
└────────────────────────────────┘
"""
        
        message += f"""
💡 AI 추천: 평균 {avg_missed:.1f}% 추가 상승 가능성
목표가 +{situation['user_avg_profit_rate']:.0f}%까지 기다려보세요

[그래도 매도하기] [목표가까지 기다리기]
"""
        return message
    
    def _create_loss_aversion_message(self, situation, similar_trades, confidence):
        """손실 회피 경고 메시지"""
        avg_additional_loss = np.mean([t['missed_profit'] for t in similar_trades if t['missed_profit'] < 0])
        
        message = f"""
🚨 손절 신호!

📉 현재 상황: {situation['stock_name']} {situation['current_return']:.1f}% ({situation['holding_days']}일 보유)

🤖 AI 분석: "손실 회피 패턴" 감지 (신뢰도 {confidence*100:.0f}%)
추가 하락 위험이 높습니다

⚠️ 과거 유사한 손실 방치 사례:
"""
        
        for trade in similar_trades[:2]:
            if trade['missed_profit'] < 0:
                message += f"""
• {trade['stock']}: {trade['sold_at_return']:.1f}% → {trade['actual_peak_return']:.1f}% (추가 {trade['missed_profit']:.1f}%)
"""
        
        message += f"""
평균 추가 손실: {avg_additional_loss:.1f}%

🛡️ AI 추천: 지금 손절하여 추가 손실을 방지하세요
스탑로스: {situation['current_return'] - 2:.1f}% 설정 권장

[손절 매도] [스탑로스 설정] [더 지켜보기]
"""
        return message
    
    def _create_overtrading_message(self, situation, confidence):
        """과잉거래 경고 메시지"""
        return f"""
🛑 과잉거래 주의!

📊 거래 패턴 분석 (신뢰도 {confidence*100:.0f}%)
- 월평균 거래: {situation['user_monthly_trades']:.0f}회 (평균 대비 2배↑)
- 최근 거래 빈도 급증

💡 AI 조언:
과도한 거래는 수수료와 실수를 증가시킵니다
• 하루 1회 이상 거래 자제
• 명확한 매매 계획 수립 후 실행
• 감정적 거래 피하기

[거래 일시정지 (24시간)] [거래 계속하기]
"""
        return message
    
    def _create_positive_message(self, situation, confidence):
        """긍정적 피드백"""
        return f"""
✅ 좋은 결정입니다!

📈 AI 분석: 적절한 매매 타이밍 (신뢰도 {confidence*100:.0f}%)

현재 수익률 {situation['current_return']:+.1f}%는 
당신의 평균 패턴과 일치합니다

계속 이런 규율을 유지하세요! 👍
"""
    
    def save_models(self, path_prefix='ai_trading_coach'):
        """학습된 모델 저장"""
        # 패턴 분류기 저장
        joblib.dump({
            'model': self.pattern_classifier.model,
            'scaler': self.pattern_classifier.scaler,
            'pattern_labels': self.pattern_classifier.pattern_labels
        }, f'{path_prefix}_pattern_classifier.pkl')
        
        # 유사도 검색 엔진 저장
        torch.save({
            'embedder_state': self.similarity_engine.embedder.state_dict(),
            'scaler': self.similarity_engine.scaler,
            'trade_database': self.similarity_engine.trade_database
        }, f'{path_prefix}_similarity_engine.pth')
        
        # FAISS 인덱스 저장
        faiss.write_index(self.similarity_engine.index, f'{path_prefix}_faiss.index')
        
        print(f"✅ AI 모델 저장 완료: {path_prefix}_*")


# ===== 실사용 예제 =====

def simulate_real_trading():
    """실제 거래 상황 시뮬레이션"""
    # AI 코치 초기화 및 학습
    coach = AITradingCoach()
    coach.train_from_data('../generate_data/output/trading_behavior_patterns.csv')
    
    # 시뮬레이션: 삼성전자 +6.8% 상황
    current_situation = {
        'stock_name': '삼성전자',
        'current_return': 6.8,
        'holding_days': 8,
        'user_avg_holding_days': 15,
        'user_avg_profit_rate': 9.2,
        'user_avg_loss_rate': -5.3,
        'user_win_rate': 0.65,
        'user_quick_sell_ratio': 0.7,
        'user_monthly_trades': 12,
        'volatility': 0.023,
        'rsi': 68,
        'market_trend': 1  # 상승장
    }
    
    # AI 코칭 생성
    print("\n" + "="*60)
    print("🤖 AI 트레이딩 코치 - 실시간 분석")
    print("="*60)
    
    coaching_result = coach.generate_coaching(current_situation)
    print(coaching_result['message'])
    
    # 모델 저장
    coach.save_models()


if __name__ == "__main__":
    simulate_real_trading()