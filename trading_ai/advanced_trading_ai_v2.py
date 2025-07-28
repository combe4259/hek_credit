# advanced_trading_ai_v2.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, roc_auc_score, mean_squared_error, 
                           classification_report, confusion_matrix, precision_recall_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 시드 고정 (재현성 확보)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
import random
random.seed(RANDOM_STATE)

class AdvancedTradingAI:

    def __init__(self):
        # 모델들  
        self.sell_probability_model = None      # 매도 확률 예측 (기존)
        self.action_classifier = None          # 3-Class 액션 예측 (새로운!)
        self.ensemble_model = None             # 앙상블 모델
        
        # 데이터 처리
        self.scaler = RobustScaler()  # 이상치에 강한 스케일러
        self.feature_scaler = StandardScaler()
        self.stock_encoder = LabelEncoder()
        self.sector_encoder = LabelEncoder()
        
        # 개인 매매 이력
        self.trading_history = []
        self.loss_patterns = []
        self.profit_patterns = defaultdict(list)
        
        # 설정
        self.is_trained = False
        self.feature_names = None
        self.model_performance = {}
        self.random_state = RANDOM_STATE
        
        # 최적 임계값
        self.optimal_threshold = 0.5

    def _generate_loss_pattern_cases(self, df):
        """과거 손실 패턴 사례 생성"""
        self.loss_patterns = [
            {
                'case_id': 'LOSS_001',
                'date': '2024-03-15',
                'stock': 'NVIDIA',
                'initial_loss': -0.042,
                'final_loss': -0.128,
                'holding_days': 15,
                'pattern_description': '손실 상황에서 홀딩 → 추가 하락',
                'market_condition': '하락장',
                'similar_cases': ['LOSS_005', 'LOSS_012']
            }
        ]

    def load_trading_data(self, csv_path="../generate_data/output/trading_patterns_augmented.csv"):
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
            raise

    def _preprocess_csv_data(self, df):
        """CSV 데이터 전처리 (generate_data 형식 → AI 모델 형식)"""
        print("🔄 데이터 형식 변환 중...")
        
        # 필요한 컬럼 매핑
        processed_df = pd.DataFrame()
        
        # 기본 정보
        processed_df['user_id'] = df['investor_profile']
        processed_df['ticker'] = 'NVDA'
        processed_df['stock_name'] = 'NVIDIA'
        processed_df['sector'] = '전자'
        processed_df['market_cap'] = '대형주'
        
        # 시간 정보 - 실제 데이터 기반 추정
        processed_df['buy_date'] = pd.to_numeric(df.get('timestamp', 0))
        
        # 투자자 프로필별 일관된 거래 시간 패턴
        profile_time_map = {
            'Conservative': (10, 30),      # 10:30 - 안정적인 시간
            'Aggressive': (9, 15),         # 9:15 - 장 초반
            'Technical_Trader': (11, 0),   # 11:00 - 지표 확인 후
            'Momentum_Trader': (14, 30),   # 14:30 - 모멘텀 확인
            'Swing_Trader': (13, 0)        # 13:00 - 중간 시점
        }
        
        # 프로필별로 시간 할당 (변형 프로필도 기본 프로필 시간 사용)
        def get_trading_time(profile_name):
            base_profile = profile_name.split('_variant')[0]
            return profile_time_map.get(base_profile, (10, 0))
        
        processed_df['buy_hour'] = df['investor_profile'].apply(lambda x: get_trading_time(x)[0])
        processed_df['buy_minute'] = df['investor_profile'].apply(lambda x: get_trading_time(x)[1])
        
        # 시장 상황 - 기술적 지표 기반 추정
        def estimate_market_condition(row):
            rsi = row.get('rsi', 50)
            macd_signal = row.get('macd_signal', 0)
            volatility = row.get('volatility_reaction', 0.5)
            
            # RSI와 MACD를 종합하여 시장 상황 판단
            if rsi > 60 and macd_signal > 0:
                return '상승장'
            elif rsi < 40 and macd_signal < 0:
                return '하락장'
            else:
                return '횡보장'
        
        processed_df['market_condition'] = df.apply(estimate_market_condition, axis=1)
        
        # 보유 기간 - 액션과 수익률 기반 추정
        def estimate_holding_days(row):
            action = row.get('action', 'HOLD')
            return_1d = row.get('return_1d', 0)
            return_7d = row.get('return_7d', 0)
            return_30d = row.get('return_30d', 0)
            
            # 수익률 변화 패턴으로 보유 기간 추정
            if action == 'SELL':
                # 매도한 경우, 수익률에 따라 보유 기간 추정
                if abs(return_1d) > 0.05:
                    return np.random.randint(1, 5)  # 단기
                elif abs(return_7d) > 0.1:
                    return np.random.randint(5, 15)  # 중기
                else:
                    return np.random.randint(15, 30)  # 장기
            elif action == 'BUY':
                return 1  # 매수는 시작
            else:  # HOLD
                return np.random.randint(5, 20)  # 중간 정도
        
        # 시드 고정된 랜덤 생성
        np.random.seed(self.random_state)
        processed_df['holding_days'] = df.apply(estimate_holding_days, axis=1)
        
        # 거래 결과
        processed_df['final_profit_rate'] = df['return_1d'].fillna(0)
        processed_df['max_profit_rate'] = df[['return_1d', 'return_7d']].max(axis=1)
        processed_df['min_profit_rate'] = df[['return_1d', 'return_7d']].min(axis=1)
        processed_df['profit_volatility'] = df.get('volatility_reaction', 0.02)
        
        # 매매 결정 - 3가지 액션 모두 학습! (업그레이드)
        # BUY=0, HOLD=1, SELL=2 (진짜 3-Class AI)
        action_mapping = {'BUY': 0, 'HOLD': 1, 'SELL': 2}
        processed_df['action_class'] = df['action'].map(action_mapping)
        
        # 기존 매도 예측도 유지 (하위 호환성)
        processed_df['sold'] = (df['action'] == 'SELL').astype(int)
        
        print(f"\n🤖 업그레이드된 AI 학습 모드:")
        print(f"   - 입력: 순수 시장 데이터 (RSI, 수익률, 변동성 등)")
        print(f"   - 정답: BUY(0), HOLD(1), SELL(2) - 3가지 액션 모두!")
        print(f"   - AI 목표: '언제 매수/보유/매도하는지' 패턴 학습")
        
        # 액션 분포 확인
        action_counts = df['action'].value_counts()
        print(f"\n📈 액션 분포:")
        for action, count in action_counts.items():
            percentage = count / len(df) * 100
            print(f"   - {action}: {count:,}개 ({percentage:.1f}%)")
        
        processed_df['sell_reason'] = df.get('reasoning', 'holding')
        
        # 수익률 구간 - 미래 정보 제거
        # processed_df['profit_zone'] = processed_df['final_profit_rate'].apply(self._get_profit_zone)
        
        # 손실 패턴 - 미래 정보 제거
        # processed_df['is_loss_pattern'] = (
        #     (processed_df['max_profit_rate'] > 0.05) &
        #     (processed_df['final_profit_rate'] < -0.05)
        # ).astype(int)
        
        print(f"✅ 데이터 변환 완료: {len(processed_df)}개 레코드")
        
        # 데이터 품질 체크
        print(f"\n🔍 데이터 품질 체크:")
        print(f"   - 매도 비율: {processed_df['sold'].mean()*100:.1f}%")
        print(f"   - 수익률 분포: 평균 {processed_df['final_profit_rate'].mean():.3f}, 표준편차 {processed_df['final_profit_rate'].std():.3f}")
        print(f"   - 보유기간 분포: 평균 {processed_df['holding_days'].mean():.1f}일")
        
        # 🎯 기술적 지표들 복사 (핵심!)
        technical_indicators = ['rsi', 'macd_signal', 'bb_position', 'volume_ratio', 'daily_return', 'gap']
        for indicator in technical_indicators:
            if indicator in df.columns:
                processed_df[indicator] = df[indicator]
                print(f"✅ {indicator} 복사됨")
            else:
                print(f"⚠️ {indicator} 누락")
        
        # 손실 패턴 생성
        self._generate_loss_pattern_cases(processed_df)
        
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
        
        # 2. 수익률 특징 - 미래 정보 제거
        # df['profit_to_max_ratio'] = df['final_profit_rate'] / (df['max_profit_rate'] + 0.001)
        # df['drawdown'] = df['max_profit_rate'] - df['final_profit_rate']
        # df['profit_per_day'] = df['final_profit_rate'] / (df['holding_days'] + 1)
        # df['is_profitable'] = (df['final_profit_rate'] > 0).astype(int)
        
        # 3. 변동성 특징 - 미래 정보 제거
        # df['volatility_ratio'] = df['profit_volatility'] / (abs(df['final_profit_rate']) + 0.001)
        # df['extreme_move'] = (abs(df['final_profit_rate']) > 0.1).astype(int)
        
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
        
        # 7. 추가 특징 (금융공학적 관점) - 미래 정보 제거
        # df['sharpe_ratio'] = df['profit_per_day'] / (df['profit_volatility'] + 0.001)
        # df['risk_adjusted_return'] = df['final_profit_rate'] / (df['profit_volatility'] + 0.001)
        
        # 8. 🎯 기술적 지표 보존 확인 (중요!)
        technical_indicators = ['rsi', 'macd_signal', 'bb_position', 'volume_ratio', 'daily_return', 'gap']
        missing_indicators = [ind for ind in technical_indicators if ind not in df.columns]
        if missing_indicators:
            print(f"⚠️ 경고: 기술적 지표 누락 - {missing_indicators}")
        else:
            print(f"✅ 기술적 지표 보존됨: {technical_indicators}")
        
        return df

    def train_models(self, test_size=0.2, csv_path="../generate_data/output/trading_patterns_augmented.csv"):
        """모든 모델 훈련 (개선된 버전)"""
        print("🤖 고급 매매 패턴 AI 모델 훈련 시작...")
        
        # CSV 데이터 로드 및 전처리
        df = self.load_trading_data(csv_path)
        df = self.create_features(df)
        
        # 특징 선택 - 순수 시장 데이터만 (투자자 성향 데이터 제거, 실제 지표 추가)
        # 🚨 미래 수익률 정보 제거! (final_profit_rate, max_profit_rate, min_profit_rate)
        # 🚨 buy_minute 제거! (너무 세밀한 정보로 과적합 유발)
        feature_cols = [
            # 기본 시장 정보
            'sector_encoded', 'market_cap_score', 'market_condition_encoded',
            
            # 시간 정보 (분 단위 제외)
            'buy_hour', 'is_closing_hour', 'is_morning_hour',
            
            # 변동성 정보만 (미래 정보 아님)
            'profit_volatility',
            
            # 🎯 실제 기술적 지표들 (핵심 추가!)
            'rsi',           # RSI 지표 (과매수/과매도)
            'macd_signal',   # MACD 신호 (0 또는 1)
            'bb_position',   # 볼린저밴드 위치 (0-1)
            'volume_ratio',  # 거래량 비율 (급증 여부)
            'daily_return',  # 일일 수익률
            'gap',           # 갭 상승/하락
        ]
        
        # 먼저 processed_df에서 기본 특징들을 만들고 나서 feature_cols 설정해야 함
                
        print(f"\n📋 사용할 특징들:")
        for i, feature in enumerate(feature_cols, 1):
            print(f"   {i:2d}. {feature}")
        
        # 디버깅: 특징 생성 확인
        print(f"\n🔍 특징 생성 확인:")
        print(f"   - 전체 데이터: {len(df)}개")
        print(f"   - 최종 특징 수: {len(feature_cols)}개")
        print(f"   - 데이터 누수 제거 완료!")
        
        # ❌ 불필요한 중복 제거 - 기술적 지표는 이미 feature_cols에 포함됨
        # ❌ 투자자 성향 패턴 제거 - 의미없는 개인차 데이터 (일반화 불가)
        
        # df에서 존재하는 특징들만 선택
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"\n⚠️ 사용 불가능한 특징: {set(feature_cols) - set(available_features)}")
        
        X = df[available_features].copy()
        self.feature_names = X.columns.tolist()
        
        print(f"\n🤖 최종 AI 입력 데이터:")
        print(f"   - 특징 수: {len(self.feature_names)}개")
        print(f"   - 데이터 크기: {X.shape}")
        print(f"   - 누수 제거: holding_days, is_short_term 등 action 기반 특징 모두 제거")
        
        # 타겟 변수 - 두 가지 예측 모델
        y_sell = df['sold'].astype(int)           # 기존: 매도 vs 비매도
        y_action = df['action_class'].astype(int) # 새로운: BUY/HOLD/SELL
        
        print(f"   - 매도 데이터: {sum(y_sell)}개 ({sum(y_sell)/len(y_sell)*100:.1f}%)")
        print(f"   - 3-Class 데이터: BUY={sum(y_action==0)}, HOLD={sum(y_action==1)}, SELL={sum(y_action==2)}")
        
        # 시계열 분할 (금융 데이터의 특성 고려)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 데이터 분할 - 누수 체크
        print(f"\n📊 데이터 분할 전 최종 체크:")
        print(f"   - X 변수들: {list(X.columns)}")
        print(f"   - y 변수: 매도={sum(y_sell)}, 보유={len(y_sell)-sum(y_sell)}")
        
        # 누수 체크: action 관련 컬럼이 X에 있는지 확인
        leak_keywords = ['action', 'sell', 'buy', 'hold', 'sold']
        potential_leaks = [col for col in X.columns if any(keyword in col.lower() for keyword in leak_keywords)]
        if potential_leaks:
            print(f"   ⚠️ 의심스러운 컬럼: {potential_leaks}")
        else:
            print(f"   ✅ 누수 체크 통과!")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_sell, test_size=test_size, random_state=self.random_state, stratify=y_sell
        )
        
        # SMOTE 적용
        print("\n📊 데이터 균형 맞추기...")
        min_class_samples = min(sum(y_train == 0), sum(y_train == 1))
        
        if min_class_samples > 5:
            smote = SMOTE(random_state=self.random_state, k_neighbors=min(5, min_class_samples-1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"   - SMOTE 후: 매도 {sum(y_train_balanced)}개, 보유 {len(y_train_balanced)-sum(y_train_balanced)}개")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            print("   - 클래스 샘플이 부족하여 SMOTE 미적용")
        
        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 1. 기존 매도 모델 (Binary)
        print("\n⏳ 1번째 AI: 매도 vs 비매도 학습...")
        
        self.sell_probability_model = xgb.XGBClassifier(
            n_estimators=100,        # 500→100 (과적합 방지)
            max_depth=4,             # 6→4 (단순화)
            learning_rate=0.1,       # 0.03→0.1 
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,      # 3→5 (보수적)
            reg_alpha=0.5,           # 0.1→0.5 (정규화)
            reg_lambda=2,            # 1→2 (정규화)
            random_state=self.random_state,
            n_jobs=-1,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        
        # 모델 학습
        self.sell_probability_model.fit(
            X_train_scaled, 
            y_train_balanced,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # 1번째 모델 성능 평가
        self._evaluate_model_performance(X_test_scaled, y_test, X_train_scaled, y_train_balanced)
        
        # 최적 임계값 찾기
        self._find_optimal_threshold(X_test_scaled, y_test)
        
        # 2. 새로운 3-Class 액션 모델
        print("\n⏳ 2번째 AI: BUY/HOLD/SELL 3-Class 학습...")
        self._train_action_classifier(X, y_action, test_size)
        
        self.is_trained = True
        print("\n✅ 모든 모델 훈련 완료!")
        
        # 특징 중요도 출력
        self._print_feature_importance()
        
        # ✨ 실무 검증: 시간 분할 테스트
        self._time_series_validation(X, y_sell)
        
        return True
    
    def _train_action_classifier(self, X, y_action, test_size):
        """새로운 3-Class 액션 모델 훈련"""
        from sklearn.metrics import classification_report, confusion_matrix
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_action, test_size=test_size, random_state=self.random_state, stratify=y_action
        )
        
        # 스케일링 (기존 scaler 사용)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 3-Class XGBoost 모델
        self.action_classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            objective='multi:softprob',  # 3클래스 분류
            num_class=3
        )
        
        # 학습
        self.action_classifier.fit(X_train_scaled, y_train)
        
        # 예측 및 성능 평가
        y_pred = self.action_classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎆 3-Class AI 성능:")
        print(f"   - 정확도: {accuracy:.3f}")
        
        # 다중분류 리포트
        class_names = ['BUY', 'HOLD', 'SELL']
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        print(f"\n📄 각 액션별 성능:")
        for action in class_names:
            if action in report:
                metrics = report[action]
                print(f"   {action:4s}: 정밀도 {metrics['precision']:.3f}, 재현률 {metrics['recall']:.3f}, F1 {metrics['f1-score']:.3f}")
        
        # 혼동행렬
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📈 3-Class 혼동행렬:")
        print(f"        예측")
        print(f"       BUY HOLD SELL")
        for i, actual in enumerate(['BUY', 'HOLD', 'SELL']):
            if i < len(cm):
                print(f"{actual:4s} {cm[i][0] if 0 < len(cm[i]) else 0:4d} {cm[i][1] if 1 < len(cm[i]) else 0:4d} {cm[i][2] if 2 < len(cm[i]) else 0:4d}")
        
        # 3-Class 특징 중요도
        print(f"\n🎯 3-Class AI가 중요하게 보는 요소 Top 5:")
        importances = self.action_classifier.feature_importances_
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        for idx, (_, row) in enumerate(feature_df.head(5).iterrows(), 1):
            print(f"   {idx}. {row['feature']}: {row['importance']:.1%}")
    
    def _time_series_validation(self, X, y):
        """시간 분할 검증 - AI가 진짜 학습했는지 확인"""
        print("\n" + "="*60)
        print("📈 실무 검증: 시간 분할 테스트")
        print("🎯 목표: '과거 데이터로 학습 → 미래 데이터 예측'이 가능한지 확인")
        print("="*60)
        
        # 데이터를 시간 순으로 정렬 (이미 시간순이라고 가정)
        n_samples = len(X)
        
        # 70% 과거 데이터로 학습, 30% 미래 데이터로 테스트
        split_point = int(n_samples * 0.7)
        
        X_past = X.iloc[:split_point]
        y_past = y.iloc[:split_point]
        X_future = X.iloc[split_point:]
        y_future = y.iloc[split_point:]
        
        print(f"📅 데이터 분할:")
        print(f"   - 과거 데이터 (학습용): {len(X_past):,}개 ({split_point}번까지)")
        print(f"   - 미래 데이터 (테스트용): {len(X_future):,}개 ({split_point}번부터)")
        
        # 과거 데이터로만 새로운 모델 훈련
        print(f"\n🎓 과거 데이터로 새 모델 훈련 중...")
        
        # 데이터 스케일링
        scaler_past = RobustScaler()
        X_past_scaled = scaler_past.fit_transform(X_past)
        X_future_scaled = scaler_past.transform(X_future)
        
        # 간단한 모델로 학습 (빠른 테스트를 위해)
        past_model = xgb.XGBClassifier(
            n_estimators=50,  # 빠른 테스트
            max_depth=4,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        past_model.fit(X_past_scaled, y_past)
        
        # 미래 데이터로 예측
        y_future_pred = past_model.predict(X_future_scaled)
        y_future_proba = past_model.predict_proba(X_future_scaled)[:, 1]
        
        # 성능 비교
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        future_accuracy = accuracy_score(y_future, y_future_pred)
        future_auc = roc_auc_score(y_future, y_future_proba)
        
        # 기존 모델 성능 (랜덤 분할)
        original_auc = self.model_performance.get('auc_score', 0)
        original_accuracy = self.model_performance.get('accuracy', 0)
        
        print(f"\n📈 성능 비교 결과:")
        print(f"" + "-"*50)
        print(f"🎲 기존 모델 (랜덤 분할):")
        print(f"   - AUC: {original_auc:.3f}")
        print(f"   - 정확도: {original_accuracy:.3f}")
        print(f"")
        print(f"📅 시간 분할 모델 (과거→미래):")
        print(f"   - AUC: {future_auc:.3f}")
        print(f"   - 정확도: {future_accuracy:.3f}")
        print(f"" + "-"*50)
        
        # 성능 하락 계산
        auc_drop = original_auc - future_auc
        accuracy_drop = original_accuracy - future_accuracy
        
        print(f"📉 성능 변화:")
        print(f"   - AUC 변화: {auc_drop:+.3f} ({auc_drop/original_auc*100:+.1f}%)")
        print(f"   - 정확도 변화: {accuracy_drop:+.3f} ({accuracy_drop/original_accuracy*100:+.1f}%)")
        
        # 결과 해석
        print(f"\n🧐 실무 해석:")
        
        if abs(auc_drop) < 0.05 and abs(accuracy_drop) < 0.05:
            print(f"   ✅ 우수: AI가 시간에 무관한 일반화된 패턴을 학습했습니다!")
            print(f"   → 미래 데이터에도 비슷한 성능 유지")
        elif abs(auc_drop) < 0.1 and abs(accuracy_drop) < 0.1:
            print(f"   🟡 양호: 약간의 성능 하락이 있지만 수용 가능한 수준")
            print(f"   → 시장 환경 변화에 약간 민감")
        elif abs(auc_drop) < 0.2:
            print(f"   🟠 주의: 성능 하락이 있음. 과적합 가능성")
            print(f"   → 더 많은 데이터나 정규화 필요")
        else:
            print(f"   🔴 위험: 심각한 성능 하락. 과적합 의심")
            print(f"   → 모델 재설계 필요")
        
        print(f"\n📊 추가 분석:")
        
        # 미래 데이터에서의 특징 중요도
        past_importance = past_model.feature_importances_
        original_importance = self.sell_probability_model.feature_importances_
        
        print(f"   - 특징 중요도 일관성: {np.corrcoef(past_importance, original_importance)[0,1]:.3f}")
        print(f"     (값이 0.8 이상이면 일관된 학습 패턴)")
        
        # 미래 데이터의 클래스 분포
        future_sell_rate = y_future.mean()
        past_sell_rate = y_past.mean()
        
        print(f"   - 과거 매도 비율: {past_sell_rate:.1%}")
        print(f"   - 미래 매도 비율: {future_sell_rate:.1%}")
        print(f"   - 패턴 동일성: {'Yes' if abs(future_sell_rate - past_sell_rate) < 0.1 else 'No'}")
        
        print("\n" + "="*60)
        
        return {
            'future_auc': future_auc,
            'future_accuracy': future_accuracy,
            'auc_drop': auc_drop,
            'accuracy_drop': accuracy_drop,
            'feature_consistency': np.corrcoef(past_importance, original_importance)[0,1]
        }

    def _evaluate_model_performance(self, X_test, y_test, X_train, y_train):
        """모델 성능 종합 평가"""
        # 예측
        y_pred = self.sell_probability_model.predict(X_test)
        y_pred_proba = self.sell_probability_model.predict_proba(X_test)[:, 1]
        
        # 메트릭 계산
        auc_score = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 AI 학습 결과:")
        print(f"   - AUC: {auc_score:.3f}")
        print(f"   - 정확도: {accuracy:.3f}")
        
        # 성능 평가
        if auc_score > 0.95:
            print("   ⚠️ 경고: AUC {:.3f} - 과적합 의심! 데이터 누수 체크 필요".format(auc_score))
        elif auc_score > 0.8:
            print("   ✅ 우수: 좋은 성능")
        elif auc_score > 0.6:
            print("   ✅ 양호: 현실적인 AI 성능")
        else:
            print("   ❌ 주의: 성능 부족 - 데이터나 모델 개선 필요")
        
        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n혼동 행렬:")
        print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Precision과 Recall
        if cm[1,1] + cm[0,1] > 0:
            precision = cm[1,1] / (cm[1,1] + cm[0,1])
            print(f"   - Precision: {precision:.3f}")
        
        if cm[1,1] + cm[1,0] > 0:
            recall = cm[1,1] / (cm[1,1] + cm[1,0])
            print(f"   - Recall: {recall:.3f}")
        
        # 교차 검증
        cv_scores = cross_val_score(
            self.sell_probability_model, 
            X_train, 
            y_train, 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1
        )
        print(f"\n🔄 5-Fold 교차 검증 AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        self.model_performance = {
            'auc_score': auc_score,
            'accuracy': accuracy,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'confusion_matrix': cm.tolist()
        }

    def _find_optimal_threshold(self, X_test, y_test):
        """최적 임계값 찾기"""
        y_pred_proba = self.sell_probability_model.predict_proba(X_test)[:, 1]
        
        # Precision-Recall 곡선에서 최적 임계값 찾기
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # F1 스코어가 최대인 지점 찾기
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores[:-1])
        self.optimal_threshold = thresholds[optimal_idx]
        
        print(f"\n🎯 최적 임계값: {self.optimal_threshold:.3f}")
        print(f"   - F1 Score: {f1_scores[optimal_idx]:.3f}")

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
                        current_time: str,
                        market_data: Dict,
                        user_history: Optional[Dict] = None) -> Dict:
        """실시간 매매 의사결정 예측 (개선된 버전)"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 시드 고정으로 일관된 예측
        np.random.seed(self.random_state)
        
        # 시간 파싱
        hour, minute = map(int, current_time.split(':'))
        
        # 기본 특징 생성
        features = {
            'sector': market_data['sector'],
            'market_cap': market_data['market_cap'],
            'buy_hour': hour,
            'buy_minute': minute,
            'holding_days': holding_days,
            # 미래 정보 제거 - 실시간 예측 시에는 미래를 모름
            # 'final_profit_rate': current_profit_rate,
            # 'max_profit_rate': max(current_profit_rate, current_profit_rate * 1.1),
            # 'min_profit_rate': min(0, current_profit_rate * 0.9),
            'profit_volatility': market_data.get('daily_volatility', 0.02),
            'market_condition': market_data['market_condition'],
            
            # 🎯 실시간 기술적 지표 추가 (핵심 수정!)
            'rsi': market_data.get('rsi', 50),                    # RSI 지표 (기본값: 50)
            'macd_signal': market_data.get('macd_signal', 0),     # MACD 신호 (기본값: 0)
            'bb_position': market_data.get('bb_position', 0.5),   # 볼린저밴드 위치 (기본값: 0.5)
            'volume_ratio': market_data.get('volume_ratio', 1.0), # 거래량 비율 (기본값: 1.0)
            'daily_return': market_data.get('daily_return', 0),   # 일일 수익률 (기본값: 0)
            'gap': market_data.get('gap', 0)                      # 갭 상승/하락 (기본값: 0)
        }
        
        # 데이터프레임 생성 및 특징 엔지니어링
        df = pd.DataFrame([features])
        # df['profit_zone'] = df['final_profit_rate'].apply(self._get_profit_zone)  # 미래 정보 제거
        df = self.create_features(df)
        
        # 원-핫 인코딩 추가
        time_slot_dummies = pd.get_dummies(df['time_slot'], prefix='time')
        # profit_zone_dummies = pd.get_dummies(df['profit_zone'], prefix='zone')  # 미래 정보 제거
        
        # 필요한 컬럼만 선택
        numeric_cols = [col for col in self.feature_names if col in df.columns]
        X = pd.concat([df[numeric_cols], time_slot_dummies], axis=1)
        
        # 모든 특징이 있는지 확인하고 없는 것은 0으로 채움
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        
        # 예측
        X_scaled = self.scaler.transform(X)
        
        # 매도 확률 (일관된 예측)
        sell_probability = self.sell_probability_model.predict_proba(X_scaled)[0, 1]
        
        # 최적 임계값 사용
        sell_decision = sell_probability > self.optimal_threshold
        
        # 3-Class 액션 예측 (BUY/HOLD/SELL)
        action_prediction = None
        if hasattr(self, 'action_classifier') and self.action_classifier is not None:
            action_proba = self.action_classifier.predict_proba(X_scaled)[0]
            action_pred = self.action_classifier.predict(X_scaled)[0]
            action_mapping = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}
            
            action_prediction = {
                'BUY_prob': f"{action_proba[0]:.0%}",
                'HOLD_prob': f"{action_proba[1]:.0%}",
                'SELL_prob': f"{action_proba[2]:.0%}",
                'predicted_action': action_mapping[action_pred],
                'confidence': f"{max(action_proba):.0%}"
            }
        
        # 과거 유사 패턴 검색
        similar_loss_pattern = self._find_similar_loss_pattern(
            stock_name, current_profit_rate, holding_days
        )
        
        # 종합 분석 및 추천
        recommendation = self._generate_recommendation_v2(
            sell_probability, sell_decision, similar_loss_pattern, 
            current_profit_rate, holding_days
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
                'optimal_threshold': f"{self.optimal_threshold:.0%}",
                'decision': '매도' if sell_decision else '보유',
                'profit_zone': self._get_profit_zone(current_profit_rate),
                'similar_loss_pattern': similar_loss_pattern,
                'action_prediction': action_prediction
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
                'message': f"유사한 손실 패턴 발견 (유사도 {best_match['similarity']*100:.0f}%)",
                'case': best_match['pattern'],
                'recommendation': "즉시 손절 검토"
            }
        
        return None

    def _generate_recommendation_v2(self, sell_prob, sell_decision, loss_pattern, 
                                   current_profit, holding_days):
        """종합 추천 생성 (개선된 버전)"""
        reasons = []
        action = "보유"
        urgency = "낮음"
        
        # 매도 결정 기반
        if sell_decision:
            action = "매도"
            if sell_prob > 0.8:
                urgency = "매우 높음"
                reasons.append(f"매우 높은 매도 확률 ({sell_prob:.0%})")
            elif sell_prob > 0.6:
                urgency = "높음"
                reasons.append(f"높은 매도 확률 ({sell_prob:.0%})")
            else:
                urgency = "중간"
                reasons.append(f"임계값 초과 ({sell_prob:.0%} > {self.optimal_threshold:.0%})")
        
        # 수익률 기반 추가 분석
        if current_profit > 0.15:
            reasons.append(f"높은 수익률 ({current_profit:.1%}) - 이익 실현 고려")
            if action == "보유":
                action = "매도 고려"
                urgency = "중간"
        elif current_profit < -0.05:
            reasons.append(f"손실 확대 중 ({current_profit:.1%})")
            if action == "보유":
                action = "손절 고려"
                urgency = "중간"
        
        # 장기 보유
        if holding_days > 30:
            reasons.append(f"장기 보유 중 ({holding_days}일)")
        
        # 손실 패턴
        if loss_pattern:
            action = "즉시 손절"
            urgency = "매우 높음"
            reasons.append(loss_pattern['message'])
        
        # 최종 추천
        emoji_map = {
            "매우 높음": "🔴",
            "높음": "🟠",
            "중간": "🟡",
            "낮음": "🟢"
        }
        
        return {
            'action': action,
            'urgency': urgency,
            'reasons': reasons[:3],  # 최대 3개
            'summary': f"{emoji_map.get(urgency, '🟡')} {action} 권장"
        }

    def save_model(self, filepath='trading_ai_model.pkl'):
        """모델 저장"""
        if not self.is_trained:
            raise ValueError("훈련된 모델이 없습니다.")
        
        model_data = {
            'sell_probability_model': self.sell_probability_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'optimal_threshold': self.optimal_threshold,
            'model_performance': self.model_performance,
            'random_state': self.random_state,
            'action_classifier': getattr(self, 'action_classifier', None)  # 3-Class 모델
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ 모델 저장 완료: {filepath}")

    def load_model(self, filepath='trading_ai_model.pkl'):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.sell_probability_model = model_data['sell_probability_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.optimal_threshold = model_data['optimal_threshold']
        self.model_performance = model_data['model_performance']
        self.random_state = model_data['random_state']
        self.action_classifier = model_data.get('action_classifier', None)  # 3-Class 모델
        self.is_trained = True
        
        print(f"✅ 모델 로드 완료: {filepath}")


# 사용 예시
if __name__ == "__main__":
    # AI 모델 생성 및 훈련
    ai = AdvancedTradingAI()
    
    # CSV 파일에서 데이터 로드하여 모델 훈련
    try:
        ai.train_models(csv_path="../generate_data/output/trading_patterns_augmented.csv")
        
        # 모델 저장
        ai.save_model('trained_trading_ai_v2.pkl')
        
        print("\n" + "="*60)
        print("🎯 실시간 매매 의사결정 예측 테스트")
        print("="*60)
        
        # 시나리오 1: NVIDIA +6.8% (보유 8일차), 14:30
        result1 = ai.predict_realtime(
            ticker="NVDA",
            stock_name="NVIDIA",
            current_profit_rate=0.068,
            holding_days=8,
            current_time="14:30",
            market_data={
                'sector': '전자',
                'market_cap': '대형주',
                'daily_volatility': 0.021,
                'market_condition': '상승장',
                # 🎯 실제 기술적 지표 추가 (시나리오 1: 수익 중)
                'rsi': 58,              # 중립 구간
                'macd_signal': 1,       # 매수 신호
                'bb_position': 0.65,    # 상단 근처
                'volume_ratio': 1.2,    # 거래량 증가
                'daily_return': 0.015,  # 1.5% 상승
                'gap': 0.008            # 0.8% 갭업
            }
        )
        
        print(f"\n📊 시나리오 1 - 수익 중")
        print(f"종목: {result1['stock_name']} ({result1['ticker']})")
        print(f"현재 상태: {result1['current_status']['profit_rate']} ({result1['current_status']['holding_days']})")
        print(f"시간: {result1['current_status']['time']}")
        print(f"\n분석:")
        print(f"  - 매도 확률: {result1['analysis']['sell_probability']}")
        print(f"  - 임계값: {result1['analysis']['optimal_threshold']}")
        print(f"  - 결정: {result1['analysis']['decision']}")
        print(f"\n추천: {result1['recommendation']['summary']}")
        for reason in result1['recommendation']['reasons']:
            print(f"  - {reason}")
        
        # 시나리오 2: NVIDIA -4.2% (보유 15일차), 10:30
        result2 = ai.predict_realtime(
            ticker="NVDA",
            stock_name="NVIDIA",
            current_profit_rate=-0.042,
            holding_days=15,
            current_time="10:30",
            market_data={
                'sector': '전자',
                'market_cap': '대형주',
                'daily_volatility': 0.035,
                'market_condition': '하락장',
                # 🎯 실제 기술적 지표 추가 (시나리오 2: 손실 중)
                'rsi': 35,              # 과매도 구간
                'macd_signal': 0,       # 매도 신호
                'bb_position': 0.25,    # 하단 근처
                'volume_ratio': 1.8,    # 거래량 급증 (공포매도)
                'daily_return': -0.025, # -2.5% 하락
                'gap': -0.012           # -1.2% 갭다운  
            }
        )
        
        print(f"\n\n📊 시나리오 2 - 손실 중")
        print(f"종목: {result2['stock_name']} ({result2['ticker']})")
        print(f"현재 상태: {result2['current_status']['profit_rate']} ({result2['current_status']['holding_days']})")
        print(f"시간: {result2['current_status']['time']}")
        print(f"\n분석:")
        print(f"  - 매도 확률: {result2['analysis']['sell_probability']}")
        print(f"  - 임계값: {result2['analysis']['optimal_threshold']}")
        print(f"  - 결정: {result2['analysis']['decision']}")
        print(f"\n추천: {result2['recommendation']['summary']}")
        for reason in result2['recommendation']['reasons']:
            print(f"  - {reason}")
            
    except FileNotFoundError:
        print("\n❌ CSV 파일을 찾을 수 없습니다.")
        print("📌 먼저 generate_data 폴더에서 데이터를 생성해주세요:")
        print("   cd ../generate_data")
        print("   python main.py")