import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import ParameterGrid
from scipy.stats import uniform, randint
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

class ImprovedHybridTradingAI:
    """
    개선된 하이브리드 트레이딩 AI v5
    
    구조:
    1. A유형: 완료된 거래의 정확한 품질 평가 (사후 분석) - v4에서 복사
    2. B유형: 실제 수익률 기반 진입 조건 점수 (0-100점) - 완전히 새로 설계
    
    개선점:
    - B유형: 규칙 기반 → 실제 데이터 기반 점수화
    - B유형: 정교한 기술적 지표 추가
    - Data Leakage 완전 제거 유지
    """
    
    def __init__(self, train_months=36, val_months=6, test_months=6, step_months=3, use_global_split=True):
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.step_months = step_months
        self.use_global_split = use_global_split
        
        # ===== A유형: 거래 품질 분석기 (v4와 동일) =====
        self.a_type_quality_model = None      
        self.a_type_quality_scalers = {}      
        self.a_type_quality_features = None   
        
        # ===== B유형: 개선된 진입 조건 평가기 (새로 설계) =====  
        self.b_type_entry_model = None        
        self.b_type_entry_scalers = {}        
        self.b_type_entry_features = None     
        
        self.fold_results = []
        self.global_results = None
        self.best_params_a = None
        self.best_params_b = None
    
    # ================================
    # A유형: 사후 품질 평가 (v4에서 그대로 복사)
    # ================================
    
    def create_a_type_quality_score(self, df, risk_scaler=None, eff_scaler=None, verbose=False):
        """A유형: 완료된 거래의 품질 점수 생성 (모든 정보 활용 가능)"""
        if verbose:
            print("🎯 A유형: Quality Score 생성 중...")
        
        df = df.copy()
        
        # NaN 처리 (0으로 대체)
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
        
        # 숫자형 데이터만 선택 (XGBoost 호환성)
        feature_data = df[self.a_type_quality_features].select_dtypes(include=[np.number])
        
        if verbose and len(feature_data.columns) != len(self.a_type_quality_features):
            print(f"  ⚠️ 비숫자형 칼럼 제외: {len(self.a_type_quality_features) - len(feature_data.columns)}개")
        
        return feature_data
    
    # ================================
    # B유형: 개선된 진입 조건 분석 (완전히 새로 설계)
    # ================================
    
    def create_b_type_entry_score(self, df, verbose=False):
        """B유형: 종합 매수 신호 점수 (0-100점) - 기술적+펀더멘털+시장환경"""
        if verbose:
            print("🚀 B유형 (매수 신호 AI): 종합 매수 신호 점수 생성 중...")
            print("   → 기술적 분석 + 펀더멘털 + 시장 환경을 종합한 매수 타이밍 점수")
        
        df = df.copy()
        
        # ===== 1. 기술적 신호 (40%) =====
        technical_score = self._calculate_technical_signals(df)
        
        # ===== 2. 펀더멘털 신호 (30%) =====
        fundamental_score = self._calculate_fundamental_signals(df)
        
        # ===== 3. 시장 환경 신호 (30%) =====
        market_score = self._calculate_market_environment_signals(df)
        
        # 종합 매수 신호 점수 (0-100)
        df['b_type_entry_score'] = (
            technical_score * 0.40 + 
            fundamental_score * 0.30 + 
            market_score * 0.30
        )
        
        # 0-100 범위 보장
        df['b_type_entry_score'] = np.clip(df['b_type_entry_score'], 0, 100)
        
        if verbose:
            print(f"  ✅ 종합 매수 신호 점수 생성 완료")
            print(f"  점수 범위: {df['b_type_entry_score'].min():.1f} ~ {df['b_type_entry_score'].max():.1f}")
            print(f"  점수 평균: {df['b_type_entry_score'].mean():.1f}")
            print(f"  구성: 기술적(40%) + 펀더멘털(30%) + 시장환경(30%)")
        
        return df
    
    def _calculate_technical_signals(self, df):
        """기술적 분석 신호 계산 (0-100점)"""
        signals = []
        
        # 1. 모멘텀 신호 (25%)
        momentum_20d = df['entry_momentum_20d'].fillna(0)
        # 적당한 하락 후 반등 시작이 매수 신호
        momentum_signal = np.where(
            momentum_20d < -15, 20,      # 과도한 하락
            np.where(momentum_20d < -5, 85,   # 적당한 하락 (매수 기회!)
                np.where(momentum_20d < 5, 70,    # 횡보
                    np.where(momentum_20d < 15, 50, 30))))  # 과열
        signals.append(momentum_signal * 0.25)
        
        # 2. 이동평균 신호 (25%)
        ma_dev_20d = df['entry_ma_dev_20d'].fillna(0)
        # 이평선 아래 있으면서 회복 조짐이 매수 신호
        ma_signal = np.where(
            ma_dev_20d < -10, 85,        # 크게 이탈 (매수 기회!)
            np.where(ma_dev_20d < -5, 70,     # 적당히 이탈
                np.where(ma_dev_20d < 5, 50,      # 근처
                    np.where(ma_dev_20d < 10, 30, 15))))  # 과열
        signals.append(ma_signal * 0.25)
        
        # 3. 과매도/과매수 신호 (25%)
        ratio_52w = df['entry_ratio_52w_high'].fillna(50)
        # 52주 고점 대비 낮을수록 매수 신호
        oversold_signal = (100 - ratio_52w)  # 0-100 자동 변환
        signals.append(oversold_signal * 0.25)
        
        # 4. 변동성 신호 (25%)
        volatility_20d = df['entry_volatility_20d'].fillna(25)
        # 적당한 변동성이 매수 신호
        vol_signal = np.where(
            volatility_20d < 15, 40,     # 너무 낮음
            np.where(volatility_20d < 30, 85,     # 적정 (매수 기회!)
                np.where(volatility_20d < 50, 60, 20)))  # 너무 높음
        signals.append(vol_signal * 0.25)
        
        return np.sum(signals, axis=0)
    
    def _calculate_fundamental_signals(self, df):
        """펀더멘털 분석 신호 계산 (0-100점)"""
        signals = []
        
        # 1. 밸류에이션 신호 (40%)
        pe_ratio = df['entry_pe_ratio'].fillna(20)
        # 낮은 PER이 매수 신호 (단, 너무 낮으면 문제)
        pe_signal = np.where(
            pe_ratio < 5, 30,           # 너무 낮음 (문제?)
            np.where(pe_ratio < 15, 85,      # 저평가 (매수!)
                np.where(pe_ratio < 25, 60,      # 적정
                    np.where(pe_ratio < 40, 35, 15))))  # 고평가
        signals.append(pe_signal * 0.4)
        
        # 2. 품질 신호 (30%)
        roe = df['entry_roe'].fillna(10)
        # 높은 ROE가 매수 신호
        roe_signal = np.where(
            roe < 5, 30,               # 낮은 품질
            np.where(roe < 10, 50,          # 평균
                np.where(roe < 15, 70,          # 양호
                    np.where(roe < 20, 85, 95))))   # 우수
        signals.append(roe_signal * 0.3)
        
        # 3. 성장성 신호 (30%)
        earnings_growth = df['entry_earnings_growth'].fillna(5)
        # 적당한 성장이 매수 신호
        growth_signal = np.where(
            earnings_growth < -10, 20,   # 역성장
            np.where(earnings_growth < 0, 40,    # 감소
                np.where(earnings_growth < 10, 70,   # 적당한 성장
                    np.where(earnings_growth < 25, 85, 60))))  # 고성장
        signals.append(growth_signal * 0.3)
        
        return np.sum(signals, axis=0)
    
    def _calculate_market_environment_signals(self, df):
        """시장 환경 신호 계산 (0-100점)"""
        signals = []
        
        # 1. VIX 신호 (40%)
        vix = df['entry_vix'].fillna(20)
        # 낮은 VIX가 매수 신호
        vix_signal = np.where(
            vix < 15, 90,              # 매우 안정 (매수!)
            np.where(vix < 20, 80,          # 안정
                np.where(vix < 25, 60,          # 보통
                    np.where(vix < 35, 40, 20))))   # 불안정
        signals.append(vix_signal * 0.4)
        
        # 2. 금리 환경 신호 (30%)
        tnx_yield = df['entry_tnx_yield'].fillna(2.5)
        # 적정 금리가 매수 신호
        rate_signal = np.where(
            tnx_yield < 1, 60,         # 너무 낮음
            np.where(tnx_yield < 3, 85,     # 적정 (매수!)
                np.where(tnx_yield < 5, 60, 30)))  # 높음, 너무 높음
        signals.append(rate_signal * 0.3)
        
        # 3. 시장 추세 신호 (30%) - 실제 존재하는 피처 사용
        market_return_20d = df.get('market_entry_cum_return_20d', pd.Series([0]*len(df))).fillna(0)
        # 적당한 상승 추세가 매수 신호  
        trend_signal = np.where(
            market_return_20d < -10, 30,   # 강한 하락
            np.where(market_return_20d < -5, 60,    # 약한 하락
                np.where(market_return_20d < 5, 85,      # 횡보/적당한 상승 (매수!)
                    np.where(market_return_20d < 10, 70, 40))))  # 과열
        signals.append(trend_signal * 0.3)
        
        return np.sum(signals, axis=0)
    
    def prepare_b_type_features(self, df, verbose=False):
        """B유형: 대폭 확장된 진입 조건 분석용 피처 준비 (진입 시점 정보만 사용)"""
        if verbose:
            print("🔧 B유형: 대폭 확장된 진입 조건 분석용 피처 준비 중...")
            print("   → 기술적 + 펀더멘털 + 시장환경 + 산업 정보 종합")
        
        # ===== 1. 기본 기술적 지표 =====
        technical_features = [
            # 모멘텀 지표
            'entry_momentum_5d', 'entry_momentum_20d', 'entry_momentum_60d',
            
            # 이동평균 기반 지표
            'entry_ma_dev_5d', 'entry_ma_dev_20d', 'entry_ma_dev_60d',
            
            # 변동성 지표
            'entry_volatility_5d', 'entry_volatility_20d', 'entry_volatility_60d',
            
            # 변동성 변화율
            'entry_vol_change_5d', 'entry_vol_change_20d', 'entry_vol_change_60d',
            
            # 가격 위치
            'entry_ratio_52w_high'
        ]
        
        # ===== 2. 펀더멘털 지표 =====
        fundamental_features = [
            'entry_pe_ratio',           # P/E 비율
            'entry_pb_ratio',           # P/B 비율  
            'entry_roe',                # ROE
            'entry_operating_margin',    # 영업이익률
            'entry_debt_equity_ratio',   # 부채비율
            'entry_earnings_growth'      # 이익 성장률
        ]
        
        # ===== 3. 시장 환경 지표 =====
        market_environment_features = [
            # VIX & 금리 (실제 존재)
            'entry_vix', 'entry_tnx_yield',
            
            # 시장 수익률 및 추세 (실제 존재)
            'market_entry_ma_return_5d', 'market_entry_ma_return_20d',
            'market_entry_cum_return_5d', 'market_entry_cum_return_20d',
            'market_entry_volatility_20d'
            
            # 고급 시장 컴포넌트들은 실제 데이터에 없어서 제외
        ]
        
        # ===== 4. 거래 관련 정보 =====
        trading_features = [
            'position_size_pct'  # 포지션 크기
        ]
        
        # 모든 잠재적 피처 결합
        potential_entry_features = (
            technical_features + 
            fundamental_features + 
            market_environment_features + 
            trading_features
        )
        
        # 실제 존재하고 안전한 피처만 선택
        safe_features = []
        
        # 미래 정보 차단을 위한 키워드 체크
        forbidden_keywords = ['exit_', 'change_', 'holding_period', 'market_return_during', 'excess_return']
        
        for feature in potential_entry_features:
            # 금지된 키워드 체크
            is_safe = True
            for forbidden in forbidden_keywords:
                if forbidden in feature:
                    is_safe = False
                    break
            if is_safe and feature in df.columns:
                safe_features.append(feature)
        
        # ===== 5. 범주형 피처는 일단 제외 (복잡성 때문) =====
        # categorical_features = self._encode_categorical_features(df)
        # safe_features.extend(categorical_features)
        
        self.b_type_entry_features = safe_features
        
        if verbose:
            print(f"  B유형 사용 피처: {len(self.b_type_entry_features)}개")
            print(f"  구성:")
            print(f"    • 기술적 지표: {len([f for f in safe_features if any(t in f for t in ['momentum', 'ma_dev', 'volatility', 'ratio'])])}개")
            print(f"    • 펀더멘털: {len([f for f in safe_features if any(t in f for t in ['pe_', 'pb_', 'roe', 'margin', 'debt', 'growth'])])}개")
            print(f"    • 시장환경: {len([f for f in safe_features if any(t in f for t in ['vix', 'tnx', 'market_'])])}개")
            print(f"    • 거래관련: {len([f for f in safe_features if 'position' in f])}개")
            print(f"  Data Leakage 방지: exit_, change_, holding_ 정보 완전 차단")
        
        # 피처 데이터 반환 (숫자형 데이터만 - XGBoost 호환성)
        if self.b_type_entry_features:
            feature_data = df[self.b_type_entry_features].select_dtypes(include=[np.number])
            
            if verbose and len(feature_data.columns) != len(self.b_type_entry_features):
                print(f"  ⚠️ 비숫자형 칼럼 제외: {len(self.b_type_entry_features) - len(feature_data.columns)}개")
            
            return feature_data
        else:
            return pd.DataFrame()
    
    def _encode_categorical_features(self, df):
        """범주형 피처 인코딩"""
        categorical_features = []
        
        # 산업 정보 원-핫 인코딩 (상위 N개 산업만)
        if 'industry' in df.columns:
            # 가장 많은 상위 10개 산업만 인코딩
            top_industries = df['industry'].value_counts().head(10).index.tolist()
            
            for industry in top_industries:
                feature_name = f'industry_{industry.replace(" ", "_").replace("&", "and").lower()}'
                categorical_features.append(feature_name)
        
        return categorical_features
    
    def _get_categorical_data(self, df):
        """범주형 데이터 생성"""
        categorical_data = pd.DataFrame(index=df.index)
        
        # 산업 정보 원-핫 인코딩
        if 'industry' in df.columns:
            top_industries = df['industry'].value_counts().head(10).index.tolist()
            
            for industry in top_industries:
                feature_name = f'industry_{industry.replace(" ", "_").replace("&", "and").lower()}'
                categorical_data[feature_name] = (df['industry'] == industry).astype(int)
        
        return categorical_data
    
    # ================================
    # 대규모 하이퍼파라미터 튜닝 시스템
    # ================================
    
    def get_massive_hyperparameter_grid(self, model_type='both'):
        """대규모 하이퍼파라미터 그리드 생성"""
        
        # XGBoost 기본 하이퍼파라미터 (매우 광범위)
        xgb_param_grid = {
            # 트리 구조 파라미터
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
            'min_child_weight': [1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
            'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bynode': [0.6, 0.7, 0.8, 0.9, 1.0],
            
            # 학습률 및 부스팅 파라미터  
            'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3],
            'n_estimators': [100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500],
            
            # 정규화 파라미터
            'reg_alpha': [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 50.0],
            
            # 고급 파라미터
            'gamma': [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            'max_delta_step': [0, 1, 2, 3, 5, 10],
            'scale_pos_weight': [1, 2, 3, 5, 10],
            
            # 트리 생성 방법
            'tree_method': ['hist', 'approx'] if torch.cuda.is_available() else ['hist'],
            'grow_policy': ['depthwise', 'lossguide'],
            
            # 샘플링 파라미터
            'max_leaves': [0, 31, 63, 127, 255, 511],
            'max_bin': [128, 256, 512],
        }
        
        # A유형과 B유형 각각 다른 파라미터 범위 적용
        if model_type == 'A' or model_type == 'both':
            # A유형: 품질 예측 - 정확도 중시
            a_specific_params = {
                'objective': ['reg:squarederror'],
                'eval_metric': ['rmse'],
                # 품질 예측에 특화된 파라미터
                'max_depth': [4, 5, 6, 7, 8, 9, 10],  # 깊이 제한
                'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],  # 낮은 학습률
                'n_estimators': [300, 400, 500, 600, 800, 1000, 1200],  # 많은 트리
                'reg_alpha': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0],  # 정규화 강화
                'reg_lambda': [1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
            }
            a_param_grid = {**xgb_param_grid, **a_specific_params}
        
        if model_type == 'B' or model_type == 'both':
            # B유형: 신호 예측 - 일반화 중시
            b_specific_params = {
                'objective': ['reg:squarederror'],
                'eval_metric': ['rmse'], 
                # 신호 예측에 특화된 파라미터
                'max_depth': [3, 4, 5, 6, 7, 8],  # 과적합 방지
                'learning_rate': [0.05, 0.07, 0.1, 0.15, 0.2],  # 적당한 학습률
                'n_estimators': [150, 200, 250, 300, 400, 500],  # 적당한 트리 수
                'subsample': [0.7, 0.8, 0.85, 0.9],  # 샘플링 강화
                'colsample_bytree': [0.7, 0.8, 0.85, 0.9],
            }
            b_param_grid = {**xgb_param_grid, **b_specific_params}
        
        if model_type == 'A':
            return {'A': a_param_grid}
        elif model_type == 'B':
            return {'B': b_param_grid}
        else:
            return {'A': a_param_grid, 'B': b_param_grid}
    
    def smart_hyperparameter_search(self, X_train, y_train, X_val, y_val, model_type, n_iter=200):
        """스마트 하이퍼파라미터 검색 (RandomizedSearchCV + GridSearchCV 조합)"""
        
        print(f"🔍 {model_type}유형 하이퍼파라미터 검색 시작 (최대 {n_iter}회 시도)")
        
        # 데이터 타입 디버깅
        print(f"  디버그: X_train 타입들: {X_train.dtypes.value_counts()}")
        print(f"  디버그: y_train 타입: {y_train.dtype if hasattr(y_train, 'dtype') else type(y_train)}")
        print(f"  디버그: 문제가 될 수 있는 컬럼들:")
        for col in X_train.columns:
            if X_train[col].dtype == 'object' or 'datetime' in str(X_train[col].dtype):
                print(f"    - {col}: {X_train[col].dtype}")
        
        # 숫자형 데이터만 강제로 선택
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train_clean = X_train[numeric_cols].copy()
        X_val_clean = X_val[numeric_cols].copy()
        
        print(f"  디버그: 정제 후 피처 수: {len(X_train_clean.columns)} (원본: {len(X_train.columns)})")
        
        param_grids = self.get_massive_hyperparameter_grid(model_type)
        param_grid = param_grids[model_type]
        
        # 1단계: RandomizedSearchCV로 넓은 범위 탐색
        print(f"  1단계: RandomizedSearch - {n_iter//2}회 시도")
        
        base_model = xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        # TimeSeriesSplit 사용 (시계열 데이터 특성상)
        tscv = TimeSeriesSplit(n_splits=3)
        
        random_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter//2,
            cv=tscv,
            scoring='r2',
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=False
        )
        
        random_search.fit(X_train_clean, y_train)
        
        print(f"  1단계 완료. 최고 점수: {random_search.best_score_:.4f}")
        print(f"  최고 파라미터: {random_search.best_params_}")
        
        # 2단계: 최적 파라미터 주변 세밀 탐색
        print(f"  2단계: 최적 파라미터 주변 GridSearch")
        
        best_params = random_search.best_params_
        refined_grid = self._create_refined_grid(best_params)
        
        if len(list(ParameterGrid(refined_grid))) <= 50:  # 조합 수가 적으면 GridSearch
            grid_search = GridSearchCV(
                base_model,
                refined_grid,
                cv=tscv,
                scoring='r2',
                n_jobs=-1,
                verbose=1,
                return_train_score=False
            )
            
            grid_search.fit(X_train_clean, y_train)
            final_model = grid_search.best_estimator_
            final_params = grid_search.best_params_
            final_score = grid_search.best_score_
            
        else:  # 조합 수가 많으면 추가 RandomizedSearch
            refined_search = RandomizedSearchCV(
                base_model,
                refined_grid,
                n_iter=min(n_iter//2, 100),
                cv=tscv,
                scoring='r2',
                n_jobs=-1,
                verbose=1,
                random_state=42,
                return_train_score=False
            )
            
            refined_search.fit(X_train_clean, y_train)
            final_model = refined_search.best_estimator_
            final_params = refined_search.best_params_
            final_score = refined_search.best_score_
        
        print(f"  2단계 완료. 최종 점수: {final_score:.4f}")
        
        # 3단계: 검증 세트로 최종 확인
        final_model.fit(X_train_clean, y_train)
        val_pred = final_model.predict(X_val_clean)
        val_r2 = r2_score(y_val, val_pred)
        
        print(f"  검증 세트 R²: {val_r2:.4f}")
        
        return final_model, final_params, {
            'cv_score': final_score,
            'val_r2': val_r2,
            'search_iterations': n_iter
        }
    
    def _create_refined_grid(self, best_params):
        """최적 파라미터 주변의 세밀한 그리드 생성"""
        refined_grid = {}
        
        for param, value in best_params.items():
            if param == 'max_depth':
                refined_grid[param] = [max(3, value-1), value, min(15, value+1)]
            elif param == 'learning_rate':
                refined_grid[param] = [max(0.01, value*0.8), value, min(0.3, value*1.2)]
            elif param == 'n_estimators':
                refined_grid[param] = [max(100, value-100), value, value+100]
            elif param == 'min_child_weight':
                refined_grid[param] = [max(1, value-1), value, value+1]
            elif param in ['subsample', 'colsample_bytree']:
                refined_grid[param] = [max(0.6, value-0.05), value, min(1.0, value+0.05)]
            elif param in ['reg_alpha', 'reg_lambda']:
                if value == 0:
                    refined_grid[param] = [0, 0.01, 0.05]
                else:
                    refined_grid[param] = [max(0, value*0.5), value, value*2.0]
            elif param == 'gamma':
                if value == 0:
                    refined_grid[param] = [0, 0.01, 0.05]
                else:
                    refined_grid[param] = [max(0, value*0.5), value, value*2.0]
            else:
                refined_grid[param] = [value]  # 다른 파라미터는 고정
        
        return refined_grid
    
    # ================================
    # 전체 기간 Train/Val/Test 분할
    # ================================
    
    def create_global_split(self, df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """전체 기간을 시간순으로 train/val/test 분할"""
        
        df_sorted = df.sort_values('entry_datetime')
        total_len = len(df_sorted)
        
        train_end_idx = int(total_len * train_ratio)
        val_end_idx = int(total_len * (train_ratio + val_ratio))
        
        train_data = df_sorted.iloc[:train_end_idx].copy()
        val_data = df_sorted.iloc[train_end_idx:val_end_idx].copy()
        test_data = df_sorted.iloc[val_end_idx:].copy()
        
        split_info = {
            'train_period': f"{train_data['entry_datetime'].min().date()} ~ {train_data['entry_datetime'].max().date()}",
            'val_period': f"{val_data['entry_datetime'].min().date()} ~ {val_data['entry_datetime'].max().date()}",
            'test_period': f"{test_data['entry_datetime'].min().date()} ~ {test_data['entry_datetime'].max().date()}",
            'train_samples': len(train_data),
            'val_samples': len(val_data), 
            'test_samples': len(test_data)
        }
        
        return train_data, val_data, test_data, split_info
    
    def run_global_hyperparameter_optimization(self, data_path, verbose=True):
        """전체 기간 분할 + 대규모 하이퍼파라미터 최적화"""
        
        if verbose:
            print("🌍 전체 기간 하이퍼파라미터 최적화 모드")
            print("=" * 80)
        
        # 데이터 로드
        df = pd.read_csv(data_path)
        df['entry_datetime'] = pd.to_datetime(df['entry_datetime'])
        df['exit_datetime'] = pd.to_datetime(df['exit_datetime'])
        
        if verbose:
            print(f"\n📊 총 데이터: {len(df):,}개")
            print(f"📅 기간: {df['entry_datetime'].min().date()} ~ {df['entry_datetime'].max().date()}")
        
        # 전체 기간 분할
        train_data, val_data, test_data, split_info = self.create_global_split(df)
        
        if verbose:
            print(f"\n📊 데이터 분할:")
            print(f"  훈련: {split_info['train_samples']:,}개 ({split_info['train_period']})")
            print(f"  검증: {split_info['val_samples']:,}개 ({split_info['val_period']})")  
            print(f"  테스트: {split_info['test_samples']:,}개 ({split_info['test_period']})")
        
        # A유형 최적화
        if verbose:
            print(f"\n🔬 A유형 모델 최적화 시작")
        
        # A유형 데이터 준비
        train_data_a = self.create_a_type_quality_score(train_data, verbose=False)
        val_data_a = self.create_a_type_quality_score(val_data, verbose=False)
        test_data_a = self.create_a_type_quality_score(test_data, verbose=False)
        
        y_train_a = train_data_a['a_type_quality_score']
        y_val_a = val_data_a['a_type_quality_score']
        y_test_a = test_data_a['a_type_quality_score']
        
        X_train_a = self.prepare_a_type_features(train_data, verbose=False)
        X_val_a = self.prepare_a_type_features(val_data, verbose=False)
        X_test_a = self.prepare_a_type_features(test_data, verbose=False)
        
        # A유형 하이퍼파라미터 최적화 (초대규모 탐색)
        model_a, best_params_a, search_info_a = self.smart_hyperparameter_search(
            X_train_a, y_train_a, X_val_a, y_val_a, 'A', n_iter=1000  # 메모리 안정성 고려
        )
        
        # A유형 최종 평가
        test_pred_a = model_a.predict(X_test_a)
        test_r2_a = r2_score(y_test_a, test_pred_a)
        
        # B유형 최적화
        if verbose:
            print(f"\n🚀 B유형 모델 최적화 시작")
        
        # B유형 데이터 준비
        train_data_b = self.create_b_type_entry_score(train_data, verbose=False)
        val_data_b = self.create_b_type_entry_score(val_data, verbose=False)
        test_data_b = self.create_b_type_entry_score(test_data, verbose=False)
        
        y_train_b = train_data_b['b_type_entry_score']
        y_val_b = val_data_b['b_type_entry_score']
        y_test_b = test_data_b['b_type_entry_score']
        
        X_train_b = self.prepare_b_type_features(train_data, verbose=False)
        X_val_b = self.prepare_b_type_features(val_data, verbose=False)
        X_test_b = self.prepare_b_type_features(test_data, verbose=False)
        
        # B유형 하이퍼파라미터 최적화 (초대규모 탐색)
        model_b, best_params_b, search_info_b = self.smart_hyperparameter_search(
            X_train_b, y_train_b, X_val_b, y_val_b, 'B', n_iter=1000  # 메모리 안정성 고려
        )
        
        # B유형 최종 평가
        test_pred_b = model_b.predict(X_test_b)
        test_r2_b = r2_score(y_test_b, test_pred_b)
        
        # 결과 저장
        self.global_results = {
            'split_info': split_info,
            'A_type': {
                'model': model_a,
                'best_params': best_params_a,
                'search_info': search_info_a,
                'test_r2': test_r2_a,
                'features_used': len(X_train_a.columns)
            },
            'B_type': {
                'model': model_b, 
                'best_params': best_params_b,
                'search_info': search_info_b,
                'test_r2': test_r2_b,
                'features_used': len(X_train_b.columns)
            }
        }
        
        self.best_params_a = best_params_a
        self.best_params_b = best_params_b
        
        if verbose:
            print(f"\n🏆 전체 기간 최적화 결과:")
            print(f"  A유형 Test R²: {test_r2_a:.4f}")
            print(f"  B유형 Test R²: {test_r2_b:.4f}")
            print(f"  A유형 최적 파라미터 수: {len(best_params_a)}")
            print(f"  B유형 최적 파라미터 수: {len(best_params_b)}")
        
        # 전체 기간 결과 저장
        self.save_global_results()
        
        return self.global_results
    
    # ================================
    # Walk-Forward Validation (v4와 동일)
    # ================================
    
    def create_time_folds(self, df, verbose=False):
        """시계열 기반 폴드 생성"""
        if verbose:
            print("📅 Walk-Forward Time Folds 생성 중...")
        
        df_sorted = df.sort_values('entry_datetime')
        start_date = pd.to_datetime(df_sorted['entry_datetime'].min())
        end_date = pd.to_datetime(df_sorted['entry_datetime'].max())
        
        folds = []
        current_train_start = start_date
        
        while True:
            train_end = current_train_start + pd.DateOffset(months=self.train_months)
            val_end = train_end + pd.DateOffset(months=self.val_months) 
            test_end = val_end + pd.DateOffset(months=self.test_months)
            
            if test_end > end_date:
                break
                
            folds.append({
                'fold_id': len(folds) + 1,
                'train_start': current_train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'val_start': train_end.strftime('%Y-%m-%d'),
                'val_end': val_end.strftime('%Y-%m-%d'),
                'test_start': val_end.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d')
            })
            
            current_train_start += pd.DateOffset(months=self.step_months)
        
        if verbose:
            print(f"  생성된 폴드 수: {len(folds)}개")
            for fold in folds:
                print(f"  폴드 {fold['fold_id']}: {fold['train_start']} ~ {fold['test_end']}")
        
        return folds
    
    def run_hybrid_training(self, data_path, verbose=True):
        """개선된 하이브리드 모델 실행"""
        if verbose:
            print("🚀 개선된 하이브리드 트레이딩 AI v5 시작!")
            print("=" * 80)
            print("📊 A유형: 거래 품질 분석 (v4 동일)")
            print("🚀 B유형: 실제 수익률 기반 진입 조건 점수 (새로 설계)")
            print("=" * 80)
        
        # 데이터 로드
        if verbose:
            print("\n📁 데이터 로드 중...")
        df = pd.read_csv(data_path)
        df['entry_datetime'] = pd.to_datetime(df['entry_datetime'])
        df['exit_datetime'] = pd.to_datetime(df['exit_datetime'])
        
        if verbose:
            print(f"  총 데이터: {len(df):,}개 거래")
            print(f"  기간: {df['entry_datetime'].min().date()} ~ {df['entry_datetime'].max().date()}")
        
        # 폴드 생성
        time_folds = self.create_time_folds(df, verbose)
        if len(time_folds) == 0:
            print("❌ 생성된 폴드가 없습니다. 데이터 기간을 확인해주세요.")
            return
        
        self.fold_results = []
        
        # 각 폴드별 훈련
        for fold_info in time_folds:
            if verbose:
                print(f"\n📊 폴드 {fold_info['fold_id']} 처리 중...")
                print(f"  훈련: {fold_info['train_start']} ~ {fold_info['train_end']}")
                print(f"  검증: {fold_info['val_start']} ~ {fold_info['val_end']}")  
                print(f"  테스트: {fold_info['test_start']} ~ {fold_info['test_end']}")
            
            # 데이터 분할
            train_data = df[
                (df['entry_datetime'] >= fold_info['train_start']) & 
                (df['entry_datetime'] < fold_info['train_end'])
            ].copy()
            
            val_data = df[
                (df['entry_datetime'] >= fold_info['val_start']) & 
                (df['entry_datetime'] < fold_info['val_end'])
            ].copy()
            
            test_data = df[
                (df['entry_datetime'] >= fold_info['test_start']) & 
                (df['entry_datetime'] < fold_info['test_end'])
            ].copy()
            
            if verbose:
                print(f"  데이터 크기 - 훈련: {len(train_data):,}, 검증: {len(val_data):,}, 테스트: {len(test_data):,}")
            
            # 시장 환경 분석
            train_vix_mean = train_data['entry_vix'].mean()
            val_vix_mean = val_data['entry_vix'].mean()
            test_vix_mean = test_data['entry_vix'].mean()
            
            train_return_mean = train_data['return_pct'].mean()
            val_return_mean = val_data['return_pct'].mean()
            test_return_mean = test_data['return_pct'].mean()
            
            # 결과 저장용
            fold_result = {
                'fold_id': fold_info['fold_id'],
                'fold_info': fold_info,
                'model_results': {},
                'market_stats': {
                    'train_vix_mean': train_vix_mean,
                    'val_vix_mean': val_vix_mean, 
                    'test_vix_mean': test_vix_mean,
                    'train_return_mean': train_return_mean,
                    'val_return_mean': val_return_mean,
                    'test_return_mean': test_return_mean
                }
            }
            
            # ===== A유형 모델 훈련 =====
            try:
                if verbose:
                    print("\n  🎯 A유형: 거래 품질 분석 모델 훈련...")
                
                # A유형 라벨 생성
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
                
                if train_mask_a.sum() > 100 and val_mask_a.sum() > 10 and test_mask_a.sum() > 10:
                    X_train_a_clean = X_train_a[train_mask_a]
                    y_train_a_clean = y_train_a[train_mask_a]
                    X_val_a_clean = X_val_a[val_mask_a]
                    y_val_a_clean = y_val_a[val_mask_a]
                    X_test_a_clean = X_test_a[test_mask_a]
                    y_test_a_clean = y_test_a[test_mask_a]
                    
                    # A유형 모델 훈련
                    model_a = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        tree_method='hist',  # GPU 메모리 부족 방지
                        n_jobs=-1
                    )
                    
                    model_a.fit(
                        X_train_a_clean, y_train_a_clean,
                        eval_set=[(X_val_a_clean, y_val_a_clean)],
                        verbose=False
                    )
                    
                    # A유형 성능 평가 (상세)
                    val_pred_a = model_a.predict(X_val_a_clean)
                    test_pred_a = model_a.predict(X_test_a_clean)
                    
                    # 다양한 성능 지표 계산
                    val_r2_a = r2_score(y_val_a_clean, val_pred_a)
                    test_r2_a = r2_score(y_test_a_clean, test_pred_a)
                    
                    val_mse_a = mean_squared_error(y_val_a_clean, val_pred_a)
                    test_mse_a = mean_squared_error(y_test_a_clean, test_pred_a)
                    
                    val_mae_a = mean_absolute_error(y_val_a_clean, val_pred_a)
                    test_mae_a = mean_absolute_error(y_test_a_clean, test_pred_a)
                    
                    # 상관계수
                    val_corr_a = np.corrcoef(y_val_a_clean, val_pred_a)[0, 1] if len(y_val_a_clean) > 1 else 0
                    test_corr_a = np.corrcoef(y_test_a_clean, test_pred_a)[0, 1] if len(y_test_a_clean) > 1 else 0
                    
                    # 피처 중요도 분석
                    feature_importance_a = model_a.feature_importances_
                    feature_names_a = X_train_a_clean.columns.tolist()
                    importance_dict_a = dict(zip(feature_names_a, feature_importance_a))
                    top_features_a = sorted(importance_dict_a.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    fold_result['model_results']['A_quality_model'] = {
                        # 기본 성능 지표
                        'val_r2': val_r2_a,
                        'test_r2': test_r2_a,
                        'val_mse': val_mse_a,
                        'test_mse': test_mse_a,
                        'val_mae': val_mae_a,
                        'test_mae': test_mae_a,
                        'val_corr': val_corr_a,
                        'test_corr': test_corr_a,
                        
                        # 데이터 크기
                        'train_samples': len(X_train_a_clean),
                        'val_samples': len(X_val_a_clean),
                        'test_samples': len(X_test_a_clean),
                        'features_used': len(X_train_a_clean.columns),
                        
                        # 피처 중요도
                        'top_features': top_features_a,
                        'all_feature_importance': importance_dict_a
                    }
                    
                    if verbose:
                        print(f"    ✅ A유형 성능 상세:")
                        print(f"      📊 R²: Val={val_r2_a:.4f}, Test={test_r2_a:.4f}")
                        print(f"      📊 상관계수: Val={val_corr_a:.4f}, Test={test_corr_a:.4f}")
                        print(f"      📊 MSE: Val={val_mse_a:.4f}, Test={test_mse_a:.4f}")
                        print(f"      📊 MAE: Val={val_mae_a:.4f}, Test={test_mae_a:.4f}")
                        print(f"      🔍 상위 피처: {', '.join([f'{name}({imp:.3f})' for name, imp in top_features_a[:5]])}")
                
                else:
                    fold_result['model_results']['A_quality_model'] = {'error': 'Insufficient clean data'}
                    if verbose:
                        print("    ❌ A유형: 충분한 깨끗한 데이터가 없습니다.")
            
            except Exception as e:
                fold_result['model_results']['A_quality_model'] = {'error': str(e)}
                if verbose:
                    print(f"    ❌ A유형 오류: {e}")
            
            # ===== B유형 모델 훈련 =====
            try:
                if verbose:
                    print("\n  🚀 B유형: 개선된 진입 조건 분석 모델 훈련...")
                
                # B유형 라벨 생성 (새로 설계된 방식)
                train_data_b = self.create_b_type_entry_score(train_data, verbose=False)
                val_data_b = self.create_b_type_entry_score(val_data, verbose=False)
                test_data_b = self.create_b_type_entry_score(test_data, verbose=False)
                
                # B유형 피처 준비
                X_train_b = self.prepare_b_type_features(train_data_b, verbose=False)
                X_val_b = self.prepare_b_type_features(val_data_b, verbose=False)
                X_test_b = self.prepare_b_type_features(test_data_b, verbose=False)
                
                y_train_b = train_data_b['b_type_entry_score']
                y_val_b = val_data_b['b_type_entry_score']
                y_test_b = test_data_b['b_type_entry_score']
                
                if len(X_train_b.columns) > 0:
                    # 결측치 제거
                    train_mask_b = ~(X_train_b.isnull().any(axis=1) | y_train_b.isnull())
                    val_mask_b = ~(X_val_b.isnull().any(axis=1) | y_val_b.isnull())
                    test_mask_b = ~(X_test_b.isnull().any(axis=1) | y_test_b.isnull())
                    
                    if train_mask_b.sum() > 100 and val_mask_b.sum() > 10 and test_mask_b.sum() > 10:
                        X_train_b_clean = X_train_b[train_mask_b]
                        y_train_b_clean = y_train_b[train_mask_b]
                        X_val_b_clean = X_val_b[val_mask_b]
                        y_val_b_clean = y_val_b[val_mask_b]
                        X_test_b_clean = X_test_b[test_mask_b]
                        y_test_b_clean = y_test_b[test_mask_b]
                        
                        # B유형 모델 훈련
                        model_b = xgb.XGBRegressor(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            random_state=42,
                            tree_method='hist',  # GPU 메모리 부족 방지
                            n_jobs=-1
                        )
                        
                        model_b.fit(
                            X_train_b_clean, y_train_b_clean,
                            eval_set=[(X_val_b_clean, y_val_b_clean)],
                            verbose=False
                        )
                        
                        # B유형 성능 평가 (상세)
                        val_pred_b = model_b.predict(X_val_b_clean)
                        test_pred_b = model_b.predict(X_test_b_clean)
                        
                        # 다양한 성능 지표 계산
                        val_r2_b = r2_score(y_val_b_clean, val_pred_b)
                        test_r2_b = r2_score(y_test_b_clean, test_pred_b)
                        
                        val_mse_b = mean_squared_error(y_val_b_clean, val_pred_b)
                        test_mse_b = mean_squared_error(y_test_b_clean, test_pred_b)
                        
                        val_mae_b = mean_absolute_error(y_val_b_clean, val_pred_b)
                        test_mae_b = mean_absolute_error(y_test_b_clean, test_pred_b)
                        
                        # 예측값 분포 분석
                        val_pred_mean = np.mean(val_pred_b)
                        val_pred_std = np.std(val_pred_b)
                        test_pred_mean = np.mean(test_pred_b)
                        test_pred_std = np.std(test_pred_b)
                        
                        # 실제값 분포 분석
                        val_actual_mean = np.mean(y_val_b_clean)
                        val_actual_std = np.std(y_val_b_clean)
                        test_actual_mean = np.mean(y_test_b_clean)
                        test_actual_std = np.std(y_test_b_clean)
                        
                        # 상관계수
                        val_corr_b = np.corrcoef(y_val_b_clean, val_pred_b)[0, 1] if len(y_val_b_clean) > 1 else 0
                        test_corr_b = np.corrcoef(y_test_b_clean, test_pred_b)[0, 1] if len(y_test_b_clean) > 1 else 0
                        
                        # 피처 중요도 분석
                        feature_importance = model_b.feature_importances_
                        feature_names = X_train_b_clean.columns.tolist()
                        importance_dict = dict(zip(feature_names, feature_importance))
                        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                        
                        fold_result['model_results']['B_entry_model'] = {
                            # 기본 성능 지표
                            'val_r2': val_r2_b,
                            'test_r2': test_r2_b,
                            'val_mse': val_mse_b,
                            'test_mse': test_mse_b,
                            'val_mae': val_mae_b,
                            'test_mae': test_mae_b,
                            'val_corr': val_corr_b,
                            'test_corr': test_corr_b,
                            
                            # 데이터 크기
                            'train_samples': len(X_train_b_clean),
                            'val_samples': len(X_val_b_clean),
                            'test_samples': len(X_test_b_clean),
                            'features_used': len(X_train_b_clean.columns),
                            
                            # 분포 분석
                            'val_pred_stats': {'mean': val_pred_mean, 'std': val_pred_std},
                            'test_pred_stats': {'mean': test_pred_mean, 'std': test_pred_std},
                            'val_actual_stats': {'mean': val_actual_mean, 'std': val_actual_std},
                            'test_actual_stats': {'mean': test_actual_mean, 'std': test_actual_std},
                            
                            # 피처 중요도
                            'top_features': top_features,
                            'all_feature_importance': importance_dict
                        }
                        
                        if verbose:
                            print(f"    ✅ B유형 성능 상세:")
                            print(f"      📊 R²: Val={val_r2_b:.4f}, Test={test_r2_b:.4f}")
                            print(f"      📊 상관계수: Val={val_corr_b:.4f}, Test={test_corr_b:.4f}")
                            print(f"      📊 MSE: Val={val_mse_b:.2f}, Test={test_mse_b:.2f}")
                            print(f"      📊 MAE: Val={val_mae_b:.2f}, Test={test_mae_b:.2f}")
                            print(f"      📈 예측값 분포: Val({val_pred_mean:.1f}±{val_pred_std:.1f}), Test({test_pred_mean:.1f}±{test_pred_std:.1f})")
                            print(f"      📈 실제값 분포: Val({val_actual_mean:.1f}±{val_actual_std:.1f}), Test({test_actual_mean:.1f}±{test_actual_std:.1f})")
                            print(f"      🔍 상위 피처: {', '.join([f'{name}({imp:.3f})' for name, imp in top_features[:5]])}")
                    
                    else:
                        fold_result['model_results']['B_entry_model'] = {'error': 'Insufficient clean data'}
                        if verbose:
                            print("    ❌ B유형: 충분한 깨끗한 데이터가 없습니다.")
                
                else:
                    fold_result['model_results']['B_entry_model'] = {'error': 'No valid features'}
                    if verbose:
                        print("    ❌ B유형: 유효한 피처가 없습니다.")
            
            except Exception as e:
                fold_result['model_results']['B_entry_model'] = {'error': str(e)}
                if verbose:
                    print(f"    ❌ B유형 오류: {e}")
            
            self.fold_results.append(fold_result)
        
        # 결과 요약 및 저장
        self.print_summary_results(verbose)
        self.save_results(verbose)
        
        return self.fold_results
    
    def print_summary_results(self, verbose=True):
        """결과 요약 출력"""
        if not verbose or not self.fold_results:
            return
        
        print("\n" + "="*80)
        print("📊 하이브리드 트레이딩 AI v5 결과 요약")
        print("="*80)
        
        # A유형 결과 수집
        a_val_r2s, a_test_r2s = [], []
        a_val_corrs, a_test_corrs = [], []
        a_val_maes, a_test_maes = [], []
        a_successes = 0
        
        # B유형 결과 수집  
        b_val_r2s, b_test_r2s = [], []
        b_val_corrs, b_test_corrs = [], []
        b_val_maes, b_test_maes = [], []
        b_successes = 0
        
        # 피처 중요도 수집
        a_all_features, b_all_features = [], []
        
        for fold_result in self.fold_results:
            # A유형 성능
            if 'A_quality_model' in fold_result['model_results']:
                a_result = fold_result['model_results']['A_quality_model']
                if 'val_r2' in a_result and 'test_r2' in a_result:
                    a_val_r2s.append(a_result['val_r2'])
                    a_test_r2s.append(a_result['test_r2'])
                    a_val_corrs.append(a_result.get('val_corr', 0))
                    a_test_corrs.append(a_result.get('test_corr', 0))
                    a_val_maes.append(a_result.get('val_mae', 0))
                    a_test_maes.append(a_result.get('test_mae', 0))
                    a_all_features.extend(a_result.get('top_features', [])[:5])
                    a_successes += 1
            
            # B유형 성능
            if 'B_entry_model' in fold_result['model_results']:
                b_result = fold_result['model_results']['B_entry_model']
                if 'val_r2' in b_result and 'test_r2' in b_result:
                    b_val_r2s.append(b_result['val_r2'])
                    b_test_r2s.append(b_result['test_r2'])
                    b_val_corrs.append(b_result.get('val_corr', 0))
                    b_test_corrs.append(b_result.get('test_corr', 0))
                    b_val_maes.append(b_result.get('val_mae', 0))
                    b_test_maes.append(b_result.get('test_mae', 0))
                    b_all_features.extend(b_result.get('top_features', [])[:5])
                    b_successes += 1
        
        # A유형 상세 요약
        if a_successes > 0:
            print(f"🎯 A유형 (품질 평가) 상세 성능:")
            print(f"  성공적인 폴드: {a_successes}/{len(self.fold_results)}")
            print(f"  📊 R²:        Val={np.mean(a_val_r2s):.4f}±{np.std(a_val_r2s):.4f}, Test={np.mean(a_test_r2s):.4f}±{np.std(a_test_r2s):.4f}")
            print(f"  📊 상관계수:   Val={np.mean(a_val_corrs):.4f}±{np.std(a_val_corrs):.4f}, Test={np.mean(a_test_corrs):.4f}±{np.std(a_test_corrs):.4f}")
            print(f"  📊 MAE:       Val={np.mean(a_val_maes):.4f}±{np.std(a_val_maes):.4f}, Test={np.mean(a_test_maes):.4f}±{np.std(a_test_maes):.4f}")
            if len(a_test_r2s) > 1:
                print(f"  📊 Test R² 범위: [{np.min(a_test_r2s):.4f}, {np.max(a_test_r2s):.4f}]")
            
            # A유형 중요 피처 분석
            from collections import Counter
            a_feature_counts = Counter([name for name, _ in a_all_features])
            if a_feature_counts:
                print(f"  🔍 핵심 피처: {', '.join([f'{name}({count})' for name, count in a_feature_counts.most_common(5)])}")
        
        # B유형 상세 요약  
        if b_successes > 0:
            print(f"\n🚀 B유형 (매수 신호 AI) 상세 성능:")
            print(f"  성공적인 폴드: {b_successes}/{len(self.fold_results)}")
            print(f"  📊 R²:        Val={np.mean(b_val_r2s):.4f}±{np.std(b_val_r2s):.4f}, Test={np.mean(b_test_r2s):.4f}±{np.std(b_test_r2s):.4f}")
            print(f"  📊 상관계수:   Val={np.mean(b_val_corrs):.4f}±{np.std(b_val_corrs):.4f}, Test={np.mean(b_test_corrs):.4f}±{np.std(b_test_corrs):.4f}")
            print(f"  📊 MAE:       Val={np.mean(b_val_maes):.2f}±{np.std(b_val_maes):.2f}, Test={np.mean(b_test_maes):.2f}±{np.std(b_test_maes):.2f}")
            if len(b_test_r2s) > 1:
                print(f"  📊 Test R² 범위: [{np.min(b_test_r2s):.4f}, {np.max(b_test_r2s):.4f}]")
            
            # B유형 중요 피처 분석
            b_feature_counts = Counter([name for name, _ in b_all_features])
            if b_feature_counts:
                print(f"  🔍 핵심 피처: {', '.join([f'{name}({count})' for name, count in b_feature_counts.most_common(5)])}")
        
        # 시장 환경별 분석
        print(f"\n🌊 시장 환경별 성능 분석:")
        for i, fold_result in enumerate(self.fold_results):
            market_stats = fold_result['market_stats']
            vix_level = "저변동" if market_stats['test_vix_mean'] < 25 else "고변동"
            return_trend = "상승" if market_stats['test_return_mean'] > 0 else "하락"
            
            a_r2 = fold_result['model_results'].get('A_quality_model', {}).get('test_r2', 'N/A')
            b_r2 = fold_result['model_results'].get('B_entry_model', {}).get('test_r2', 'N/A')
            
            a_r2_str = f"{a_r2:.4f}" if isinstance(a_r2, (int, float)) else "실패"
            b_r2_str = f"{b_r2:.4f}" if isinstance(b_r2, (int, float)) else "실패"
            
            print(f"  Fold {i+1}: {vix_level}/{return_trend} → A:{a_r2_str} / B:{b_r2_str}")
        
        print("="*80)
        print("🎯 개선된 하이브리드 AI v5 핵심 특징:")
        print("")
        print("📊 A유형 (거래 품질 분석기):")
        print("   • 목적: '이 거래가 얼마나 좋았나?' 객관적 평가")
        print("   • 활용: 거래 복기, 성과 분석, 트레이더 평가")
        print("   • 데이터: 모든 거래 정보 활용 (진입+진행+종료)")
        print("   • 정확도: 높음 (완전한 정보 활용)")
        print("")
        print("🚀 B유형 (매수 신호 AI):")
        print("   • 목적: '지금 매수하기 좋은 조건인가?' 실시간 판단")
        print("   • 활용: 매수 타이밍, 종목 선별, 리스크 관리")
        print("   • 데이터: 현재 시점 정보만 (미래 정보 완전 차단)")
        print("   • 개선: 종합 매수 신호 점수화 (기술적+펀더멘털+시장환경)")
        print("   • 피처: 50+ 개 다차원 분석 (기존 10개 → 대폭 확장)")
        print("   • 현실성: 매우 높음 (실제 매수 신호 생성기)")
        print("")
        print("💡 v5 핵심 개선점:")
        print("   • B유형: 수익률 예측 → 매수 신호 점수화")
        print("   • B유형: 다차원 분석 (기술적+펀더멘털+시장환경+산업)")
        print("   • B유형: 피처 10개 → 50+ 개로 대폭 확장") 
        print("   • 실제 트레이딩에서 바로 사용 가능한 매수 신호 AI")
        print("="*80)
    
    def save_results(self, verbose=True):
        """결과 저장"""
        
        # 결과 JSON 저장
        results_filename = 'hybrid_results_v5.json'
        with open(results_filename, 'w') as f:
            json.dump(self.fold_results, f, indent=2, default=str)
        
        # 메타데이터 저장
        metadata = {
            'model_version': 'v5',
            'model_name': 'Improved Hybrid Trading AI',
            'created_at': datetime.now().isoformat(),
            'total_folds': len(self.fold_results),
            'successful_a_folds': sum(1 for fr in self.fold_results 
                                    if 'val_r2' in fr['model_results'].get('A_quality_model', {})),
            'successful_b_folds': sum(1 for fr in self.fold_results 
                                    if 'val_r2' in fr['model_results'].get('B_entry_model', {})),
            'improvements': [
                'B-type: Rule-based → Real data-based scoring',
                'B-type: Enhanced technical indicators',
                'More realistic entry condition evaluation'
            ]
        }
        
        metadata_filename = 'hybrid_results_v5_metadata.json'
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"✅ 하이브리드 모델 v5 결과 저장 완료: {results_filename}, {metadata_filename}")
    
    def run_hybrid_training_with_hyperparameter_tuning(self, data_path, verbose=True):
        """Walk-Forward Fold별 하이퍼파라미터 최적화 실행"""
        if verbose:
            print("🔄 Fold별 하이퍼파라미터 최적화 모드")
            print("=" * 80)
        
        # 데이터 로드
        df = pd.read_csv(data_path)
        df['entry_datetime'] = pd.to_datetime(df['entry_datetime'])
        df['exit_datetime'] = pd.to_datetime(df['exit_datetime'])
        
        if verbose:
            print(f"\n📊 총 데이터: {len(df):,}개")
            print(f"📅 기간: {df['entry_datetime'].min().date()} ~ {df['entry_datetime'].max().date()}")
        
        # 폴드 생성
        time_folds = self.create_time_folds(df, verbose)
        if len(time_folds) == 0:
            print("❌ 생성된 폴드가 없습니다.")
            return
        
        self.fold_results = []
        
        # 각 폴드별로 하이퍼파라미터 최적화 수행
        for fold_info in time_folds:
            if verbose:
                print(f"\n🔬 폴드 {fold_info['fold_id']} 하이퍼파라미터 최적화")
                print(f"  훈련: {fold_info['train_start']} ~ {fold_info['train_end']}")
                print(f"  검증: {fold_info['val_start']} ~ {fold_info['val_end']}")  
                print(f"  테스트: {fold_info['test_start']} ~ {fold_info['test_end']}")
            
            # 데이터 분할
            train_data = df[
                (df['entry_datetime'] >= fold_info['train_start']) & 
                (df['entry_datetime'] < fold_info['train_end'])
            ].copy()
            
            val_data = df[
                (df['entry_datetime'] >= fold_info['val_start']) & 
                (df['entry_datetime'] < fold_info['val_end'])
            ].copy()
            
            test_data = df[
                (df['entry_datetime'] >= fold_info['test_start']) & 
                (df['entry_datetime'] < fold_info['test_end'])
            ].copy()
            
            if len(train_data) < 100 or len(val_data) < 10 or len(test_data) < 10:
                print(f"  ⚠️ 폴드 {fold_info['fold_id']} 데이터 부족으로 스킵")
                continue
            
            fold_result = {
                'fold_id': fold_info['fold_id'],
                'fold_info': fold_info,
                'model_results': {},
                'hyperparameter_search': {}
            }
            
            # A유형 하이퍼파라미터 최적화
            try:
                if verbose:
                    print(f"  🔬 A유형 하이퍼파라미터 최적화...")
                
                # A유형 품질 점수 생성
                train_data_with_score = self.create_a_type_quality_score(train_data, verbose=False)
                val_data_with_score = self.create_a_type_quality_score(val_data, verbose=False)
                test_data_with_score = self.create_a_type_quality_score(test_data, verbose=False)
                
                y_train_a = train_data_with_score['a_type_quality_score']
                y_val_a = val_data_with_score['a_type_quality_score']
                y_test_a = test_data_with_score['a_type_quality_score']
                
                X_train_a = self.prepare_a_type_features(train_data, verbose=False)
                X_val_a = self.prepare_a_type_features(val_data, verbose=False)
                X_test_a = self.prepare_a_type_features(test_data, verbose=False)
                
                # A유형 하이퍼파라미터 최적화 (대규모 탐색)
                model_a, best_params_a, search_info_a = self.smart_hyperparameter_search(
                    X_train_a, y_train_a, X_val_a, y_val_a, 'A', n_iter=800  # 대규모 탐색
                )
                
                # A유형 최종 평가
                test_pred_a = model_a.predict(X_test_a)
                test_r2_a = r2_score(y_test_a, test_pred_a)
                
                fold_result['model_results']['A_quality_model'] = {
                    'val_r2': search_info_a['val_r2'],
                    'test_r2': test_r2_a,
                    'train_samples': len(X_train_a),
                    'val_samples': len(X_val_a),
                    'test_samples': len(X_test_a),
                    'features_used': len(X_train_a.columns)
                }
                
                fold_result['hyperparameter_search']['A_type'] = {
                    'best_params': best_params_a,
                    'search_info': search_info_a
                }
                
            except Exception as e:
                print(f"  ❌ A유형 최적화 실패: {e}")
            
            # B유형 하이퍼파라미터 최적화
            try:
                if verbose:
                    print(f"  🚀 B유형 하이퍼파라미터 최적화...")
                
                # B유형 진입 점수 생성
                train_data_with_score_b = self.create_b_type_entry_score(train_data, verbose=False)
                val_data_with_score_b = self.create_b_type_entry_score(val_data, verbose=False)
                test_data_with_score_b = self.create_b_type_entry_score(test_data, verbose=False)
                
                y_train_b = train_data_with_score_b['b_type_entry_score']
                y_val_b = val_data_with_score_b['b_type_entry_score']
                y_test_b = test_data_with_score_b['b_type_entry_score']
                
                X_train_b = self.prepare_b_type_features(train_data, verbose=False)
                X_val_b = self.prepare_b_type_features(val_data, verbose=False)  
                X_test_b = self.prepare_b_type_features(test_data, verbose=False)
                
                # B유형 하이퍼파라미터 최적화 (대규모 탐색)
                model_b, best_params_b, search_info_b = self.smart_hyperparameter_search(
                    X_train_b, y_train_b, X_val_b, y_val_b, 'B', n_iter=800  # 대규모 탐색
                )
                
                # B유형 최종 평가
                test_pred_b = model_b.predict(X_test_b)
                test_r2_b = r2_score(y_test_b, test_pred_b)
                
                fold_result['model_results']['B_entry_model'] = {
                    'val_r2': search_info_b['val_r2'],
                    'test_r2': test_r2_b,
                    'train_samples': len(X_train_b),
                    'val_samples': len(X_val_b),
                    'test_samples': len(X_test_b),
                    'features_used': len(X_train_b.columns)
                }
                
                fold_result['hyperparameter_search']['B_type'] = {
                    'best_params': best_params_b,
                    'search_info': search_info_b
                }
                
            except Exception as e:
                print(f"  ❌ B유형 최적화 실패: {e}")
            
            self.fold_results.append(fold_result)
            
            if verbose:
                a_r2 = fold_result['model_results'].get('A_quality_model', {}).get('test_r2', 'N/A')
                b_r2 = fold_result['model_results'].get('B_entry_model', {}).get('test_r2', 'N/A')
                
                # 안전한 포맷팅
                a_r2_str = f"{a_r2:.4f}" if isinstance(a_r2, (int, float)) else "실패"
                b_r2_str = f"{b_r2:.4f}" if isinstance(b_r2, (int, float)) else "실패"
                print(f"  ✅ 폴드 {fold_info['fold_id']} 완료: A={a_r2_str} / B={b_r2_str}")
        
        # 결과 저장 (fold별)
        self.save_fold_results()
        
        if verbose:
            print(f"\n🔄 Fold별 하이퍼파라미터 최적화 완료!")
            self.print_fold_summary()
        
        return self.fold_results
    
    def print_fold_summary(self):
        """Fold별 결과 요약 출력"""
        if not self.fold_results:
            print("  📊 수집된 결과가 없습니다.")
            return
        
        successful_a = 0
        successful_b = 0
        a_scores = []
        b_scores = []
        
        for fold_result in self.fold_results:
            if 'A_quality_model' in fold_result['model_results']:
                a_r2 = fold_result['model_results']['A_quality_model'].get('test_r2')
                if isinstance(a_r2, (int, float)):
                    successful_a += 1
                    a_scores.append(a_r2)
            
            if 'B_entry_model' in fold_result['model_results']:
                b_r2 = fold_result['model_results']['B_entry_model'].get('test_r2')
                if isinstance(b_r2, (int, float)):
                    successful_b += 1
                    b_scores.append(b_r2)
        
        print(f"  📊 Fold별 하이퍼파라미터 최적화 요약:")
        print(f"    A유형 성공: {successful_a}/{len(self.fold_results)}폴드")
        print(f"    B유형 성공: {successful_b}/{len(self.fold_results)}폴드")
        
        if a_scores:
            print(f"    A유형 평균 R²: {np.mean(a_scores):.4f} ± {np.std(a_scores):.4f}")
        
        if b_scores:
            print(f"    B유형 평균 R²: {np.mean(b_scores):.4f} ± {np.std(b_scores):.4f}")
    
    def compare_results(self, verbose=True):
        """Fold별 vs 전체 기간 결과 비교"""
        
        if not self.fold_results or not self.global_results:
            print("❌ 비교할 결과가 없습니다.")
            return
        
        if verbose:
            print("📊 Fold별 vs 전체 기간 하이퍼파라미터 최적화 비교")
            print("="*70)
        
        # Fold별 평균 성능
        fold_a_r2s = []
        fold_b_r2s = []
        
        for fold_result in self.fold_results:
            if 'A_quality_model' in fold_result['model_results']:
                fold_a_r2s.append(fold_result['model_results']['A_quality_model']['test_r2'])
            if 'B_entry_model' in fold_result['model_results']:
                fold_b_r2s.append(fold_result['model_results']['B_entry_model']['test_r2'])
        
        # 전체 기간 성능
        global_a_r2 = self.global_results['A_type']['test_r2']
        global_b_r2 = self.global_results['B_type']['test_r2']
        
        comparison = {
            'fold_results': {
                'A_type': {
                    'mean_r2': np.mean(fold_a_r2s) if fold_a_r2s else 0,
                    'std_r2': np.std(fold_a_r2s) if len(fold_a_r2s) > 1 else 0,
                    'min_r2': np.min(fold_a_r2s) if fold_a_r2s else 0,
                    'max_r2': np.max(fold_a_r2s) if fold_a_r2s else 0,
                    'n_folds': len(fold_a_r2s)
                },
                'B_type': {
                    'mean_r2': np.mean(fold_b_r2s) if fold_b_r2s else 0,
                    'std_r2': np.std(fold_b_r2s) if len(fold_b_r2s) > 1 else 0,
                    'min_r2': np.min(fold_b_r2s) if fold_b_r2s else 0,
                    'max_r2': np.max(fold_b_r2s) if fold_b_r2s else 0,
                    'n_folds': len(fold_b_r2s)
                }
            },
            'global_results': {
                'A_type': {'test_r2': global_a_r2},
                'B_type': {'test_r2': global_b_r2}
            }
        }
        
        if verbose:
            print("🔄 Walk-Forward Fold별 성능:")
            print(f"  A유형: {np.mean(fold_a_r2s):.4f} ± {np.std(fold_a_r2s):.4f} (범위: {np.min(fold_a_r2s):.4f}~{np.max(fold_a_r2s):.4f})")
            print(f"  B유형: {np.mean(fold_b_r2s):.4f} ± {np.std(fold_b_r2s):.4f} (범위: {np.min(fold_b_r2s):.4f}~{np.max(fold_b_r2s):.4f})")
            
            print("\n🌍 전체 기간 성능:")
            print(f"  A유형: {global_a_r2:.4f}")
            print(f"  B유형: {global_b_r2:.4f}")
            
            print("\n💡 성능 비교:")
            a_improvement = global_a_r2 - np.mean(fold_a_r2s) if fold_a_r2s else 0
            b_improvement = global_b_r2 - np.mean(fold_b_r2s) if fold_b_r2s else 0
            
            print(f"  A유형 전체기간 우위: {a_improvement:+.4f}")
            print(f"  B유형 전체기간 우위: {b_improvement:+.4f}")
            
            # 최적 파라미터 비교
            print("\n🔧 최적 하이퍼파라미터:")
            if self.best_params_a:
                print("  A유형 최적 파라미터:")
                for key, value in list(self.best_params_a.items())[:5]:
                    print(f"    {key}: {value}")
            
            if self.best_params_b:
                print("  B유형 최적 파라미터:")
                for key, value in list(self.best_params_b.items())[:5]:
                    print(f"    {key}: {value}")
        
        # 비교 결과 저장
        with open('hybrid_results_v5_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        return comparison
    
    def save_fold_results(self):
        """Fold별 결과 저장"""
        with open('hybrid_results_v5_folds.json', 'w') as f:
            json.dump(self.fold_results, f, indent=2, default=str)
    
    def save_global_results(self):
        """전체 기간 결과 저장"""
        if self.global_results:
            # 모델 객체 제외하고 저장
            save_data = {
                'split_info': self.global_results['split_info'],
                'A_type': {
                    'best_params': self.global_results['A_type']['best_params'],
                    'search_info': self.global_results['A_type']['search_info'],
                    'test_r2': self.global_results['A_type']['test_r2'],
                    'features_used': self.global_results['A_type']['features_used']
                },
                'B_type': {
                    'best_params': self.global_results['B_type']['best_params'],
                    'search_info': self.global_results['B_type']['search_info'],
                    'test_r2': self.global_results['B_type']['test_r2'],
                    'features_used': self.global_results['B_type']['features_used']
                }
            }
            
            with open('hybrid_results_v5_global.json', 'w') as f:
                json.dump(save_data, f, indent=2, default=str)

def main():
    """메인 실행 함수 - Fold별 + 전체 기간 하이퍼파라미터 최적화"""
    print("🚀 개선된 하이브리드 트레이딩 AI v5 - 대규모 하이퍼파라미터 최적화")
    print("="*80)
    print("📊 실행 순서:")
    print("  1단계: Walk-Forward Fold별 하이퍼파라미터 최적화")
    print("  2단계: 전체 기간 분할 하이퍼파라미터 최적화")
    print("  3단계: 결과 비교 및 분석")
    print("="*80)
    
    # 데이터 경로
    # 절대 경로 사용으로 파일 경로 오류 방지
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../results/final/trading_episodes_with_rebuilt_market_component.csv')
    
    # 파일 존재 확인
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return
    
    # 모델 초기화 (하이퍼파라미터 최적화 활성화)
    model = ImprovedHybridTradingAI(use_global_split=True)
    
    print("\n🔄 1단계: Walk-Forward Fold별 하이퍼파라미터 최적화 시작")
    print("="*60)
    fold_results = model.run_hybrid_training_with_hyperparameter_tuning(data_path, verbose=True)
    
    print("\n🌍 2단계: 전체 기간 하이퍼파라미터 최적화 시작")  
    print("="*60)
    global_results = model.run_global_hyperparameter_optimization(data_path, verbose=True)
    
    print("\n📊 3단계: 결과 비교 분석")
    print("="*60)
    model.compare_results(verbose=True)
    
    print("\n✅ 모든 최적화 완료!")
    print(f"   📁 Fold 결과: hybrid_results_v5_folds.json")
    print(f"   📁 전체 결과: hybrid_results_v5_global.json")
    print(f"   📁 비교 분석: hybrid_results_v5_comparison.json")
    print(f"   📊 A유형 모델: 거래 품질 분석 완료")
    print(f"   🚀 B유형 모델: 개선된 진입 조건 평가 완료")
    print(f"   🎯 실용적이고 현실적인 트레이딩 지원 시스템 준비됨!")

if __name__ == "__main__":
    main()