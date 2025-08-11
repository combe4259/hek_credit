# config.py
from dataclasses import dataclass
from typing import Dict

# 기본 설정
SYMBOL = "NVDA"
PERIOD = "1y"  # 1년 데이터
INITIAL_CAPITAL = 100000  # 초기 자금 10만 달러

# 매매 임계값
PROFIT_THRESHOLD_1 = 0.05  # 5% 수익
PROFIT_THRESHOLD_2 = 0.20  # 20% 수익
LOSS_THRESHOLD_1 = -0.03   # -3% 손실
LOSS_THRESHOLD_2 = -0.10   # -10% 손실
VOLATILITY_THRESHOLD = 0.05  # 5% 일일 변동성

# 기술적 지표 설정
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
SMA_SHORT = 3
SMA_MEDIUM = 5
SMA_LONG = 10  # 60에서 10으로 변경하여 더 많은 거래 데이터 생성
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# 거래량 설정
VOLUME_SPIKE_THRESHOLD = 1.5  # 50% 이상 거래량 증가
HIGH_VOLUME_THRESHOLD = 2.0   # 2배 이상 거래량 급증

# 갭 설정
GAP_UP_THRESHOLD = 0.02      # 2% 이상 갭업
GAP_DOWN_THRESHOLD = -0.02   # 2% 이상 갭다운

# 캔들 패턴 설정
LONG_CANDLE_THRESHOLD = 0.03  # 3% 이상 몸통
DOJI_THRESHOLD = 0.005        # 0.5% 이하는 도지

# 출력 파일 경로
OUTPUT_DIR = 'output'
OUTPUT_CSV = f'{OUTPUT_DIR}/trading_patterns.csv'
OUTPUT_CHART = f'{OUTPUT_DIR}/pattern_analysis.png'

# 8가지 핵심 패턴 이름 (영문/한글)
PATTERN_NAMES = [
    'technical_indicator_reliance',  # 기술적 지표 의존도
    'chart_pattern_recognition',     # 차트 패턴 인식
    'volume_reaction',              # 거래량 반응
    'candle_analysis',              # 캔들 분석
    'profit_taking_tendency',       # 수익 실현 성향
    'stop_loss_tendency',           # 손절 성향
    'volatility_reaction',          # 변동성 반응
    'time_based_trading'            # 시간대별 거래
]

PATTERN_KOREAN_NAMES = [
    '기술적지표 의존도',
    '차트패턴 인식',
    '거래량 반응',
    '캔들 분석',
    '수익 실현 성향',
    '손절 성향',
    '변동성 반응',
    '시간대별 거래'
]

@dataclass
class InvestorProfile:
    """투자자 프로필 데이터 클래스"""
    name: str
    profit_taking: float        # 0-1: 0=빠른익절, 1=장기보유
    stop_loss: float           # 0-1: 0=빠른손절, 1=참고버티기
    volatility_reaction: float  # 0-1: 0=패닉매매, 1=침착함
    technical_reliance: float   # 0-1: 0=지표무시, 1=지표의존
    time_sensitivity: float     # 0-1: 0=충동적, 1=계획적
    pattern_sensitivity: float  # 0-1: 0=패턴무시, 1=패턴민감
    volume_sensitivity: float   # 0-1: 0=거래량무시, 1=거래량추종
    candle_sensitivity: float   # 0-1: 0=캔들무시, 1=캔들민감

# 5가지 투자자 프로필 정의
INVESTOR_PROFILES = [
    InvestorProfile(
        name='Conservative',
        profit_taking=0.2,      # 빠른 익절
        stop_loss=0.8,          # 빠른 손절
        volatility_reaction=0.9, # 침착함
        technical_reliance=0.8,  # 지표 의존
        time_sensitivity=0.8,    # 계획적
        pattern_sensitivity=0.7, # 패턴 인식
        volume_sensitivity=0.6,  # 보통 거래량 반응
        candle_sensitivity=0.7   # 캔들 패턴 인식
    ),
    InvestorProfile(
        name='Aggressive',
        profit_taking=0.8,      # 장기 보유
        stop_loss=0.3,          # 버티기
        volatility_reaction=0.3, # 패닉 성향
        technical_reliance=0.4,  # 직감 의존
        time_sensitivity=0.3,    # 충동적
        pattern_sensitivity=0.4, # 패턴 무시
        volume_sensitivity=0.8,  # 거래량 추종
        candle_sensitivity=0.5   # 보통 캔들 반응
    ),
    InvestorProfile(
        name='Technical_Trader',
        profit_taking=0.6,
        stop_loss=0.7,
        volatility_reaction=0.7,
        technical_reliance=0.9,  # 지표 완전 의존
        time_sensitivity=0.8,    # 계획적
        pattern_sensitivity=0.9, # 패턴 완전 인식
        volume_sensitivity=0.7,
        candle_sensitivity=0.8   # 캔들 분석 중시
    ),
    InvestorProfile(
        name='Momentum_Trader',
        profit_taking=0.4,
        stop_loss=0.4,
        volatility_reaction=0.2, # 변동성 따라가기
        technical_reliance=0.6,
        time_sensitivity=0.4,    # 반충동적
        pattern_sensitivity=0.5,
        volume_sensitivity=0.9,  # 거래량 완전 추종
        candle_sensitivity=0.6
    ),
    InvestorProfile(
        name='Swing_Trader',
        profit_taking=0.7,
        stop_loss=0.6,
        volatility_reaction=0.8,
        technical_reliance=0.7,
        time_sensitivity=0.7,
        pattern_sensitivity=0.8,
        volume_sensitivity=0.5,
        candle_sensitivity=0.7
    )
]

# 더 많은 프로필 자동 생성 (기존 5개 + 추가 45개 = 총 50개)
import numpy as np

# 시드 고정 (재현성 확보)
RANDOM_SEED = 100
np.random.seed(RANDOM_SEED)

# 각 매개변수의 변형을 만들어 추가 프로필 생성
for i in range(45):
    # 랜덤하게 값을 조정하여 다양한 프로필 생성
    np.random.seed(RANDOM_SEED + i)  # 재현 가능한 랜덤값
    
    base_profile = np.random.choice(INVESTOR_PROFILES[:5])  # 기존 5개 프로필 중 하나를 기반으로
    
    # 각 속성을 ±0.2 범위에서 조정
    new_profile = InvestorProfile(
        name=f'{base_profile.name}_variant_{i}',
        profit_taking=max(0, min(1, base_profile.profit_taking + np.random.uniform(-0.2, 0.2))),
        stop_loss=max(0, min(1, base_profile.stop_loss + np.random.uniform(-0.2, 0.2))),
        volatility_reaction=max(0, min(1, base_profile.volatility_reaction + np.random.uniform(-0.2, 0.2))),
        technical_reliance=max(0, min(1, base_profile.technical_reliance + np.random.uniform(-0.2, 0.2))),
        time_sensitivity=max(0, min(1, base_profile.time_sensitivity + np.random.uniform(-0.2, 0.2))),
        pattern_sensitivity=max(0, min(1, base_profile.pattern_sensitivity + np.random.uniform(-0.2, 0.2))),
        volume_sensitivity=max(0, min(1, base_profile.volume_sensitivity + np.random.uniform(-0.2, 0.2))),
        candle_sensitivity=max(0, min(1, base_profile.candle_sensitivity + np.random.uniform(-0.2, 0.2)))
    )
    
    INVESTOR_PROFILES.append(new_profile)

# 패턴 분석용 설정
PATTERN_NAMES = [
    'profit_taking_tendency',
    'stop_loss_tendency', 
    'volatility_reaction',
    'time_based_trading',
    'technical_indicator_reliance',
    'chart_pattern_recognition',
    'volume_reaction',
    'candle_analysis'
]

PATTERN_KOREAN_NAMES = [
    '수익실현 성향',
    '손절 성향',
    '변동성 반응', 
    '시간대별 매매',
    '기술지표 의존',
    '차트패턴 인식',
    '거래량 반응',
    '캔들 분석'
]

# 파일 출력 설정 (위에서 이미 정의됨, 중복 제거)