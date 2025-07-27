# patterns/chart_patterns.py
import pandas as pd
import numpy as np
from config import *

class ChartPatternIdentifier:
    """차트 패턴 인식 클래스"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def identify_all_patterns(self) -> pd.DataFrame:
        """모든 차트 패턴 인식"""
        df = self.data.copy()
        
        # 캔들스틱 패턴
        self._identify_candle_patterns(df)
        
        # 갭 패턴
        self._identify_gap_patterns(df)
        
        # 지지/저항 패턴
        self._identify_support_resistance(df)
        
        # 추세 패턴
        self._identify_trend_patterns(df)
        
        return df
    
    def _identify_candle_patterns(self, df: pd.DataFrame):
        """캔들스틱 패턴 인식"""
        # 캔들 몸통 크기
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Body_Ratio'] = df['Body_Size'] / df['Close']
        
        # 장대양봉/음봉
        df['Is_Long_Green'] = (
            (df['Close'] > df['Open']) & 
            (df['Body_Ratio'] > LONG_CANDLE_THRESHOLD)
        )
        df['Is_Long_Red'] = (
            (df['Close'] < df['Open']) & 
            (df['Body_Ratio'] > LONG_CANDLE_THRESHOLD)
        )
        
        # 도지 캔들
        df['Is_Doji'] = df['Body_Ratio'] < DOJI_THRESHOLD
        
        # 해머/행맨
        df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['Shadow_Ratio'] = df['Lower_Shadow'] / (df['High'] - df['Low'])
        
        df['Is_Hammer'] = (
            (df['Shadow_Ratio'] > 0.6) & 
            (df['Body_Ratio'] < 0.3) &
            (df['Close'] > df['Open'])
        )
        
        df['Is_Hanging_Man'] = (
            (df['Shadow_Ratio'] > 0.6) & 
            (df['Body_Ratio'] > 0.3) &
            (df['Close'] < df['Open'])
        )
    
    def _identify_gap_patterns(self, df: pd.DataFrame):
        """갭 패턴 인식"""
        # 갭 크기 계산
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Gap'] = df['Gap'].fillna(0)
        
        # 갭업/갭다운
        df['Is_Gap_Up'] = df['Gap'] > GAP_UP_THRESHOLD
        df['Is_Gap_Down'] = df['Gap'] < GAP_DOWN_THRESHOLD
        
        # 갭 강도
        df['Gap_Strength'] = abs(df['Gap'])
    
    def _identify_support_resistance(self, df: pd.DataFrame):
        """지지/저항선 인식"""
        window = 10
        
        # 국소 고점/저점 찾기
        df['Local_High'] = df['High'].rolling(window=window, center=True).max() == df['High']
        df['Local_Low'] = df['Low'].rolling(window=window, center=True).min() == df['Low']
        
        # 이동평균선과의 거리 (지지/저항 역할)
        df['Distance_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Distance_SMA60'] = (df['Close'] - df['SMA_60']) / df['SMA_60']
        
        # 볼린저 밴드 터치
        df['Touch_BB_Upper'] = abs(df['Close'] - df['BB_Upper']) / df['Close'] < 0.01
        df['Touch_BB_Lower'] = abs(df['Close'] - df['BB_Lower']) / df['Close'] < 0.01
    
    def _identify_trend_patterns(self, df: pd.DataFrame):
        """추세 패턴 인식"""
        # 이동평균선 정렬
        df['MA_Bullish_Align'] = (
            (df['Close'] > df['SMA_5']) & 
            (df['SMA_5'] > df['SMA_20']) & 
            (df['SMA_20'] > df['SMA_60'])
        )
        
        df['MA_Bearish_Align'] = (
            (df['Close'] < df['SMA_5']) & 
            (df['SMA_5'] < df['SMA_20']) & 
            (df['SMA_20'] < df['SMA_60'])
        )
        
        # 추세 강도 (기울기)
        period = 5
        df['SMA20_Slope'] = df['SMA_20'].rolling(window=period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else 0
        )
        
        # 골든크로스/데드크로스
        df['Golden_Cross'] = (
            (df['SMA_5'] > df['SMA_20']) & 
            (df['SMA_5'].shift(1) <= df['SMA_20'].shift(1))
        )
        
        df['Dead_Cross'] = (
            (df['SMA_5'] < df['SMA_20']) & 
            (df['SMA_5'].shift(1) >= df['SMA_20'].shift(1))
        )
    
    def get_pattern_strength(self, idx: int, pattern_type: str) -> float:
        """특정 패턴의 강도 반환 (0-1)"""
        if idx >= len(self.data):
            return 0.0
            
        row = self.data.iloc[idx]
        
        if pattern_type == 'candle':
            # 캔들 패턴 강도
            score = 0.0
            if row.get('Is_Long_Green', False):
                score += row.get('Body_Ratio', 0) * 10
            if row.get('Is_Long_Red', False):
                score += row.get('Body_Ratio', 0) * 10
            if row.get('Is_Hammer', False):
                score += 0.3
            if row.get('Is_Hanging_Man', False):
                score += 0.3
                
            return min(score, 1.0)
            
        elif pattern_type == 'gap':
            # 갭 패턴 강도
            return min(row.get('Gap_Strength', 0) * 10, 1.0)
            
        elif pattern_type == 'support_resistance':
            # 지지/저항 패턴 강도
            score = 0.0
            if row.get('Touch_BB_Upper', False) or row.get('Touch_BB_Lower', False):
                score += 0.4
            if abs(row.get('Distance_SMA20', 0)) < 0.02:  # 2% 이내
                score += 0.3
            if abs(row.get('Distance_SMA60', 0)) < 0.02:
                score += 0.3
                
            return min(score, 1.0)
            
        elif pattern_type == 'trend':
            # 추세 패턴 강도
            score = 0.0
            if row.get('MA_Bullish_Align', False) or row.get('MA_Bearish_Align', False):
                score += 0.5
            if row.get('Golden_Cross', False) or row.get('Dead_Cross', False):
                score += 0.4
            slope_strength = abs(row.get('SMA20_Slope', 0)) * 100
            score += min(slope_strength, 0.3)
            
            return min(score, 1.0)
            
        return 0.0