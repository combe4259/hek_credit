# data/data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from ta.utils import dropna
from datetime import datetime
from config import *

class DataLoader:
    """주식 데이터 로드 및 기술적 지표 계산"""
    
    def __init__(self, symbol: str = SYMBOL, period: str = PERIOD):
        self.symbol = symbol
        self.period = period
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """주식 데이터 로드"""

        max_retries = 3
        retry_delay = 2  # seconds
        
        print(f"📈 {self.symbol} 데이터 로드 중... (기간: {self.period})")
        
        for attempt in range(max_retries):
            try:
                # yfinance를 사용하여 데이터 로드
                ticker = yf.Ticker(self.symbol)
                df = ticker.history(period=self.period, interval="1d", auto_adjust=True)
                
                # 인덱스 초기화 및 날짜 컬럼 추가
                df = df.reset_index()
                
                # 컬럼명 정규화 (대소문자 통일 및 공백 제거)
                df.columns = [str(col).lower().strip() for col in df.columns]
                
                # 필요한 컬럼 매핑 (다양한 네이밍 컨벤션 대응)
                column_mapping = {
                    'open': ['open', 'opening', '시가'],
                    'high': ['high', 'high price', '고가'],
                    'low': ['low', 'low price', '저가'],
                    'close': ['close', 'closing', 'adj close', 'adjusted close', '종가'],
                    'volume': ['volume', 'vol', '거래량']
                }
                
                # 컬럼 매핑 적용
                normalized_columns = {}
                for std_col, possible_cols in column_mapping.items():
                    for col in possible_cols:
                        if col in df.columns:
                            normalized_columns[std_col] = col
                            break
                
                # 필요한 컬럼이 모두 있는지 확인
                missing_columns = [col for col in column_mapping.keys() if col not in normalized_columns]
                if missing_columns:
                    raise ValueError(f"필수 컬럼을 찾을 수 없습니다: {', '.join(missing_columns)}")
                
                # 컬럼명 표준화
                df = df.rename(columns={v: k for k, v in normalized_columns.items()})
                
                # 필요한 컬럼만 선택
                df = df[['date'] + list(column_mapping.keys())]
                
                # 결측값 처리
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)
                
                # 기본 정보 출력
                date_min = df['date'].min().strftime('%Y-%m-%d') if hasattr(df['date'], 'min') else 'N/A'
                date_max = df['date'].max().strftime('%Y-%m-%d') if hasattr(df['date'], 'max') else 'N/A'
                print(f"✅ {self.symbol} 데이터 로드 완료: {date_min} ~ {date_max} (총 {len(df)}일치)")
                
                self.data = df
                return df
                
            except Exception as e:
                error_msg = str(e)
                print(f"X 데이터 로드 실패 (시도 {attempt + 1}/{max_retries}): {error_msg}")
                
                # 디버깅을 위해 사용 가능한 컬럼 출력
                if 'df' in locals():
                    print(f"사용 가능한 컬럼들: {df.columns.tolist()}")
                
                if attempt == max_retries - 1:
                    print("X 모든 시도가 실패했습니다.")
                    raise

                import time
                time.sleep(retry_delay)
    
    def calculate_technical_indicators(self) -> pd.DataFrame:
        """기술 지표 계산"""
        if self.data is None or self.data.empty:
            raise ValueError("데이터가 로드되지 않았습니다. 먼저 load_data()를 호출해주세요.")
            
        print("기술적 지표 계산 중...")
        df = self.data.copy()
        
        # 데이터 정제
        df = df.dropna()
        if df.empty:
            raise ValueError("계산할 데이터가 없습니다.")
        
        # 1. 추세 지표
        # 이동평균선
        df['sma_5'] = SMAIndicator(close=df['close'], window=SMA_SHORT).sma_indicator()
        df['sma_20'] = SMAIndicator(close=df['close'], window=SMA_MEDIUM).sma_indicator()
        df['sma_60'] = SMAIndicator(close=df['close'], window=SMA_LONG).sma_indicator()
        
        # 지수이동평균
        df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
        
        # 2. 모멘텀 지표
        # RSI
        rsi_indicator = RSIIndicator(close=df['close'], window=RSI_PERIOD)
        df['rsi'] = rsi_indicator.rsi()
        
        # MACD
        macd_indicator = MACD(
            close=df['close'],
            window_slow=MACD_SLOW,
            window_fast=MACD_FAST,
            window_sign=MACD_SIGNAL
        )
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()
        
        # 스토캐스틱 오실레이터
        stoch_indicator = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        df['stoch_k'] = stoch_indicator.stoch()
        df['stoch_d'] = stoch_indicator.stoch_signal()
        
        # 3. 변동성 지표
        # 볼린저 밴드
        bb_indicator = BollingerBands(close=df['close'], window=SMA_MEDIUM, window_dev=2)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        atr_indicator = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        df['atr'] = atr_indicator.average_true_range()
        
        # 4. 거래량 지표
        # 거래량 이동평균
        df['volume_ma'] = df['volume'].rolling(window=SMA_MEDIUM).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # OBV (On-Balance Volume)
        obv_indicator = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = obv_indicator.on_balance_volume()
        
        # 5. 기타 유용한 지표
        # 일일 수익률
        df['daily_return'] = df['close'].pct_change() * 100  # 백분율로 표시
        
        # 가격 변동률
        df['price_change'] = (df['close'] - df['open']) / df['open'] * 100  # 백분율로 표시
        
        # 갭 계산 (오늘 시가 - 전일 종가) / 전일 종가
        df['gap'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * 100  # 백분율로 표시
        
        # 시간대 정보 (요일별 거래 행태 분석용)
        df['is_monday'] = df['date'].dt.weekday == 0
        df['is_friday'] = df['date'].dt.weekday == 4
        
        # 6. 결측값 처리
        # 선형 보간으로 결측값 처리
        df.interpolate(method='linear', inplace=True)
        
        # 그래도 남아있는 결측값은 0 또는 이전 값으로 채우기
        df.fillna(method='ffill', inplace=True)  # 이전 값으로 채우기
        df.fillna(method='bfill', inplace=True)  # 이후 값으로 채우기
        df.fillna(0, inplace=True)  # 그래도 남아있으면 0으로 채우기
        
        self.data = df
        print("✅ 기술적 지표 계산 완료")
        return df
    
    def get_processed_data(self) -> pd.DataFrame:
        """전처리된 데이터 반환"""
        if self.data is None:
            self.load_data()
            
        self.calculate_technical_indicators()
        return self.data
    
    def get_data_info(self) -> dict:
        """데이터 정보 반환"""
        if self.data is None:
            return {}
            
        return {
            'symbol': self.symbol,
            'period': self.period,
            'start_date': self.data.index.min(),
            'end_date': self.data.index.max(),
            'total_days': len(self.data),
            'columns': list(self.data.columns)
        }