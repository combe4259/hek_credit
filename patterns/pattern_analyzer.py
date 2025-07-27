import pandas as pd
import numpy as np
from typing import Dict
from config import *
from patterns.chart_patterns import ChartPatternIdentifier

class PatternAnalyzer:
    """8가지 매매 패턴 분석기"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.chart_patterns = ChartPatternIdentifier(data).identify_all_patterns()
        
    def analyze_all_patterns(self, current_idx: int) -> Dict[str, float]:
        """8가지 매매 패턴 종합 분석"""
        if current_idx < SMA_LONG:  # 충분한 데이터가 없으면 스킵
            return {}
        
        patterns = {}
        
        # 1. 수익실현 성향
        patterns['profit_taking_tendency'] = self._analyze_profit_taking(current_idx)
        
        # 2. 손절 성향
        patterns['stop_loss_tendency'] = self._analyze_stop_loss(current_idx)
        
        # 3. 변동성 반응
        patterns['volatility_reaction'] = self._analyze_volatility_reaction(current_idx)
        
        # 4. 시간대별 매매
        patterns['time_based_trading'] = self._analyze_time_based_trading(current_idx)
        
        # 5. 기술적 지표 의존도
        patterns['technical_indicator_reliance'] = self._analyze_technical_reliance(current_idx)
        
        # 6. 차트 패턴 인식
        patterns['chart_pattern_recognition'] = self._analyze_chart_pattern_recognition(current_idx)
        
        # 7. 거래량 반응
        patterns['volume_reaction'] = self._analyze_volume_reaction(current_idx)
        
        # 8. 캔들 분석
        patterns['candle_analysis'] = self._analyze_candle_analysis(current_idx)
        
        return patterns
    
    def _analyze_profit_taking(self, idx: int) -> float:
        """수익실현 성향 분석 (0-1: 0=단타, 1=장기보유)"""
        return self._analyze_behavior(idx, "profit")
    
    def _analyze_stop_loss(self, idx: int) -> float:
        """손절 성향 분석 (0-1: 0=빠른손절, 1=버팀)"""
        return self._analyze_behavior(idx, "loss")
    
    def _analyze_behavior(self, idx: int, mode: str = "profit") -> float:
        """행동 성향 분석"""
        lookback = 20
        scenarios = 0          # 상황(수익 또는 손실) 발생 횟수
        reaction_signals = 0   # 빠르게 반응한(판) 신호 횟수

        for i in range(max(0, idx-lookback), idx):
            if mode == "profit":
                # 최근 10일 저점 대비 수익률
                recent_low = self.data.iloc[max(0, i-10):i+1]['Low'].min()
                current_price = self.data.iloc[i]['Close']
                change_pct = (current_price - recent_low) / recent_low

                # 5% 이상 수익
                if change_pct > PROFIT_THRESHOLD_1:
                    scenarios += 1

                    # RSI 과매도 구간 (팔 가능성 ↑)
                    rsi = self.data.iloc[i]['RSI']
                    if rsi < RSI_OVERSOLD:
                        reaction_signals += 1

                    # 볼린저 상단 근처면 (익절 가능성 ↑)
                    bb_position = self.data.iloc[i]['BB_Position']
                    if bb_position > 0.9:
                        reaction_signals += 1

            elif mode == "loss":
                # 최근 10일 고점 대비 손실률
                recent_high = self.data.iloc[max(0, i-10):i+1]['High'].max()
                current_price = self.data.iloc[i]['Close']
                change_pct = (current_price - recent_high) / recent_high

                # -3% 이상 손실
                if change_pct < LOSS_THRESHOLD_1:
                    scenarios += 1

                    # RSI 과매도 구간 (겁먹고 손절 가능성 ↑)
                    rsi = self.data.iloc[i]['RSI']
                    if rsi < RSI_OVERSOLD:
                        reaction_signals += 1

                    # 볼린저 하단 근처면 (손절 가능성 ↑)
                    bb_position = self.data.iloc[i]['BB_Position']
                    if bb_position < 0.2:
                        reaction_signals += 1

        # 상황이 없으면 중립값 반환
        if scenarios == 0:
            return 0.5

        # 빠른 반응 비율 계산
        reaction_ratio = reaction_signals / (scenarios * 2)  # 각 상황당 최대 2신호
        return 1 - min(reaction_ratio, 1.0)

    def _analyze_volatility_reaction(self, idx: int) -> float:
        """변동성 반응 패턴 (0-1: 0=패닉매매, 1=침착함)"""
        current_return = abs(self.data.iloc[idx]['Daily_Return'])
        
        # 하루 5% 이상 변동 시
        if current_return > VOLATILITY_THRESHOLD:
            rsi = self.data.iloc[idx]['RSI']
            macd_hist = self.data.iloc[idx]['MACD_Hist']
            
            # 극단적 지표 상황에서의 반응 측정
            panic_signals = 0
            
            if rsi > 80 or rsi < 20:  # 극단적 RSI
                panic_signals += 1
            
            if abs(macd_hist) > self.data['MACD_Hist'].std() * 2:  # 극단적 MACD
                panic_signals += 1
                
            # 거래량 급증도 패닉 신호
            volume_ratio = self.data.iloc[idx]['Volume_Ratio']
            if volume_ratio > HIGH_VOLUME_THRESHOLD:
                panic_signals += 1
            
            # 패닉 신호 많을수록 패닉 매매 성향
            panic_score = panic_signals / 3
            return 1 - panic_score
        
        return 0.7  # 평상시는 비교적 침착
    
    def _analyze_time_based_trading(self, idx: int) -> float:
        """시간대별 매매 성향 (0-1: 0=충동적, 1=계획적)"""
        # 거래량 급증을 통한 충동적 매매 추정
        volume_ratio = self.data.iloc[idx]['Volume_Ratio']
        
        # 갭 상황을 충동 매매로 가정
        gap = abs(self.data.iloc[idx]['Gap']) if idx > 0 else 0
        
        # 주초/주말 효과 (월요일/금요일)
        is_monday = self.data.iloc[idx]['Is_Monday']
        is_friday = self.data.iloc[idx]['Is_Friday']
        
        impulse_score = 0
        if volume_ratio > HIGH_VOLUME_THRESHOLD:  # 거래량 2배 이상 급증
            impulse_score += 0.4
        if gap > GAP_UP_THRESHOLD:  # 큰 갭
            impulse_score += 0.3
        if is_monday or is_friday:  # 주초/주말 효과
            impulse_score += 0.2
        
        return 1 - min(impulse_score, 1.0)
    
    def _analyze_technical_reliance(self, idx: int) -> float:
        """기술적 지표 의존도 (0-1: 0=지표무시, 1=지표의존)"""
        # 여러 지표의 일치도 확인
        indicators = []
        
        # 이동평균 신호
        sma_signal = 1 if self.data.iloc[idx]['Close'] > self.data.iloc[idx]['SMA_20'] else -1
        indicators.append(sma_signal)
        
        # RSI 신호
        rsi = self.data.iloc[idx]['RSI']
        if rsi > RSI_OVERBOUGHT:
            rsi_signal = -1  # 매도 신호
        elif rsi < RSI_OVERSOLD:
            rsi_signal = 1   # 매수 신호
        else:
            rsi_signal = 0   # 중립
        indicators.append(rsi_signal)

        # MACD 신호
        macd_signal = 1 if self.data.iloc[idx]['MACD'] > self.data.iloc[idx]['MACD_Signal'] else -1
        indicators.append(macd_signal)
        
        # 볼린저 밴드 신호
        bb_position = self.data.iloc[idx]['BB_Position']
        if bb_position > 0.8:
            bb_signal = -1  # 상단 근처 매도
        elif bb_position < 0.2:
            bb_signal = 1   # 하단 근처 매수
        else:
            bb_signal = 0   # 중립
        indicators.append(bb_signal)
            
        # 지표들의 일치도 계산
        non_zero_indicators = [x for x in indicators if x != 0]
        if len(non_zero_indicators) == 0:
            return 0.5
        
        # 같은 방향 신호의 비율
        agreement = sum(1 for x in non_zero_indicators if x == non_zero_indicators[0])
        agreement_ratio = agreement / len(non_zero_indicators)
        
        return agreement_ratio
    
    def _analyze_chart_pattern_recognition(self, idx: int) -> float:
        """차트 패턴 인식 민감도 (0-1: 0=패턴무시, 1=패턴민감)"""
        if idx >= len(self.chart_patterns):
            return 0.5  # Default value if index is out of bounds
            
        row = self.chart_patterns.iloc[idx]
        pattern_scores = []
        
        # 캔들 패턴 강도 (장대양봉/음봉, 도지 등)
        candle_strength = 0.0
        if 'Is_Long_Green' in row and row['Is_Long_Green']:
            candle_strength = max(candle_strength, 1.0)
        if 'Is_Long_Red' in row and row['Is_Long_Red']:
            candle_strength = max(candle_strength, 1.0)
        if 'Is_Doji' in row and row['Is_Doji']:
            candle_strength = max(candle_strength, 0.8)
        pattern_scores.append(candle_strength)
        
        # 지지/저항 패턴 강도
        support_resistance_strength = 0.0
        if 'Near_Support' in row and row['Near_Support']:
            support_resistance_strength = max(support_resistance_strength, 0.7)
        if 'Near_Resistance' in row and row['Near_Resistance']:
            support_resistance_strength = max(support_resistance_strength, 0.7)
        pattern_scores.append(support_resistance_strength)
        
        # 추세 패턴 강도 (이동평균선 정배열/역배열)
        trend_strength = 0.0
        if 'SMA_5' in row and 'SMA_20' in row and 'SMA_60' in row:
            if row['SMA_5'] > row['SMA_20'] > row['SMA_60']:  # 상승 추세
                trend_strength = 1.0
            elif row['SMA_5'] < row['SMA_20'] < row['SMA_60']:  # 하락 추세
                trend_strength = 1.0
        pattern_scores.append(trend_strength)
        
        # 갭 패턴 강도
        gap_strength = 0.0
        if 'Is_Gap_Up' in row and row['Is_Gap_Up']:
            gap_strength = 1.0
        elif 'Is_Gap_Down' in row and row['Is_Gap_Down']:
            gap_strength = 1.0
        pattern_scores.append(gap_strength)
        
        # 평균 패턴 민감도 (0-1 범위로 정규화)
        valid_scores = [s for s in pattern_scores if s > 0]
        if not valid_scores:
            return 0.5  # Default value if no patterns detected
        return np.mean(valid_scores)
    
    def _analyze_volume_reaction(self, idx: int) -> float:
        """거래량 반응 민감도 (0-1: 0=거래량무시, 1=거래량추종)"""
        volume_ratio = self.data.iloc[idx]['Volume_Ratio']
        price_change = abs(self.data.iloc[idx]['Daily_Return'])
        
        # 거래량과 가격 변동의 상관관계 확인
        if volume_ratio > VOLUME_SPIKE_THRESHOLD:  # 거래량 50% 이상 증가
            if price_change > 0.02:  # 2% 이상 가격 변동
                # 거래량-가격 동조성이 높을수록 거래량 추종 성향
                sync_score = min(volume_ratio / 3, 1.0)
                return sync_score
        
        # 최근 거래량 패턴 분석
        lookback = 10
        recent_volume_ratios = []
        recent_price_changes = []
        
        for i in range(max(0, idx-lookback), idx):
            recent_volume_ratios.append(self.data.iloc[i]['Volume_Ratio'])
            recent_price_changes.append(abs(self.data.iloc[i]['Daily_Return']))
        
        # 거래량과 가격 변동의 상관관계
        if len(recent_volume_ratios) > 3:
            correlation = np.corrcoef(recent_volume_ratios, recent_price_changes)[0, 1]
            if not np.isnan(correlation):
                return (correlation + 1) / 2  # -1~1을 0~1로 변환
        
        return 0.4  # 기본 거래량 민감도
    
    def _analyze_candle_analysis(self, idx: int) -> float:
        """캔들 패턴 반응도 (0-1: 0=캔들무시, 1=캔들민감)"""
        current_data = self.data.iloc[idx]
        
        # 현재 캔들의 특성
        is_long_green = current_data.get('Is_Long_Green', False)
        is_long_red = current_data.get('Is_Long_Red', False)
        is_doji = current_data.get('Is_Doji', False)
        is_hammer = current_data.get('Is_Hammer', False)
        is_hanging_man = current_data.get('Is_Hanging_Man', False)
        
        body_ratio = current_data.get('Body_Ratio', 0)
        
        candle_score = 0
        
        # 강한 캔들 패턴일수록 높은 점수
        if is_long_green or is_long_red:
            candle_score += min(body_ratio * 15, 0.6)  # 몸통 클수록 민감
        
        if is_hammer:
            candle_score += 0.3
            
        if is_hanging_man:
            candle_score += 0.3
            
        if is_doji:
            candle_score += 0.2  # 도지도 의미있는 패턴
        
        # 연속 캔들 패턴 확인
        if idx > 0:
            prev_data = self.data.iloc[idx-1]
            
            # 연속 상승/하락 캔들
            if (is_long_green and prev_data.get('Is_Long_Green', False)) or \
            (is_long_red and prev_data.get('Is_Long_Red', False)):
                candle_score += 0.2
        
        return min(candle_score, 1.0)
