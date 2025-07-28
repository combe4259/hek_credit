# simulation/trading_simulator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from config import *
from patterns.pattern_analyzer import PatternAnalyzer

class TradingSimulator:
    """매매 시뮬레이션 실행기"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.pattern_analyzer = PatternAnalyzer(data)
        self.pattern_dataset = []
        
    def simulate_all_profiles(self) -> List[Dict]:
        """모든 투자자 프로필에 대해 시뮬레이션 실행"""
        print("🎯 매매 시나리오 생성 중...")
        
        for profile in INVESTOR_PROFILES:
            print(f"  📊 {profile.name} 프로필 시뮬레이션...")
            self._simulate_single_profile(profile)
        
        print(f"✅ 총 {len(self.pattern_dataset)}개의 매매 패턴 데이터 생성")
        return self.pattern_dataset
    
    def _simulate_single_profile(self, profile: InvestorProfile):
        """단일 투자자 프로필 시뮬레이션"""
        portfolio_value = INITIAL_CAPITAL
        position = None  # 현재 포지션 {'price': float, 'timestamp': datetime}
        
        # 충분한 데이터 확보 후 시작
        start_idx = max(SMA_LONG, 10)  # 더 빠른 시작
        
        for i in range(start_idx, len(self.data) - 1):
            current_price = self.data.iloc[i]['close']
            current_time = self.data.index[i]
            
            # 8가지 패턴 분석
            patterns = self.pattern_analyzer.analyze_all_patterns(i)
            if not patterns:
                continue
            
            # 투자자 프로필에 따른 매매 결정
            decision = self._make_trading_decision(patterns, profile, current_price, position, i)
            
            # 모든 결정을 기록 (HOLD 포함)
            if True:  # 항상 기록
                # 미래 수익률 계산
                future_returns = self._calculate_future_returns(i)
                
                # 데이터 기록
                record = self._create_record(
                    timestamp=current_time,
                    profile=profile,
                    price=current_price,
                    decision=decision,
                    patterns=patterns,
                    market_data=self.data.iloc[i],
                    future_returns=future_returns
                )
                
                self.pattern_dataset.append(record)
                
                # 포지션 업데이트
                if decision['action'] == 'BUY':
                    position = {'price': current_price, 'timestamp': current_time}
                elif decision['action'] == 'SELL':
                    position = None
    
    def _make_trading_decision(self, patterns: Dict, profile: InvestorProfile, 
                             price: float, position: Optional[Dict], idx: int) -> Dict:
        """투자자 프로필과 패턴을 바탕으로 매매 결정"""
        
        # 포지션 보유 중인 경우
        if position:
            return_pct = (price - position['price']) / position['price']
            
            # 수익 실현 결정
            if return_pct > PROFIT_THRESHOLD_1:  # 5% 이상 수익
                # 프로필의 수익실현 성향과 현재 패턴 비교
                should_sell_profit = self._should_take_profit(patterns, profile, return_pct)
                if should_sell_profit:
                    return {
                        'action': 'SELL',
                        'reasoning': f"수익실현 {return_pct:.2%} (성향:{profile.profit_taking:.2f})"
                    }
            
            # 손절 결정
            if return_pct < LOSS_THRESHOLD_1:  # -3% 이상 손실
                should_stop_loss = self._should_stop_loss(patterns, profile, return_pct)
                if should_stop_loss:
                    return {
                        'action': 'SELL',
                        'reasoning': f"손절 {return_pct:.2%} (성향:{profile.stop_loss:.2f})"
                    }
        
        # 매수 신호 검토 (포지션 없을 때)
        else:
            buy_signals = self._evaluate_buy_signals(patterns, profile, idx)
            if buy_signals['should_buy']:
                return {
                    'action': 'BUY',
                    'reasoning': f"매수신호 {buy_signals['signal_count']}개: {buy_signals['reasons']}"
                }
        
        return {'action': 'HOLD', 'reasoning': '관망'}
    
    def _should_take_profit(self, patterns: Dict, profile: InvestorProfile, return_pct: float) -> bool:
        """수익 실현 여부 결정"""
        # 기본 성향
        base_tendency = profile.profit_taking
        
        # 현재 패턴의 수익실현 성향
        pattern_tendency = patterns.get('profit_taking_tendency', 0.5)
        
        # 수익률에 따른 가중치 (수익이 클수록 매도 압박 증가)
        profit_pressure = min(return_pct / PROFIT_THRESHOLD_2, 1.0)  # 20%에서 최대
        
        # 종합 판단 (낮을수록 매도 성향)
        combined_score = (base_tendency * 0.6 + pattern_tendency * 0.4) * (1 - profit_pressure * 0.3)
        threshold = 0.4 + (return_pct - PROFIT_THRESHOLD_1) * 2  # 수익률 높을수록 낮은 임계값
        
        return combined_score < threshold
    
    def _should_stop_loss(self, patterns: Dict, profile: InvestorProfile, return_pct: float) -> bool:
        """손절 여부 결정"""
        # 기본 성향
        base_tendency = profile.stop_loss
        
        # 현재 패턴의 손절 성향
        pattern_tendency = patterns.get('stop_loss_tendency', 0.5)
        
        # 손실률에 따른 압박 증가
        loss_pressure = min(abs(return_pct) / abs(LOSS_THRESHOLD_2), 1.0)  # -10%에서 최대
        
        # 종합 판단 (낮을수록 손절 성향)
        combined_score = (base_tendency * 0.6 + pattern_tendency * 0.4) * (1 - loss_pressure * 0.4)
        threshold = 0.5 - (abs(return_pct) - abs(LOSS_THRESHOLD_1)) * 3  # 손실 클수록 낮은 임계값
        
        return combined_score < threshold
    
    def _evaluate_buy_signals(self, patterns: Dict, profile: InvestorProfile, idx: int) -> Dict:
        """매수 신호 평가"""
        signals = []
        reasons = []
        
        # 1. 기술적 지표 신호
        if patterns.get('technical_indicator_reliance', 0) > profile.technical_reliance:
            signals.append('technical')
            reasons.append('기술적지표')
        
        # 2. 거래량 신호
        if patterns.get('volume_reaction', 0) > profile.volume_sensitivity:
            signals.append('volume')
            reasons.append('거래량급증')
        
        # 3. 캔들 패턴 신호
        if patterns.get('candle_analysis', 0) > profile.candle_sensitivity:
            signals.append('candle')
            reasons.append('강세캔들')
        
        # 4. 차트 패턴 신호
        if patterns.get('chart_pattern_recognition', 0) > profile.pattern_sensitivity:
            signals.append('pattern')
            reasons.append('차트패턴')
        
        # 5. 변동성 기회 (모멘텀 트레이더의 경우)
        if profile.name == 'Momentum_Trader':
            volatility = patterns.get('volatility_reaction', 0)
            if volatility < 0.4:  # 변동성이 클 때 진입
                signals.append('momentum')
                reasons.append('모멘텀')
        
        # 매수 결정 (프로필에 따라 필요한 신호 개수 다름)
        required_signals = self._get_required_signals(profile)
        should_buy = len(signals) >= required_signals
        
        return {
            'should_buy': should_buy,
            'signal_count': len(signals),
            'reasons': '+'.join(reasons[:3])  # 최대 3개만 표시
        }
    
    def _get_required_signals(self, profile: InvestorProfile) -> int:
        """프로필별 필요한 매수 신호 개수"""
        if profile.name == 'Conservative':
            return 2  # 신중한 매수 (3에서 2로 완화)
        elif profile.name == 'Aggressive':
            return 1  # 공격적 매수
        elif profile.name == 'Technical_Trader':
            return 1  # 기술적 근거 중시 (2에서 1로 완화)
        elif profile.name == 'Momentum_Trader':
            return 1  # 모멘텀 중시 (2에서 1로 완화)
        else:  # Swing_Trader
            return 1  # 균형잡힌 접근 (2에서 1로 완화)
    
    def _calculate_future_returns(self, current_idx: int) -> Dict:
        """미래 수익률 계산"""
        current_price = self.data.iloc[current_idx]['close']
        returns = {}
        
        for days, label in [(1, '1d'), (7, '7d'), (30, '30d')]:
            future_idx = min(current_idx + days, len(self.data) - 1)
            future_price = self.data.iloc[future_idx]['close']
            returns[label] = (future_price - current_price) / current_price
        
        return returns
    
    def _create_record(self, timestamp: datetime, profile: InvestorProfile, 
                      price: float, decision: Dict, patterns: Dict, 
                      market_data: pd.Series, future_returns: Dict) -> Dict:
        """매매 기록 생성"""
        return {
            # 기본 정보
            'timestamp': timestamp,
            'investor_profile': profile.name,
            'price': price,
            'action': decision['action'],
            'reasoning': decision['reasoning'],
            
            # 8가지 패턴 점수
            'profit_taking_tendency': patterns.get('profit_taking_tendency', 0.5),
            'stop_loss_tendency': patterns.get('stop_loss_tendency', 0.5),
            'volatility_reaction': patterns.get('volatility_reaction', 0.5),
            'time_based_trading': patterns.get('time_based_trading', 0.5),
            'technical_indicator_reliance': patterns.get('technical_indicator_reliance', 0.5),
            'chart_pattern_recognition': patterns.get('chart_pattern_recognition', 0.5),
            'volume_reaction': patterns.get('volume_reaction', 0.5),
            'candle_analysis': patterns.get('candle_analysis', 0.5),
            
            # 시장 상황
            'rsi': market_data.get('rsi', 50),
            'macd_signal': 1 if market_data.get('macd', 0) > market_data.get('macd_signal', 0) else 0,
            'bb_position': market_data.get('bb_position', 0.5),
            'volume_ratio': market_data.get('volume_ratio', 1.0),
            'daily_return': market_data.get('daily_return', 0),
            'gap': market_data.get('gap', 0),
            
            # 결과 라벨
            'return_1d': future_returns['1d'],
            'return_7d': future_returns['7d'],
            'return_30d': future_returns['30d']
        }