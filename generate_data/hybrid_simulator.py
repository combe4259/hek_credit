#!/usr/bin/env python3
"""
하이브리드 매매 시뮬레이터
실제 주가 데이터 + 고급 매매 결정 로직
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from config import *
from patterns.pattern_analyzer import PatternAnalyzer
from advanced_data_generator import AdvancedDataGenerator

# 시드 고정 (재현성 확보)
np.random.seed(RANDOM_SEED)

class HybridTradingSimulator:
    """실제 데이터 + 고급 매매 로직을 결합한 시뮬레이터"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.pattern_analyzer = PatternAnalyzer(data)
        self.advanced_generator = AdvancedDataGenerator()
        self.pattern_dataset = []
        
    def simulate_all_profiles(self) -> List[Dict]:
        """모든 투자자 프로필에 대해 하이브리드 시뮬레이션 실행"""
        print("🎯 하이브리드 매매 시나리오 생성 중...")
        
        # Advanced 방식의 투자자 프로필 생성
        for i, profile in enumerate(INVESTOR_PROFILES):
            # Advanced 스타일 프로필로 변환
            advanced_profile = self._convert_to_advanced_profile(profile)
            print(f"  📊 {profile.name} 프로필 시뮬레이션...")
            self._simulate_single_profile(profile, advanced_profile)
        
        print(f"✅ 총 {len(self.pattern_dataset)}개의 매매 패턴 데이터 생성")
        return self.pattern_dataset
    
    def _convert_to_advanced_profile(self, profile):
        """기존 프로필을 Advanced 스타일로 변환"""
        return {
            'user_id': profile.name,
            'profit_targets': {  # 수익률 구간별 매도 확률
                '0-5%': 1 - profile.profit_taking * 0.8,
                '5-10%': 1 - profile.profit_taking * 0.6,
                '10-20%': 1 - profile.profit_taking * 0.4,
                '20%+': 1 - profile.profit_taking * 0.2
            },
            'loss_thresholds': {  # 손실률별 손절 확률
                '0--3%': 1 - profile.stop_loss * 0.9,
                '-3--5%': 1 - profile.stop_loss * 0.7,
                '-5--10%': 1 - profile.stop_loss * 0.5,
                '-10%+': 1 - profile.stop_loss * 0.3
            },
            'panic_threshold': 0.05 * (1 - profile.volatility_reaction),
            'time_sensitivity': profile.time_sensitivity
        }
    
    def _simulate_single_profile(self, profile, advanced_profile):
        """단일 투자자 프로필 시뮬레이션"""
        position = None
        trades_count = 0
        
        # 충분한 데이터 확보 후 시작
        start_idx = max(SMA_LONG, 10)
        
        for i in range(start_idx, len(self.data) - 1):
            current_price = self.data.iloc[i]['close']
            current_time = self.data.index[i]
            
            # 8가지 패턴 분석
            patterns = self.pattern_analyzer.analyze_all_patterns(i)
            if not patterns:
                continue
            
            # 하이브리드 매매 결정
            decision = self._make_hybrid_decision(
                patterns, profile, advanced_profile, 
                current_price, position, i
            )
            
            # 매매 결정이 있을 때만 기록 (HOLD 제외 옵션)
            # 시드 고정된 랜덤 생성
            rng = np.random.RandomState(RANDOM_SEED * 1000 + i)
            if decision['action'] != 'HOLD' or rng.random() < 0.1:  # HOLD는 10%만 기록
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
                trades_count += 1
                
                # 포지션 업데이트
                if decision['action'] == 'BUY' and position is None:
                    position = {'price': current_price, 'timestamp': current_time, 'index': i}
                elif decision['action'] == 'SELL' and position is not None:
                    position = None
        
        print(f"    - {trades_count}개 거래 생성 (BUY/SELL 위주)")
    
    def _make_hybrid_decision(self, patterns, profile, advanced_profile, 
                             price, position, idx):
        """하이브리드 매매 결정 (실제 데이터 + 고급 로직)"""
        
        # 포지션이 있을 때
        if position:
            return_pct = (price - position['price']) / position['price']
            holding_days = idx - position['index']
            
            # Advanced 스타일 매도 결정
            sell_decision = self._advanced_sell_decision(
                advanced_profile, return_pct, holding_days, patterns
            )
            
            if sell_decision['sell']:
                return {
                    'action': 'SELL',
                    'reasoning': sell_decision['reason']
                }
        
        # 포지션이 없을 때
        else:
            # 기술적 지표 기반 매수 결정
            buy_score = self._calculate_buy_score(patterns, profile, idx)
            market_data = self.data.iloc[idx]
            
            # 매수 이유 결정
            if market_data['rsi'] < 30:
                reasoning = f"RSI 과매도({market_data['rsi']:.0f})"
            elif market_data['bb_position'] < 0.2:
                reasoning = f"볼린저 하단({market_data['bb_position']:.2f})"
            else:
                reasoning = f"매수 점수: {buy_score:.2f}"
            
            if buy_score > 0.65:  # 기술적 지표가 강할 때만 매수
                return {
                    'action': 'BUY',
                    'reasoning': reasoning
                }
        
        # HOLD 결정 (대부분 기록하지 않음)
        return {'action': 'HOLD', 'reasoning': '관망'}
    
    def _advanced_sell_decision(self, profile, return_pct, holding_days, patterns):
        """현실적인 매도 결정 - 감정과 노이즈 포함"""
        rng = np.random.RandomState(RANDOM_SEED * 1000 + holding_days * 10)
        
        # 🎯 1. 수익 구간 (탐욕과 후회 포함)
        if return_pct > 0:
            base_sell_prob = 0
            zone = ''
            
            if return_pct < 0.05:
                zone = '0-5%'
                base_sell_prob = profile['profit_targets'].get(zone, 0.2)
            elif return_pct < 0.10:
                zone = '5-10%'
                base_sell_prob = profile['profit_targets'].get(zone, 0.4)
            elif return_pct < 0.20:
                zone = '10-20%'
                base_sell_prob = profile['profit_targets'].get(zone, 0.6)
            else:
                zone = '20%+'
                base_sell_prob = profile['profit_targets'].get(zone, 0.8)
            
            # 🎯 탐욕 요소 (더 오를 것 같아서 안 파는 심리)
            if return_pct > 0.15:  # 15% 이상 수익
                greed_factor = min(0.3, (return_pct - 0.15) * 2)  # 탐욕으로 매도 연기
                base_sell_prob *= (1 - greed_factor)
                
            # 🎯 FOMO 역작용 (너무 일찍 팔았다는 후회 반영)
            if holding_days < 3:  # 단기간 수익
                early_sell_hesitation = 0.8  # 20% 매도 확률 감소
                base_sell_prob *= early_sell_hesitation
            
            # 🎯 세금 고려 (현실적!)
            if holding_days < 365:  # 1년 미만 보유 (단기투자세)
                tax_hesitation = rng.uniform(0.85, 0.95)  # 5-15% 매도 확률 감소
                base_sell_prob *= tax_hesitation
            
            # 🎯 감정적 노이즈
            emotional_noise = rng.normal(0, 0.1)  # ±10% 감정적 변동
            final_sell_prob = max(0, min(1, base_sell_prob + emotional_noise))
            
            if rng.random() < final_sell_prob:
                return {'sell': True, 'reason': f'수익실현_{zone}'}
        
        # 🎯 2. 손실 구간 (공포와 고집 포함)
        else:
            zone = ''
            base_sell_prob = 0
            
            if return_pct > -0.03:
                zone = '0--3%'
                base_sell_prob = 0.1  # 작은 손실은 잘 안 팖
            elif return_pct > -0.05:
                zone = '-3--5%'  
                base_sell_prob = 0.2
            elif return_pct > -0.10:
                zone = '-5--10%'
                base_sell_prob = 0.4
            else:
                zone = '-10%+'
                base_sell_prob = 0.7  # 큰 손실은 대부분 손절
            
            # 🎯 손실 회피 심리 (Loss Aversion)
            loss_aversion = abs(return_pct) * 1.5  # 손실을 1.5배 더 크게 느낌
            if loss_aversion > 0.15:  # 15% 이상 손실 느낌
                panic_boost = min(0.3, loss_aversion - 0.15)
                base_sell_prob += panic_boost
            
            # 🎯 물타기 심리 (평균단가 낮추기)
            if return_pct < -0.08 and holding_days > 14:  # 8% 이상 손실, 2주 이상 보유
                averaging_down_prob = 0.2  # 20% 확률로 더 버팀
                if rng.random() < averaging_down_prob:
                    base_sell_prob *= 0.5  # 매도 확률 절반으로
            
            # 🎯 심리적 저항선 (예: -20%, -50%)
            psychological_levels = [-0.20, -0.30, -0.50]
            for level in psychological_levels:
                if abs(return_pct - level) < 0.02:  # ±2% 범위
                    psychological_resistance = 1.3  # 심리적 저항으로 30% 더 매도
                    base_sell_prob *= psychological_resistance
                    break
            
            # 🎯 시간 압박 (손실이 오래될수록 포기)
            if holding_days > 60:  # 2개월 이상 손실
                time_pressure = min(0.2, (holding_days - 60) / 300)  # 최대 20% 추가
                base_sell_prob += time_pressure
            
            # 감정적 노이즈
            emotional_noise = rng.normal(0, 0.15)  # 손실 시 더 큰 감정적 변동
            final_sell_prob = max(0, min(1, base_sell_prob + emotional_noise))
            
            if rng.random() < final_sell_prob:
                return {'sell': True, 'reason': f'손절_{zone}'}
        
        # 🎯 3. 기술적 지표 기반 매도 (추가)
        # RSI 과매수 체크 (patterns에서 가져옴)
        rsi_value = patterns.get('rsi', 50)
        if rsi_value > 70:
            if rng.random() < 0.6:  # 60% 확률로 매도
                return {'sell': True, 'reason': f'RSI_과매수({rsi_value:.0f})'}
        
        # 볼린저 상단 체크
        bb_pos = patterns.get('bb_position', 0.5)
        if bb_pos > 0.85:
            if rng.random() < 0.5:  # 50% 확률로 매도
                return {'sell': True, 'reason': f'BB_상단({bb_pos:.2f})'}
        
        # 🎯 시간 압박과 지겨움 (현실적!)
        if holding_days > 30:
            boredom_factor = (holding_days - 30) / 365  # 1년에 걸쳐 서서히 증가
            time_pressure_prob = min(0.3, boredom_factor * 0.1)  # 최대 30%까지
            
            # 주말 효과 (월요일에 더 매도 성향)
            day_of_week_effect = rng.uniform(0.8, 1.2)  # 요일별 변동
            
            final_time_prob = time_pressure_prob * day_of_week_effect
            if rng.random() < final_time_prob:
                return {'sell': True, 'reason': '장기보유_청산'}
        
        # 🎯 급작스러운 외부 이벤트 (1% 확률)
        if rng.random() < 0.01:
            emergency_reasons = ['긴급자금필요', '가족이벤트', '다른투자기회']
            reason = rng.choice(emergency_reasons)
            return {'sell': True, 'reason': reason}
        
        return {'sell': False, 'reason': None}
    
    def _calculate_buy_score(self, patterns, profile, idx):
        """기술적 지표 중심의 매수 점수 계산 (0-1)"""
        # 시드 고정 (재현 가능한 랜덤)
        rng = np.random.RandomState(RANDOM_SEED + idx * 7)
        
        score = 0.5  # 중립에서 시작
        market_data = self.data.iloc[idx]
        
        # 🎯 1. RSI - 가장 중요한 지표 (40% 가중치)
        rsi = market_data['rsi']
        if rsi < 25:  # 극도의 과매도
            score += 0.4
        elif rsi < RSI_OVERSOLD:  # 과매도 (30)
            score += 0.3
        elif rsi < 40:  # 약한 과매도
            score += 0.15
        elif rsi > 70:  # 과매수 - 매수 기피
            score -= 0.3
        elif rsi > 60:  # 약한 과매수
            score -= 0.15
        
        # 🎯 2. 볼린저 밴드 (30% 가중치)
        bb_pos = market_data['bb_position']
        if bb_pos < 0.1:  # 하단 돌파
            score += 0.3
        elif bb_pos < 0.2:  # 하단 근처
            score += 0.2
        elif bb_pos > 0.9:  # 상단 돌파 - 매수 기피
            score -= 0.2
        elif bb_pos > 0.8:  # 상단 근처
            score -= 0.1
        
        # 🎯 3. MACD 신호 (15% 가중치)
        if market_data['macd'] > market_data['macd_signal']:
            score += 0.15  # 골든크로스
        else:
            score -= 0.1  # 데드크로스
        
        # 🎯 4. 거래량 (10% 가중치)
        volume_ratio = market_data['volume_ratio']
        if volume_ratio > 1.5 and market_data['daily_return'] < 0:  # 하락 시 거래량 급증
            score += 0.1  # 바닥 신호
        elif volume_ratio < 0.7:  # 거래량 감소
            score -= 0.05
        
        # 🎯 5. 노이즈 추가 (20% 영향)
        # 감정적 요소
        emotional_noise = rng.normal(0, 0.1)
        score += emotional_noise
        # 랜덤 요소 (10% 확률로 반대 행동)
        if rng.random() < 0.1:
            score = 1 - score
        
        # 프로필 성향 반영 (소폭)
        profile_effect = 1 + (profile.profit_taking - 0.5) * 0.2
        score *= profile_effect
        
        return max(0, min(1.0, score))  # 0-1 범위로 제한
    
    def _calculate_recent_performance(self, idx):
        """최근 시장 성과 계산 (감정적 영향)"""
        lookback = min(10, idx)  # 최근 10일
        if lookback < 2:
            return 0
        
        recent_returns = []
        for i in range(max(0, idx - lookback), idx):
            if i < len(self.data):
                daily_return = self.data.iloc[i].get('daily_return', 0)
                recent_returns.append(daily_return)
        
        return sum(recent_returns) if recent_returns else 0
    
    def _calculate_future_returns(self, current_idx):
        """미래 수익률 계산"""
        current_price = self.data.iloc[current_idx]['close']
        returns = {}
        
        for days, label in [(1, '1d'), (7, '7d'), (30, '30d')]:
            future_idx = min(current_idx + days, len(self.data) - 1)
            future_price = self.data.iloc[future_idx]['close']
            returns[label] = (future_price - current_price) / current_price
        
        return returns
    
    def _create_record(self, timestamp, profile, price, decision, 
                      patterns, market_data, future_returns):
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