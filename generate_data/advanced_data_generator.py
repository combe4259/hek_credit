#!/usr/bin/env python3
"""
고급 데이터 생성기 (advanced_trading_ai.py에서 이전)
실제 시장 환경을 반영한 복합적인 매매 패턴 데이터 생성
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
import os
import sys

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *

# 시드 고정 (재현성 확보)
np.random.seed(RANDOM_SEED)

class AdvancedDataGenerator:
    """고급 매매 패턴 데이터 생성기"""

    def __init__(self):
        # 개인 매매 이력
        self.trading_history = []
        self.loss_patterns = []
        self.profit_patterns = defaultdict(list)  # 수익률 구간별 행동 기록

    def generate_realistic_market_data(self, n_users=500, n_stocks=50, n_days=365):
        """실제 시장 환경을 반영한 데이터 생성"""
        print("📊 실제 시장 환경 데이터 생성 중...")

        # 종목 정보 생성
        stocks = self._generate_stock_info(n_stocks)

        # 사용자별 거래 데이터 생성
        all_trades = []

        for user_id in range(n_users):
            # 사용자 고유 특성
            user_profile = {
                'user_id': f'user_{user_id}',
                'profit_targets': {  # 수익률 구간별 매도 확률
                    '0-5%': 0.1 + (user_id % 4) * 0.1,  # 0.1, 0.2, 0.3, 0.4
                    '5-10%': 0.3 + (user_id % 5) * 0.1,  # 0.3-0.7
                    '10-20%': 0.5 + (user_id % 5) * 0.08,  # 0.5-0.9
                    '20%+': 0.8 + (user_id % 3) * 0.067  # 0.8-1.0
                },
                'loss_thresholds': {  # 손실률별 손절 확률
                    '0--3%': 0.1 + (user_id % 3) * 0.067,  # 0.1-0.3
                    '-3--5%': 0.2 + (user_id % 4) * 0.075,  # 0.2-0.5
                    '-5--10%': 0.4 + (user_id % 5) * 0.08,  # 0.4-0.8
                    '-10%+': 0.7 + (user_id % 4) * 0.075  # 0.7-1.0
                },
                'time_preferences': {  # 시간대별 거래 성향
                    'morning': (user_id % 10) / 10.0,      # 0.0-0.9
                    'mid_morning': ((user_id + 1) % 10) / 10.0,  # 0.1-1.0
                    'lunch': ((user_id + 2) % 10) / 10.0,        # 0.2-1.0
                    'afternoon': ((user_id + 3) % 10) / 10.0,    # 0.3-1.0
                    'closing': ((user_id + 4) % 10) / 10.0       # 0.4-1.0
                },
                'panic_threshold': 0.03 + (user_id % 6) * 0.0083,  # 0.03-0.08
                'fomo_tendency': (user_id % 10) / 10.0,          # 0.0-0.9
                'loss_aversion': 0.3 + (user_id % 7) * 0.1       # 0.3-0.9
            }

            # 사용자별 거래 생성
            user_trades = self._generate_user_trades(
                user_profile, stocks, n_days
            )
            all_trades.extend(user_trades)

        df = pd.DataFrame(all_trades)
        print(f"✅ 총 {len(df):,}개 거래 데이터 생성 완료")

        # 손실 패턴 사례 생성
        self._generate_loss_pattern_cases(df)

        return df

    def _generate_stock_info(self, n_stocks):
        """종목 정보 생성"""
        sectors = ['전자', '화학', '금융', '바이오', '자동차', '건설', '유통', '엔터']

        stocks = []
        for i in range(n_stocks):
            stock = {
                'ticker': f'STOCK_{i:03d}',
                'name': f'종목{i}',
                'sector': np.random.choice(sectors),
                'market_cap': np.random.choice(['대형주', '중형주', '소형주'],
                                               p=[0.3, 0.4, 0.3]),
                'avg_volatility': np.random.uniform(0.01, 0.05),
                'beta': np.random.uniform(0.5, 1.5)
            }

            # 실제 종목명 예시 추가
            if i == 0:
                stock.update({'ticker': '005930', 'name': '삼성전자',
                              'sector': '전자', 'market_cap': '대형주'})
            elif i == 1:
                stock.update({'ticker': '051910', 'name': 'LG화학',
                              'sector': '화학', 'market_cap': '대형주'})

            stocks.append(stock)

        return stocks

    def _generate_user_trades(self, user_profile, stocks, n_days):
        """사용자별 거래 데이터 생성"""
        trades = []
        # 사용자ID에 기반한 일관된 거래 수
        base_trades = 50
        user_id_num = int(user_profile['user_id'].split('_')[1])
        n_trades = base_trades + (user_id_num % 150)  # 50-199 범위

        for _ in range(n_trades):
            # 종목 선택 (사용자별 일관성)
            stock_idx = (_ + user_id_num) % len(stocks)
            stock = stocks[stock_idx]

            # 거래 시작일 (사용자별 일관성)
            buy_date = (_ * 7 + user_id_num * 3) % max(1, n_days - 30)

            # 매수 시간 (사용자 선호도 반영)
            buy_hour, buy_minute = self._get_trading_time(user_profile['time_preferences'])

            # 시장 상황 (거래일에 기반한 일관성)
            market_conditions = ['상승장', '하락장', '횡보장']
            market_idx = (buy_date + _) % 3
            market_condition = market_conditions[market_idx]

            # 보유 기간 및 수익률 시뮬레이션
            trade_result = self._simulate_trade(
                user_profile, stock, market_condition
            )

            # 거래 데이터 구성
            trade = {
                'user_id': user_profile['user_id'],
                'ticker': stock['ticker'],
                'stock_name': stock['name'],
                'sector': stock['sector'],
                'market_cap': stock['market_cap'],
                'buy_date': buy_date,
                'buy_hour': buy_hour,
                'buy_minute': buy_minute,
                'market_condition': market_condition,
                **trade_result
            }

            trades.append(trade)

        return trades

    def _get_trading_time(self, time_preferences):
        """시간대별 선호도에 따른 거래 시간 생성"""
        time_slots = [
            (9, 0, 10, 0, 'morning'),
            (10, 0, 11, 0, 'mid_morning'),
            (11, 0, 13, 0, 'lunch'),
            (13, 0, 14, 0, 'afternoon'),
            (14, 0, 15, 30, 'closing')
        ]

        # 선호도에 따른 가중치 적용
        weights = [time_preferences[slot[4]] for slot in time_slots]
        weights = np.array(weights) / sum(weights)

        # 시간대 선택 (가중치 기반 결정적 선택)
        cumulative_weights = np.cumsum(weights)
        rand_val = sum(weights) * 0.5  # 중간값 사용
        chosen_slot = np.searchsorted(cumulative_weights, rand_val)
        start_h, start_m, end_h, end_m, _ = time_slots[chosen_slot]

        # 구체적 시간 생성 (스롯 중간 시간)
        total_minutes = (end_h - start_h) * 60 + (end_m - start_m)
        random_minutes = total_minutes // 2  # 중간 시간 사용

        hour = start_h + random_minutes // 60
        minute = start_m + random_minutes % 60

        return hour, minute

    def _simulate_trade(self, user_profile, stock, market_condition):
        """거래 시뮬레이션"""
        # 초기 설정 (사용자 ID 기반 일관성)
        user_id_num = int(user_profile['user_id'].split('_')[1])
        holding_days = 1 + (user_id_num * 7) % 59  # 1~59일 범위
        current_profit = 0
        max_profit = 0
        min_profit = 0
        daily_profits = []

        # 매도 여부
        sold = False
        sell_reason = None
        sell_hour = None
        sell_minute = None

        # 보유 기간 동안 가격 변동 시뮬레이션
        for day in range(1, holding_days + 1):
            # 일별 수익률 변화
            daily_change = self._calculate_daily_return(
                stock, market_condition, day
            )
            current_profit += daily_change
            daily_profits.append(current_profit)

            max_profit = max(max_profit, current_profit)
            min_profit = min(min_profit, current_profit)

        # 최종일에 매도 결정 (한 번만)
        intraday_volatility = stock['avg_volatility'] * np.random.uniform(0.5, 2)

        sell_decision = self._decide_sell(
            user_profile, current_profit, holding_days,
            max_profit, intraday_volatility
        )

        # 추가로 30% 확률로는 무조건 보유 중 (매도하지 않음)
        # 사용자 ID 기반 결정적 선택
        if (user_id_num + holding_days) % 10 < 3:  # 약 30%
            sold = False
            sell_reason = 'holding'
        elif sell_decision['sell']:
            sold = True
            sell_reason = sell_decision['reason']
            sell_hour, sell_minute = self._get_trading_time(
                user_profile['time_preferences']
            )
        else:
            sold = False
            sell_reason = 'holding'

        # buy_hour 생성
        buy_hour, buy_minute = self._get_trading_time(user_profile['time_preferences'])

        # 수익률 구간 계산
        profit_zone = self._get_profit_zone(current_profit)

        return {
            'holding_days': holding_days,
            'final_profit_rate': round(current_profit, 4),
            'max_profit_rate': round(max_profit, 4),
            'min_profit_rate': round(min_profit, 4),
            'profit_volatility': round(np.std(daily_profits), 4),
            'profit_zone': profit_zone,
            'sold': 1 if sold else 0,  # 명시적으로 0/1로 변환
            'sell_reason': sell_reason or 'holding',
            'sell_hour': sell_hour if sell_hour else buy_hour,
            'sell_minute': sell_minute if sell_minute else buy_minute,
            'is_loss_pattern': 1 if (max_profit > 0.05 and current_profit < -0.05) else 0
        }

    def _calculate_daily_return(self, stock, market_condition, day):
        """일별 수익률 계산"""
        base_return = 0

        # 종목과 날짜에 기반한 일관된 수익률
        stock_hash = hash(stock['ticker']) % 1000
        day_factor = (day * stock_hash) % 100 / 100.0 - 0.5
        
        if market_condition == '상승장':
            base_return = 0.002 + stock['avg_volatility'] * day_factor
        elif market_condition == '하락장':
            base_return = -0.002 + stock['avg_volatility'] * day_factor
        else:  # 횡보장
            base_return = 0 + stock['avg_volatility'] * 0.8 * day_factor

        # 베타 적용
        base_return *= stock['beta']

        # 특별 이벤트 (종목과 날짜에 기반한 결정적 이벤트)
        event_hash = (stock_hash + day * 13) % 100
        if event_hash < 5:  # 5%
            event_returns = [-0.1, -0.05, 0.05, 0.1]
            event_idx = event_hash % 4
            base_return += event_returns[event_idx]

        return base_return

    def _decide_sell(self, user_profile, current_profit, holding_days,
                     max_profit, volatility):
        """매도 결정 로직"""
        # 수익률 구간 확인
        profit_zone = self._get_profit_zone(current_profit)
        
        # user_id에서 숫자 추출 (없으면 0 사용)
        user_id_num = int(''.join(filter(str.isdigit, user_profile.get('user_id', '0'))) or '0')

        # 수익 상황에서의 매도
        if current_profit > 0:
            if profit_zone in user_profile['profit_targets']:
                sell_prob = user_profile['profit_targets'][profit_zone] * 0.7  # 확률 낮춤
                # 사용자 ID와 보유일 기반 결정
                if (holding_days * 100 + user_id_num) % 100 < sell_prob * 100:
                    return {'sell': True, 'reason': f'profit_taking_{profit_zone}'}

        # 손실 상황에서의 매도
        else:
            loss_zone = self._get_loss_zone(current_profit)
            if loss_zone in user_profile['loss_thresholds']:
                sell_prob = user_profile['loss_thresholds'][loss_zone] * 0.6  # 확률 낮춤
                # 사용자 ID와 보유일 기반 결정
                if (holding_days * 100 + user_id_num + 50) % 100 < sell_prob * 100:
                    return {'sell': True, 'reason': f'stop_loss_{loss_zone}'}

        # 패닉셀 체크
        if volatility > user_profile['panic_threshold']:
            # 사용자 ID 기반 결정
            user_id_num = int(user_profile['user_id'].split('_')[1])
            if (volatility * 1000 + user_id_num) % 100 < 20:  # 20%
                return {'sell': True, 'reason': 'panic_sell'}

        # 고점 대비 하락 (그리드)
        if max_profit > 0.1 and (max_profit - current_profit) > 0.05:
            # 보유일 기반 결정
            if (holding_days * 10) % 100 < 30:  # 30%
                return {'sell': True, 'reason': 'drawdown_sell'}

        # 장기 보유
        if holding_days > 60:
            # 보유일 기반 결정
            prob = 0.03 * (holding_days - 60)
            if (holding_days * 100) % 100 < prob * 100:
                return {'sell': True, 'reason': 'time_based'}

        return {'sell': False, 'reason': None}

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

    def _get_loss_zone(self, profit_rate):
        """손실률 구간 분류"""
        if profit_rate > -0.03:
            return '0--3%'
        elif profit_rate > -0.05:
            return '-3--5%'
        elif profit_rate > -0.10:
            return '-5--10%'
        else:
            return '-10%+'

    def _generate_loss_pattern_cases(self, df):
        """과거 손실 패턴 사례 생성"""
        # 실제 손실 사례 추가 (하드코딩된 예시)
        self.loss_patterns = [
            {
                'case_id': 'LOSS_001',
                'date': '2024-03-15',
                'stock': 'LG화학',
                'initial_loss': -0.042,
                'final_loss': -0.128,
                'holding_days': 15,
                'pattern_description': '손실 상황에서 홀딩 → 추가 하락',
                'market_condition': '하락장',
                'similar_cases': ['LOSS_005', 'LOSS_012']
            },
            {
                'case_id': 'LOSS_002',
                'date': '2024-02-20',
                'stock': '카카오',
                'initial_loss': -0.03,
                'final_loss': -0.15,
                'holding_days': 25,
                'pattern_description': '실적 발표 후 급락 미대응',
                'market_condition': '횡보장',
                'similar_cases': ['LOSS_008']
            }
        ]

        # 시뮬레이션 데이터에서 손실 패턴 추출
        loss_trades = df[df['is_loss_pattern'] == 1].head(20)
        for idx, trade in loss_trades.iterrows():
            self.loss_patterns.append({
                'case_id': f'LOSS_{idx:03d}',
                'date': f'2024-{(idx % 12) + 1:02d}-{(idx % 28) + 1:02d}',
                'stock': trade['stock_name'],
                'initial_loss': trade['min_profit_rate'],
                'final_loss': trade['final_profit_rate'],
                'holding_days': trade['holding_days'],
                'pattern_description': '고점 대비 큰 폭 하락',
                'market_condition': trade['market_condition'],
                'similar_cases': []
            })

    def save_advanced_dataset(self, filename="output/advanced_trading_data.csv"):
        """고급 데이터셋 생성 및 저장"""
        print("🚀 고급 매매 패턴 데이터셋 생성 시작...")

        # 데이터 생성
        df = self.generate_realistic_market_data(n_users=300, n_stocks=30, n_days=365)

        # 디렉토리 생성
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # CSV 저장
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ 고급 데이터셋 저장 완료: {filename}")

        # 메타데이터 저장
        meta_filename = os.path.splitext(filename)[0] + '_metadata.json'
        metadata = {
            'generation_time': datetime.now().isoformat(),
            'total_records': len(df),
            'data_type': '고급 매매 패턴 데이터',
            'features': list(df.columns),
            'loss_patterns': len(self.loss_patterns),
            'description': '실제 시장 환경을 반영한 복합적 매매 패턴 데이터'
        }

        with open(meta_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"✅ 메타데이터 저장: {meta_filename}")

        # 통계 출력
        print(f"\n📊 데이터셋 요약:")
        print(f"   - 총 거래 수: {len(df):,}개")
        print(f"   - 매도 거래: {df['sold'].sum():,}개 ({df['sold'].mean():.1%})")
        print(f"   - 손실 패턴: {df['is_loss_pattern'].sum():,}개")
        print(f"   - 평균 수익률: {df['final_profit_rate'].mean():.2%}")
        print(f"   - 평균 보유일: {df['holding_days'].mean():.1f}일")

        return df

def main():
    """메인 실행 함수"""
    generator = AdvancedDataGenerator()
    generator.save_advanced_dataset()

if __name__ == "__main__":
    main()
