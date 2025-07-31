#!/usr/bin/env python3
"""
개인 투자자의 매매 패턴을 분석하고 행동 개선을 위한 AI 학습 데이터 생성
FAR-Trans 데이터를 활용한 투자 행동 패턴 분석
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os

class TradingPatternAnalyzer:
    """개인별 매매 패턴 분석 및 AI 학습 데이터 생성"""
    
    def __init__(self, transactions_path, customers_path):
        self.transactions_df = pd.read_csv(transactions_path)
        self.customers_df = pd.read_csv(customers_path)
        self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df['timestamp'])
        
        # 고객별 패턴 저장
        self.customer_patterns = defaultdict(dict)
        
    def analyze_customer_patterns(self):
        """각 고객의 매매 패턴 분석"""
        print(" 고객별 매매 패턴 분석 중...")
        
        # 고객별로 거래 분석
        for customer_id in self.transactions_df['customerID'].unique():
            customer_trades = self.transactions_df[
                self.transactions_df['customerID'] == customer_id
            ].sort_values('timestamp')
            
            if len(customer_trades) < 5:  # 충분한 거래가 있는 고객만
                continue
            
            # 1. Buy-Sell 매칭으로 실제 거래 결과 계산
            trading_results = self._match_buy_sell_pairs(customer_trades)
            
            # 2. 보유 기간 패턴 분석
            holding_patterns = self._analyze_holding_periods(trading_results)
            
            # 3. 거래 타이밍 패턴 분석
            timing_patterns = self._analyze_timing_patterns(customer_trades)
            
            # 4. 손익 실현 패턴 분석
            profit_loss_patterns = self._analyze_profit_loss_behavior(trading_results)
            
            # 5. 거래 빈도 패턴
            frequency_patterns = self._analyze_trading_frequency(customer_trades)
            
            self.customer_patterns[customer_id] = {
                'trading_results': trading_results,
                'holding_patterns': holding_patterns,
                'timing_patterns': timing_patterns,
                'profit_loss_patterns': profit_loss_patterns,
                'frequency_patterns': frequency_patterns,
                'customer_info': self._get_customer_info(customer_id)
            }
        
        print(f" {len(self.customer_patterns)}명 고객 패턴 분석 완료")
        
    def _match_buy_sell_pairs(self, trades):
        """Buy-Sell 쌍을 매칭하여 실제 거래 결과 계산"""
        results = []
        positions = defaultdict(list)  # 종목별 보유 포지션
        
        for _, trade in trades.iterrows():
            isin = trade['ISIN']
            
            if trade['transactionType'] == 'Buy':
                # 매수 포지션 추가
                positions[isin].append({
                    'buy_date': trade['timestamp'],
                    'buy_price': trade['totalValue'] / trade['units'],
                    'units': trade['units'],
                    'buy_id': trade['transactionID']
                })
            
            elif trade['transactionType'] == 'Sell' and positions[isin]:
                # 매도 시 FIFO로 포지션 정리
                sell_units = trade['units']
                sell_price = trade['totalValue'] / trade['units']
                
                while sell_units > 0 and positions[isin]:
                    position = positions[isin][0]
                    
                    if position['units'] <= sell_units:
                        # 포지션 전체 매도
                        holding_days = (trade['timestamp'] - position['buy_date']).days
                        return_rate = (sell_price - position['buy_price']) / position['buy_price'] * 100
                        
                        results.append({
                            'isin': isin,
                            'buy_date': position['buy_date'],
                            'sell_date': trade['timestamp'],
                            'holding_days': holding_days,
                            'buy_price': position['buy_price'],
                            'sell_price': sell_price,
                            'return_rate': return_rate,
                            'units': position['units'],
                            'profit_loss': 'profit' if return_rate > 0 else 'loss'
                        })
                        
                        sell_units -= position['units']
                        positions[isin].pop(0)
                    else:
                        # 포지션 일부 매도
                        holding_days = (trade['timestamp'] - position['buy_date']).days
                        return_rate = (sell_price - position['buy_price']) / position['buy_price'] * 100
                        
                        results.append({
                            'isin': isin,
                            'buy_date': position['buy_date'],
                            'sell_date': trade['timestamp'],
                            'holding_days': holding_days,
                            'buy_price': position['buy_price'],
                            'sell_price': sell_price,
                            'return_rate': return_rate,
                            'units': sell_units,
                            'profit_loss': 'profit' if return_rate > 0 else 'loss'
                        })
                        
                        position['units'] -= sell_units
                        sell_units = 0
        
        return results
    
    def _analyze_holding_periods(self, trading_results):
        """보유 기간 패턴 분석"""
        if not trading_results:
            return {}
        
        holding_days = [r['holding_days'] for r in trading_results]
        profit_trades = [r for r in trading_results if r['profit_loss'] == 'profit']
        loss_trades = [r for r in trading_results if r['profit_loss'] == 'loss']
        
        return {
            'avg_holding_days': np.mean(holding_days),
            'median_holding_days': np.median(holding_days),
            'profit_avg_holding': np.mean([r['holding_days'] for r in profit_trades]) if profit_trades else 0,
            'loss_avg_holding': np.mean([r['holding_days'] for r in loss_trades]) if loss_trades else 0,
            'quick_sell_ratio': len([h for h in holding_days if h < 7]) / len(holding_days) if holding_days else 0
        }
    
    def _analyze_timing_patterns(self, trades):
        """거래 타이밍 패턴 분석"""
        patterns = {
            'avg_days_between_trades': 0,
            'weekend_trader': False,
            'morning_trader': False,
            'reactive_trader': False
        }
        
        if len(trades) < 2:
            return patterns
        
        # 거래 간격 계산
        trade_dates = trades['timestamp'].sort_values()
        intervals = [(trade_dates.iloc[i+1] - trade_dates.iloc[i]).days 
                    for i in range(len(trade_dates)-1)]
        patterns['avg_days_between_trades'] = np.mean(intervals)
        
        # 요일 패턴 (실제로는 시간 데이터가 필요하지만 날짜로 추정)
        weekdays = trades['timestamp'].dt.dayofweek
        patterns['weekend_trader'] = (weekdays >= 5).sum() > 0  # 주말 거래자
        
        # 연속 거래 패턴 (하루에 여러 번 거래)
        daily_trades = trades.groupby(trades['timestamp'].dt.date).size()
        patterns['reactive_trader'] = (daily_trades > 2).sum() > len(daily_trades) * 0.1
        
        return patterns
    
    def _analyze_profit_loss_behavior(self, trading_results):
        """손익 실현 패턴 분석"""
        if not trading_results:
            return {}
        
        profits = [r for r in trading_results if r['profit_loss'] == 'profit']
        losses = [r for r in trading_results if r['profit_loss'] == 'loss']
        
        patterns = {
            'win_rate': len(profits) / len(trading_results) if trading_results else 0,
            'avg_profit_rate': np.mean([r['return_rate'] for r in profits]) if profits else 0,
            'avg_loss_rate': np.mean([r['return_rate'] for r in losses]) if losses else 0
        }
        
        # profit_loss_ratio 안전하게 계산
        if profits and losses:
            avg_profit = np.mean([r['return_rate'] for r in profits])
            avg_loss = np.mean([r['return_rate'] for r in losses])
            if avg_loss != 0:
                patterns['profit_loss_ratio'] = abs(avg_profit) / abs(avg_loss)
            else:
                patterns['profit_loss_ratio'] = float('inf') if avg_profit > 0 else 0
        else:
            patterns['profit_loss_ratio'] = 0
        
        # 조기 익절 패턴 (작은 수익에 만족)
        if profits:
            small_profits = [r for r in profits if r['return_rate'] < 5]
            patterns['early_profit_taking'] = len(small_profits) / len(profits)
        
        # 손실 회피 패턴 (큰 손실 보유)
        if losses:
            big_losses = [r for r in losses if r['return_rate'] < -10]
            patterns['loss_aversion'] = len(big_losses) / len(losses)
        
        return patterns
    
    def _analyze_trading_frequency(self, trades):
        """거래 빈도 패턴 분석"""
        # 월별 거래 횟수
        monthly_trades = trades.groupby(trades['timestamp'].dt.to_period('M')).size()
        
        # 종목별 거래 집중도
        stock_concentration = trades['ISIN'].value_counts(normalize=True)
        
        return {
            'avg_monthly_trades': monthly_trades.mean(),
            'max_monthly_trades': monthly_trades.max(),
            'favorite_stock_concentration': stock_concentration.iloc[0] if len(stock_concentration) > 0 else 0,
            'num_stocks_traded': trades['ISIN'].nunique(),
            'overtrading_score': monthly_trades.std() / monthly_trades.mean() if monthly_trades.mean() > 0 else 0
        }
    
    def _get_customer_info(self, customer_id):
        """고객 정보 조회"""
        customer_info = self.customers_df[self.customers_df['customerID'] == customer_id]
        if not customer_info.empty:
            return customer_info.iloc[-1].to_dict()  # 최신 정보
        return {}
    
    def generate_ai_training_data(self):
        """AI 학습용 데이터 생성"""
        print("\n🤖 AI 학습 데이터 생성 중...")
        
        training_data = []
        
        for customer_id, patterns in self.customer_patterns.items():
            # 각 거래 결과에 대한 학습 데이터 생성
            for result in patterns['trading_results']:
                # 거래 시점의 컨텍스트 생성
                context = {
                    'customer_id': customer_id,
                    'customer_type': patterns['customer_info'].get('customerType', 'Unknown'),
                    'risk_level': patterns['customer_info'].get('riskLevel', 'Unknown'),
                    
                    # 거래 정보
                    'holding_days': result['holding_days'],
                    'return_rate': result['return_rate'],
                    'profit_loss': result['profit_loss'],
                    
                    # 고객 패턴 정보
                    'avg_holding_days': patterns['holding_patterns']['avg_holding_days'],
                    'quick_sell_ratio': patterns['holding_patterns']['quick_sell_ratio'],
                    'win_rate': patterns['profit_loss_patterns']['win_rate'],
                    'avg_profit_rate': patterns['profit_loss_patterns']['avg_profit_rate'],
                    'avg_loss_rate': patterns['profit_loss_patterns']['avg_loss_rate'],
                    'early_profit_taking': patterns['profit_loss_patterns'].get('early_profit_taking', 0),
                    'loss_aversion': patterns['profit_loss_patterns'].get('loss_aversion', 0),
                    
                    # 거래 빈도 패턴
                    'avg_monthly_trades': patterns['frequency_patterns']['avg_monthly_trades'],
                    'overtrading_score': patterns['frequency_patterns']['overtrading_score'],
                    
                    # 타이밍 패턴
                    'reactive_trader': patterns['timing_patterns']['reactive_trader'],
                    'avg_days_between_trades': patterns['timing_patterns']['avg_days_between_trades']
                }
                
                # 행동 개선 제안 라벨 생성
                if result['profit_loss'] == 'profit' and result['return_rate'] < 3:
                    context['improvement_needed'] = 'hold_longer_for_profit'
                elif result['profit_loss'] == 'loss' and result['return_rate'] < -5:
                    context['improvement_needed'] = 'cut_loss_earlier'
                elif result['holding_days'] < 3:
                    context['improvement_needed'] = 'avoid_impulsive_trading'
                else:
                    context['improvement_needed'] = 'good_decision'
                
                training_data.append(context)
        
        # DataFrame 변환
        df = pd.DataFrame(training_data)
        
        print(f"✅ AI 학습 데이터 생성 완료: {len(df)}개 레코드")
        print(f"\n📊 행동 개선 필요 분포:")
        print(df['improvement_needed'].value_counts())
        
        return df
    
    def save_pattern_analysis(self, output_dir='output'):
        """패턴 분석 결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        # AI 학습 데이터 저장
        ai_data = self.generate_ai_training_data()
        ai_data_path = os.path.join(output_dir, 'trading_behavior_patterns.csv')
        ai_data.to_csv(ai_data_path, index=False, encoding='utf-8-sig')
        
        # 고객별 패턴 요약 저장
        pattern_summary = []
        for customer_id, patterns in self.customer_patterns.items():
            # profit_loss_patterns가 비어있을 수 있음
            win_rate = patterns['profit_loss_patterns'].get('win_rate', 0) if patterns.get('profit_loss_patterns') else 0
            
            summary = {
                'customer_id': customer_id,
                'customer_type': patterns['customer_info'].get('customerType', 'Unknown'),
                'risk_level': patterns['customer_info'].get('riskLevel', 'Unknown'),
                'total_trades': len(patterns['trading_results']) if patterns.get('trading_results') else 0,
                'win_rate': win_rate,
                'avg_return': np.mean([r['return_rate'] for r in patterns['trading_results']]) if patterns.get('trading_results') else 0,
                'behavior_score': self._calculate_behavior_score(patterns)
            }
            pattern_summary.append(summary)
        
        summary_df = pd.DataFrame(pattern_summary)
        summary_path = os.path.join(output_dir, 'customer_pattern_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 분석 결과 저장 완료:")
        print(f"   - AI 학습 데이터: {ai_data_path}")
        print(f"   - 고객 패턴 요약: {summary_path}")
        
        return ai_data
    
    def _calculate_behavior_score(self, patterns):
        """투자 행동 점수 계산 (0-100)"""
        score = 50  # 기본 점수
        
        # 승률에 따른 가산
        if patterns.get('profit_loss_patterns'):
            score += patterns['profit_loss_patterns'].get('win_rate', 0) * 20
            
            # 조기 익절 패턴 감점
            score -= patterns['profit_loss_patterns'].get('early_profit_taking', 0) * 10
            
            # 손실 회피 패턴 감점
            score -= patterns['profit_loss_patterns'].get('loss_aversion', 0) * 15
        
        # 과도한 거래 감점
        if patterns.get('frequency_patterns') and patterns['frequency_patterns'].get('overtrading_score', 0) > 1:
            score -= 10
        
        return max(0, min(100, score))


if __name__ == "__main__":
    # 실제 데이터 경로
    transactions_path = "/Users/inter4259/Downloads/FAR-Trans/transactions.csv"
    customers_path = "/Users/inter4259/Downloads/FAR-Trans/customer_information.csv"
    
    # 분석기 초기화
    analyzer = TradingPatternAnalyzer(transactions_path, customers_path)
    
    # 고객 패턴 분석
    analyzer.analyze_customer_patterns()
    
    # AI 학습 데이터 생성 및 저장
    ai_training_data = analyzer.save_pattern_analysis()