#!/usr/bin/env python3
"""
데이터 품질 및 노이즈 수준 분석
"""

import pandas as pd
import numpy as np

def analyze_data_quality():
    # CSV 읽기
    df = pd.read_csv('output/trading_patterns_augmented.csv')
    
    print('📊 데이터 패턴 분석')
    print('='*50)
    
    # 액션별 기술적 지표 평균값
    print('\n🎯 액션별 평균 지표값:')
    for action in ['BUY', 'HOLD', 'SELL']:
        action_df = df[df['action'] == action]
        print(f'\n{action} (n={len(action_df)}):')
        print(f'  - RSI: {action_df["rsi"].mean():.1f} (±{action_df["rsi"].std():.1f})')
        print(f'  - BB위치: {action_df["bb_position"].mean():.2f} (±{action_df["bb_position"].std():.2f})')
        print(f'  - 일일수익률: {action_df["daily_return"].mean():.3f} (±{action_df["daily_return"].std():.3f})')
        print(f'  - 볼륨비율: {action_df["volume_ratio"].mean():.2f} (±{action_df["volume_ratio"].std():.2f})')
    
    # 시간대별 액션 분포
    print('\n⏰ 시간대별 액션 비율:')
    df['buy_hour'] = df['timestamp'] % 7 + 9  # 9시-15시 변환
    for hour in range(9, 16):
        hour_df = df[df['buy_hour'] == hour]
        if len(hour_df) > 0:
            buy_ratio = (hour_df['action'] == 'BUY').mean() * 100
            sell_ratio = (hour_df['action'] == 'SELL').mean() * 100
            print(f'{hour}시: BUY {buy_ratio:.1f}%, SELL {sell_ratio:.1f}%')
    
    # 노이즈 수준 분석
    print('\n🎲 노이즈 분석 (엔트로피 - 높을수록 더 무작위):')
    # 같은 RSI 범위에서 액션의 다양성
    for rsi_range in [(20,30), (30,40), (40,60), (60,70), (70,80)]:
        range_df = df[(df['rsi'] >= rsi_range[0]) & (df['rsi'] < rsi_range[1])]
        if len(range_df) > 0:
            actions = range_df['action'].value_counts(normalize=True)
            entropy = -sum(p * np.log(p) if p > 0 else 0 for p in actions.values)
            print(f'RSI {rsi_range}: 엔트로피 {entropy:.2f} (샘플: {len(range_df)}개)')
    
    # 패턴의 일관성 검사
    print('\n📈 패턴 일관성 검사:')
    
    # RSI가 낮은데(과매도) BUY하지 않는 비율
    oversold_df = df[df['rsi'] < 30]
    if len(oversold_df) > 0:
        not_buy_ratio = (oversold_df['action'] != 'BUY').mean() * 100
        print(f'RSI < 30에서 BUY하지 않는 비율: {not_buy_ratio:.1f}%')
    
    # RSI가 높은데(과매수) SELL하지 않는 비율
    overbought_df = df[df['rsi'] > 70]
    if len(overbought_df) > 0:
        not_sell_ratio = (overbought_df['action'] != 'SELL').mean() * 100
        print(f'RSI > 70에서 SELL하지 않는 비율: {not_sell_ratio:.1f}%')
    
    # 수익률과 액션의 상관관계
    print('\n💰 수익률과 액션의 관계:')
    for action in ['BUY', 'HOLD', 'SELL']:
        action_df = df[df['action'] == action]
        avg_return = action_df['return_1d'].mean() * 100
        print(f'{action}: 평균 1일 수익률 {avg_return:.2f}%')
    
    # 랜덤 시드의 영향 분석
    print('\n🎯 데이터 생성의 결정론적 패턴 검사:')
    # timestamp가 같으면 비슷한 행동을 하는지
    timestamp_groups = df.groupby('timestamp')['action'].apply(lambda x: x.value_counts(normalize=True).to_dict())
    
    # 각 timestamp에서 가장 빈번한 액션의 비율
    dominant_ratios = []
    for ts_actions in timestamp_groups:
        if isinstance(ts_actions, dict) and ts_actions:
            max_ratio = max(ts_actions.values())
            dominant_ratios.append(max_ratio)
    
    avg_dominance = np.mean(dominant_ratios) if dominant_ratios else 0
    print(f'평균 지배적 액션 비율: {avg_dominance:.2f} (낮을수록 더 다양함)')
    
    # 결론
    print('\n🔍 분석 결론:')
    if entropy < 0.5:
        print('⚠️ 노이즈가 부족합니다. 데이터가 너무 예측 가능합니다.')
    elif avg_dominance > 0.7:
        print('⚠️ 특정 조건에서 너무 일관된 행동을 보입니다.')
    else:
        print('✅ 적절한 수준의 노이즈와 다양성이 있습니다.')

if __name__ == "__main__":
    analyze_data_quality()