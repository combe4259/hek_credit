#!/usr/bin/env python3
"""
데이터 증강 (Data Augmentation)
기존 2,432개 데이터를 50,000개로 증강
"""

import pandas as pd
import numpy as np
from config import RANDOM_SEED

class TradingDataAugmenter:
    """매매 데이터 증강기"""
    
    def __init__(self, target_size=50000):
        self.target_size = target_size
        self.random_state = np.random.RandomState(RANDOM_SEED)
        
    def augment_dataset(self, csv_path='output/advanced_trading_data.csv'):
        """데이터셋 증강"""
        print(f"🔄 데이터 증강 시작: 목표 {self.target_size:,}개")
        
        # 원본 데이터 로드
        original_df = pd.read_csv(csv_path)
        print(f"📊 원본 데이터: {len(original_df):,}개")
        
        # 증강 비율 계산
        augment_ratio = self.target_size / len(original_df)
        print(f"🎯 증강 배수: {augment_ratio:.1f}배")
        
        augmented_data = []
        
        # 원본 데이터 추가
        augmented_data.append(original_df)
        print("✅ 원본 데이터 추가")
        
        # 증강 데이터 생성
        remaining_size = self.target_size - len(original_df)
        
        # 1. 노이즈 변형 (30%)
        noise_size = int(remaining_size * 0.3)
        noise_data = self._generate_noise_variations(original_df, noise_size)
        augmented_data.append(noise_data)
        print(f"✅ 노이즈 변형: {len(noise_data):,}개")
        
        # 2. 시간 이동 (25%)  
        time_size = int(remaining_size * 0.25)
        time_data = self._generate_time_variations(original_df, time_size)
        augmented_data.append(time_data)
        print(f"✅ 시간 변형: {len(time_data):,}개")
        
        # 3. 수익률 스케일링 (25%)
        scale_size = int(remaining_size * 0.25)
        scale_data = self._generate_scale_variations(original_df, scale_size)
        augmented_data.append(scale_data)
        print(f"✅ 스케일 변형: {len(scale_data):,}개")
        
        # 4. 혼합 변형 (20%)
        mix_size = remaining_size - noise_size - time_size - scale_size
        mix_data = self._generate_mixed_variations(original_df, mix_size)
        augmented_data.append(mix_data)
        print(f"✅ 혼합 변형: {len(mix_data):,}개")
        
        # 최종 결합
        final_df = pd.concat(augmented_data, ignore_index=True)
        
        # 셔플
        final_df = final_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        print(f"🎉 최종 데이터: {len(final_df):,}개")
        
        # 저장
        output_path = 'output/advanced_trading_data_augmented.csv'
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"💾 저장 완료: {output_path}")
        
        # 통계 출력
        self._print_statistics(final_df)
        
        return final_df
    
    def _generate_noise_variations(self, df, size):
        """노이즈 변형 생성"""
        variations = []
        
        for _ in range(size):
            # 랜덤 샘플 선택
            sample = df.sample(1, random_state=self.random_state).iloc[0].copy()
            
            # 🎯 기술적 지표에 노이즈 추가
            sample['rsi'] = self._add_bounded_noise(sample['rsi'], 5, 0, 100)  # RSI ±5
            sample['bb_position'] = self._add_bounded_noise(sample['bb_position'], 0.1, 0, 1)  # BB ±0.1
            sample['volume_ratio'] = self._add_bounded_noise(sample['volume_ratio'], 0.2, 0.1, 5)  # 거래량 ±20%
            sample['daily_return'] = self._add_bounded_noise(sample['daily_return'], 0.01, -0.1, 0.1)  # ±1%
            sample['gap'] = self._add_bounded_noise(sample['gap'], 0.005, -0.05, 0.05)  # ±0.5%
            
            # 🎯 수익률 정보에 미세 노이즈
            sample['return_1d'] = self._add_bounded_noise(sample['return_1d'], 0.005, -0.2, 0.2)
            sample['return_7d'] = self._add_bounded_noise(sample['return_7d'], 0.01, -0.5, 0.5)  
            sample['return_30d'] = self._add_bounded_noise(sample['return_30d'], 0.02, -1, 1)
            
            # 🎯 패턴 점수에 노이즈 (투자자 성향)
            pattern_cols = ['profit_taking_tendency', 'stop_loss_tendency', 'volatility_reaction',
                           'time_based_trading', 'technical_indicator_reliance', 
                           'chart_pattern_recognition', 'volume_reaction', 'candle_analysis']
            
            for col in pattern_cols:
                if col in sample.index:
                    sample[col] = self._add_bounded_noise(sample[col], 0.1, 0, 1)
            
            variations.append(sample)
            
            # 진행률 표시
            if len(variations) % 1000 == 0:
                print(f"  📈 노이즈 변형 진행: {len(variations):,}/{size:,}")
        
        return pd.DataFrame(variations)
    
    def _generate_time_variations(self, df, size):
        """시간 기반 변형 생성"""
        variations = []
        
        for _ in range(size):
            sample = df.sample(1, random_state=self.random_state).iloc[0].copy()
            
            # 🎯 타임스탬프 이동 (±30일)
            time_shift = self.random_state.randint(-30, 31)
            sample['timestamp'] = max(0, sample['timestamp'] + time_shift)
            
            # 🎯 시간대에 따른 미세 조정
            # 다른 시간대에서는 패턴이 살짝 다를 수 있음
            time_factor = self.random_state.uniform(0.95, 1.05)  # ±5% 변동
            
            # 기술적 지표에 시간 요소 반영
            sample['rsi'] = self._add_bounded_noise(sample['rsi'], 2 * time_factor, 0, 100)
            sample['volume_ratio'] = sample['volume_ratio'] * time_factor
            
            variations.append(sample)
            
            if len(variations) % 1000 == 0:
                print(f"  ⏰ 시간 변형 진행: {len(variations):,}/{size:,}")
        
        return pd.DataFrame(variations)
    
    def _generate_scale_variations(self, df, size):
        """스케일 변형 생성 (수익률 조정)"""
        variations = []
        
        for _ in range(size):
            sample = df.sample(1, random_state=self.random_state).iloc[0].copy()
            
            # 🎯 수익률 스케일링 (80%-120%)
            scale_factor = self.random_state.uniform(0.8, 1.2)
            
            # 수익률 관련 컬럼들 스케일링
            return_cols = ['return_1d', 'return_7d', 'return_30d', 'daily_return', 'gap']
            for col in return_cols:
                if col in sample.index:
                    sample[col] = sample[col] * scale_factor
            
            # 🎯 수익률에 따라 액션도 조정될 수 있음
            if scale_factor < 0.9 and sample['action'] == 'BUY':
                # 수익률이 낮아지면 매수 -> 보유로 변경 가능성
                if self.random_state.random() < 0.3:  # 30% 확률
                    sample['action'] = 'HOLD'
                    sample['reasoning'] = '관망'
            
            variations.append(sample)
            
            if len(variations) % 1000 == 0:
                print(f"  📊 스케일 변형 진행: {len(variations):,}/{size:,}")
        
        return pd.DataFrame(variations)
    
    def _generate_mixed_variations(self, df, size):
        """혼합 변형 생성 (여러 기법 조합)"""
        variations = []
        
        for _ in range(size):
            sample = df.sample(1, random_state=self.random_state).iloc[0].copy()
            
            # 🎯 복합적 변형
            # 1. 노이즈 추가
            sample['rsi'] = self._add_bounded_noise(sample['rsi'], 3, 0, 100)
            sample['bb_position'] = self._add_bounded_noise(sample['bb_position'], 0.05, 0, 1)
            
            # 2. 시간 이동
            sample['timestamp'] = sample['timestamp'] + self.random_state.randint(-15, 16)
            
            # 3. 미세 스케일링
            scale = self.random_state.uniform(0.9, 1.1)
            sample['daily_return'] = sample['daily_return'] * scale
            sample['volume_ratio'] = sample['volume_ratio'] * scale
            
            # 4. 투자자 성향 미세 조정
            sentiment_shift = self.random_state.uniform(-0.05, 0.05)
            pattern_cols = ['profit_taking_tendency', 'stop_loss_tendency']  # 핵심 성향만
            for col in pattern_cols:
                if col in sample.index:
                    sample[col] = np.clip(sample[col] + sentiment_shift, 0, 1)
            
            variations.append(sample)
            
            if len(variations) % 1000 == 0:
                print(f"  🎭 혼합 변형 진행: {len(variations):,}/{size:,}")
        
        return pd.DataFrame(variations)
    
    def _add_bounded_noise(self, value, noise_std, min_val, max_val):
        """제한된 범위 내에서 노이즈 추가"""
        noise = self.random_state.normal(0, noise_std)
        new_value = value + noise
        return np.clip(new_value, min_val, max_val)
    
    def _print_statistics(self, df):
        """통계 정보 출력"""
        print("\n📈 증강된 데이터셋 통계:")
        print(f"   - 총 레코드: {len(df):,}개")
        
        # 액션 분포
        print("\n🎯 액션 분포:")
        action_counts = df['action'].value_counts()
        for action, count in action_counts.items():
            percentage = count / len(df) * 100
            print(f"   - {action}: {count:,}개 ({percentage:.1f}%)")
        
        # 수익률 분포
        return_cols = ['return_1d', 'return_7d', 'return_30d']
        for col in return_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(f"   - {col}: 평균 {mean_val:.3f}, 표준편차 {std_val:.3f}")
        
        # RSI 분포
        if 'rsi' in df.columns:
            rsi_mean = df['rsi'].mean()
            rsi_std = df['rsi'].std()
            print(f"   - RSI: 평균 {rsi_mean:.1f}, 표준편차 {rsi_std:.1f}")

if __name__ == "__main__":
    augmenter = TradingDataAugmenter(target_size=50000)
    augmented_df = augmenter.augment_dataset('output/advanced_trading_data.csv')
    print("🎉 고급 데이터 증강 완료!")