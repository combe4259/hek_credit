import pandas as pd
import numpy as np

class MarketComponentRebuilder:
    """
    Market Component 완전 재설계
    - Entry + Exit + 보유기간 전체 고려
    - 올바른 VIX/금리 해석
    - 시장 방향성 및 추세 반영
    """
    
    def __init__(self, data_path='../results/final/enriched_trading_episodes_with_advanced_market_components.csv'):
        self.data_path = data_path
        
    def load_data(self):
        """데이터 로드"""
        print("📊 데이터 로드 중...")
        df = pd.read_csv(self.data_path)
        df['entry_datetime'] = pd.to_datetime(df['entry_datetime'])
        df['exit_datetime'] = pd.to_datetime(df['exit_datetime'])
        
        print(f"총 거래: {len(df):,}개")
        return df
    
    def calculate_vix_score(self, df):
        """
        VIX 기반 시장 점수 계산
        - 낮은 VIX = 좋은 시장 (높은 점수)
        - 높은 VIX = 나쁜 시장 (낮은 점수)
        - Entry + Exit + 변화량 모두 고려
        """
        print("\n📈 VIX 기반 시장 점수 계산...")
        
        # 1. Entry VIX 점수 (낮을수록 좋음)
        # VIX 10 = 1.0 (최고), VIX 80 = 0.0 (최악)
        entry_vix_score = np.clip((80 - df['entry_vix']) / 70, 0, 1)
        
        # 2. Exit VIX 점수
        exit_vix_score = np.clip((80 - df['exit_vix']) / 70, 0, 1)
        
        # 3. VIX 변화 점수 (VIX 하락 = 시장 개선 = 좋음)
        # VIX가 크게 하락(-20) = 1.0, VIX가 크게 상승(+20) = 0.0
        vix_change_score = np.clip(0.5 - df['change_vix'] / 40, 0, 1)
        
        # 4. 종합 VIX 점수 (가중평균)
        vix_score = (
            entry_vix_score * 0.4 +     # 진입 시점 40%
            exit_vix_score * 0.4 +      # 청산 시점 40%  
            vix_change_score * 0.2      # 변화 추세 20%
        )
        
        print(f"  VIX 점수 범위: {vix_score.min():.3f} ~ {vix_score.max():.3f}")
        print(f"  평균: {vix_score.mean():.3f}")
        
        return vix_score
    
    def calculate_rate_score(self, df):
        """
        금리 기반 시장 점수 계산
        - 적정 금리(2-3%) = 좋은 시장
        - 너무 높거나 낮으면 = 나쁜 시장
        - 금리 안정성도 고려
        """
        print("\n💰 금리 기반 시장 점수 계산...")
        
        # 1. Entry 금리 점수 (2-3%가 최적)
        optimal_rate = 2.5
        entry_rate_distance = np.abs(df['entry_tnx_yield'] - optimal_rate)
        entry_rate_score = np.clip(1 - entry_rate_distance / 3, 0, 1)  # 3%차이까지 허용
        
        # 2. Exit 금리 점수
        exit_rate_distance = np.abs(df['exit_tnx_yield'] - optimal_rate)  
        exit_rate_score = np.clip(1 - exit_rate_distance / 3, 0, 1)
        
        # 3. 금리 변화 점수 (안정성 중시)
        # 금리 변화가 적을수록 좋음
        rate_stability_score = np.clip(1 - np.abs(df['change_tnx_yield']) / 2, 0, 1)
        
        # 4. 금리 방향성 점수
        # 현재 금리가 높으면 하락이 좋고, 낮으면 상승이 좋음
        rate_direction_score = np.where(
            df['entry_tnx_yield'] > optimal_rate,
            np.clip(0.5 - df['change_tnx_yield'] / 2, 0, 1),  # 높으면 하락 선호
            np.clip(0.5 + df['change_tnx_yield'] / 2, 0, 1)   # 낮으면 상승 선호
        )
        
        # 5. 종합 금리 점수
        rate_score = (
            entry_rate_score * 0.3 +        # 진입시 적정성 30%
            exit_rate_score * 0.3 +         # 청산시 적정성 30%
            rate_stability_score * 0.2 +    # 안정성 20%
            rate_direction_score * 0.2      # 방향성 20%
        )
        
        print(f"  금리 점수 범위: {rate_score.min():.3f} ~ {rate_score.max():.3f}")
        print(f"  평균: {rate_score.mean():.3f}")
        
        return rate_score
    
    def calculate_momentum_score(self, df):
        """
        기존 주식 모멘텀 기반 점수
        - 상승 모멘텀 = 좋은 시장
        - 낮은 변동성 = 좋은 시장
        """
        print("\n🚀 모멘텀 기반 시장 점수 계산...")
        
        if 'entry_momentum_20d' not in df.columns or 'entry_volatility_20d' not in df.columns:
            print("  모멘텀/변동성 데이터 없음 - 스킵")
            return np.ones(len(df)) * 0.5  # 중립 점수
            
        # 1. 모멘텀 점수 (20일 기준)
        momentum_20d = df['entry_momentum_20d'].fillna(0)
        # 모멘텀 +20% = 1.0, -20% = 0.0
        momentum_score = np.clip((momentum_20d + 20) / 40, 0, 1)
        
        # 2. 변동성 점수 (낮을수록 좋음)
        volatility_20d = df['entry_volatility_20d'].fillna(50)
        # 변동성 10% = 1.0, 100% = 0.0
        volatility_score = np.clip((100 - volatility_20d) / 90, 0, 1)
        
        # 3. 종합 모멘텀 점수
        momentum_total_score = (
            momentum_score * 0.6 +      # 모멘텀 60%
            volatility_score * 0.4      # 변동성 40%
        )
        
        print(f"  모멘텀 점수 범위: {momentum_total_score.min():.3f} ~ {momentum_total_score.max():.3f}")
        print(f"  평균: {momentum_total_score.mean():.3f}")
        
        return momentum_total_score
    
    def calculate_holding_period_weight(self, df):
        """
        보유기간에 따른 가중치 계산
        - 장기 보유일수록 Exit 시점 중요도 증가
        - 단기 보유일수록 Entry 시점 중요도 증가  
        """
        print("\n⏱️  보유기간 가중치 계산...")
        
        holding_days = df['holding_period_days']
        
        # 로그 변환으로 장기/단기 구분
        # 1일 = 0.0, 30일 = 0.5, 365일+ = 1.0
        period_weight = np.clip(np.log(holding_days + 1) / np.log(366), 0, 1)
        
        print(f"  기간 가중치 범위: {period_weight.min():.3f} ~ {period_weight.max():.3f}")
        
        return period_weight
        
    def create_comprehensive_market_score(self, df):
        """종합 시장 점수 생성"""
        print("\n🧠 종합 시장 점수 생성 중...")
        print("="*60)
        
        # 각 컴포넌트 계산
        vix_score = self.calculate_vix_score(df)
        rate_score = self.calculate_rate_score(df) 
        momentum_score = self.calculate_momentum_score(df)
        period_weight = self.calculate_holding_period_weight(df)
        
        # 보유기간에 따른 동적 가중치
        # 단기: VIX 50%, Rate 30%, Momentum 20%
        # 장기: VIX 40%, Rate 40%, Momentum 20%
        
        vix_weight = 0.5 - period_weight * 0.1      # 0.4 ~ 0.5
        rate_weight = 0.3 + period_weight * 0.1     # 0.3 ~ 0.4  
        momentum_weight = 0.2                       # 고정 0.2
        
        # 종합 점수 계산
        comprehensive_score = (
            vix_score * vix_weight +
            rate_score * rate_weight + 
            momentum_score * momentum_weight
        )
        
        print(f"\n📊 종합 시장 점수 통계:")
        print(f"  범위: {comprehensive_score.min():.4f} ~ {comprehensive_score.max():.4f}")
        print(f"  평균: {comprehensive_score.mean():.4f}")
        print(f"  표준편차: {comprehensive_score.std():.4f}")
        
        # return_pct와 상관관계 확인
        correlation = comprehensive_score.corr(df['return_pct'])
        print(f"  return_pct와 상관관계: {correlation:.4f}")
        
        if correlation > 0.05:
            print(f"  ✅ 양의 상관관계 - 정상적!")
        elif correlation < -0.05:
            print(f"  ⚠️ 음의 상관관계 - 여전히 이상함")
        else:
            print(f"  ⚠️ 무상관 - 효과 미약")
            
        return comprehensive_score, {
            'vix_score': vix_score,
            'rate_score': rate_score, 
            'momentum_score': momentum_score,
            'period_weight': period_weight
        }
    
    def validate_market_score(self, df, market_score):
        """시장 점수 검증"""
        print(f"\n🔍 시장 점수 검증")
        print("-"*40)
        
        # 1. 시장 상황별 점수 분포
        print("시장 상황별 평균 점수:")
        
        # VIX 구간별
        vix_low = df['entry_vix'] < 15
        vix_medium = (df['entry_vix'] >= 15) & (df['entry_vix'] < 30) 
        vix_high = df['entry_vix'] >= 30
        
        print(f"  VIX 낮음(<15): {market_score[vix_low].mean():.4f}")
        print(f"  VIX 보통(15-30): {market_score[vix_medium].mean():.4f}")
        print(f"  VIX 높음(30+): {market_score[vix_high].mean():.4f}")
        
        # 2020년 코로나 시기 확인
        covid_period = (df['entry_datetime'] >= '2020-03-01') & (df['entry_datetime'] <= '2020-04-30')
        normal_period = ~covid_period
        
        print(f"\n기간별 평균 점수:")
        print(f"  코로나 시기(2020.3-4): {market_score[covid_period].mean():.4f}")
        print(f"  평상시: {market_score[normal_period].mean():.4f}")
        
        if market_score[covid_period].mean() < market_score[normal_period].mean():
            print(f"  ✅ 코로나 시기 점수가 낮음 - 정상적!")
        else:
            print(f"  ❌ 코로나 시기 점수가 높음 - 이상함")
    
    def run_rebuild(self):
        """전체 재구축 실행"""
        print("🚀 Market Component 완전 재구축 시작")
        print("="*70)
        
        # 1. 데이터 로드
        df = self.load_data()
        
        # 2. 새로운 시장 점수 계산
        market_score, components = self.create_comprehensive_market_score(df)
        
        # 3. 검증
        self.validate_market_score(df, market_score)
        
        # 4. 데이터에 추가
        df['new_market_component'] = market_score
        for name, component in components.items():
            df[f'market_{name}'] = component
            
        # 5. 저장
        output_path = '../results/final/trading_episodes_with_rebuilt_market_component.csv'
        df.to_csv(output_path, index=False)
        
        print(f"\n💾 저장 완료: {output_path}")
        print("="*70)
        print("✅ Market Component 재구축 완료!")
        print("="*70)
        
        return output_path

if __name__ == "__main__":
    rebuilder = MarketComponentRebuilder()
    output_path = rebuilder.run_rebuild()