import pandas as pd
import numpy as np

def prepare_baseline_data_v2(input_file, output_file):
    """Baseline 모델을 위한 데이터 준비 v2 - 시장 초과 수익률 기반 라벨링"""
    
    print("📊 Loading enriched data with fundamentals...")
    df = pd.read_csv(input_file)
    print(f"Total episodes: {len(df):,}")
    
    # 시장 초과 수익률 계산
    if 'excess_return' not in df.columns:
        if 'market_return_during_holding' in df.columns:
            df['excess_return'] = df['return_pct'] - df['market_return_during_holding']
            print("✅ Calculated excess_return from market_return_during_holding")
        else:
            print("❌ ERROR: market_return_during_holding column not found!")
            return None
    
    # 결측치 확인
    print(f"\n📊 Market data coverage:")
    market_coverage = df['market_return_during_holding'].notna().sum()
    print(f"  - Episodes with market data: {market_coverage:,} ({market_coverage/len(df)*100:.1f}%)")
    
    # 시장 데이터가 있는 것만 사용
    df_with_market = df[df['excess_return'].notna()].copy()
    print(f"  - Using {len(df_with_market):,} episodes with complete market data")
    
    # 1. Baseline features 선택 (v2: 시장 지표 포함)
    baseline_features = [
        # 기본 정보
        'holding_period_days',
        'position_size_pct',
        
        # Entry 시점 기술적 지표
        'entry_momentum_5d',
        'entry_momentum_20d',
        'entry_momentum_60d',
        'entry_volatility_5d', 
        'entry_volatility_20d',
        'entry_volatility_60d',
        'entry_ma_dev_5d',
        'entry_ma_dev_20d',
        'entry_ma_dev_60d',
        'entry_vol_change_5d',
        'entry_vol_change_20d',
        'entry_vol_change_60d',
        'entry_ratio_52w_high',
        
        # 시장 상황 지표 (v2 추가)
        'market_entry_ma_return_5d',
        'market_entry_ma_return_20d',
        'market_entry_cum_return_5d',
        'market_entry_cum_return_20d',
        'market_entry_volatility_20d',
        
        # 재무 지표 (있으면 포함)
        'entry_pe_ratio',
        'entry_pb_ratio',
        'entry_roe',
        'entry_operating_margin',
        'entry_debt_equity_ratio',
        
        # 카테고리 정보
        'symbol',
        'industry'
    ]
    
    # 사용 가능한 features만 선택
    available_features = [f for f in baseline_features if f in df_with_market.columns]
    print(f"\n📋 Available features: {len(available_features)}/{len(baseline_features)}")
    
    # 2. 여러 Target 생성 옵션
    # 주요 라벨: 시장 초과 수익률 > 0
    df_with_market['label'] = (df_with_market['excess_return'] > 0).astype(int)
    
    # 추가 라벨링 옵션들
    df_with_market['label_excess_1pct'] = (df_with_market['excess_return'] > 1.0).astype(int)
    df_with_market['label_excess_2pct'] = (df_with_market['excess_return'] > 2.0).astype(int)
    df_with_market['label_simple_0.5pct'] = (df_with_market['return_pct'] > 0.005).astype(int)
    
    # 3. 필요한 컬럼만 선택
    columns_to_keep = available_features + ['return_pct', 'excess_return', 'market_return_during_holding',
                                           'label', 'label_excess_1pct', 'label_excess_2pct', 'label_simple_0.5pct']
    baseline_df = df_with_market[columns_to_keep].copy()
    
    # 4. 결측치 처리
    print("\n🔧 Handling missing values...")
    missing_before = baseline_df.isnull().sum().sum()
    
    # 수치형 컬럼만 선택하여 결측치 처리
    numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns
    
    # 재무 지표는 평균값으로, 기술적 지표는 0으로 채우기
    financial_cols = ['entry_pe_ratio', 'entry_pb_ratio', 'entry_roe', 
                     'entry_operating_margin', 'entry_debt_equity_ratio']
    
    for col in numeric_cols:
        if col in financial_cols and col in baseline_df.columns:
            # 재무 지표는 평균값으로 채우기
            baseline_df[col] = baseline_df[col].fillna(baseline_df[col].mean())
        else:
            # 기술적 지표는 0으로 채우기
            baseline_df[col] = baseline_df[col].fillna(0)
    
    missing_after = baseline_df.isnull().sum().sum()
    print(f"Missing values: {missing_before} → {missing_after}")
    
    # 5. 라벨 분포 확인
    print("\n📈 Label distributions:")
    print("\n1. Main label (excess_return > 0):")
    print(baseline_df['label'].value_counts())
    print(baseline_df['label'].value_counts(normalize=True))
    
    print("\n2. Excess 1% label:")
    print(baseline_df['label_excess_1pct'].value_counts(normalize=True))
    
    print("\n3. Simple 0.5% label (original):")
    print(baseline_df['label_simple_0.5pct'].value_counts(normalize=True))
    
    # 6. 시장 대비 성과 통계
    print("\n📊 Market-adjusted performance statistics:")
    print(f"Average return: {baseline_df['return_pct'].mean():.2f}%")
    print(f"Average market return: {baseline_df['market_return_during_holding'].mean():.2f}%")
    print(f"Average excess return: {baseline_df['excess_return'].mean():.2f}%")
    
    # 라벨별 평균 수익률
    print("\n📊 Returns by label:")
    for label_col in ['label', 'label_excess_1pct', 'label_simple_0.5pct']:
        print(f"\n{label_col}:")
        for val in [0, 1]:
            mask = baseline_df[label_col] == val
            if mask.sum() > 0:
                avg_return = baseline_df[mask]['return_pct'].mean()
                avg_excess = baseline_df[mask]['excess_return'].mean()
                print(f"  Label={val}: avg_return={avg_return:.2f}%, avg_excess={avg_excess:.2f}%")
    
    # 7. 저장
    baseline_df.to_csv(output_file, index=False)
    print(f"\n✅ Baseline data v2 saved to: {output_file}")
    print(f"   Shape: {baseline_df.shape}")
    
    return baseline_df

def create_train_test_splits_v2(baseline_df, test_size=0.2, val_size=0.1):
    """Train/Validation/Test 분할 - v2"""
    
    # 시간 순서 유지를 위해 인덱스 기준 분할
    n = len(baseline_df)
    
    # 시간순 분할 (과거 데이터로 학습, 최근 데이터로 테스트)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    train_df = baseline_df.iloc[:train_end]
    val_df = baseline_df.iloc[train_end:val_end]
    test_df = baseline_df.iloc[val_end:]
    
    print(f"\n📊 Data splits (temporal):")
    print(f"Train: {len(train_df):,} ({len(train_df)/n:.1%})")
    print(f"Val:   {len(val_df):,} ({len(val_df)/n:.1%})")
    print(f"Test:  {len(test_df):,} ({len(test_df)/n:.1%})")
    
    # 라벨 분포 확인
    print(f"\n🎯 Label distribution per split (main label):")
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        pos_ratio = split_df['label'].mean()
        print(f"{name}: {pos_ratio:.1%} positive (beat market)")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # 1. Baseline 데이터 준비 v2
    baseline_df = prepare_baseline_data_v2(
        '../../results/final/enriched_trading_episodes_with_fundamentals.csv',
        '../data/baseline_trading_episodes_v2.csv'
    )
    
    if baseline_df is not None:
        # 2. Train/Val/Test 분할
        train_df, val_df, test_df = create_train_test_splits_v2(baseline_df)
        
        # 분할 데이터 저장
        train_df.to_csv('../data/baseline_train_v2.csv', index=False)
        val_df.to_csv('../data/baseline_val_v2.csv', index=False)
        test_df.to_csv('../data/baseline_test_v2.csv', index=False)
        
        print("\n✅ All v2 files created successfully!")