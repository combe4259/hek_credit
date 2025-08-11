import pandas as pd
import numpy as np

def prepare_baseline_data(input_file, output_file):
    """Baseline 모델을 위한 데이터 준비"""
    
    print("📊 Loading enriched data...")
    df = pd.read_csv(input_file)
    print(f"Total episodes: {len(df):,}")
    
    # 1. Baseline features 선택 (간단한 것만)
    baseline_features = [
        # 기본 정보
        'holding_period_days',
        'position_size_pct',
        
        # Entry 시점 기술적 지표
        'entry_momentum_5d',
        'entry_momentum_20d',
        'entry_volatility_5d', 
        'entry_volatility_20d',
        'entry_ma_dev_5d',
        'entry_ma_dev_20d',
        'entry_vol_change_5d',
        'entry_vol_change_20d',
        'entry_ratio_52w_high',
        
        # 카테고리 정보 (나중에 인코딩 필요)
        'symbol',
        'industry'
    ]
    
    # 2. Target 생성
    df['label'] = (df['return_pct'] > 0.005).astype(int)
    
    # 3. 필요한 컬럼만 선택
    columns_to_keep = baseline_features + ['return_pct', 'label']
    baseline_df = df[columns_to_keep].copy()
    
    # 4. 결측치 처리
    print("\n🔧 Handling missing values...")
    missing_before = baseline_df.isnull().sum().sum()
    
    # 0으로 채우기 (보유기간에 따라 계산 안 된 지표들)
    numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns
    baseline_df[numeric_cols] = baseline_df[numeric_cols].fillna(0)
    
    missing_after = baseline_df.isnull().sum().sum()
    print(f"Missing values: {missing_before} → {missing_after}")
    
    # 5. 데이터 검증
    print("\n📈 Data validation:")
    print(f"Shape: {baseline_df.shape}")
    print(f"\nLabel distribution:")
    print(baseline_df['label'].value_counts())
    print(f"\nLabel ratio:")
    print(baseline_df['label'].value_counts(normalize=True))
    
    # 6. Feature 통계
    print("\n📊 Feature statistics:")
    for col in baseline_features[:5]:  # 처음 5개만 출력
        if col in numeric_cols:
            mean_val = baseline_df[col].mean()
            std_val = baseline_df[col].std()
            print(f"{col}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    # 7. 저장
    baseline_df.to_csv(output_file, index=False)
    print(f"\n✅ Baseline data saved to: {output_file}")
    
    # 8. 추가 정보
    print("\n💡 Next steps:")
    print("1. One-hot encode 'industry' column")
    print("2. Label encode 'symbol' or drop it")
    print("3. Scale features if using algorithms sensitive to scale")
    print("4. Split into train/val/test sets")
    
    return baseline_df

def create_train_test_splits(baseline_df, test_size=0.2, val_size=0.1):
    """Train/Validation/Test 분할"""
    from sklearn.model_selection import train_test_split
    
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
    print(f"\n🎯 Label distribution per split:")
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        pos_ratio = split_df['label'].mean()
        print(f"{name}: {pos_ratio:.1%} positive")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # 1. Baseline 데이터 준비
    baseline_df = prepare_baseline_data(
        '../results/final/enriched_trading_episodes.csv',
        '../results/final/baseline_trading_episodes.csv'
    )
    
    # 2. Train/Val/Test 분할 (선택적)
    train_df, val_df, test_df = create_train_test_splits(baseline_df)
    
    # 분할 데이터 저장
    train_df.to_csv('../results/final/baseline_train.csv', index=False)
    val_df.to_csv('../results/final/baseline_val.csv', index=False) 
    test_df.to_csv('../results/final/baseline_test.csv', index=False)