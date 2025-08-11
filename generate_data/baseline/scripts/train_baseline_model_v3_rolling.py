import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def prepare_features(df):
    """특성 준비 - V2와 동일 (시장 지표 포함)"""
    baseline_features = [
        'holding_period_days',
        'position_size_pct',
        'entry_momentum_5d',
        'entry_momentum_20d',
        'entry_volatility_5d',
        'entry_volatility_20d',
        'entry_ma_dev_5d',
        'entry_ma_dev_20d',
        'entry_vol_change_5d',
        'entry_vol_change_20d',
        'entry_ratio_52w_high',
        'market_entry_ma_return_5d',
        'market_entry_ma_return_20d',
        'market_entry_volatility_20d',
    ]
    
    available_features = [f for f in baseline_features if f in df.columns]
    X = df[available_features].copy()
    y = df['label']  # 시장 초과 수익률 기반
    
    return X, y

def rolling_window_train(data_file, window_months=6, step_months=1):
    """Rolling Window 방식으로 모델 학습 및 평가"""
    
    print("🔄 Rolling Window 학습 (V3)")
    print(f"   Window: {window_months}개월, Step: {step_months}개월")
    print("=" * 60)
    
    # 전체 데이터 로드
    df = pd.read_csv(data_file)
    
    # 원본 데이터에서 날짜 정보 가져오기
    original_df = pd.read_csv('../../results/final/enriched_trading_episodes_with_fundamentals.csv')
    
    # 시장 데이터가 있는 것만 필터링 (v2와 동일)
    original_df = original_df[original_df['market_return_during_holding'].notna()].reset_index(drop=True)
    
    # 날짜 정보 추가
    df['entry_datetime'] = original_df['entry_datetime']
    df['entry_date'] = pd.to_datetime(df['entry_datetime'])
    
    # 날짜 범위 확인
    min_date = df['entry_date'].min()
    max_date = df['entry_date'].max()
    print(f"\n📅 데이터 기간: {min_date.strftime('%Y-%m')} ~ {max_date.strftime('%Y-%m')}")
    
    # Rolling Window 설정
    results = []
    window_delta = timedelta(days=window_months * 30)  # 대략적인 월 계산
    step_delta = timedelta(days=step_months * 30)
    
    # 최소 학습 시작 시점 (첫 window 이후부터)
    current_end = min_date + window_delta
    
    print("\n🚀 Rolling Window 학습 시작...")
    
    while current_end < max_date - timedelta(days=30):  # 테스트용 최소 1달 남기기
        # 학습 기간
        train_start = current_end - window_delta
        train_end = current_end
        
        # 테스트 기간 (다음 1개월)
        test_start = train_end
        test_end = test_start + timedelta(days=30)
        
        # 데이터 필터링
        train_mask = (df['entry_date'] >= train_start) & (df['entry_date'] < train_end)
        test_mask = (df['entry_date'] >= test_start) & (df['entry_date'] < test_end)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        if len(train_df) < 100 or len(test_df) < 50:
            current_end += step_delta
            continue
        
        # 특성 준비
        X_train, y_train = prepare_features(train_df)
        X_test, y_test = prepare_features(test_df)
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 모델 학습 (간단하게 XGBoost만)
        model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            scale_pos_weight=sum(y_train==0)/sum(y_train==1) if sum(y_train==1) > 0 else 1
        )
        model.fit(X_train_scaled, y_train)
        
        # 평가
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # 메트릭 계산
        result = {
            'train_period': f"{train_start.strftime('%Y-%m')} ~ {train_end.strftime('%Y-%m')}",
            'test_period': f"{test_start.strftime('%Y-%m')} ~ {test_end.strftime('%Y-%m')}",
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_label_ratio': y_train.mean(),
            'test_label_ratio': y_test.mean(),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred) if sum(y_pred) > 0 else 0,
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0
        }
        
        results.append(result)
        
        # 진행 상황 출력
        print(f"\n📊 Window {len(results)}:")
        print(f"   Train: {result['train_period']} ({result['train_size']:,}개, 초과비율: {result['train_label_ratio']:.1%})")
        print(f"   Test:  {result['test_period']} ({result['test_size']:,}개, 초과비율: {result['test_label_ratio']:.1%})")
        print(f"   F1: {result['f1']:.4f}, AUC: {result['auc']:.4f}")
        
        # 다음 window로 이동
        current_end += step_delta
    
    # 결과 분석
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("📊 Rolling Window 결과 요약")
    print("=" * 60)
    
    # 전체 평균 성능
    print("\n📈 평균 성능:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"   {metric}: {mean_val:.4f} (±{std_val:.4f})")
    
    # 시간에 따른 성능 변화
    print("\n📊 시간별 F1 Score:")
    for _, row in results_df.tail(5).iterrows():
        print(f"   {row['test_period']}: {row['f1']:.4f}")
    
    # 라벨 분포 안정성
    print("\n🎯 라벨 분포 변동성:")
    print(f"   Train 라벨 비율: {results_df['train_label_ratio'].mean():.1%} (±{results_df['train_label_ratio'].std():.1%})")
    print(f"   Test 라벨 비율: {results_df['test_label_ratio'].mean():.1%} (±{results_df['test_label_ratio'].std():.1%})")
    
    # 결과 저장
    import os
    os.makedirs('../results', exist_ok=True)
    results_df.to_csv('../results/rolling_window_results.csv', index=False)
    print(f"\n✅ 결과가 ../results/rolling_window_results.csv에 저장되었습니다")
    
    return results_df

def compare_with_static_model(rolling_results):
    """정적 모델(V2)과 Rolling Window 비교"""
    
    print("\n\n🔍 정적 모델 vs Rolling Window 비교")
    print("=" * 60)
    
    print("\n📊 비교 결과:")
    print(f"정적 모델 (V2):")
    print(f"   - 고정된 Train/Test 분할")
    print(f"   - Train 라벨: 15.7%, Test 라벨: 58.8%")
    print(f"   - F1 Score: 0.4178")
    
    print(f"\nRolling Window (V3):")
    print(f"   - 동적 학습 (최근 {6}개월)")
    print(f"   - 평균 F1 Score: {rolling_results['f1'].mean():.4f}")
    print(f"   - 최고 F1 Score: {rolling_results['f1'].max():.4f}")
    print(f"   - 최저 F1 Score: {rolling_results['f1'].min():.4f}")
    
    # 안정성 비교
    label_diff = abs(rolling_results['train_label_ratio'] - rolling_results['test_label_ratio'])
    print(f"\n📊 안정성:")
    print(f"   - V2 라벨 차이: {abs(0.157 - 0.588):.1%}")
    print(f"   - V3 평균 라벨 차이: {label_diff.mean():.1%}")

if __name__ == "__main__":
    # 1. Rolling Window 학습
    results = rolling_window_train(
        '../data/baseline_trading_episodes_v2.csv',
        window_months=6,
        step_months=1
    )
    
    # 2. 정적 모델과 비교
    if results is not None and len(results) > 0:
        compare_with_static_model(results)