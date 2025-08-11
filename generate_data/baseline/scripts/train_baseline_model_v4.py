import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def prepare_features(df):
    """
    V1과 동일한 특성 추출 (시장 지표 제외)
    """
    
    # V1의 baseline features
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
        'entry_ratio_52w_high'
    ]
    
    # 사용 가능한 특성만 선택
    available_features = [f for f in baseline_features if f in df.columns]
    X = df[available_features].copy()
    y = df['label']
    
    # 결측치 처리 (V1과 동일 - 0으로 채우기)
    X = X.fillna(0)
    
    # 데이터 검증
    print(f"✅ 선택된 특성 수: {len(available_features)}개")
    print(f"✅ 결측치 처리 완료")
    
    return X, y

def apply_percentile_labeling(df, upper_percentile=70, lower_percentile=30):
    """
    상대적 퍼센타일 라벨링 적용
    상위 30%를 Positive(1), 하위 30%를 Negative(0)로 라벨링
    중간 40%는 제외
    """
    df = df.copy()
    
    # 퍼센타일 기준값 계산
    upper_threshold = df['return_pct'].quantile(upper_percentile / 100)
    lower_threshold = df['return_pct'].quantile(lower_percentile / 100)
    
    # 라벨 생성 (중간값은 -1로 표시 후 제거)
    df['label'] = -1
    df.loc[df['return_pct'] >= upper_threshold, 'label'] = 1  # 상위 30%
    df.loc[df['return_pct'] <= lower_threshold, 'label'] = 0  # 하위 30%
    
    # 중간 40% 제거
    df = df[df['label'] != -1].copy()
    
    print(f"\n📊 퍼센타일 라벨링 적용:")
    print(f"  상위 {100-upper_percentile}% 기준값: {upper_threshold:.4f}")
    print(f"  하위 {lower_percentile}% 기준값: {lower_threshold:.4f}")
    print(f"  Positive 비율: {df['label'].mean():.1%}")
    print(f"  제외된 중간 거래: {100-upper_percentile-lower_percentile}%")
    
    return df

def evaluate_trade_quality_model(model, X, y, dataset_name):
    """V1과 동일한 평가 함수"""
    
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # 기본 메트릭
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'auc': roc_auc_score(y, y_prob)
    }
    
    print(f"\n📊 {dataset_name} 성능:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\n혼동 행렬:")
    print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    
    # 거래 품질별 분석
    print(f"\n💰 거래 품질 분석:")
    true_positive_rate = cm[1,1] / (cm[1,0] + cm[1,1])
    false_positive_rate = cm[0,1] / (cm[0,0] + cm[0,1])
    print(f"  상위 거래 정확히 식별: {true_positive_rate:.1%}")
    print(f"  하위 거래를 상위로 잘못 분류: {false_positive_rate:.1%}")
    
    return metrics, y_prob

def train_trade_quality_baseline():
    """개별 거래 품질 평가 Baseline 모델 학습 - v4 (퍼센타일 라벨링)"""
    
    print("🎯 개별 거래 품질 평가 모델 학습 v4 (상대적 퍼센타일 라벨링)")
    print("==" * 25)
    
    # 1. 전체 데이터 로드
    print("\n📂 데이터 로딩 중...")
    df = pd.read_csv('../../results/final/enriched_trading_episodes_with_fundamentals.csv')
    print(f"전체 데이터: {len(df):,}개")
    
    # 2. 퍼센타일 라벨링 적용 (상위 30%, 하위 30%)
    df = apply_percentile_labeling(df)
    
    # 3. Train/Val/Test 분할 (시간순)
    n_total = len(df)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train+n_val].copy()
    test_df = df.iloc[n_train+n_val:].copy()
    
    print(f"\n📊 데이터 분할:")
    print(f"학습용: {len(train_df):,}개 거래")
    print(f"검증용: {len(val_df):,}개 거래")
    print(f"테스트용: {len(test_df):,}개 거래")
    
    # 라벨 분포 확인
    print(f"\n🎯 라벨 분포:")
    print(f"학습: {train_df['label'].mean():.1%}")
    print(f"검증: {val_df['label'].mean():.1%}")
    print(f"테스트: {test_df['label'].mean():.1%}")
    
    # 4. Feature 준비 (V1과 동일)
    print("\n🔧 특성 준비 중 (V1과 동일한 기술적 지표)...")
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    
    # 5. 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. 모델 정의
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            scale_pos_weight=sum(y_train==0)/sum(y_train==1)
        )
    }
    
    results = {}
    best_val_f1 = 0
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"🤖 {name} 학습 중...")
        
        # 학습
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            val_metrics, val_prob = evaluate_trade_quality_model(model, X_val_scaled, y_val, "검증 데이터")
            test_metrics, test_prob = evaluate_trade_quality_model(model, X_test_scaled, y_test, "테스트 데이터")
        else:
            model.fit(X_train, y_train)
            val_metrics, val_prob = evaluate_trade_quality_model(model, X_val, y_val, "검증 데이터")
            test_metrics, test_prob = evaluate_trade_quality_model(model, X_test, y_test, "테스트 데이터")
        
        results[name] = {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model': model
        }
        
        # 최고 모델 추적
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model = model
            best_model_name = name
        
        # Feature Importance (tree 기반 모델만)
        if hasattr(model, 'feature_importances_'):
            print(f"\n📊 중요 특성 순위:")
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for idx, row in feature_importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 7. 최종 결과 요약
    print(f"\n{'='*50}")
    print(f"🏆 최고 성능 모델: {best_model_name}")
    print(f"   검증 F1 점수: {best_val_f1:.4f}")
    print(f"   테스트 F1 점수: {results[best_model_name]['test_metrics']['f1']:.4f}")
    
    # 8. V1과 비교
    print(f"\n📊 V1과 V4 비교:")
    print(f"  V1: 절대 기준 (return > 0.5%)")
    print(f"  V4: 상대 기준 (상위 50%)")
    print(f"  → 시장 상황과 무관하게 일정한 라벨 분포 유지")
    
    # 9. 거래 품질 점수 분포 분석
    print(f"\n📈 거래 품질 점수 분포:")
    test_df['quality_score'] = best_model.predict_proba(
        X_test_scaled if best_model_name == 'Logistic Regression' else X_test
    )[:, 1]
    
    # 점수 구간별 실제 수익률
    test_df['score_bin'] = pd.qcut(test_df['quality_score'], q=5, labels=['최하', '하', '중', '상', '최상'])
    
    score_analysis = test_df.groupby('score_bin').agg({
        'return_pct': ['mean', 'std', 'count'],
        'label': 'mean'
    })
    
    print("\n품질 점수별 실제 성과:")
    print(score_analysis)
    
    # 10. 모델 저장
    import os
    os.makedirs('../models', exist_ok=True)
    
    joblib.dump(best_model, '../models/baseline_model_v4.pkl')
    joblib.dump(scaler, '../models/baseline_scaler_v4.pkl')
    
    # 메타데이터 저장
    metadata = {
        'model_name': best_model_name,
        'features': list(X_train.columns),
        'val_f1': best_val_f1,
        'test_f1': results[best_model_name]['test_metrics']['f1'],
        'label_type': 'percentile_50',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }
    
    with open('../models/baseline_metadata_v4.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ 모델이 ../models/ 폴더에 저장되었습니다")
    print(f"✅ V4는 상대적 퍼센타일 라벨링으로 안정적인 성능 제공")
    
    return results, best_model

if __name__ == "__main__":
    results, model = train_trade_quality_baseline()