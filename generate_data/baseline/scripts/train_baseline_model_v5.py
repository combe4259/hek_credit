import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
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
    """평가 함수"""
    
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
    
    return metrics, y_prob

def train_trade_quality_baseline():
    """개별 거래 품질 평가 Baseline 모델 학습 - v5 (GridSearch 추가)"""
    
    print("🎯 개별 거래 품질 평가 모델 학습 v5 (GridSearch 하이퍼파라미터 튜닝)")
    print("==" * 30)
    
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
    
    # 4. Feature 준비
    print("\n🔧 특성 준비 중...")
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    
    # 5. 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. GridSearch로 모델별 하이퍼파라미터 튜닝
    print("\n🔍 GridSearch로 하이퍼파라미터 튜닝 시작...")
    
    # GridSearch 설정
    cv_folds = 3  # 시간 절약을 위해 3-fold
    scoring = 'f1'  # F1 score로 최적화
    
    models_params = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs'],
                'class_weight': ['balanced']
            },
            'use_scaled': True
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 300, 500],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 5],
                'class_weight': ['balanced']
            },
            'use_scaled': False
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, scale_pos_weight=sum(y_train==0)/sum(y_train==1)),
            'params': {
                'n_estimators': [100, 300, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            },
            'use_scaled': False
        }
    }
    
    best_models = {}
    results = {}
    
    for name, config in models_params.items():
        print(f"\n{'='*50}")
        print(f"🔍 {name} GridSearch 진행 중...")
        
        # GridSearch 실행
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # 스케일링된 데이터 또는 원본 데이터 사용
        X_train_use = X_train_scaled if config['use_scaled'] else X_train
        X_val_use = X_val_scaled if config['use_scaled'] else X_val
        X_test_use = X_test_scaled if config['use_scaled'] else X_test
        
        # GridSearch 실행
        grid_search.fit(X_train_use, y_train)
        
        # 최적 모델
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        print(f"\n🏆 {name} 최적 파라미터:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\n📊 교차검증 최고 F1 점수: {grid_search.best_score_:.4f}")
        
        # 검증 및 테스트 데이터로 평가
        val_metrics, val_prob = evaluate_trade_quality_model(best_model, X_val_use, y_val, "검증 데이터")
        test_metrics, test_prob = evaluate_trade_quality_model(best_model, X_test_use, y_test, "테스트 데이터")
        
        results[name] = {
            'cv_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model': best_model,
            'use_scaled': config['use_scaled']
        }
        
        # Feature Importance (tree 기반 모델만)
        if hasattr(best_model, 'feature_importances_'):
            print(f"\n📊 중요 특성 순위:")
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for idx, row in feature_importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 7. 최고 성능 모델 선택
    best_model_name = max(results.keys(), key=lambda k: results[k]['val_metrics']['f1'])
    best_result = results[best_model_name]
    
    print(f"\n{'='*60}")
    print(f"🏆 최고 성능 모델: {best_model_name}")
    print(f"   교차검증 F1: {best_result['cv_score']:.4f}")
    print(f"   검증 F1: {best_result['val_metrics']['f1']:.4f}")
    print(f"   테스트 F1: {best_result['test_metrics']['f1']:.4f}")
    
    # 8. 모든 모델 성능 비교
    print(f"\n📊 모델별 성능 비교 (테스트 데이터):")
    print(f"{'Model':<20} {'F1':<8} {'AUC':<8} {'Precision':<10} {'Recall':<8}")
    print("-" * 55)
    for name, result in results.items():
        metrics = result['test_metrics']
        print(f"{name:<20} {metrics['f1']:<8.4f} {metrics['auc']:<8.4f} {metrics['precision']:<10.4f} {metrics['recall']:<8.4f}")
    
    # 9. 거래 품질 점수 분포 분석
    best_model = best_result['model']
    use_scaled = best_result['use_scaled']
    X_test_final = X_test_scaled if use_scaled else X_test
    
    print(f"\n📈 거래 품질 점수 분포 ({best_model_name}):")
    test_df['quality_score'] = best_model.predict_proba(X_test_final)[:, 1]
    
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
    
    joblib.dump(best_model, '../models/baseline_model_v5.pkl')
    joblib.dump(scaler, '../models/baseline_scaler_v5.pkl')
    
    # 메타데이터 저장
    metadata = {
        'model_name': best_model_name,
        'best_params': best_result['best_params'],
        'features': list(X_train.columns),
        'cv_f1': best_result['cv_score'],
        'val_f1': best_result['val_metrics']['f1'],
        'test_f1': best_result['test_metrics']['f1'],
        'use_scaled_data': use_scaled,
        'label_type': 'percentile_30_70',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }
    
    with open('../models/baseline_metadata_v5.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ 모델이 ../models/ 폴더에 저장되었습니다")
    print(f"✅ V5는 GridSearch로 최적화된 하이퍼파라미터 사용")
    
    return results, best_model

if __name__ == "__main__":
    results, model = train_trade_quality_baseline()