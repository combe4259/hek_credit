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
    개별 거래 품질 평가를 위한 순수 기술적 지표만 추출
    - symbol, industry 제외 (종목 특성 배제)
    - 거래 시점의 기술적 상태만으로 품질 평가
    """
    
    # Baseline features - 순수하게 거래 품질과 관련된 지표만
    baseline_features = [
        # 거래 특성
        'holding_period_days',      # 보유 기간
        'position_size_pct',        # 포지션 크기 (리스크 노출도)
        
        # 진입 시점 모멘텀 (단기/중기 추세)
        'entry_momentum_5d',        # 단기 모멘텀
        'entry_momentum_20d',       # 중기 모멘텀
        
        # 진입 시점 변동성 (리스크 수준)
        'entry_volatility_5d',      # 단기 변동성
        'entry_volatility_20d',     # 중기 변동성
        
        # 진입 시점 이동평균 괴리 (과매수/과매도 상태)
        'entry_ma_dev_5d',          # 단기 MA 괴리
        'entry_ma_dev_20d',         # 중기 MA 괴리
        
        # 진입 시점 거래량 변화 (시장 관심도)
        'entry_vol_change_5d',      # 단기 거래량 변화
        'entry_vol_change_20d',     # 중기 거래량 변화
        
        # 진입 시점 상대 위치 (타이밍)
        'entry_ratio_52w_high',     # 52주 고점 대비 위치
    ]
    
    X = df[baseline_features].copy()
    y = df['label']
    
    # 데이터 검증
    print(f"✅ 선택된 특성 수: {len(baseline_features)}개")
    print(f"✅ 결측치: {X.isnull().sum().sum()}개")
    
    return X, y

def evaluate_trade_quality_model(model, X, y, dataset_name):
    """개별 거래 품질 평가 모델 성능 측정"""
    
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
    print(f"  우수 거래 정확히 식별: {true_positive_rate:.1%}")
    print(f"  불량 거래를 우수로 잘못 분류: {false_positive_rate:.1%}")
    
    return metrics, y_prob

def train_trade_quality_baseline():
    """개별 거래 품질 평가 Baseline 모델 학습"""
    
    print("🎯 개별 거래 품질 평가 모델 학습")
    print("=" * 50)
    
    # 1. 데이터 로드
    print("\n📂 데이터 로딩 중...")
    train_df = pd.read_csv('../data/baseline_train.csv')
    val_df = pd.read_csv('../data/baseline_val.csv')
    test_df = pd.read_csv('../data/baseline_test.csv')
    
    print(f"학습용: {len(train_df):,}개 거래")
    print(f"검증용: {len(val_df):,}개 거래")
    print(f"테스트용: {len(test_df):,}개 거래")
    
    # 2. Feature 준비 (symbol, industry 제외)
    print("\n🔧 특성 준비 중 (기술적 지표만 사용)...")
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    
    # 3. 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 모델 정의
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'  # 클래스 불균형 처리
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
            scale_pos_weight=sum(y_train==0)/sum(y_train==1)  # 클래스 불균형 처리
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
    
    # 5. 최종 결과 요약
    print(f"\n{'='*50}")
    print(f"🏆 최고 성능 모델: {best_model_name}")
    print(f"   검증 F1 점수: {best_val_f1:.4f}")
    print(f"   테스트 F1 점수: {results[best_model_name]['test_metrics']['f1']:.4f}")
    
    # 6. 거래 품질 점수 분포 분석
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
    
    print("\n품질 점수별 실제 수익률:")
    print(score_analysis)
    
    # 7. 모델 저장
    import os
    os.makedirs('../models', exist_ok=True)
    
    joblib.dump(best_model, '../models/baseline_model.pkl')
    joblib.dump(scaler, '../models/baseline_scaler.pkl')
    
    # 메타데이터 저장
    metadata = {
        'model_name': best_model_name,
        'features': list(X_train.columns),
        'val_f1': best_val_f1,
        'test_f1': results[best_model_name]['test_metrics']['f1'],
        'label_threshold': 0.005,  # return_pct > 0.5%
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }
    
    with open('../models/baseline_metadata.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ 모델이 ../models/ 폴더에 저장되었습니다")
    print(f"✅ 이 baseline은 기술적 지표만으로 개별 거래 품질을 평가합니다")
    print(f"✅ 종목/산업 편향 없음 - 순수 기술적 분석")
    
    return results, best_model

def analyze_trade_quality_patterns(model_path='../models/baseline_model.pkl'):
    """거래 품질 패턴 심층 분석"""
    
    print("\n🔍 거래 품질 패턴 분석 중...")
    print("=" * 50)
    
    # 모델과 데이터 로드
    model = joblib.load(model_path)
    scaler = joblib.load('../models/baseline_scaler.pkl')
    test_df = pd.read_csv('../data/baseline_test.csv')
    
    X_test, y_test = prepare_features(test_df)
    
    # 품질 점수 계산
    if hasattr(model, 'coef_'):  # Logistic Regression
        X_test_scaled = scaler.transform(X_test)
        quality_scores = model.predict_proba(X_test_scaled)[:, 1]
    else:
        quality_scores = model.predict_proba(X_test)[:, 1]
    
    test_df['quality_score'] = quality_scores
    
    # 1. 고품질 거래 특성
    print("\n📊 고품질 거래 특성 (상위 20%):")
    high_quality = test_df[test_df['quality_score'] >= test_df['quality_score'].quantile(0.8)]
    low_quality = test_df[test_df['quality_score'] <= test_df['quality_score'].quantile(0.2)]
    
    features = prepare_features(test_df)[0].columns
    for feature in features[:5]:  # 상위 5개 특성만
        high_avg = high_quality[feature].mean()
        low_avg = low_quality[feature].mean()
        print(f"  {feature}: 상위={high_avg:.2f}, 하위={low_avg:.2f}, 차이={high_avg-low_avg:.2f}")
    
    # 2. 실제 수익률과의 관계
    print("\n💰 품질 점수와 실제 수익률 관계:")
    correlation = test_df['quality_score'].corr(test_df['return_pct'])
    print(f"  상관계수: {correlation:.4f}")
    
    # 3. 거래 품질별 추천
    print("\n🎯 거래 품질별 분석:")
    for score_range, label in [(0.8, '최상급'), (0.6, '우수'), (0.4, '보통'), (0.2, '미흡')]:
        mask = test_df['quality_score'] >= score_range
        if mask.sum() > 0:
            subset = test_df[mask]
            print(f"  {label} 거래 (점수 ≥ {score_range:.1f}):")
            print(f"    - 거래 수: {len(subset):,}개")
            print(f"    - 평균 수익률: {subset['return_pct'].mean():.2%}")
            print(f"    - 승률: {(subset['return_pct'] > 0.005).mean():.1%}")

if __name__ == "__main__":
    # 1. 개별 거래 품질 평가 모델 학습
    results, model = train_trade_quality_baseline()
    
    # 2. 거래 품질 패턴 분석
    analyze_trade_quality_patterns()