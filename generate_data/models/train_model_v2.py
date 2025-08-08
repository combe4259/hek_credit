import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_quality_score_label(df, w1=0.4, w2=0.3, w3=0.3):
    """
    리스크 조정 품질 점수 라벨 생성
    """
    df = df.copy()
    
    # NaN 값 먼저 처리
    df = df.fillna(0)
    
    # 1. 리스크 조정 수익률
    risk_adj_return = df['return_pct'] / (df['position_size_pct'] * df['holding_period_days'] + 0.01)
    
    # 2. 시장 조건 점수 (모멘텀/변동성 비율)
    market_condition_score = df['entry_momentum_5d'] / (df['entry_volatility_5d'] + 0.01)
    
    # 3. 보유 기간 점수 (짧을수록 좋음)
    holding_period_score = np.log(df['holding_period_days'] + 1)
    
    # 무한대와 NaN 처리
    risk_adj_return = np.nan_to_num(risk_adj_return, nan=0.0, posinf=1.0, neginf=-1.0)
    market_condition_score = np.nan_to_num(market_condition_score, nan=0.0, posinf=1.0, neginf=-1.0)
    holding_period_score = np.nan_to_num(holding_period_score, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # MinMax 정규화
    scaler_risk = MinMaxScaler()
    scaler_market = MinMaxScaler()
    scaler_holding = MinMaxScaler()
    
    # 정규화할 때도 안전하게 처리
    try:
        normalized_risk_adj_return = scaler_risk.fit_transform(risk_adj_return.reshape(-1, 1)).flatten()
    except:
        normalized_risk_adj_return = np.zeros_like(risk_adj_return)
    
    try:
        normalized_market_condition = scaler_market.fit_transform(market_condition_score.reshape(-1, 1)).flatten()
    except:
        normalized_market_condition = np.zeros_like(market_condition_score)
    
    try:
        normalized_holding_period = scaler_holding.fit_transform(holding_period_score.reshape(-1, 1)).flatten()
    except:
        normalized_holding_period = np.zeros_like(holding_period_score)
    
    # 보유기간은 역수로 변환 (짧을수록 높은 점수)
    normalized_holding_period = 1 - normalized_holding_period
    
    # 최종 품질 점수 계산
    quality_score = (w1 * normalized_risk_adj_return + 
                    w2 * normalized_market_condition + 
                    w3 * normalized_holding_period)
    
    # 최종 NaN 처리
    quality_score = np.nan_to_num(quality_score, nan=0.5)
    df['quality_score'] = quality_score
    
    print(f"\n📊 Quality Score 생성:")
    print(f"  가중치: Risk({w1:.1f}), Market({w2:.1f}), Holding({w3:.1f})")
    print(f"  평균: {quality_score.mean():.4f}")
    print(f"  표준편차: {quality_score.std():.4f}")
    print(f"  범위: {quality_score.min():.4f} ~ {quality_score.max():.4f}")
    
    # 정규화 객체들도 반환 (예측시 필요)
    scalers = {
        'risk': scaler_risk,
        'market': scaler_market, 
        'holding': scaler_holding
    }
    
    return df, scalers

def prepare_advanced_features(df):
    """
    정교한 특징 엔지니어링 (데이터 리키지 제거)
    Quality Score 계산에 사용된 변수들 제외:
    - return_pct, position_size_pct, holding_period_days, entry_momentum_5d, entry_volatility_5d
    """
    
    # 기본 특징들 (리키지 변수 제외)
    features = [
        # === 진입 시점 기술적 지표 ===
        'entry_momentum_20d', 'entry_momentum_60d',  # entry_momentum_5d 제외
        'entry_volatility_20d', 'entry_volatility_60d',  # entry_volatility_5d 제외
        'entry_ma_dev_5d', 'entry_ma_dev_20d', 'entry_ma_dev_60d',
        'entry_vol_change_5d', 'entry_vol_change_20d', 'entry_vol_change_60d',
        'entry_ratio_52w_high',
        
        # === 청산 시점 기술적 지표 ===
        'exit_momentum_5d', 'exit_momentum_20d', 'exit_momentum_60d',
        'exit_volatility_5d', 'exit_volatility_20d', 'exit_volatility_60d',
        'exit_ma_dev_5d', 'exit_ma_dev_20d', 'exit_ma_dev_60d',
        'exit_vol_change_5d', 'exit_vol_change_20d', 'exit_vol_change_60d',
        'exit_ratio_52w_high',
        
        # === 시장 상황 지표 ===
        'market_entry_ma_return_5d', 'market_entry_ma_return_20d',
        'market_entry_cum_return_5d', 'market_entry_cum_return_20d',
        'market_entry_volatility_20d', 'market_return_during_holding',
        
        # === 재무 지표 ===
        'entry_pe_ratio', 'entry_pb_ratio', 'entry_roe',
        'entry_operating_margin', 'entry_debt_equity_ratio',
        
        # === 변화량 지표 ===
        'change_momentum_5d', 'change_momentum_20d', 'change_momentum_60d',
        'change_ma_dev_5d', 'change_ma_dev_20d', 'change_ma_dev_60d',
        'change_volatility_5d', 'change_volatility_20d', 'change_volatility_60d',
        'change_ratio_52w_high'
    ]
    
    # 사용 가능한 특징만 선택
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].copy()
    
    # === 고급 특징 엔지니어링 (데이터 리키지 제거) ===
    
    # 1. 리스크 조정 지표들 (리키지 변수 사용 안 함)
    X['momentum_volatility_ratio_20d'] = df['entry_momentum_20d'] / (df['entry_volatility_20d'] + 0.01)
    X['momentum_volatility_ratio_60d'] = df['entry_momentum_60d'] / (df['entry_volatility_60d'] + 0.01)
    
    # 2. 시장 대비 상대 성과 (return_pct 사용 안 함)
    if 'market_entry_volatility_20d' in df.columns:
        X['relative_volatility'] = df['entry_volatility_20d'] / (df['market_entry_volatility_20d'] + 0.01)
    
    # 3. 타이밍 지표들
    X['entry_timing_score'] = (df['entry_momentum_20d'] * df['entry_ratio_52w_high']) / (df['entry_volatility_20d'] + 0.01)
    X['exit_timing_score'] = (df['exit_momentum_20d'] * df['exit_ratio_52w_high']) / (df['exit_volatility_20d'] + 0.01)
    
    # 4. 변화율 지표들 (holding_period_days 사용 안 함)
    X['momentum_change_ratio'] = df['change_momentum_20d'] / (df['entry_momentum_20d'] + 0.01)
    X['volatility_stability'] = df['entry_volatility_20d'] / (df['change_volatility_20d'].abs() + 0.01)
    
    # 6. 시장 조건 종합 점수
    if all(col in df.columns for col in ['market_entry_ma_return_20d', 'market_entry_volatility_20d']):
        X['market_condition_score'] = df['market_entry_ma_return_20d'] / (df['market_entry_volatility_20d'] + 0.01)
    
    # 7. 재무 건전성 종합 점수
    if all(col in df.columns for col in ['entry_roe', 'entry_debt_equity_ratio', 'entry_operating_margin']):
        X['financial_health_score'] = (df['entry_roe'] * df['entry_operating_margin']) / (df['entry_debt_equity_ratio'] + 0.01)
    
    # 5. 기술적 강도 지표
    X['technical_strength'] = (df['entry_momentum_20d'] * df['entry_ratio_52w_high']) / (df['entry_ma_dev_20d'].abs() + 0.01)
    
    # 6. 진입-청산 비교 지표
    X['momentum_consistency'] = df['exit_momentum_20d'] / (df['entry_momentum_20d'] + 0.01)
    X['volatility_trend'] = df['exit_volatility_20d'] / (df['entry_volatility_20d'] + 0.01)
    
    # 결측치 처리
    X = X.fillna(0)
    
    # 무한대 값 처리
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"\n🔧 고급 특징 엔지니어링 완료 (데이터 리키지 제거):")
    print(f"  기본 특징: {len(available_features)}개")
    print(f"  엔지니어링된 특징: {len(X.columns) - len(available_features)}개")
    print(f"  총 특징 수: {len(X.columns)}개")
    print(f"  제외된 리키지 변수: return_pct, position_size_pct, holding_period_days, entry_momentum_5d, entry_volatility_5d")
    
    return X

def evaluate_regression_model(model, X, y, dataset_name):
    """회귀 모델 평가"""
    
    y_pred = model.predict(X)
    
    # 기본 메트릭
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mse)
    
    # 상관계수
    correlation = np.corrcoef(y, y_pred)[0, 1]
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation
    }
    
    print(f"\n📊 {dataset_name} 성능:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  상관계수: {correlation:.4f}")
    
    return metrics, y_pred

def optimize_quality_score_weights():
    """가중치 최적화를 통해 상/하위 20% 구간 간 수익률 차이 최대화"""
    
    print("🎯 Quality Score 가중치 최적화")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n📂 데이터 로딩 중...")
    df = pd.read_csv('../results/final/enriched_trading_episodes_with_fundamentals.csv')
    
    # Train/Val 분할 (검증 데이터로 최적화)
    n_total = len(df)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    
    val_df = df.iloc[n_train:n_train+n_val].copy()
    print(f"검증 데이터: {len(val_df):,}개")
    
    # 여러 초기값 후보들
    initial_weight_candidates = [
        [0.4, 0.3, 0.3],  # 현재
        [0.7, 0.2, 0.1],  # Risk 위주
        [0.2, 0.7, 0.1],  # Market 위주
        [0.6, 0.35, 0.05],  # 수정된 가중치
        [0.34, 0.33, 0.33]  # 균등
    ]
    
    # 기본 초기값
    initial_weights = [0.4, 0.3, 0.3]
    
    def calculate_bucket_return_difference(weights):
        """가중치에 따른 상/하위 20% 구간 수익률 차이 계산"""
        if len(weights) != 3:
            print(f"Warning: weights length is {len(weights)}, expected 3. Weights: {weights}")
            return 0
        w1, w2, w3 = weights
        
        # Quality score 계산
        quality_scores, _ = create_quality_score_label(val_df, w1, w2, w3)
        
        # 상/하위 20% 구간 분할
        val_df_sorted = quality_scores.sort_values('quality_score')
        n = len(val_df_sorted)
        bottom_20 = val_df_sorted.iloc[:int(n*0.2)]
        top_20 = val_df_sorted.iloc[int(n*0.8):]
        
        # 수익률 차이 (상위 - 하위)
        return_diff = top_20['return_pct'].mean() - bottom_20['return_pct'].mean()
        
        return return_diff
    
    def objective_function(weights):
        """최적화 목적함수 (차이를 최대화하므로 음수 반환)"""
        return -calculate_bucket_return_difference(weights)
    
    # 2. 제약조건 설정
    from scipy.optimize import differential_evolution
    from scipy.optimize import LinearConstraint
    
    # 각 가중치는 0 이상 1 이하
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    
    # 가중치 합이 1이 되도록 제약 (LinearConstraint 사용)
    # A @ x = b 형태: [1, 1, 1] @ [w1, w2, w3] = 1
    constraints = LinearConstraint([[1, 1, 1]], [1], [1])
    
    # 현재 성능 확인
    current_diff = calculate_bucket_return_difference(initial_weights)
    print(f"\n📊 현재 가중치 성능:")
    print(f"  가중치: Risk({initial_weights[0]:.1f}), Market({initial_weights[1]:.1f}), Holding({initial_weights[2]:.1f})")
    print(f"  수익률 차이: {current_diff:.4f} (상위 20% - 하위 20%)")
    
    # 3. 여러 초기값으로 최적화 시도
    print(f"\n🔍 가중치 최적화 중 (여러 초기값으로 시도)...")
    
    best_result = None
    best_diff = current_diff
    best_weights = initial_weights
    
    for i, candidate in enumerate(initial_weight_candidates):
        print(f"\n--- 시도 {i+1}/{len(initial_weight_candidates)}: {candidate} ---")
        
        try:
            # 각 후보로 성능 확인
            candidate_diff = calculate_bucket_return_difference(candidate)
            print(f"후보 성능: {candidate_diff:.4f}")
            
            # Differential Evolution 실행
            result = differential_evolution(
                objective_function,
                bounds=bounds,
                constraints=constraints,
                seed=42+i,  # 다른 시드 사용
                maxiter=30,  # 반복횟수 줄임
                popsize=10,  # 인구 크기 줄임
                polish=True,
                disp=False  # 출력 줄임
            )
            
            if result.success:
                opt_diff = calculate_bucket_return_difference(result.x)
                print(f"최적화 결과: {result.x} → 성능: {opt_diff:.4f}")
                
                if opt_diff > best_diff:
                    best_result = result
                    best_diff = opt_diff
                    best_weights = result.x
                    print(f"✅ 새로운 최고 성능!")
            else:
                print(f"❌ 최적화 실패: {result.message}")
                
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            continue
    
    # 최종 결과 설정
    if best_result and best_result.success:
        result = best_result
        optimal_weights = best_weights
        optimal_diff = best_diff
    else:
        print(f"\n🔄 모든 최적화 시도 실패 - 수동으로 후보 중 최고 선택")
        # 후보들 중 가장 좋은 것 선택
        best_candidate = initial_weights
        for candidate in initial_weight_candidates:
            candidate_diff = calculate_bucket_return_difference(candidate)
            if candidate_diff > best_diff:
                best_diff = candidate_diff
                best_candidate = candidate
        
        optimal_weights = best_candidate
        optimal_diff = best_diff
        result = type('obj', (object,), {'success': best_diff > current_diff})()
    
    print(f"\n🏆 최종 최적화 결과:")
    print(f"  최적 가중치: Risk({optimal_weights[0]:.3f}), Market({optimal_weights[1]:.3f}), Holding({optimal_weights[2]:.3f})")
    print(f"  최적화된 수익률 차이: {optimal_diff:.4f}")
    print(f"  개선폭: {optimal_diff - current_diff:.4f}")
    
    return optimal_weights, current_diff, optimal_diff, result.success if hasattr(result, 'success') else True

def train_quality_score_model_with_optimized_weights(optimal_weights):
    """최적화된 가중치로 거래 품질 점수 예측 모델 학습"""
    
    print("🎯 거래 품질 점수 예측 모델 학습 V2 (최적화된 가중치)")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n📂 데이터 로딩 중...")
    df = pd.read_csv('../results/final/enriched_trading_episodes_with_fundamentals.csv')
    print(f"전체 데이터: {len(df):,}개")
    
    # 2. Quality Score 라벨 생성 (최적화된 가중치 사용)
    print("\n🏷️ Quality Score 라벨 생성 중 (최적화된 가중치)...")
    df, scalers = create_quality_score_label(df, optimal_weights[0], optimal_weights[1], optimal_weights[2])
    
    # NaN 체크
    print(f"Quality Score NaN 확인: {df['quality_score'].isna().sum()}개")
    
    # 3. 고급 특징 엔지니어링
    print("\n⚙️ 고급 특징 엔지니어링 중...")
    X = prepare_advanced_features(df)
    y = df['quality_score']
    
    # 특징과 라벨 최종 확인
    print(f"\n🔍 최종 데이터 확인:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  X에 NaN: {X.isna().sum().sum()}개")
    print(f"  X에 Inf: {np.isinf(X).sum().sum()}개")
    print(f"  y에 NaN: {y.isna().sum()}개")
    print(f"  y에 Inf: {np.isinf(y).sum()}개")
    
    # 4. Train/Val/Test 분할 (시간순)
    print("\n📊 데이터 분할 중...")
    n_total = len(X)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    
    X_train = X.iloc[:n_train]
    X_val = X.iloc[n_train:n_train+n_val]
    X_test = X.iloc[n_train+n_val:]
    
    y_train = y.iloc[:n_train]
    y_val = y.iloc[n_train:n_train+n_val]
    y_test = y.iloc[n_train+n_val:]
    
    print(f"  학습용: {len(X_train):,}개")
    print(f"  검증용: {len(X_val):,}개")
    print(f"  테스트용: {len(X_test):,}개")
    
    # Quality Score 분포 확인
    print(f"\n📊 Quality Score 분포:")
    for name, data in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        print(f"  {name}: 평균 {data.mean():.4f}, 표준편차 {data.std():.4f}")
    
    # 5. 특징 스케일링
    print("\n🔧 특징 스케일링 중...")
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # 6. 회귀 모델들 정의 (간소화된 파라미터)
    models_params = {
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [300],
                'max_depth': [15],
                'min_samples_split': [5],
                'min_samples_leaf': [2]
            },
            'use_scaled': False
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=42),
            'params': {
                'n_estimators': [300],
                'max_depth': [6],
                'learning_rate': [0.1],
                'subsample': [0.9],
                'colsample_bytree': [0.9]
            },
            'use_scaled': False
        },
        'Ridge': {
            'model': Ridge(random_state=42),
            'params': {
                'alpha': [1.0]
            },
            'use_scaled': True
        }
    }
    
    # 7. GridSearch로 각 모델 최적화
    print("\n🔍 모델 학습 중...")
    
    best_models = {}
    results = {}
    
    for name, config in models_params.items():
        print(f"\n{'='*50}")
        print(f"🔍 {name} 학습 중...")
        
        # GridSearch
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # 데이터 선택
        X_train_use = X_train_scaled if config['use_scaled'] else X_train
        X_val_use = X_val_scaled if config['use_scaled'] else X_val
        X_test_use = X_test_scaled if config['use_scaled'] else X_test
        
        # 학습
        grid_search.fit(X_train_use, y_train)
        best_model = grid_search.best_estimator_
        
        print(f"\n🏆 {name} 최적 파라미터:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        # 평가
        val_metrics, val_pred = evaluate_regression_model(best_model, X_val_use, y_val, f"{name} 검증")
        test_metrics, test_pred = evaluate_regression_model(best_model, X_test_use, y_test, f"{name} 테스트")
        
        results[name] = {
            'cv_score': -grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model': best_model,
            'use_scaled': config['use_scaled']
        }
        
        # Feature Importance
        if hasattr(best_model, 'feature_importances_'):
            print(f"\n📊 상위 10개 중요 특성:")
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 8. 최고 성능 모델 선택
    best_model_name = max(results.keys(), key=lambda k: results[k]['val_metrics']['r2'])
    best_result = results[best_model_name]
    
    print(f"\n{'='*60}")
    print(f"🏆 최고 성능 모델: {best_model_name}")
    print(f"   검증 R²: {best_result['val_metrics']['r2']:.4f}")
    print(f"   테스트 R²: {best_result['test_metrics']['r2']:.4f}")
    print(f"   테스트 상관계수: {best_result['test_metrics']['correlation']:.4f}")
    
    # 9. 모든 모델 성능 비교
    print(f"\n📊 모델별 성능 비교 (테스트 데이터):")
    print(f"{'Model':<15} {'R²':<8} {'RMSE':<8} {'상관계수':<8}")
    print("-" * 45)
    for name, result in results.items():
        metrics = result['test_metrics']
        print(f"{name:<15} {metrics['r2']:<8.4f} {metrics['rmse']:<8.4f} {metrics['correlation']:<8.4f}")
    
    # 10. 품질 점수 분포 분석
    best_model = best_result['model']
    use_scaled = best_result['use_scaled']
    X_test_final = X_test_scaled if use_scaled else X_test
    
    pred_scores = best_model.predict(X_test_final)
    
    print(f"\n📈 예측 품질 점수 분석:")
    print(f"  실제 점수 범위: {y_test.min():.4f} ~ {y_test.max():.4f}")
    print(f"  예측 점수 범위: {pred_scores.min():.4f} ~ {pred_scores.max():.4f}")
    
    # 점수 구간별 분석
    test_df = df.iloc[n_train+n_val:].copy()
    test_df['predicted_score'] = pred_scores
    test_df['score_bin'] = pd.qcut(pred_scores, q=5, labels=['최하', '하', '중', '상', '최상'])
    
    score_analysis = test_df.groupby('score_bin').agg({
        'return_pct': ['mean', 'std'],
        'quality_score': ['mean'],
        'predicted_score': ['mean']
    })
    
    print("\n📊 예측 점수별 실제 성과:")
    print(score_analysis)
    
    # 11. 모델 저장
    print(f"\n💾 모델 저장 중...")
    
    # 최고 모델 저장
    joblib.dump(best_model, 'quality_model_v2.pkl')
    joblib.dump(feature_scaler, 'feature_scaler_v2.pkl')
    joblib.dump(scalers, 'label_scalers_v2.pkl')
    
    # 특성 정보 저장
    feature_info = {
        'feature_names': list(X.columns),
        'model_name': best_model_name,
        'best_params': best_result['best_params'],
        'use_scaled_data': use_scaled,
        'test_r2': best_result['test_metrics']['r2'],
        'test_correlation': best_result['test_metrics']['correlation'],
        'optimized_weights': optimal_weights.tolist()
    }
    
    with open('model_metadata_v2.json', 'w') as f:
        import json
        json.dump(feature_info, f, indent=2)
    
    print(f"✅ 모델이 저장되었습니다:")
    print(f"  - quality_model_v2.pkl")
    print(f"  - feature_scaler_v2.pkl") 
    print(f"  - label_scalers_v2.pkl")
    print(f"  - model_metadata_v2.json")
    
    return results, best_model, scalers

if __name__ == "__main__":
    # 1. 가중치 최적화
    optimal_weights, current_diff, optimal_diff, success = optimize_quality_score_weights()
    
    # 2. 최적화 결과 저장
    optimization_results = {
        'original_weights': [0.4, 0.3, 0.3],
        'optimal_weights': optimal_weights.tolist() if hasattr(optimal_weights, 'tolist') else list(optimal_weights),
        'original_return_diff': float(current_diff),
        'optimal_return_diff': float(optimal_diff),
        'improvement': float(optimal_diff - current_diff),
        'optimization_success': bool(success)
    }
    
    import json
    with open('weight_optimization_results_v2.json', 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    print(f"\n✅ 가중치 최적화 결과가 weight_optimization_results_v2.json에 저장되었습니다.")
    
    # 3. 최적화된 가중치로 모델 학습
    results, model, scalers = train_quality_score_model_with_optimized_weights(optimal_weights)