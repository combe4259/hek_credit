import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BuySignalPredictor:
    """
    매수 신호 예측 모델 - 2단계 가중치 최적화

    1단계: 가중치 최적화 (Grid Search)
    2단계: 하이퍼파라미터 최적화 (GridSearchCV)
    """

    def __init__(self, train_months=36, val_months=6, test_months=6, step_months=3):
        self.model = None
        self.features = None
        self.is_trained = False

        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.step_months = step_months

        self.fold_results = []
        self.best_params = None
        
        # 최적화된 가중치 저장
        self.best_weights = {'technical': 0.40, 'fundamental': 0.30, 'market': 0.30}  # 기본값

    def optimize_weights(self, df_train, verbose=True):
        """1단계: 가중치 최적화 (0.1~0.8 범위, 0.05 단위)"""
        if verbose:
            print("1단계: 매수 신호 가중치 최적화 시작")
        
        # 0.1부터 0.8까지 0.05 단위로 체계적 가중치 조합 생성
        weight_combinations = []
        step = 0.05
        weights_range = np.arange(0.1, 0.85, step)  # 0.1, 0.15, 0.2, ... 0.8
        
        for tech in weights_range:
            for fund in weights_range:
                market = 1.0 - tech - fund
                # market도 0.1~0.8 범위 내에 있어야 함
                if 0.1 <= market <= 0.8:
                    weight_combinations.append({
                        'technical': round(tech, 2),
                        'fundamental': round(fund, 2), 
                        'market': round(market, 2)
                    })
        
        if verbose:
            print(f"총 {len(weight_combinations)}개 가중치 조합 테스트")
        
        best_score = -float('inf')
        best_weights = None
        weight_results = []
        
        # 각 가중치 조합 평가
        for i, weights in enumerate(weight_combinations):
            # 해당 가중치로 라벨 생성
            df_with_labels = self._create_labels_with_weights(df_train, weights)
            
            # 빠른 평가를 위한 기본 XGBoost 모델
            X = self.prepare_features(df_with_labels, verbose=False)
            y = df_with_labels['buy_signal_score']
            
            # 간단한 Cross Validation
            from sklearn.model_selection import cross_val_score
            base_model = xgb.XGBRegressor(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=100,  # 빠른 평가를 위해 적게
                random_state=42,
                tree_method='gpu_hist',
                gpu_id=0
            )
            
            cv_scores = cross_val_score(base_model, X, y, cv=3, scoring='r2')
            avg_score = cv_scores.mean()
            
            weight_results.append({
                'weights': weights,
                'cv_r2': avg_score,
                'cv_std': cv_scores.std()
            })
            
            if avg_score > best_score:
                best_score = avg_score
                best_weights = weights
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  진행상황: {i+1}/{len(weight_combinations)} 완료 (현재 최고: R² = {best_score:.4f})")
        
        self.best_weights = best_weights
        
        if verbose:
            print(f"\n최적 가중치: {best_weights}")
            print(f"최고 성능: R² = {best_score:.4f}")
            
            # 상위 5개 조합 출력
            sorted_results = sorted(weight_results, key=lambda x: x['cv_r2'], reverse=True)
            print(f"\n상위 5개 가중치 조합:")
            for i, result in enumerate(sorted_results[:5], 1):
                w = result['weights']
                print(f"  {i}. T:{w['technical']:.2f} F:{w['fundamental']:.2f} M:{w['market']:.2f} → R²={result['cv_r2']:.4f}")
        
        return weight_results
    
    def _create_labels_with_weights(self, df, weights):
        """주어진 가중치로 라벨 생성"""
        df = df.copy()
        
        # 각 신호별 점수 계산
        technical_score = self._calculate_technical_signals(df)
        fundamental_score = self._calculate_fundamental_signals(df)
        market_score = self._calculate_market_environment_signals(df)
        
        # 가중치 적용
        df['buy_signal_score'] = (
            technical_score * weights['technical'] +
            fundamental_score * weights['fundamental'] +
            market_score * weights['market']
        )
        
        df['buy_signal_score'] = np.clip(df['buy_signal_score'], 0, 100)
        
        return df

    def create_entry_signal_score(self, df, weights=None, verbose=False):
        if verbose:
            print("매수 신호 점수 생성")

        df = df.copy()

        # 가중치 설정 (최적화된 가중치 또는 주어진 가중치 사용)
        if weights is None:
            weights = self.best_weights

        # 1. 기술적 신호
        technical_score = self._calculate_technical_signals(df)

        # 2. 펀더멘털 신호
        fundamental_score = self._calculate_fundamental_signals(df)

        # 3. 시장 환경 신호
        market_score = self._calculate_market_environment_signals(df)

        # 최적화된 가중치로 매수 신호 점수 계산
        df['buy_signal_score'] = (
            technical_score * weights['technical'] +
            fundamental_score * weights['fundamental'] +
            market_score * weights['market']
        )

        df['buy_signal_score'] = np.clip(df['buy_signal_score'], 0, 100)

        if verbose:
            print(f"  매수 신호 점수 생성")
            print(f"  점수 범위: {df['buy_signal_score'].min():.1f} ~ {df['buy_signal_score'].max():.1f}")
            print(f"  점수 평균: {df['buy_signal_score'].mean():.1f}")

        return df

    def _calculate_technical_signals(self, df):
        """기술적 분석 신호 계산"""
        signals = []

        # 1. 모멘텀 신호 (25%)
        momentum_20d = df['entry_momentum_20d'].fillna(0)
        # 매수 신호 = 적당한 하락 후 반등
        momentum_signal = np.where(
            momentum_20d < -15, 20,      # 과도한 하락
            np.where(momentum_20d < -5, 85,   # 적당한 하락
                     np.where(momentum_20d < 5, 70,    # 횡보
                              np.where(momentum_20d < 15, 50, 30))))  # 과열
        signals.append(momentum_signal * 0.25)

        # 2. 이동평균 신호 (25%)
        ma_dev_20d = df['entry_ma_dev_20d'].fillna(0)
        # 매수 신호 = 이평선 아래
        ma_signal = np.where(
            ma_dev_20d < -10, 85,        # 이평선 크게 이탈
            np.where(ma_dev_20d < -5, 70,     # 이탈
                     np.where(ma_dev_20d < 5, 50,      # 이평선 근처
                              np.where(ma_dev_20d < 10, 30, 15))))  # 과열
        signals.append(ma_signal * 0.25)

        # 3. 과매도/과매수 신호 (25%)
        ratio_52w = df['entry_ratio_52w_high'].fillna(0)
        # 매수 신호 = 52주 고점 대비 낮을수록
        oversold_signal = (100 - ratio_52w)
        signals.append(oversold_signal * 0.25)

        # 4. 변동성 신호 (25%)
        volatility_20d = df['entry_volatility_20d'].fillna(0)
        # 매수 신호 = 적당한 변동성이
        vol_signal = np.where(
            volatility_20d < 15, 40,     # 너무 낮음
            np.where(volatility_20d < 30, 85,     # 적정 변동성
                     np.where(volatility_20d < 50, 60, 20)))  # 위험
        signals.append(vol_signal * 0.25)

        return np.sum(signals, axis=0)

    def _calculate_fundamental_signals(self, df):
        """펀더멘털 분석 신호 계산)"""
        signals = []

        # 1. 밸류에이션 신호 (40%)
        pe_ratio = df['entry_pe_ratio'].fillna(0)
        # 매수 신호 = 낮은 PER
        pe_signal = np.where(
            pe_ratio < 5, 30,           # 낮음
            np.where(pe_ratio < 15, 85,      # 저평가
                     np.where(pe_ratio < 25, 60,      # 적정 가치
                              np.where(pe_ratio < 40, 35, 15))))  # 고평가
        signals.append(pe_signal * 0.4)

        # 2. 품질 신호 (30%)
        roe = df['entry_roe'].fillna(0)
        # 매수 신호 = 높은 ROE
        roe_signal = np.where(
            roe < 5, 30,               # 낮은 품질
            np.where(roe < 10, 50,          # 평균적 품질
                     np.where(roe < 15, 70,          # 양호한 품질
                              np.where(roe < 20, 85, 95))))   # 우수한 품질
        signals.append(roe_signal * 0.3)

        # 3. 성장성 신호 (30%)
        earnings_growth = df['entry_earnings_growth'].fillna(0)
        # 매수 신호 = 적당한 성장
        growth_signal = np.where(
            earnings_growth < -10, 20,   # 역성장
            np.where(earnings_growth < 0, 40,    # 감소
                     np.where(earnings_growth < 10, 70,   # 적당한 성장
                              np.where(earnings_growth < 25, 85, 60))))  # 고성장
        signals.append(growth_signal * 0.3)

        return np.sum(signals, axis=0)

    def _calculate_market_environment_signals(self, df):
        """시장 환경 신호 계산 """
        signals = []

        # 1. VIX 신호 (40%)
        vix = df['entry_vix'].fillna(0)
        # 매수 신호 = 낮은 VIX
        vix_signal = np.where(
            vix < 15, 90,              # 매우 안정
            np.where(vix < 20, 80,          # 안정
                     np.where(vix < 25, 60,          # 보통
                              np.where(vix < 35, 40, 20))))   # 불안정
        signals.append(vix_signal * 0.4)

        # 2. 금리 환경 신호 (30%)
        tnx_yield = df['entry_tnx_yield'].fillna(0)
        # 매수 신호 =  적정 금리
        rate_signal = np.where(
            tnx_yield < 1, 60,         # 너무 낮음
            np.where(tnx_yield < 3, 85,     # 적정 금리
                     np.where(tnx_yield < 5, 60, 30)))  # 높음
        signals.append(rate_signal * 0.3)

        # 3. 시장 추세 신호 (30%)
        market_return_20d = df.get('market_entry_cum_return_20d', pd.Series([0]*len(df))).fillna(0)
        # 매수 신호 = 적당한 상승 추세
        trend_signal = np.where(
            market_return_20d < -10, 30,   # 강한 하락
            np.where(market_return_20d < -5, 60,    # 약한 하락
                     np.where(market_return_20d < 5, 85,      # 횡보/적당한 상승
                              np.where(market_return_20d < 10, 70, 40))))  # 과열
        signals.append(trend_signal * 0.3)

        return np.sum(signals, axis=0)

    def prepare_features(self, df, verbose=False):
        """매수 신호 예측 피처 """
        if verbose:
            print("매수 신호 예측용 피처")

        #라벨링 포함된 피처 제외
        excluded_features = {
            'entry_momentum_20d', 'entry_ma_dev_20d', 'entry_ratio_52w_high',
            'entry_volatility_20d', 'entry_pe_ratio', 'entry_roe',
            'entry_earnings_growth', 'entry_vix', 'entry_tnx_yield',
            'market_entry_cum_return_20d', 'buy_signal_score'
        }

        # 사용 가능한 피처
        available_features = []

        # ===== 1. 기본 기술적 지표  =====
        technical_features = [
            # 다른 기간 모멘텀 지표
            'entry_momentum_5d', 'entry_momentum_60d',

            # 다른 기간 이동평균 기반 지표
            'entry_ma_dev_5d', 'entry_ma_dev_60d',

            # 다른 기간 변동성 지표
            'entry_volatility_5d', 'entry_volatility_60d',

            # 변동성 변화율
            'entry_vol_change_5d', 'entry_vol_change_20d', 'entry_vol_change_60d'
        ]
        available_features.extend([col for col in technical_features if col in df.columns])

        # ===== 2. 추가 펀더멘털 지표 =====
        additional_fundamental_features = [
            'entry_pb_ratio',           # P/B 비율
            'entry_operating_margin',   # 영업이익률
            'entry_debt_equity_ratio'   # 부채비율
        ]
        available_features.extend([col for col in additional_fundamental_features if col in df.columns])

        # ===== 3. 시장 환경 지표  =====
        additional_market_features = [
            # 다른 기간 시장 수익률
            'market_entry_ma_return_5d', 'market_entry_ma_return_20d',
            'market_entry_cum_return_5d',
            'market_entry_volatility_20d'
        ]
        available_features.extend([col for col in additional_market_features if col in df.columns])

        # ===== 4. 거래 관련 정보 =====
        trading_features = [
            'position_size_pct'  # 포지션 크기
        ]
        available_features.extend([col for col in trading_features if col in df.columns])


        self.features = [col for col in available_features
                         if col in df.columns and col not in excluded_features]

        if verbose:
            print(f"  사용 피처: {len(self.features)}개")

        feature_data = df[self.features].select_dtypes(include=[np.number])

        if verbose and len(feature_data.columns) != len(self.features):
            print(f"  비숫자형 컬럼 제외: {len(self.features) - len(feature_data.columns)}개")

        return feature_data

    def train_model_with_weight_optimization(self, df, hyperparameter_search=False, verbose=False):
        """2단계 최적화: 가중치 → 하이퍼파라미터"""
        if verbose:
            print("2단계 최적화 매수 신호 모델 훈련 시작")
            print("=" * 50)
        
        # 펀더멘털 데이터가 있는 것만 필터링
        df_filtered = df[
            df['entry_pe_ratio'].notna() | 
            df['entry_roe'].notna() | 
            df['entry_earnings_growth'].notna()
        ].copy()
        
        if verbose:
            filter_ratio = len(df_filtered) / len(df) * 100
            print(f"펀더멘털 데이터 필터링: {len(df_filtered):,}개 ({filter_ratio:.1f}%)")
        
        # Train/Val 분할
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df_filtered, test_size=0.2, random_state=42)
        
        # 1단계: 가중치 최적화
        if verbose:
            print(f"\n1단계: 가중치 최적화")
        weight_results = self.optimize_weights(train_df, verbose=verbose)
        
        # 2단계: 최적 가중치로 하이퍼파라미터 최적화
        if verbose:
            print(f"\n2단계: 하이퍼파라미터 최적화")
            print(f"최적 가중치 사용: {self.best_weights}")
        
        # 최적 가중치로 전체 데이터 라벨링
        df_with_score = self.create_entry_signal_score(df_filtered, weights=self.best_weights, verbose=verbose)
        
        # 피처 준비
        X = self.prepare_features(df_with_score, verbose=verbose)
        y = df_with_score['buy_signal_score']
        
        # 하이퍼파라미터 최적화
        if hyperparameter_search:
            best_params = self._optimize_hyperparameters(X, y, verbose=verbose)
        else:
            # 기본 파라미터
            best_params = {
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 400,
                'subsample': 0.8,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.1,
                'reg_lambda': 1.5,
                'random_state': 42
            }
        
        # 최종 모델 훈련
        best_params.update({
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        })
        self.model = xgb.XGBRegressor(**best_params)
        self.model.fit(X, y)
        
        # 성능 평가
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        self.is_trained = True
        
        if verbose:
            print(f"\n2단계 최적화 완료")
            print(f"  최종 R² Score: {r2:.4f}")
            print(f"  최종 RMSE: {rmse:.4f}")
            print(f"  사용된 가중치: {self.best_weights}")
        
        return {
            'model': self.model,
            'r2_score': r2,
            'rmse': rmse,
            'best_params': best_params,
            'best_weights': self.best_weights,
            'weight_results': weight_results,
            'feature_count': len(self.features)
        }

    def train_model(self, df, hyperparameter_search=False, verbose=False):
        """매수 신호 예측 모델 훈련"""
        if verbose:
            print("매수 신호 모델 훈련 시작")

        # 펀더멘털 데이터가 있는 것만 필터링
        df_filtered = df[
            df['entry_pe_ratio'].notna() |
            df['entry_roe'].notna() |
            df['entry_earnings_growth'].notna()
            ].copy()

        if verbose:
            filter_ratio = len(df_filtered) / len(df) * 100
            print(f"펀더멘털 데이터 필터링: {len(df_filtered):,}개 ({filter_ratio:.1f}%)")

        # 매수 신호 점수 생성
        df_with_score = self.create_entry_signal_score(df_filtered, verbose=verbose)

        # 피처 준비
        X = self.prepare_features(df_with_score, verbose=verbose)
        y = df_with_score['buy_signal_score']

        # 하이퍼파라미터 최적화
        if hyperparameter_search:
            best_params = self._optimize_hyperparameters(X, y, verbose=verbose)
        else:
            # 기본 파라미터
            best_params = {
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 400,
                'subsample': 0.8,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.1,
                'reg_lambda': 1.5,
                'random_state': 42
            }

        # 최종 모델 훈련
        best_params.update({
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        })
        self.model = xgb.XGBRegressor(**best_params)
        self.model.fit(X, y)

        # 성능 평가
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        self.is_trained = True

        if verbose:
            print(f"  매수 신호 모델 훈련 완료")
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")

        return {
            'model': self.model,
            'r2_score': r2,
            'rmse': rmse,
            'best_params': best_params,
            'feature_count': len(self.features)
        }

    def predict_entry_signal(self, df, verbose=False):
        """매수 신호 강도 예측 """
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다. train_model()을 먼저 실행하세요.")

        if verbose:
            print("매수 신호 강도 예측")

        # 피처 준비
        X = self.prepare_features(df, verbose=False)

        # 예측
        predictions = self.model.predict(X)

        # 0-100 범위 보장
        predictions = np.clip(predictions, 0, 100)

        if verbose:
            print(f"  {len(predictions)}개 종목의 매수 신호 예측 완료")
            print(f"  신호 강도 범위: {predictions.min():.1f} ~ {predictions.max():.1f}")
            print(f"  평균 신호 강도: {predictions.mean():.1f}")

        return predictions

    def get_signal_interpretation(self, score):
        """매수 신호 점수 해석"""
        if score >= 80:
            return "매우 강한 매수 신호"
        elif score >= 70:
            return "강한 매수 신호"
        elif score >= 60:
            return "중간 매수 신호"
        elif score >= 50:
            return "약한 매수 신호"
        else:
            return "매수 부적합"

    def _optimize_hyperparameters(self, X, y, verbose=False):
        """하이퍼파라미터 최적화"""
        if verbose:
            print("  하이퍼파라미터 최적화 시작")

        param_grid = {
            'max_depth': [4, 5, 6, 7],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [300, 400, 500],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1.0, 1.5, 2.0]
        }


        base_model = xgb.XGBRegressor(
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0
        )
        tscv = TimeSeriesSplit(n_splits=3)



        search = GridSearchCV(
            base_model, param_grid,
            cv=tscv, scoring='r2',
            verbose=1
        )
        search.fit(X, y)

        if verbose:
            print(f"  최적 R² Score: {search.best_score_:.4f}")

        return search.best_params_

    def save_model(self, filename=None):
        """모델 저장"""
        if not self.is_trained:
            raise ValueError("훈련된 모델이 없음")

        if filename is None:
            filename = f"buy_signal_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        save_data = {
            'model': self.model,
            'features': self.features,
            'model_type': 'BUY_SIGNAL_PREDICTOR',
            'created_at': datetime.now().isoformat()
        }

        joblib.dump(save_data, filename)
        print(f"Buy Signal 모델 저장: {filename}")
        return filename

    def load_model(self, filename):
        """모델 로드"""
        save_data = joblib.load(filename)

        self.model = save_data['model']
        self.features = save_data['features']
        self.is_trained = True

        print(f"Buy Signal 모델 로드: {filename}")
        return True

    # ================================
    # Walk-Forward 학습 파이프라인
    # ================================

    def create_time_folds_deprecated(self, df, verbose=False):
        """시계열 데이터를 위한 Walk-Forward 폴드 생성"""
        if verbose:
            print(" Buy Signal Walk-Forward 시간 폴드 생성")

        df = df.copy()
        df['date'] = pd.to_datetime(df['entry_datetime'])
        df = df.sort_values('date').reset_index(drop=True)

        start_date = df['date'].min()
        end_date = df['date'].max()

        folds = []
        current_start = start_date
        fold_id = 1

        while current_start + pd.DateOffset(months=self.train_months + self.val_months + self.test_months) <= end_date:
            train_end = current_start + pd.DateOffset(months=self.train_months)
            val_end = train_end + pd.DateOffset(months=self.val_months)
            test_end = val_end + pd.DateOffset(months=self.test_months)

            train_mask = (df['date'] >= current_start) & (df['date'] < train_end)
            val_mask = (df['date'] >= train_end) & (df['date'] < val_end)
            test_mask = (df['date'] >= val_end) & (df['date'] < test_end)

            if train_mask.sum() > 1000 and val_mask.sum() > 100 and test_mask.sum() > 100:
                fold_info = {
                    'fold_id': fold_id,
                    'train_start': current_start.strftime('%Y-%m-%d'),
                    'train_end': train_end.strftime('%Y-%m-%d'),
                    'val_start': train_end.strftime('%Y-%m-%d'),
                    'val_end': val_end.strftime('%Y-%m-%d'),
                    'test_start': val_end.strftime('%Y-%m-%d'),
                    'test_end': test_end.strftime('%Y-%m-%d'),
                    'train_indices': df[train_mask].index.tolist(),
                    'val_indices': df[val_mask].index.tolist(),
                    'test_indices': df[test_mask].index.tolist()
                }
                folds.append(fold_info)
                fold_id += 1

            current_start += pd.DateOffset(months=self.step_months)

        if verbose:
            print(f"  생성된 폴드 수: {len(folds)}개")
            for i, fold in enumerate(folds):
                print(f"  폴드 {i+1}: {fold['train_start']} ~ {fold['test_end']}")
                print(f"    Train: {len(fold['train_indices']):,}개, Val: {len(fold['val_indices']):,}개, Test: {len(fold['test_indices']):,}개")

        return folds

    def run_walk_forward_training_deprecated(self, data_path, hyperparameter_search=True, verbose=True):
        """Buy Signal Walk-Forward 학습 및 평가"""
        if verbose:
            print("Buy Signal Walk-Forward 학습 시작")
            print("="*60)

        # 데이터 로드
        df = pd.read_csv(data_path)
        if verbose:
            print(f"📊 데이터 로드: {len(df):,}개 거래")

        # 펀더멘털 데이터가 있는 것만 필터링
        df_filtered = df[
            df['entry_pe_ratio'].notna() |
            df['entry_roe'].notna() |
            df['entry_earnings_growth'].notna()
            ].copy()

        if verbose:
            filter_ratio = len(df_filtered) / len(df) * 100
            print(f"펀더멘털 데이터 필터링: {len(df_filtered):,}개 ({filter_ratio:.1f}%)")

        # Buy Signal 점수 생성
        df = self.create_entry_signal_score(df_filtered, verbose=verbose)

        # 시간 폴드 생성
        folds = self.create_time_folds(df, verbose=verbose)

        fold_results = []

        for fold_info in tqdm(folds, desc="폴드별 학습"):
            if verbose:
                print(f"\n 폴드 {fold_info['fold_id']} 학습 중...")

            # 폴드별 데이터 분할
            train_data = df.loc[fold_info['train_indices']]
            val_data = df.loc[fold_info['val_indices']]
            test_data = df.loc[fold_info['test_indices']]

            # 피처 준비
            X_train = self.prepare_features(train_data, verbose=False)
            X_val = self.prepare_features(val_data, verbose=False)
            X_test = self.prepare_features(test_data, verbose=False)

            y_train = train_data['buy_signal_score']
            y_val = val_data['buy_signal_score']
            y_test = test_data['buy_signal_score']

            # 하이퍼파라미터 최적화
            if hyperparameter_search:
                search_result = self._optimize_hyperparameters(X_train, y_train, verbose=False)
                best_params = search_result
            else:
                # 기본 파라미터
                best_params = {
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'n_estimators': 400,
                    'subsample': 0.8,
                    'colsample_bytree': 0.9,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.5,
                    'random_state': 42
                }

            # 최적 파라미터로 모델 학습
            best_model = xgb.XGBRegressor(**best_params)
            best_model.fit(X_train, y_train)

            # 검증 및 테스트 평가
            val_pred = best_model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)

            test_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, test_pred)

            # 결과 저장
            fold_result = {
                'fold_id': fold_info['fold_id'],
                'fold_info': fold_info,
                'best_params': best_params,
                'val_r2': val_r2,
                'test_r2': test_r2,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'features_used': len(X_train.columns),
                'best_model': best_model
            }

            fold_results.append(fold_result)

            if verbose:
                print(f"  폴드 {fold_info['fold_id']} 완료: Val R² = {val_r2:.4f}, Test R² = {test_r2:.4f}")

        self.fold_results = fold_results

        # 최고 성능 최종 모델
        best_fold = max(fold_results, key=lambda x: x['test_r2'])
        self.model = best_fold['best_model']
        self.best_params = best_fold['best_params']
        self.is_trained = True

        if verbose:
            self._print_fold_summary()

        return fold_results

    def _print_fold_summary_deprecated(self):
        """폴드별 결과 요약 출력"""
        if not self.fold_results:
            print("폴드 결과가 없음")
            return

        print("\n" + "="*70)
        print("Buy Signal Walk-Forward 결과 요약")
        print("="*70)

        val_r2_scores = [result['val_r2'] for result in self.fold_results]
        test_r2_scores = [result['test_r2'] for result in self.fold_results]

        print(f"폴드별 성능:")
        for result in self.fold_results:
            print(f"  폴드 {result['fold_id']}: Val R² = {result['val_r2']:.4f}, Test R² = {result['test_r2']:.4f}")

        print(f"\n 전체 통계:")
        print(f"  Validation R²: {np.mean(val_r2_scores):.4f} ± {np.std(val_r2_scores):.4f}")
        print(f"  Test R²:       {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")
        print(f"  최고 성능:     {np.max(test_r2_scores):.4f} (폴드 {np.argmax(test_r2_scores) + 1})")
        print(f"  평균 피처 수:  {np.mean([r['features_used'] for r in self.fold_results]):.0f}개")

        print("="*70)

    def save_training_results_deprecated(self, filename=None):
        """학습 결과 저장 """
        if filename is None:
            filename = f"buy_signal_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        save_data = {
            'model_type': 'BUY_SIGNAL_PREDICTOR',
            'model_name': 'Buy Signal Predictor',
            'created_at': datetime.now().isoformat(),
            'total_folds': len(self.fold_results),
            'best_params': self.best_params,
            'fold_results': []
        }

        for result in self.fold_results:
            fold_data = {key: value for key, value in result.items()
                         if key != 'best_model'}
            save_data['fold_results'].append(fold_data)

        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"Buy Signal 학습 결과 저장: {filename}")
        return filename

def main():
    """Buy Signal Predictor 학습 파이프라인 실행"""
    print("Buy Signal Predictor - 매수 신호 예측 모델 학습")
    print("="*70)
    print("  - 실시간 매수 신호 강도 0-100점 평가")
    print("="*70)

    # 데이터 경로 설정
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../results/final/trading_episodes_with_rebuilt_market_component.csv')

    # 파일 존재 확인
    if not os.path.exists(data_path):
        print(f"데이터 파일을 찾을 수 없음: {data_path}")
        return

    # 모델 초기화 및 학습
    predictor = BuySignalPredictor(
        train_months=18,  # 18개월 학습
        val_months=3,     # 3개월 검증
        test_months=3,    # 3개월 테스트
        step_months=3     # 3개월씩 이동
    )

    # 전체 범위 학습 실행
    try:
        # 데이터 로드
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"데이터 로드: {len(df):,}개 거래")

        df_filtered = df[
            df['entry_pe_ratio'].notna() |
            df['entry_roe'].notna() |
            df['entry_earnings_growth'].notna()
            ].copy()

        print(f"펀더멘털 데이터 필터링: {len(df_filtered):,}개 ({len(df_filtered)/len(df)*100:.1f}%)")

        # Train/Val/Test 분할 (60/20/20)
        from sklearn.model_selection import train_test_split
        train_val_df, test_df = train_test_split(df_filtered, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

        print(f"\n📊 데이터 분할:")
        print(f"  Train: {len(train_df):,}개 ({len(train_df)/len(df_filtered)*100:.1f}%)")
        print(f"  Val:   {len(val_df):,}개 ({len(val_df)/len(df_filtered)*100:.1f}%)")
        print(f"  Test:  {len(test_df):,}개 ({len(test_df)/len(df_filtered)*100:.1f}%)")

        # 2단계 최적화 모델 학습
        print(f"\n🚀 2단계 최적화 모델 학습 시작...")
        result = predictor.train_model_with_weight_optimization(train_df, hyperparameter_search=False, verbose=True)

        # 평가 함수 (최적화된 가중치 사용)
        def evaluate_model(predictor, data, name):
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            import numpy as np
            data_with_score = predictor.create_entry_signal_score(data, weights=predictor.best_weights, verbose=False)
            X = predictor.prepare_features(data_with_score, verbose=False)
            y = data_with_score['buy_signal_score']
            y_pred = predictor.model.predict(X)

            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)

            return {
                'name': name,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'y_mean': y.mean(),
                'y_std': y.std(),
                'pred_mean': y_pred.mean(),
                'pred_std': y_pred.std()
            }

        # 각 세트 평가
        train_metrics = evaluate_model(predictor, train_df, 'Train')
        val_metrics = evaluate_model(predictor, val_df, 'Val')
        test_metrics = evaluate_model(predictor, test_df, 'Test')

        # 성과 출력
        print(f"\n 성과 지표:")
        print("="*60)
        print(f"{'Dataset':<10} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Mean':>8} {'Std':>8}")
        print("-"*60)
        for metrics in [train_metrics, val_metrics, test_metrics]:
            print(f"{metrics['name']:<10} {metrics['r2']:>8.4f} {metrics['rmse']:>8.4f} {metrics['mae']:>8.4f} {metrics['y_mean']:>8.4f} {metrics['y_std']:>8.4f}")

        # 오버피팅 체크
        overfit_score = train_metrics['r2'] - val_metrics['r2']
        print(f"\n🔍 오버피팅 분석:")
        if overfit_score > 0.05:
            print(f"  ️ 오버피팅 가능성: Train-Val R² 차이 = {overfit_score:.4f}")
        else:
            print(f"    오버피팅 없음: Train-Val R² 차이 = {overfit_score:.4f}")

        # Val-Test 성능 안정성
        stability_score = abs(val_metrics['r2'] - test_metrics['r2'])
        print(f"\n📏 성능 안정성:")
        if stability_score < 0.05:
            print(f"    안정적: Val-Test R² 차이 = {stability_score:.4f}")
        else:
            print(f"   ️ 불안정: Val-Test R² 차이 = {stability_score:.4f}")

        # 모델 저장
        model_filename = predictor.save_model()

        print(f"\n Buy Signal 2단계 최적화 모델 학습 완료")
        print(f"저장된 모델: {model_filename}")
        print(f"최종 사용된 가중치: {predictor.best_weights}")
        
        # 가중치 최적화 결과도 출력
        if 'weight_results' in result:
            sorted_weights = sorted(result['weight_results'], key=lambda x: x['cv_r2'], reverse=True)
            print(f"\n상위 3개 가중치 조합:")
            for i, w_result in enumerate(sorted_weights[:3], 1):
                w = w_result['weights']
                print(f"  {i}. T:{w['technical']:.2f} F:{w['fundamental']:.2f} M:{w['market']:.2f} → R²={w_result['cv_r2']:.4f}")



        return predictor

    except Exception as e:
        print(f"학습 중 오류 발생: {str(e)}")
        return None

if __name__ == "__main__":
    main()