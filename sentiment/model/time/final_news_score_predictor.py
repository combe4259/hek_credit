# =============================================================================
# 최종 뉴스 점수 예측 모델 (최종 검증 강화 버전)
# 기간 기반 롤링 윈도우, KR-FinBERT, LightGBM, Optuna 적용
# =============================================================================
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import joblib
import warnings
from pandas.tseries.offsets import DateOffset
from sklearn.decomposition import PCA
import sys
import os
import traceback
from google.colab import drive
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class FinalNewsScorePredictor:
    """
    🎯 기간 기반 롤링 윈도우 평가를 통해 모델의 안정성을 검증하는 최종 모델
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.best_params = None
        self.pca = None
        print("✅ FinalNewsScorePredictor 클래스 초기화 완료!")

    def _composite_signal_to_score(self, composite_signal):
        composite_signal = np.clip(composite_signal, -1.0, 1.0)
        score = 7.5 * (composite_signal + 1)
        # 분산 증가를 위한 변환 - 클리핑 전에 적용
        score = np.sqrt(score) * 2.5  # 제곱근 변환으로 분산 증가
        return score  # 클리핑 제거하여 더 넓은 분포 허용

    def _score_to_category(self, score):
        if score >= 8.5: return "강력호재"
        elif score >= 7.0: return "호재"
        elif score >= 6.0: return "약간호재"
        elif score >= 4.0: return "중립"
        elif score >= 3.0: return "약간악재"
        elif score >= 1.5: return "악재"
        else: return "강력악재"

    def _create_features(self, df):
        """사전 생성된 피처를 불러와 모델 입력 데이터로 변환하는 함수"""
        print("🛠️ 사전 생성된 피처를 기반으로 최종 데이터셋 구성 중...")
        final_feature_parts = []

        # 🎯 개선: BERT 임베딩 차원 축소 (PCA)
        bert_cols = [f'finbert_{i}' for i in range(768)]
        X_bert = df[bert_cols]
        if self.pca is None:
            self.pca = PCA(n_components=0.95)  # 95% 분산 설명하는 컴포넌트 자동 선택
            X_bert_pca = self.pca.fit_transform(X_bert)
        else:
            X_bert_pca = self.pca.transform(X_bert)
        X_bert_pca = pd.DataFrame(X_bert_pca, index=df.index, columns=[f'pca_bert_{i}' for i in range(X_bert_pca.shape[1])])
        final_feature_parts.append(X_bert_pca)

        # 감성 피처
        sentiment_features = df[['positive', 'negative', 'neutral', 'sentence_count']].copy()
        total_emotion = (sentiment_features['positive'] + sentiment_features['negative']).replace(0, 1e-8)
        sentiment_features['emotion_balance'] = (sentiment_features['positive'] - sentiment_features['negative']) / total_emotion
        final_feature_parts.append(sentiment_features)

        # 🎯 수정: rule_score 피처를 제거하여 데이터 누수 방지
        ruleset_cols = ['momentum_score', 'volume_score']
        ruleset_features = df[ruleset_cols].fillna(50).copy()
        final_feature_parts.append(ruleset_features)

        # 맥락 피처
        context_features = df[['sector', 'market_cap']].copy()
        cap_q1, cap_q3 = context_features['market_cap'].quantile([0.25, 0.75])
        context_features['market_cap_group'] = pd.cut(
            context_features['market_cap'], bins=[0, cap_q1, cap_q3, np.inf],
            labels=['Small', 'Mid', 'Large'], right=False
        )
        context_features_encoded = pd.get_dummies(context_features[['sector', 'market_cap_group']], dummy_na=True)
        final_feature_parts.append(context_features_encoded)

        X = pd.concat(final_feature_parts, axis=1)

        # 🎯 개선: 피처 상관성 제거
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        X = X.drop(columns=to_drop)

        # 타겟 변수가 있는 경우에만 상관성 검사 수행
        # (훈련 시에만 실행, 예측 시에는 스킵)

        return X

    def train(self, df, model_save_path='final_model.pkl'):
        print("\n🤖 최종 모델 훈련 (시계열 교차검증) 시작...")

        df['news_date'] = pd.to_datetime(df['news_date'])
        df = df.sort_values('news_date').reset_index(drop=True)

        required_cols = ['composite_signal'] + [f'finbert_{i}' for i in range(768)]
        train_df = df.dropna(subset=required_cols).copy()

        y = self._composite_signal_to_score(train_df['composite_signal'])
        X = self._create_features(train_df)
        self.feature_names = X.columns.tolist()
        
        # 데이터 누수 검사 (훈련 시에만)
        combined_data = pd.concat([X, pd.Series(y, name='target', index=X.index)], axis=1)
        corr_with_target = combined_data.corr()['target'].abs()
        high_corr = corr_with_target[corr_with_target > 0.95].drop('target', errors='ignore').index.tolist()
        if high_corr:
            print(f"⚠️ 높은 상관 피처 (누수 의심): {high_corr}")
            X = X.drop(columns=high_corr)
            self.feature_names = X.columns.tolist()

        split_index = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        print(f"📈 전체 데이터: {len(X)}개")
        print(f"   - 훈련/튜닝 데이터: {len(X_train)}개 (과거 80%)")
        print(f"   - 최종 테스트 데이터: {len(X_test)}개 (최신 20%)")

        # 🎯 수정: 타겟 변수 분산이 충분한지 검증
        print("\n📊 타겟 분포 요약:")
        print(y_train.describe())
        if len(y_train.unique()) <= 1:
            print("❌ 타겟 변수에 유의미한 분산이 없어 모델 학습이 불가능합니다. 데이터셋을 확인해주세요.")
            return

        def objective(trial):
            params = {
                'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1,
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 5, 11),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 71),
                'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42, 'n_jobs': -1
            }

            tscv = TimeSeriesSplit(n_splits=3)
            mae_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                y_pred = model.predict(X_val_fold)
                mae_scores.append(mean_absolute_error(y_val_fold, y_pred))

            return np.mean(mae_scores)

        print("\n⏳ Optuna 튜닝 (시계열 교차검증 기반)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100, show_progress_bar=True)

        print("\n✅ 최적 파라미터:", study.best_params)
        self.best_params = study.best_params
        self.model = lgb.LGBMRegressor(**self.best_params, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        direction_accuracy = np.mean((y_test > 5) == (y_pred > 5))
        print(f"\n📊 최종 성능 (Hold-out Test) - MAE: {mae:.3f}, R²: {r2:.3f}, 방향 정확도: {direction_accuracy:.1%}")

        # 🎯 수정: 피처 중요도 시각화 오류 방지
        if hasattr(self.model, 'feature_importances_') and len(self.model.feature_importances_) > 0:
            lgb.plot_importance(self.model, max_num_features=20)
            plt.show()
        else:
            print("⚠️ 모델이 학습되지 않았거나 피처가 없습니다. 데이터와 훈련 과정을 확인하세요.")

        self.save_model(model_save_path)

    def evaluate_rolling_window(self, df, train_period_months=18, test_period_months=3):
        """
        ✨ 기간 기반 롤링 윈도우로 모델의 성능 안정성을 평가합니다.
        """
        if self.best_params is None:
            print("⚠️ 롤링 윈도우 평가를 위해 먼저 train() 함수를 실행하여 최적 파라미터를 찾아야 합니다.")
            return

        print("\n" + "="*60)
        print("🔍 기간 기반 롤링 윈도우 평가 시작...")
        print(f"(훈련 기간: {train_period_months}개월, 테스트 기간: {test_period_months}개월)")
        print("="*60)

        df['news_date'] = pd.to_datetime(df['news_date'])
        df = df.sort_values('news_date').reset_index(drop=True)

        required_cols = ['composite_signal'] + [f'finbert_{i}' for i in range(768)]
        eval_df = df.dropna(subset=required_cols).copy()

        if self.pca is None:
            print("❌ PCA 모델이 훈련되지 않아 롤링 윈도우 평가를 진행할 수 없습니다. train()을 먼저 실행해주세요.")
            return

        X = self._create_features(eval_df)
        y = self._composite_signal_to_score(eval_df['composite_signal'])

        results = []

        start_date = eval_df['news_date'].min()
        end_date = eval_df['news_date'].max()

        current_date = start_date + DateOffset(months=train_period_months)

        while current_date + DateOffset(months=test_period_months) <= end_date:
            train_start_date = current_date - DateOffset(months=train_period_months)
            train_end_date = current_date
            test_end_date = current_date + DateOffset(months=test_period_months)

            train_mask = (eval_df['news_date'] >= train_start_date) & (eval_df['news_date'] < train_end_date)
            test_mask = (eval_df['news_date'] >= train_end_date) & (eval_df['news_date'] < test_end_date)

            if train_mask.sum() < 50 or test_mask.sum() < 10:
                current_date += DateOffset(months=test_period_months)
                continue

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            print(f"\n🗓️  테스트 구간: {train_end_date.date()} ~ {test_end_date.date()-DateOffset(days=1)}")

            model = lgb.LGBMRegressor(**self.best_params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            direction_accuracy = np.mean((y_test > 5) == (y_pred > 5))

            results.append({'test_period': f"{train_end_date.date()}", 'mae': mae, 'r2': r2, 'direction_accuracy': direction_accuracy})
            print(f"   - MAE: {mae:.3f}, R²: {r2:.3f}, 방향 정확도: {direction_accuracy:.1%}")

            current_date += DateOffset(months=test_period_months)

        if not results:
            print("⚠️ 롤링 윈도우 평가를 수행하기에 데이터 기간이 충분하지 않습니다.")
            return

        results_df = pd.DataFrame(results)
        print("\n" + "-"*60)
        print("📊 롤링 윈도우 평가 최종 요약:")
        print(results_df.round(3))
        print("-" * 60)
        print(f"   - 평균 MAE: {results_df['mae'].mean():.3f}")
        print(f"   - 평균 R²: {results_df['r2'].mean():.3f}")
        print(f"   - 평균 방향 정확도: {results_df['direction_accuracy'].mean():.1%}")
        return results_df

    def save_model(self, path):
        """✨ 모델, 피처 이름, 최적 파라미터를 함께 저장"""
        model_payload = {
            'model': self.model,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'pca': self.pca
        }
        joblib.dump(model_payload, path)
        print(f"💾 모델과 설정 저장 완료: {path}")

    @classmethod
    def load_model(cls, path):
        """✨ 클래스 메서드로 모델과 설정을 로드"""
        try:
            model_payload = joblib.load(path)
            instance = cls()
            instance.model = model_payload['model']
            instance.feature_names = model_payload['feature_names']
            instance.best_params = model_payload['best_params']
            instance.pca = model_payload.get('pca')
            print(f"💾 모델 로드 완료: {path}")
            return instance
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return None

# =============================================================================
# 실행
# =============================================================================
if __name__ == "__main__":
    drive.mount('/content/drive', force_remount=True)

    try:
        DRIVE_DATA_PATH = "/content/drive/MyDrive/news_full_features_robust.csv"
        df_news = pd.read_csv(DRIVE_DATA_PATH, engine='python')

        predictor = FinalNewsScorePredictor()
        predictor.train(df_news)
        predictor.evaluate_rolling_window(df_news, train_period_months=18, test_period_months=3)

    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        traceback.print_exc()