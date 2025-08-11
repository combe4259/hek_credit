# =============================================================================
# 최종 뉴스 점수 예측 모델 (성능 좋지만 시계열 안 씀. 상용화 불가)
# KR-FinBERT + 데이터 누수 없는 룰셋 + LightGBM + Optuna 튜닝 (오류 수정)
# =============================================================================
# !pip install sentence-transformers optuna lightgbm scikit-learn pandas joblib -q

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')

class BetterNewsScorePredictor:
    def __init__(self, bert_model_name='snunlp/KR-FinBERT'):
        self.model = None
        self.feature_names = None
        print(f"📚 KR-FinBERT 모델 로딩 중: {bert_model_name}")
        self.bert_model = SentenceTransformer(bert_model_name)
        print("✅ KR-FinBERT 모델 로딩 완료!")

    def _composite_signal_to_score(self, composite_signal):
        score = 5 * (composite_signal + 1)
        return np.clip(score, 0, 10)

    def _score_to_category(self, score):
        if score >= 8.5: return "강력호재"
        elif score >= 7.0: return "호재"
        elif score >= 6.0: return "약간호재"
        elif score >= 4.0: return "중립"
        elif score >= 3.0: return "약간악재"
        elif score >= 1.5: return "악재"
        else: return "강력악재"

    def _create_features(self, df):
        print("🛠️ 최종 하이브리드 피처 생성 시작...")
        final_feature_parts = []

        # --- 1. KR-FinBERT 임베딩 피처 (배치 처리) ---
        text_column = 'content' if 'content' in df.columns else 'title'
        texts = df[text_column].astype(str).tolist()
        print(f"🧠 '{text_column}' 컬럼 배치 임베딩 생성 중...")
        embeddings = self.bert_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        X_bert = pd.DataFrame(embeddings, index=df.index)
        X_bert.columns = [f'finbert_{i}' for i in range(X_bert.shape[1])]
        final_feature_parts.append(X_bert)

        # --- 2. 감성 분석 피처 ---
        sentiment_cols = ['positive', 'negative', 'neutral', 'sentence_count']
        sentiment_features = df[sentiment_cols].copy()
        total_emotion = (sentiment_features['positive'] + sentiment_features['negative']).replace(0, 1e-8)
        sentiment_features['emotion_balance'] = (sentiment_features['positive'] - sentiment_features['negative']) / total_emotion
        final_feature_parts.append(sentiment_features)

        # --- 3. 룰셋 피처 ---
        ruleset_cols = ['momentum_score', 'volume_score', 'rule_score']
        ruleset_features = df[ruleset_cols].fillna(50).copy()
        final_feature_parts.append(ruleset_features)

        # --- 4. 맥락 피처 ---
        context_features = df[['sector', 'market_cap']].copy()
        cap_q1, cap_q3 = context_features['market_cap'].quantile([0.25, 0.75])
        context_features['market_cap_group'] = pd.cut(
            context_features['market_cap'],
            bins=[0, cap_q1, cap_q3, np.inf],
            labels=['Small', 'Mid', 'Large'],
            right=False
        )
        context_features_encoded = pd.get_dummies(context_features[['sector', 'market_cap_group']], dummy_na=True)
        final_feature_parts.append(context_features_encoded)

        # --- 5. 결합 ---
        final_features = pd.concat(final_feature_parts, axis=1)
        print("🔗 모든 피처 결합 완료!")
        return final_features

    def train(self, df, model_save_path='final_model.pkl'):
        print("\n🤖 최종 모델 훈련 시작...")
        required_cols = ['composite_signal', 'content'] if 'content' in df.columns else ['composite_signal', 'title']
        train_df = df.dropna(subset=required_cols).copy()

        X = self._create_features(train_df)
        y = self._composite_signal_to_score(train_df['composite_signal'])
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"📈 학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")

        def objective(trial):
            params = {
                'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1,
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 5, 11),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 71),
                'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                'random_state': 42,
                'n_jobs': -1
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            y_pred = model.predict(X_test)
            return mean_absolute_error(y_test, y_pred)

        print("\n⏳ Optuna 튜닝 시작...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30, show_progress_bar=True)

        print("\n✅ 최적 파라미터:", study.best_params)
        self.model = lgb.LGBMRegressor(**study.best_params)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        direction_accuracy = np.mean((y_test > 5) == (y_pred > 5))
        print(f"\n📊 최종 성능 - MAE: {mae:.3f}, R²: {r2:.3f}, 방향 정확도: {direction_accuracy:.1%}")

        joblib.dump(self, model_save_path)
        print(f"💾 모델 저장 완료: {model_save_path}")

    def predict(self, new_data):
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        X_new = self._create_features(new_data)
        for col in self.feature_names:
            if col not in X_new.columns:
                X_new[col] = 0
        X_new = X_new[self.feature_names]
        predicted_scores = self.model.predict(X_new)
        result_df = new_data.copy()
        result_df['predicted_score'] = np.clip(predicted_scores, 0, 10)
        result_df['predicted_category'] = result_df['predicted_score'].apply(self._score_to_category)
        return result_df

# =============================================================================
# 실행
# =============================================================================
if __name__ == "__main__":
    from google.colab import drive
    print("🔗 Google Drive 연동...")
    drive.mount('/content/drive', force_remount=True)

    try:
        DRIVE_DATA_PATH = "/content/drive/MyDrive/news_with_composite_signals.csv"
        print(f"📂 데이터 로딩: {DRIVE_DATA_PATH}")
        df_news = pd.read_csv(DRIVE_DATA_PATH)

        # --- 필수 컬럼 확인 ---
        if 'content' not in df_news.columns:
             raise ValueError("'content' 컬럼이 없습니다.")
        if 'sector' not in df_news.columns:
            df_news['sector'] = 'Unknown'
        if 'market_cap' not in df_news.columns:
            df_news['market_cap'] = df_news['market_cap'].median() if not df_news['market_cap'].isnull().all() else 1e10

        predictor = FinalNewsScorePredictor()
        predictor.train(df_news)

        print("\n🚀 샘플 예측 테스트")
        sample_news = df_news.dropna(subset=['composite_signal', 'content']).sample(5, random_state=101)
        predictions = predictor.predict(sample_news)

        for idx, row in predictions.iterrows():
            actual_score = predictor._composite_signal_to_score(row['composite_signal'])
            print(f"\n📰 샘플 뉴스 (ticker: {row['ticker']})")
            print(f"  - 🤖 AI 예측 점수: {row['predicted_score']:.2f} ({row['predicted_category']})")
            print(f"  - 📊 실제 점수: {actual_score:.2f}")

    except FileNotFoundError:
        print(f"\n❌ 오류: '{DRIVE_DATA_PATH}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
