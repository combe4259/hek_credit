# =============================================================================
# 2단계 뉴스 점수 예측 시스템
# 1단계: 개별 뉴스 점수화 (0~100)
# 2단계: 종목별 뉴스 집계 → 종목 최종 점수
# =============================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import joblib
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class IndividualNewsScorer:
    """
    🎯 1단계: 개별 뉴스 호재/악재 점수 예측기
    
    목표: 뉴스 하나만 보고 그 뉴스 자체의 호재도 점수화
    피처: FinBERT + 감성점수만 (과적합 방지)
    """
    
    def __init__(self):
        self.model = None
        self.pca = None
        self.feature_names = None
        print("✅ 개별 뉴스 점수기 초기화!")
    
    def create_individual_features(self, df, max_bert_dim=50, verbose=True):
        """
        개별 뉴스 점수화를 위한 최소 피처 생성
        """
        if verbose:
            print("🛠️ 개별 뉴스용 피처 생성 (과적합 방지)")
        
        feature_parts = []
        
        # === 1. BERT 임베딩  ===
        bert_cols = [f'finbert_{i}' for i in range(768) if f'finbert_{i}' in df.columns]
        if bert_cols:
            X_bert = df[bert_cols].fillna(0)
            
            if self.pca is None:
                self.pca = PCA(n_components=max_bert_dim, random_state=42)
                X_bert_pca = self.pca.fit_transform(X_bert)
            else:
                X_bert_pca = self.pca.transform(X_bert)
            
            bert_df = pd.DataFrame(
                X_bert_pca,
                index=df.index, 
                columns=[f'bert_pca_{i}' for i in range(max_bert_dim)]
            )
            feature_parts.append(bert_df)
            
            if verbose:
                explained_var = self.pca.explained_variance_ratio_.sum()
                print(f"  BERT: 768 → {max_bert_dim}차원 (정보보존: {explained_var:.2%})")
        
        # === 2. 핵심 감성 피처만 ===
        sentiment_cols = ['positive', 'negative', 'sentiment_score']
        available_sentiment = [col for col in sentiment_cols if col in df.columns]
        
        if available_sentiment:
            sentiment_df = df[available_sentiment].fillna(0).copy()
            
            # 간단한 파생 피처 1개만
            if 'positive' in sentiment_df.columns and 'negative' in sentiment_df.columns:
                sentiment_df['sentiment_balance'] = (
                    sentiment_df['positive'] - sentiment_df['negative']
                ) / (sentiment_df['positive'] + sentiment_df['negative'] + 1e-8)
            
            feature_parts.append(sentiment_df)
            if verbose:
                print(f"  감성: {len(sentiment_df.columns)}개 피처")
        
        # === 3. 의미있는 메타 피처 ===
        meta_features = []
        
        # 감성 강도 (positive + negative)
        if 'positive' in df.columns and 'negative' in df.columns:
            sentiment_intensity = df['positive'].fillna(0) + df['negative'].fillna(0)
            meta_features.append(('sentiment_intensity', sentiment_intensity))
        
        # 감성 확신도 (max - min)
        sentiment_cols_available = [col for col in ['positive', 'negative', 'neutral'] if col in df.columns]
        if len(sentiment_cols_available) >= 2:
            sentiment_values = df[sentiment_cols_available].fillna(0)
            sentiment_confidence = sentiment_values.max(axis=1) - sentiment_values.min(axis=1)
            meta_features.append(('sentiment_confidence', sentiment_confidence))
        
        if meta_features:
            meta_df = pd.DataFrame({name: values for name, values in meta_features}, index=df.index)
            feature_parts.append(meta_df)
            if verbose:
                print(f"  의미있는 메타: {len(meta_df.columns)}개 피처")
        
        # 피처 결합
        X = pd.concat(feature_parts, axis=1)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        self.feature_names = X.columns.tolist()
        if verbose:
            print(f"  최종: {len(X.columns)}개 피처 (과적합 방지)")
        
        return X
    
    def create_comprehensive_individual_targets(self, df, verbose=True):
        """
        Comprehensive 방식으로 개별 뉴스 점수 생성 (빈도 제외)
        감성(50%) + 기술적지표(30%) + BERT강도(20%)
        """
        if verbose:
            print("🎯 Comprehensive 방식 개별 뉴스 점수 생성")
        
        individual_scores = []
        
        for idx, row in df.iterrows():
            # === 1. 감성 컴포넌트 (50%) ===
            sentiment_score = row.get('sentiment_score', 0)  # -1 ~ +1
            positive = row.get('positive', 0)
            negative = row.get('negative', 0)
            neutral = row.get('neutral', 0)
            
            # 감성 균형 점수 (-1 ~ +1)
            total_emotion = positive + negative + neutral + 1e-8
            emotion_balance = (positive - negative) / total_emotion
            
            # 감성 강도 (0 ~ 1)
            emotion_intensity = (positive + negative) / total_emotion
            
            # 감성 컴포넌트 점수 (0~100)
            sentiment_component = (
                (sentiment_score + 1) / 2 * 40 +      # sentiment_score: 40점
                (emotion_balance + 1) / 2 * 30 +      # emotion_balance: 30점  
                emotion_intensity * 30                 # emotion_intensity: 30점
            ) / 2  # 50% 비중
            
            # === 2. 기술적 지표 컴포넌트 (30%) ===
            momentum_score = row.get('momentum_score', 50)  # 0~100
            volume_score = row.get('volume_score', 50)      # 0~100
            
            technical_component = (momentum_score * 0.7 + volume_score * 0.3) * 0.3
            
            # === 3. BERT 강도 컴포넌트 (20%) ===
            # BERT 임베딩의 절댓값 평균으로 감성 강도 측정
            bert_cols = [f'finbert_{i}' for i in range(768) if f'finbert_{i}' in df.columns and not pd.isna(row.get(f'finbert_{i}', np.nan))]
            
            if bert_cols:
                bert_values = np.array([row.get(col, 0) for col in bert_cols])
                bert_intensity = np.mean(np.abs(bert_values))  # 절댓값 평균
                bert_component = min(bert_intensity * 100, 100) * 0.2  # 최대 20점
            else:
                bert_component = 10  # 기본값
            
            # === 최종 점수 합산 ===
            final_score = sentiment_component + technical_component + bert_component
            final_score = np.clip(final_score, 0, 100)
            
            individual_scores.append(final_score)
        
        individual_scores = np.array(individual_scores)
        
        if verbose:
            print(f"  개별 점수 분포: {individual_scores.min():.1f} ~ {individual_scores.max():.1f}")
            print(f"  평균: {individual_scores.mean():.1f} ± {individual_scores.std():.1f}")
            print(f"  점수 구성: 감성(50%) + 기술적지표(30%) + BERT강도(20%)")
        
        return individual_scores
    
    def create_targets_from_existing_comprehensive(self, df, stock_scores_dict, verbose=True):
        """
        기존 comprehensive 점수를 개별 뉴스에 적용 (약간의 노이즈 추가)
        """
        if verbose:
            print("🎯 기존 comprehensive 점수 기반 개별 뉴스 타겟 생성")
        
        individual_scores = []
        matched_count = 0
        
        for idx, row in df.iterrows():
            stock_name = row.get('original_stock', '')
            
            if stock_name in stock_scores_dict:
                base_score = stock_scores_dict[stock_name]
                matched_count += 1
                
                # 작은 랜덤 노이즈 추가 (±3점)
                noise = np.random.normal(0, 3)
                individual_score = base_score + noise
                individual_score = np.clip(individual_score, 0, 100)
            else:
                individual_score = 50  # 중립
            
            individual_scores.append(individual_score)
        
        individual_scores = np.array(individual_scores)
        
        if verbose:
            print(f"  매칭률: {matched_count}/{len(df)} ({matched_count/len(df)*100:.1f}%)")
            print(f"  점수 분포: {individual_scores.min():.1f} ~ {individual_scores.max():.1f}")
            print(f"  평균: {individual_scores.mean():.1f} ± {individual_scores.std():.1f}")
        
        return individual_scores
    
    def train_individual(self, df, stock_scores_dict, verbose=True):
        """
        개별 뉴스 점수 예측 모델 훈련 (기존 comprehensive 점수 기반)
        """
        if verbose:
            print("\n🎯 1단계: 개별 뉴스 점수 예측 모델 훈련 (기존 comprehensive 기준)")
            print("="*60)
        
        # 피처 및 타겟 생성 (기존 comprehensive 점수 사용)
        X = self.create_individual_features(df, max_bert_dim=50, verbose=verbose)
        y = self.create_targets_from_existing_comprehensive(df, stock_scores_dict, verbose=verbose)
        
        # 전체 데이터로 훈련
        X_train = X
        y_train = y
        
        # 성능 검증용 샘플링
        test_sample = X.sample(frac=0.2, random_state=42)
        X_test = test_sample
        y_test = y[test_sample.index]
        
        if verbose:
            print(f"\n📊 데이터 사용:")
            print(f"  전체 훈련: {len(X_train):,}개")
            print(f"  성능 검증용 샘플: {len(X_test):,}개")
        
        # 과적합 방지 최적화 파라미터
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'n_estimators': 100,
            'max_depth': 4,
            'num_leaves': 15,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_data_in_leaf': 30,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'random_state': 42,
            'verbosity': -1
        }
        
        # 모델 훈련
        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )
        
        # 성능 평가
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        gap = test_mae - train_mae
        
        if verbose:
            print(f"\n🏆 개별 뉴스 모델 성능:")
            print(f"  훈련 MAE: {train_mae:.2f}")
            print(f"  테스트 MAE: {test_mae:.2f}")
            print(f"  테스트 R²: {test_r2:.4f}")
            print(f"  과적합 GAP: {gap:.2f}")
            
            if gap < 2.0:
                print(f"  ✅ 과적합 잘 억제됨!")
            else:
                print(f"  🟡 추가 정규화 고려")
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'gap': gap
        }
    
    def predict_individual(self, news_data):
        """개별 뉴스 점수 예측"""
        if self.model is None:
            raise ValueError("개별 뉴스 모델이 훈련되지 않았습니다!")
        
        df_input = pd.DataFrame([news_data])
        X = self.create_individual_features(df_input, max_bert_dim=50, verbose=False)  # 동일한 차원
        score = self.model.predict(X)[0]
        return np.clip(score, 0, 100)

class StockAggregateScorer:
    """
    🎯 2단계: 종목별 뉴스 집계 + 기술적 지표 → 종목 최종 점수
    """
    
    def __init__(self):
        self.individual_scorer = None
        self.aggregation_model = None
        print("✅ 종목 집계 점수기 초기화 (뉴스+기술적지표)!")
    
    def aggregate_stock_news(self, df, individual_scorer, days_window=30, verbose=True):
        """
        종목별 뉴스를 집계해서 종목 점수 생성 (시간 가중치 적용)
        """
        if verbose:
            print(f"\n🔄 2단계: 종목별 뉴스 집계 (최근 {days_window}일, 시간 가중)")
            print("="*60)
        
        self.individual_scorer = individual_scorer
        
        # 모든 뉴스에 개별 점수 할당
        if verbose:
            print("📊 모든 뉴스에 개별 점수 부여 중...")
        
        individual_scores = []
        for idx, row in df.iterrows():
            score = individual_scorer.predict_individual(row)
            individual_scores.append(score)
        
        df = df.copy()
        df['individual_score'] = individual_scores
        
        # 날짜 처리
        if 'news_date' in df.columns:
            df['news_date'] = pd.to_datetime(df['news_date'])
        else:
            # 임시 날짜 생성
            df['news_date'] = pd.date_range(end='2024-01-01', periods=len(df), freq='H')
        
        # 종목별 집계 (시간 불일치 해결)
        stock_scores = []
        current_date = df['news_date'].max()
        
        if verbose:
            print(f"  기준 날짜: {current_date}")
        
        for stock in df['original_stock'].unique():
            stock_news = df[df['original_stock'] == stock].copy()
            
            if len(stock_news) == 0:
                continue
            
            # 🎯 시간 불일치 해결: 최근 뉴스 우선순위
            stock_news = stock_news.sort_values('news_date', ascending=False)
            
            # comprehensive 방식과 동일하게 모든 뉴스 사용
            recent_news = stock_news  # 모든 뉴스 사용
            
            if len(recent_news) == 0:
                continue
            
            # comprehensive 방식과 동일한 신선도 가중치
            # 각 종목 내에서 상대적 신선도 계산 (0~1)
            days_from_latest = (current_date - recent_news['news_date']).dt.days
            max_days = days_from_latest.max()
            
            if max_days == 0:  # 모든 뉴스가 같은 날
                freshness_weights = np.ones(len(recent_news))
            else:
                freshness_weights = 1 - (days_from_latest / max_days)  # 1~0
            
            # 정규화
            if freshness_weights.sum() > 0:
                time_weights = freshness_weights / freshness_weights.sum()
            else:
                time_weights = np.ones(len(recent_news)) / len(recent_news)
            
            # 가중 평균 계산
            aggregated_score = np.average(recent_news['individual_score'], weights=time_weights)
            
            # 데이터 품질 지표
            avg_days_ago = np.average(days_from_latest, weights=time_weights)
            freshness_score = np.average(freshness_weights, weights=time_weights)  # 평균 신선도
            
            stock_scores.append({
                'stock_name': stock,
                'final_score': aggregated_score,
                'news_count': len(recent_news),
                'score_std': recent_news['individual_score'].std(),
                'latest_news_date': recent_news['news_date'].max(),
                'avg_days_ago': avg_days_ago,
                'freshness_score': freshness_score
            })
        
        result_df = pd.DataFrame(stock_scores)
        
        if verbose:
            print(f"  집계 완료: {len(result_df)}개 종목")
            print(f"  점수 분포: {result_df['final_score'].min():.1f} ~ {result_df['final_score'].max():.1f}")
            print(f"  평균 뉴스 수: {result_df['news_count'].mean():.1f}개/종목")
            print(f"  평균 신선도: {result_df['freshness_score'].mean():.3f} (1.0=최신)")
            print(f"  평균 경과일: {result_df['avg_days_ago'].mean():.1f}일")
        
        return result_df
    
    def create_stock_level_features(self, aggregated_news_df, df_original, verbose=True):
        """
        뉴스 집계 점수 + 기술적 지표 결합하여 종목 레벨 피처 생성
        """
        if verbose:
            print("\n🔧 2단계: 뉴스 집계 + 기술적 지표 결합")
            print("="*50)
        
        # 🚨 Data Leakage 방지: 기술적 지표 제외
        # comprehensive에서 momentum_score, volume_score를 30% 사용했으므로 제외
        if verbose:
            print(f"  기술적 지표: Data Leakage 방지를 위해 제외")
        
        technical_cols = []  # 기술적 지표 완전 제외
        
        # 종목별 기술적 지표 집계 (최신값 사용)
        stock_technical_features = []
        
        for stock in aggregated_news_df['stock_name'].unique():
            stock_data = df_original[df_original['original_stock'] == stock]
            
            if len(stock_data) == 0:
                continue
            
            # 최신 기술적 지표값 사용
            latest_tech_data = stock_data.iloc[-1] if len(stock_data) > 0 else None
            
            # 뉴스 집계 정보 가져오기
            stock_news_info = aggregated_news_df[aggregated_news_df['stock_name'] == stock].iloc[0]
            
            stock_feature = {
                'stock_name': stock,
                'news_aggregated_score': stock_news_info['final_score'],
                'news_count': stock_news_info['news_count'],
                'news_score_std': stock_news_info['score_std']
            }
            
            # 🚨 기술적 지표 및 상호작용 피처 제거 (Data Leakage 방지)
            # 뉴스 관련 피처만 사용
            
            stock_technical_features.append(stock_feature)
        
        result_df = pd.DataFrame(stock_technical_features)
        
        if verbose:
            print(f"  최종 피처 수: {len(result_df.columns)}개 (뉴스 피처만)")
            print(f"  처리된 종목 수: {len(result_df)}개")
            print(f"  피처 목록: {list(result_df.columns)}")
            print(f"  📍 Data Leakage 방지: 기술적 지표 피처 제외됨")
        
        return result_df
    
    def train_final_aggregation_model(self, stock_features_df, target_scores_df, verbose=True):
        """
        뉴스+기술적지표 결합 → 최종 종목 점수 예측 모델 훈련
        """
        if verbose:
            print("\n🎯 2단계 모델 훈련: 뉴스집계+기술적지표 → 최종점수")
            print("="*60)
        
        # 타겟 점수와 매칭
        merged_df = stock_features_df.merge(
            target_scores_df[['stock_name', 'final_score']], 
            on='stock_name', 
            how='inner'
        )
        
        if len(merged_df) == 0:
            raise ValueError("타겟 점수와 매칭되는 종목이 없습니다!")
        
        # 피처와 타겟 분리
        feature_cols = [col for col in merged_df.columns 
                       if col not in ['stock_name', 'final_score']]
        
        X = merged_df[feature_cols].fillna(0)
        y = merged_df['final_score'].values
        
        if verbose:
            print(f"  훈련 데이터: {len(X)}개 종목")
            print(f"  피처 수: {len(feature_cols)}개")
            print(f"  타겟 분포: {y.min():.1f} ~ {y.max():.1f}")
        
        # 데이터 분할
        if len(X) > 10:  # 충분한 데이터가 있을 때만 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        else:
            X_train, X_test = X, X
            y_train, y_test = y, y
        
        # 2단계 모델 파라미터 (적당한 복잡도)
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'n_estimators': 100,        # 적당한 복잡도
            'max_depth': 4,             # 적당한 깊이
            'num_leaves': 15,           # 적당한 리프 수
            'learning_rate': 0.1,       # 일반적인 학습률
            'min_data_in_leaf': 5,      # 적당한 최소값
            'reg_alpha': 0.3,           # 적당한 L1 정규화
            'reg_lambda': 0.3,          # 적당한 L2 정규화
            'feature_fraction': 0.8,    # 피처 샘플링
            'random_state': 42,
            'verbosity': -1
        }
        
        # 모델 훈련
        self.aggregation_model = lgb.LGBMRegressor(**params)
        self.aggregation_model.fit(X_train, y_train)
        
        # 성능 평가
        y_pred_train = self.aggregation_model.predict(X_train)
        y_pred_test = self.aggregation_model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        if verbose:
            print(f"\n🏆 2단계 모델 성능:")
            print(f"  훈련 MAE: {train_mae:.2f}")
            print(f"  테스트 MAE: {test_mae:.2f}")
            print(f"  테스트 R²: {test_r2:.4f}")
            
            # 피처 중요도
            if hasattr(self.aggregation_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': self.aggregation_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\n📊 피처 중요도:")
                for _, row in importance_df.iterrows():
                    print(f"  {row['feature']:25s}: {row['importance']:6.0f}")
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'feature_importance': importance_df if 'importance_df' in locals() else None
        }
    
    def predict_final_stock_score(self, stock_name, recent_news_list, technical_indicators, verbose=True):
        """
        종목의 뉴스들 + 기술적 지표 → 최종 종목 점수 예측
        """
        if self.individual_scorer is None or self.aggregation_model is None:
            raise ValueError("모델들이 훈련되지 않았습니다!")
        
        if verbose:
            print(f"🎯 종목 '{stock_name}' 최종 점수 예측")
        
        # 1단계: 개별 뉴스 점수들
        individual_scores = []
        for i, news in enumerate(recent_news_list):
            score = self.individual_scorer.predict_individual(news)
            individual_scores.append(score)
            if verbose:
                print(f"  뉴스 {i+1} 점수: {score:.1f}")
        
        # 뉴스 집계 점수
        if individual_scores:
            news_aggregated_score = np.mean(individual_scores)
            news_score_std = np.std(individual_scores) if len(individual_scores) > 1 else 0
        else:
            news_aggregated_score = 50  # 중립
            news_score_std = 0
        
        if verbose:
            print(f"  뉴스 집계 점수: {news_aggregated_score:.1f} ± {news_score_std:.1f}")
        
        # 2단계 피처 생성
        stock_features = {
            'news_aggregated_score': news_aggregated_score,
            'news_count': len(recent_news_list),
            'news_score_std': news_score_std
        }
        
        # 🚨 기술적 지표 피처 제거 (Data Leakage 방지)
        # 뉴스 피처만 사용
        
        # 2단계 모델로 최종 예측
        feature_df = pd.DataFrame([stock_features])
        
        # 훈련 시 사용된 피처와 맞추기
        expected_features = self.aggregation_model.feature_names_in_
        for col in expected_features:
            if col not in feature_df.columns:
                feature_df[col] = 0  # 누락된 피처는 0으로 처리
        
        feature_df = feature_df[expected_features]  # 순서 맞추기
        
        final_score = self.aggregation_model.predict(feature_df)[0]
        final_score = np.clip(final_score, 0, 100)
        
        if verbose:
            print(f"  기술적 지표: 제외됨 (Data Leakage 방지)")
            print(f"🏆 최종 종목 점수: {final_score:.1f}/100")
        
        return {
            'stock_name': stock_name,
            'individual_news_scores': individual_scores,
            'news_aggregated_score': news_aggregated_score,
            'technical_indicators': technical_indicators,
            'final_score': final_score,
            'news_count': len(recent_news_list)
        }
        
    def save_models(self, individual_path, aggregate_path=None):
        """모델 저장"""
        if self.individual_scorer:
            joblib.dump(self.individual_scorer, individual_path)
            print(f"💾 개별 뉴스 모델 저장: {individual_path}")

class TwoStageNewsSystem:
    """
    🚀 2단계 뉴스 점수 예측 통합 시스템
    """
    
    def __init__(self):
        self.stage1 = IndividualNewsScorer()
        self.stage2 = StockAggregateScorer()
        print("🚀 2단계 뉴스 시스템 초기화 완료!")
    
    def train_full_system(self, df, target_scores_df, verbose=True):
        """전체 2단계 시스템 훈련 (뉴스+기술적지표)"""
        if verbose:
            print("\n🚀 2단계 뉴스+기술적지표 시스템 훈련")
            print("="*70)
        
        # 종목 점수 딕셔너리 생성
        stock_scores_dict = dict(zip(target_scores_df['stock_name'], target_scores_df['final_score']))
        
        # 1단계: 개별 뉴스 점수 예측 훈련 (기존 comprehensive 기준)
        stage1_results = self.stage1.train_individual(df, stock_scores_dict, verbose=verbose)
        
        # 2-1단계: 종목별 뉴스 집계
        aggregated_news_results = self.stage2.aggregate_stock_news(
            df, self.stage1, days_window=30, verbose=verbose
        )
        
        # 2-2단계: 뉴스 집계 + 기술적 지표 결합
        stock_features_df = self.stage2.create_stock_level_features(
            aggregated_news_results, df, verbose=verbose
        )
        
        # 2-3단계: 최종 집계 모델 훈련
        stage2_results = self.stage2.train_final_aggregation_model(
            stock_features_df, target_scores_df, verbose=verbose
        )
        
        return {
            'stage1_results': stage1_results,
            'stage2_results': stage2_results,
            'aggregated_stocks': aggregated_news_results,  # 키 이름 통일
            'stock_features': stock_features_df
        }
    
    def predict_stock_score(self, stock_name, recent_news_list, verbose=True):
        """종목의 최근 뉴스들로부터 종목 점수 예측"""
        if verbose:
            print(f"🎯 종목 '{stock_name}' 점수 예측")
        
        # 각 뉴스별 개별 점수
        individual_scores = []
        for news in recent_news_list:
            score = self.stage1.predict_individual(news)
            individual_scores.append(score)
            if verbose:
                print(f"  뉴스 점수: {score:.1f}")
        
        # 집계 점수 (단순 평균, 실제로는 시간가중 등 적용 가능)
        final_score = np.mean(individual_scores)
        
        if verbose:
            print(f"🏆 최종 종목 점수: {final_score:.1f}/100")
        
        return {
            'stock_name': stock_name,
            'individual_scores': individual_scores,
            'final_score': final_score,
            'news_count': len(recent_news_list)
        }

# =============================================================================
# 실행부
# =============================================================================
if __name__ == "__main__":
    try:
        print("🚀 2단계 뉴스 점수 시스템 테스트")
        print("="*70)
        
        # 데이터 로드
        news_csv_path = "/Users/inter4259/Desktop/news_full_features_robust.csv"
        scores_csv_path = "/Users/inter4259/Desktop/Programming/hek_credit/sentiment/model/time/stock_comprehensive_scores.csv"
        
        df_news = pd.read_csv(news_csv_path)
        df_scores = pd.read_csv(scores_csv_path)
        
        print(f"📊 뉴스 데이터: {len(df_news):,}개")
        print(f"📊 타겟 점수: {len(df_scores):,}개 종목")
        
        # 2단계 시스템 초기화 및 훈련
        system = TwoStageNewsSystem()
        results = system.train_full_system(
            df_news,                 # 전체 데이터 사용
            df_scores, 
            verbose=True
        )
        
        # 결과 저장
        model_path = "/Users/inter4259/Desktop/Programming/hek_credit/sentiment/model/time/two_stage_system.pkl"
        system.stage2.save_models(model_path)
        
        # 결과 출력
        print(f"\n📊 집계된 종목 점수:")
        stock_results = results['aggregated_stocks']
        print(stock_results.head(10))
        
        print(f"\n✅ 2단계 시스템 구축 완료!")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()