# feature_table_builder.py

import pandas as pd
from technical_scorer import TechnicalScorer  # → 실제 클래스명으로 수정
from pathlib import Path

def build_feature_table(
    sentiment_csv_path: str = "news_sentiment_probabilities.csv",
    output_csv_path: str = "news_combined_features.csv"
):
    # 1. 감성분석 결과 불러오기
    df = pd.read_csv(sentiment_csv_path)
    print(f"✅ 감성분석 결과 불러옴: {len(df)}개 뉴스")

    # 2. 룰셋 점수 계산
    scorer = TechnicalScorer()
    df_with_rule = scorer.process_sentiment_data(df)

    print(f"✅ 룰셋 점수 계산 완료")

    # 3. 필요한 피처만 추출
    feature_cols = [
        "ticker", "date", "title",  # 메타 정보
        "positive", "negative", "neutral", "sentiment_score",  # 감성 피처
        "rule_score",  # 종합 점수
        "analyst_score", "momentum_score", "volume_score", "valuation_score",  # 룰셋 피처
        "rule_category"  # 라벨 또는 참고
    ]

    df_features = df_with_rule[feature_cols].copy()

    # 4. 저장
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 피처 테이블 저장 완료: {output_csv_path}")

    return df_features
