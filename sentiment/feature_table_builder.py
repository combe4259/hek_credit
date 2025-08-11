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

    # 3. AI 학습용 0~10점 스코어 추가 (감성분석 + 룰셋 결합)
    # sentiment_score (-1~1)을 0~100 스케일로 변환
    sentiment_scaled = ((df_with_rule['sentiment_score'] + 1) / 2) * 100
    
    # 감성분석(40%) + 룰셋(60%) 가중평균으로 결합
    combined_score = (sentiment_scaled * 0.4 + df_with_rule['rule_score'] * 0.6)
    df_with_rule['final_score_10'] = (combined_score / 10.0).round(1)
    
    # 종합점수 기준으로 카테고리 재분류
    def categorize_final_score(score):
        if score >= 8.0:
            return "강력호재"
        elif score >= 6.0:
            return "호재"
        elif score >= 4.0:
            return "중립"
        elif score >= 2.0:
            return "악재"
        else:
            return "강력악재"
    
    df_with_rule['final_category'] = df_with_rule['final_score_10'].apply(categorize_final_score)
    print(f"✅ 0~10점 호재/악재 스코어 생성 완료 (감성분석 40% + 룰셋 60%)")
    print(f"✅ 종합점수 기준 카테고리 재분류 완료")

    # 4. 최종 피처 선택
    feature_cols = [
        # 메타 정보
        "ticker", "news_date", "title",
        
        # 감성분석 피처 (이미 완료됨)
        "positive", "negative", "neutral", "sentiment_score", "sentence_count",
        
        # 기술적 분석 피처
        "rule_score", "analyst_score", "momentum_score", "volume_score", "valuation_score",
        
        # AI 학습 타겟 (0~10점 연속값)
        "final_score_10"
    ]
    
    # 실제 존재하는 컬럼만 선택 (date 컬럼명 체크)
    if "news_date" not in df_with_rule.columns and "date" in df_with_rule.columns:
        feature_cols[1] = "date"
    
    available_cols = [col for col in feature_cols if col in df_with_rule.columns]
    df_features = df_with_rule[available_cols].copy()

    # 5. 저장
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 피처 테이블 저장 완료: {output_csv_path}")
    print(f"📊 최종 피처 수: {len(df_features.columns)}개")
    print(f"📈 final_score_10 범위: {df_features['final_score_10'].min():.1f}~{df_features['final_score_10'].max():.1f}점")
    
    return df_features

if __name__ == "__main__":
    build_feature_table()