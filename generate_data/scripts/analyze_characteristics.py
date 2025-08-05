import pandas as pd
import numpy as np
import json
from improved_mapping import analyze_all_greek_isins, analyze_us_characteristics, get_diverse_us_stocks

def save_stock_characteristics():
    """주식 특성을 분석하고 저장하는 함수"""
    
    # 1. 그리스 ISIN 특성 분석
    print("📊 Analyzing Greek ISIN characteristics...")
    greek_profiles = analyze_all_greek_isins('/Users/inter4259/Desktop/transactions.csv', top_n=50)
    
    # 그리스 ISIN 특성 저장
    with open('../results/profiles/greek_isin_profiles.json', 'w') as f:
        json.dump(greek_profiles, f, indent=2)
    
    # DataFrame으로 변환하여 CSV 저장
    greek_df = pd.DataFrame(greek_profiles).T
    greek_df.to_csv('../results/profiles/greek_isin_profiles.csv')
    
    print(f"✅ Saved Greek ISIN profiles ({len(greek_profiles)} ISINs)")
    
    # 2. 미국 주식 특성 분석
    print("\n📈 Analyzing US stock characteristics...")
    us_tickers = get_diverse_us_stocks()
    us_profiles = analyze_us_characteristics(us_tickers)
    
    # 미국 주식 특성 저장
    with open('../results/profiles/us_stock_profiles.json', 'w') as f:
        json.dump(us_profiles, f, indent=2)
    
    # DataFrame으로 변환하여 CSV 저장
    us_df = pd.DataFrame(us_profiles).T
    us_df.to_csv('../results/profiles/us_stock_profiles.csv')
    
    print(f"✅ Saved US stock profiles ({len(us_profiles)} stocks)")
    
    # 3. 매핑 정보와 함께 요약 생성
    with open('../results/mappings/unique_isin_mapping.json', 'r') as f:
        mappings = json.load(f)
    
    # 매핑 요약 생성
    mapping_summary = []
    for isin, mapping in mappings.items():
        if isin in greek_profiles:
            summary = {
                'isin': isin,
                'us_ticker': mapping['us_ticker'],
                'isin_volatility': greek_profiles[isin]['volatility'],
                'us_volatility': mapping['us_vol'],
                'volatility_diff': abs(greek_profiles[isin]['volatility'] - mapping['us_vol']),
                'isin_return': greek_profiles[isin]['return'],
                'us_return': mapping['us_ret'],
                'isin_trades': greek_profiles[isin]['n_trades'],
                'match_score': mapping['match_score']
            }
            mapping_summary.append(summary)
    
    # 매핑 요약 저장
    summary_df = pd.DataFrame(mapping_summary)
    summary_df = summary_df.sort_values('volatility_diff')
    summary_df.to_csv('../results/mappings/mapping_quality_report.csv', index=False)
    
    print("\n📊 Mapping Quality Summary:")
    print(f"Average volatility difference: {summary_df['volatility_diff'].mean():.2f}%")
    print(f"Best match: {summary_df.iloc[0]['isin']} → {summary_df.iloc[0]['us_ticker']} (diff: {summary_df.iloc[0]['volatility_diff']:.2f}%)")
    print(f"Worst match: {summary_df.iloc[-1]['isin']} → {summary_df.iloc[-1]['us_ticker']} (diff: {summary_df.iloc[-1]['volatility_diff']:.2f}%)")
    
    return greek_profiles, us_profiles, summary_df

if __name__ == "__main__":
    greek, us, summary = save_stock_characteristics()
    print("\n✅ All characteristics saved successfully!")