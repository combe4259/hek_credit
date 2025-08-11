import yfinance as yf
import pandas as pd

def check_available_fundamentals():
    """yfinance로 가져올 수 있는 재무 지표 확인"""
    
    # 테스트할 종목들 (다양한 섹터)
    test_symbols = ['AAPL', 'JPM', 'JNJ', 'TSLA', 'WMT']
    
    print("🔍 yfinance에서 가져올 수 있는 재무 지표 확인\n")
    
    # 관심있는 재무 지표들
    financial_metrics = {
        # 밸류에이션
        'trailingPE': 'P/E Ratio (후행)',
        'forwardPE': 'P/E Ratio (선행)',
        'priceToBook': 'P/B Ratio',
        'pegRatio': 'PEG Ratio',
        'enterpriseToEbitda': 'EV/EBITDA',
        'priceToSalesTrailing12Months': 'P/S Ratio',
        
        # 수익성
        'returnOnEquity': 'ROE',
        'returnOnAssets': 'ROA',
        'profitMargins': '순이익률',
        'operatingMargins': '영업이익률',
        'grossMargins': '매출총이익률',
        
        # 재무건전성
        'debtToEquity': '부채비율 (D/E)',
        'currentRatio': '유동비율',
        'quickRatio': '당좌비율',
        
        # 성장성
        'revenueGrowth': '매출성장률',
        'earningsGrowth': '이익성장률',
        'revenueQuarterlyGrowth': '분기 매출성장률',
        
        # 배당
        'dividendYield': '배당수익률',
        'dividendRate': '연간 배당금',
        'payoutRatio': '배당성향',
        
        # 기타
        'beta': '베타 (시장민감도)',
        'trailingEps': 'EPS (후행)',
        'forwardEps': 'EPS (선행)',
        'bookValue': '주당순자산(BPS)',
        'marketCap': '시가총액',
        '52WeekChange': '52주 수익률'
    }
    
    # 각 종목별로 체크
    all_data = {}
    
    for symbol in test_symbols:
        print(f"\n📊 {symbol} 재무 지표 확인 중...")
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        symbol_data = {}
        for key, name in financial_metrics.items():
            value = info.get(key, 'N/A')
            if value != 'N/A' and value is not None:
                # 비율은 퍼센트로 변환
                if any(word in name for word in ['률', 'Ratio', 'Margin', 'Growth', 'ROE', 'ROA']):
                    if isinstance(value, (int, float)) and value < 10:  # 대부분 소수로 표현됨
                        value = f"{value*100:.2f}%"
                    else:
                        value = f"{value:.2f}%"
                elif isinstance(value, (int, float)):
                    value = f"{value:.2f}"
            symbol_data[name] = value
        
        all_data[symbol] = symbol_data
    
    # DataFrame으로 정리
    df = pd.DataFrame(all_data).T
    
    print("\n" + "="*80)
    print("📋 yfinance에서 가져올 수 있는 주요 재무 지표")
    print("="*80)
    print(df.to_string())
    
    # 가용성 통계
    print("\n📊 지표별 가용성 (N/A가 아닌 비율):")
    availability = {}
    for col in df.columns:
        available = (df[col] != 'N/A').sum()
        availability[col] = f"{available}/{len(df)} ({available/len(df)*100:.0f}%)"
    
    for metric, avail in sorted(availability.items(), key=lambda x: x[0]):
        print(f"  - {metric}: {avail}")
    
    # 거래 에피소드에 추가하기 좋은 지표 추천
    print("\n✅ 추천 지표 (가용성 높고 중요한 지표):")
    recommended = [
        'P/E Ratio (후행)',
        'P/B Ratio', 
        'ROE',
        '부채비율 (D/E)',
        '배당수익률',
        '베타 (시장민감도)',
        '52주 수익률',
        '영업이익률'
    ]
    
    for metric in recommended:
        if metric in availability:
            print(f"  - {metric}: {availability[metric]}")
    
    return df

if __name__ == "__main__":
    df = check_available_fundamentals()
    
    # CSV로 저장
    df.to_csv('../results/yfinance_fundamentals_check.csv')
    print(f"\n💾 결과가 '../results/yfinance_fundamentals_check.csv'에 저장되었습니다.")