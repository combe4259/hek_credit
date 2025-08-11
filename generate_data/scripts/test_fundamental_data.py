import yfinance as yf
import pandas as pd
import numpy as np

def test_fundamental_keys():
    """yfinance에서 원하는 재무 지표들을 정확히 가져올 수 있는지 테스트"""
    
    # 테스트할 종목들
    test_symbols = ['AAPL', 'MSFT', 'JPM', 'TSLA', 'JNJ']
    
    print("🧪 재무 지표 키 매핑 테스트\n")
    print("원하는 지표 → yfinance 키 이름:")
    print("-" * 50)
    
    # 키 매핑
    key_mapping = {
        'pe_ratio_trailing': 'trailingPE',
        'pb_ratio': 'priceToBook', 
        'roe': 'returnOnEquity',
        'operating_margin': 'operatingMargins',
        'debt_equity_ratio': 'debtToEquity',
        'beta': 'beta',
        'earnings_growth': 'earningsGrowth'
    }
    
    for our_key, yf_key in key_mapping.items():
        print(f"{our_key:20} → {yf_key}")
    
    print("\n" + "="*80)
    print("📊 각 종목별 데이터 수집 테스트")
    print("="*80)
    
    all_data = {}
    
    for symbol in test_symbols:
        print(f"\n📈 {symbol}:")
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        symbol_data = {}
        
        # 각 지표 가져오기
        for our_key, yf_key in key_mapping.items():
            raw_value = info.get(yf_key, None)
            
            # 값 처리
            if raw_value is not None:
                # ROE, margins, growth는 보통 0.xx 형태로 제공됨
                if our_key in ['roe', 'operating_margin', 'earnings_growth']:
                    if isinstance(raw_value, (int, float)) and raw_value < 10:
                        value = raw_value * 100  # 퍼센트로 변환
                    else:
                        value = raw_value
                else:
                    value = raw_value
                
                symbol_data[our_key] = value
                
                # 출력
                if our_key in ['roe', 'operating_margin', 'earnings_growth']:
                    print(f"  {our_key:20}: {value:>10.2f}%")
                else:
                    print(f"  {our_key:20}: {value:>10.2f}")
            else:
                symbol_data[our_key] = np.nan
                print(f"  {our_key:20}: {'N/A':>10}")
        
        all_data[symbol] = symbol_data
    
    # DataFrame으로 정리
    df = pd.DataFrame(all_data).T
    
    print("\n" + "="*80)
    print("📊 전체 요약 (DataFrame)")
    print("="*80)
    print(df.to_string())
    
    # 가용성 통계
    print("\n📈 지표별 가용성:")
    for col in df.columns:
        available = df[col].notna().sum()
        total = len(df)
        print(f"  {col:20}: {available}/{total} ({available/total*100:.0f}%)")
    
    # 실제 enriched_trading_episodes.csv의 심볼들 체크
    print("\n" + "="*80)
    print("🔍 실제 거래 데이터의 심볼 확인")
    print("="*80)
    
    try:
        episodes = pd.read_csv('../results/final/enriched_trading_episodes.csv')
        unique_symbols = episodes['symbol'].unique()
        print(f"\n거래 데이터의 고유 심볼 수: {len(unique_symbols)}")
        print(f"심볼 목록: {', '.join(unique_symbols[:10])}...")
        
        # 몇 개 샘플로 테스트
        print("\n📊 실제 심볼 샘플 테스트:")
        for symbol in unique_symbols[:3]:
            print(f"\n{symbol}:")
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                for our_key, yf_key in key_mapping.items():
                    value = info.get(yf_key, 'N/A')
                    if value != 'N/A' and our_key in ['roe', 'operating_margin', 'earnings_growth']:
                        if isinstance(value, (int, float)) and value < 10:
                            value = value * 100
                        print(f"  {our_key}: {value:.2f}%")
                    elif value != 'N/A':
                        print(f"  {our_key}: {value:.2f}")
                    else:
                        print(f"  {our_key}: N/A")
            except Exception as e:
                print(f"  Error: {e}")
                
    except FileNotFoundError:
        print("enriched_trading_episodes.csv 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    test_fundamental_keys()