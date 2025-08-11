import requests
import pandas as pd

def test_all_fmp_endpoints():
    """Financial Modeling Prep의 모든 관련 엔드포인트 테스트"""
    
    api_key = "4pdQ35BjBbXNe2saE8cX8zxMiQKgxvdy"
    symbol = "AAPL"
    
    print("🔍 Financial Modeling Prep - 모든 재무 비율 엔드포인트 테스트\n")
    
    # 1. Financial Ratios 엔드포인트
    print("1️⃣ Financial Ratios 엔드포인트:")
    url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}?limit=5&apikey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data:
            df = pd.DataFrame(data)
            print(f"✅ {len(data)}개 연도 데이터 수신")
            print(f"컬럼: {df.columns.tolist()[:15]}...")
            
            # 우리가 원하는 지표 확인
            our_metrics = {
                'peRatio': 'P/E Ratio',
                'priceToBookRatio': 'P/B Ratio',
                'returnOnEquity': 'ROE',
                'debtEquityRatio': 'Debt/Equity',
                'operatingProfitMargin': 'Operating Margin'
            }
            
            print("\n우리가 원하는 지표들:")
            for key, name in our_metrics.items():
                if key in df.columns:
                    print(f"  ✅ {name} ({key})")
                else:
                    print(f"  ❌ {name} ({key})")
            
            # 2018-2020 데이터 샘플
            if 'date' in df.columns:
                print("\n2018-2020년 데이터 샘플:")
                mask = df['date'].str[:4].isin(['2018', '2019', '2020'])
                sample_cols = ['date'] + [col for col in our_metrics.keys() if col in df.columns]
                if mask.any():
                    print(df[mask][sample_cols].to_string(index=False))
    
    except Exception as e:
        print(f"❌ 오류: {e}")
    
    # 2. Financial Growth 엔드포인트
    print("\n\n2️⃣ Financial Growth 엔드포인트:")
    url = f"https://financialmodelingprep.com/api/v3/financial-growth/{symbol}?limit=5&apikey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data:
            df = pd.DataFrame(data)
            print(f"✅ {len(data)}개 연도 데이터 수신")
            
            # Earnings Growth 확인
            if 'epsgrowth' in df.columns or 'netIncomeGrowth' in df.columns:
                print("\nEarnings Growth 관련:")
                growth_cols = [col for col in df.columns if 'growth' in col.lower()]
                print(f"Growth 지표들: {growth_cols[:5]}")
                
                if 'date' in df.columns:
                    sample_cols = ['date'] + [col for col in ['epsgrowth', 'netIncomeGrowth', 'revenueGrowth'] if col in df.columns]
                    print(df[sample_cols].head(3).to_string(index=False))
    
    except Exception as e:
        print(f"❌ 오류: {e}")
    
    # 3. Company Key Metrics TTM (Trailing Twelve Months)
    print("\n\n3️⃣ Key Metrics TTM 엔드포인트:")
    url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}?apikey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data and len(data) > 0:
            ttm_data = data[0]
            print("✅ TTM (Trailing Twelve Months) 데이터:")
            
            for key in ['peRatioTTM', 'priceToBookRatioTTM', 'roeTTM', 'debtEquityRatioTTM']:
                if key in ttm_data:
                    print(f"  {key}: {ttm_data[key]}")
    
    except Exception as e:
        print(f"❌ 오류: {e}")
    
    # 4. 추가: Balance Sheet (부채비율 계산용)
    print("\n\n4️⃣ Balance Sheet Statement (부채비율 직접 계산):")
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?limit=3&apikey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data:
            df = pd.DataFrame(data)
            if 'totalDebt' in df.columns and 'totalStockholdersEquity' in df.columns:
                df['calculatedDebtEquity'] = df['totalDebt'] / df['totalStockholdersEquity']
                print("✅ 부채비율 직접 계산 가능")
                print(df[['date', 'totalDebt', 'totalStockholdersEquity', 'calculatedDebtEquity']].to_string(index=False))
    
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    test_all_fmp_endpoints()