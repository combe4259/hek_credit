import requests
import pandas as pd
import json
from datetime import datetime

def test_fmp_api():
    """Financial Modeling Prep API로 과거 재무제표 테스트"""
    
    # 무료 API 키 필요 (https://site.financialmodelingprep.com/developer/docs)
    # 무료 플랜: 하루 250 요청
    api_key = "4pdQ35BjBbXNe2saE8cX8zxMiQKgxvdy"  # 여기에 API 키 입력
    
    if api_key == "YOUR_API_KEY":
        print("⚠️  Financial Modeling Prep API 키가 필요합니다!")
        print("1. https://site.financialmodelingprep.com/register 에서 가입")
        print("2. 무료 API 키 발급")
        print("3. 이 스크립트의 api_key 변수에 입력")
        return
    
    # 테스트할 종목
    symbol = "AAPL"
    
    print(f"🔍 {symbol}의 과거 재무제표 데이터 테스트\n")
    
    # 1. 연간 재무제표 (Income Statement)
    print("📊 과거 Income Statement 가져오기...")
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?limit=10&apikey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data:
            print(f"✅ {len(data)}개 연도의 데이터 수신\n")
            
            # DataFrame으로 변환
            df = pd.DataFrame(data)
            
            # 주요 지표만 선택
            key_metrics = ['date', 'revenue', 'operatingIncome', 'netIncome', 
                          'eps', 'operatingIncomeRatio']
            
            if all(col in df.columns for col in key_metrics):
                df_key = df[key_metrics].head(5)
                print("최근 5년 주요 지표:")
                print(df_key.to_string(index=False))
                
                # 2018-2022년 데이터 확인
                print("\n📅 2018-2022년 데이터 확인:")
                mask = df['date'].str[:4].isin(['2018', '2019', '2020', '2021', '2022'])
                historical_data = df[mask][key_metrics]
                
                if not historical_data.empty:
                    print(historical_data.to_string(index=False))
                else:
                    print("❌ 2018-2022년 데이터가 없습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    
    # 2. 주요 재무 비율
    print("\n📊 과거 재무 비율 (Key Metrics)...")
    url = f"https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?limit=10&apikey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data:
            df_ratios = pd.DataFrame(data)
            
            # 우리가 원하는 지표들
            wanted_metrics = ['date', 'peRatio', 'priceToBookRatio', 'pbRatio', 
                            'roe', 'returnOnEquity', 'debtEquityRatio', 'debtToEquity',
                            'operatingProfitMargin', 'netProfitMargin']
            
            # 실제로 존재하는 컬럼 확인
            print(f"\n사용 가능한 컬럼들: {df_ratios.columns.tolist()[:10]}...")
            
            available_metrics = [col for col in wanted_metrics if col in df_ratios.columns]
            
            if available_metrics:
                print("\n우리가 원하는 지표들 (2018-2022):")
                mask = df_ratios['date'].str[:4].isin(['2018', '2019', '2020', '2021', '2022'])
                historical_ratios = df_ratios[mask][available_metrics]
                
                if not historical_ratios.empty:
                    print(historical_ratios.to_string(index=False))
                    
                    # 거래 시점과 매칭 예시
                    print("\n💡 거래 시점 매칭 예시:")
                    print("거래 진입: 2018-01-02 → 2018년 재무 데이터 사용")
                    print("거래 청산: 2020-02-24 → 2020년 재무 데이터 사용")
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

def test_yfinance_historical():
    """yfinance로 가능한 과거 데이터 확인"""
    import yfinance as yf
    
    print("\n" + "="*60)
    print("📊 yfinance로 가능한 과거 재무 데이터 확인")
    print("="*60)
    
    ticker = yf.Ticker("AAPL")
    
    # 분기별 재무제표 (최근 4분기만)
    print("\n분기별 Income Statement (최근 4분기):")
    income = ticker.quarterly_financials
    if not income.empty:
        print(income.columns[:5].tolist())
        print("날짜:", [str(d)[:10] for d in income.columns[:5]])
    
    # 연간 재무제표 (최근 4년)
    print("\n연간 Income Statement (최근 4년):")
    annual = ticker.financials
    if not annual.empty:
        print("날짜:", [str(d)[:10] for d in annual.columns])

if __name__ == "__main__":
    # 1. Financial Modeling Prep 테스트
    test_fmp_api()
    
    # 2. yfinance 과거 데이터 확인
    test_yfinance_historical()