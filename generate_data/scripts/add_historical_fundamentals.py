import pandas as pd
import numpy as np
import requests
from datetime import datetime
from tqdm import tqdm
import time
import yfinance as yf

class HistoricalFundamentalsCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.cache = {}  # 캐시로 API 호출 최소화
        
    def get_financial_ratios(self, symbol, year):
        """특정 연도의 재무 비율 가져오기"""
        cache_key = f"{symbol}_ratios_{year}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        url = f"{self.base_url}/ratios/{symbol}?limit=10&apikey={self.api_key}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if data:
                df = pd.DataFrame(data)
                # 해당 연도 데이터 찾기
                mask = df['date'].str[:4] == str(year)
                if mask.any():
                    year_data = df[mask].iloc[0]
                    result = {
                        'pb_ratio': year_data.get('priceToBookRatio', np.nan),
                        'roe': year_data.get('returnOnEquity', np.nan) * 100 if year_data.get('returnOnEquity') else np.nan,
                        'operating_margin': year_data.get('operatingProfitMargin', np.nan) * 100 if year_data.get('operatingProfitMargin') else np.nan,
                        'debt_equity_ratio': year_data.get('debtEquityRatio', np.nan)
                    }
                    self.cache[cache_key] = result
                    return result
        except:
            pass
            
        return {'pb_ratio': np.nan, 'roe': np.nan, 'operating_margin': np.nan, 'debt_equity_ratio': np.nan}
    
    def get_earnings_growth(self, symbol, year):
        """특정 연도의 이익 성장률 가져오기"""
        cache_key = f"{symbol}_growth_{year}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        url = f"{self.base_url}/financial-growth/{symbol}?limit=10&apikey={self.api_key}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if data:
                df = pd.DataFrame(data)
                mask = df['date'].str[:4] == str(year)
                if mask.any():
                    year_data = df[mask].iloc[0]
                    # epsgrowth 또는 netIncomeGrowth 사용
                    growth = year_data.get('epsgrowth', year_data.get('netIncomeGrowth', np.nan))
                    if growth and growth != np.nan:
                        growth = growth * 100  # 퍼센트로 변환
                    self.cache[cache_key] = growth
                    return growth
        except:
            pass
            
        return np.nan
    
    def get_income_statement(self, symbol, year):
        """EPS 데이터 가져오기 (P/E 계산용)"""
        cache_key = f"{symbol}_income_{year}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        url = f"{self.base_url}/income-statement/{symbol}?limit=10&apikey={self.api_key}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if data:
                df = pd.DataFrame(data)
                mask = df['date'].str[:4] == str(year)
                if mask.any():
                    year_data = df[mask].iloc[0]
                    result = {
                        'eps': year_data.get('eps', np.nan),
                        'revenue': year_data.get('revenue', np.nan),
                        'net_income': year_data.get('netIncome', np.nan)
                    }
                    self.cache[cache_key] = result
                    return result
        except:
            pass
            
        return {'eps': np.nan, 'revenue': np.nan, 'net_income': np.nan}
    
    def calculate_pe_ratio(self, price, eps):
        """P/E Ratio 계산"""
        if pd.notna(eps) and eps > 0 and pd.notna(price):
            return price / eps
        return np.nan

def add_fundamentals_to_episodes(input_file, output_file):
    """거래 에피소드에 과거 재무 데이터 추가"""
    
    print("📊 Loading enriched episodes...")
    df = pd.read_csv(input_file)
    df['entry_date'] = pd.to_datetime(df['entry_datetime']).dt.date
    df['exit_date'] = pd.to_datetime(df['exit_datetime']).dt.date
    
    print(f"  - Loaded {len(df):,} episodes")
    print(f"  - Date range: {df['entry_date'].min()} to {df['exit_date'].max()}")
    
    # FMP API 초기화
    api_key = "4pdQ35BjBbXNe2saE8cX8zxMiQKgxvdy"
    collector = HistoricalFundamentalsCollector(api_key)
    
    # 고유 심볼 추출
    symbols = df['symbol'].unique()
    print(f"  - Unique symbols: {len(symbols)}")
    
    # 새로운 컬럼 초기화
    fundamental_cols = [
        'entry_pe_ratio', 'entry_pb_ratio', 'entry_roe', 'entry_operating_margin', 
        'entry_debt_equity_ratio', 'entry_earnings_growth',
        'exit_pe_ratio', 'exit_pb_ratio', 'exit_roe', 'exit_operating_margin',
        'exit_debt_equity_ratio', 'exit_earnings_growth'
    ]
    
    for col in fundamental_cols:
        df[col] = np.nan
    
    # 각 에피소드에 대해 재무 데이터 추가
    print("\n💼 Fetching historical fundamentals for each episode...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing episodes"):
        symbol = row['symbol']
        entry_year = row['entry_date'].year
        exit_year = row['exit_date'].year
        
        # Entry 시점 재무 데이터
        entry_ratios = collector.get_financial_ratios(symbol, entry_year)
        entry_growth = collector.get_earnings_growth(symbol, entry_year)
        entry_income = collector.get_income_statement(symbol, entry_year)
        
        # Entry P/E Ratio 계산
        entry_pe = collector.calculate_pe_ratio(row['entry_price'], entry_income['eps'])
        
        df.at[idx, 'entry_pe_ratio'] = entry_pe
        df.at[idx, 'entry_pb_ratio'] = entry_ratios['pb_ratio']
        df.at[idx, 'entry_roe'] = entry_ratios['roe']
        df.at[idx, 'entry_operating_margin'] = entry_ratios['operating_margin']
        df.at[idx, 'entry_debt_equity_ratio'] = entry_ratios['debt_equity_ratio']
        df.at[idx, 'entry_earnings_growth'] = entry_growth
        
        # Exit 시점 재무 데이터
        exit_ratios = collector.get_financial_ratios(symbol, exit_year)
        exit_growth = collector.get_earnings_growth(symbol, exit_year)
        exit_income = collector.get_income_statement(symbol, exit_year)
        
        # Exit P/E Ratio 계산
        exit_pe = collector.calculate_pe_ratio(row['exit_price'], exit_income['eps'])
        
        df.at[idx, 'exit_pe_ratio'] = exit_pe
        df.at[idx, 'exit_pb_ratio'] = exit_ratios['pb_ratio']
        df.at[idx, 'exit_roe'] = exit_ratios['roe']
        df.at[idx, 'exit_operating_margin'] = exit_ratios['operating_margin']
        df.at[idx, 'exit_debt_equity_ratio'] = exit_ratios['debt_equity_ratio']
        df.at[idx, 'exit_earnings_growth'] = exit_growth
        
        # API 제한 방지 (무료 플랜)
        if idx % 10 == 0:
            time.sleep(0.5)
    
    # 통계 출력
    print("\n📊 Fundamental data statistics:")
    for col in fundamental_cols:
        non_null = df[col].notna().sum()
        pct = non_null / len(df) * 100
        if non_null > 0:
            mean_val = df[col].mean()
            print(f"\n{col}:")
            print(f"  - Coverage: {non_null:,}/{len(df):,} ({pct:.1f}%)")
            print(f"  - Mean: {mean_val:.2f}")
    
    # 저장
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved enriched data with historical fundamentals to: {output_file}")
    
    # 재무 지표 변화 분석
    print("\n🔍 Fundamental changes during holding period:")
    
    # 변화량 계산
    df['pe_change'] = df['exit_pe_ratio'] - df['entry_pe_ratio']
    df['roe_change'] = df['exit_roe'] - df['entry_roe']
    df['margin_change'] = df['exit_operating_margin'] - df['entry_operating_margin']
    
    # 변화와 수익률의 상관관계
    for metric in ['pe_change', 'roe_change', 'margin_change']:
        mask = df[metric].notna() & df['return_pct'].notna()
        if mask.sum() > 100:
            corr = df[mask][metric].corr(df[mask]['return_pct'])
            print(f"  - {metric} correlation with returns: {corr:.4f}")
    
    return df

if __name__ == "__main__":
    # 실행
    enriched_df = add_fundamentals_to_episodes(
        '../results/final/enriched_trading_episodes.csv',
        '../results/final/enriched_trading_episodes_with_fundamentals.csv'
    )
    
    # 샘플 출력
    print("\n📋 Sample data (first 3 rows):")
    sample_cols = ['symbol', 'entry_date', 'entry_pe_ratio', 'entry_roe', 
                   'exit_date', 'exit_pe_ratio', 'exit_roe', 'return_pct']
    print(enriched_df[sample_cols].head(3).to_string(index=False))