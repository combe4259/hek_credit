import pandas as pd
import numpy as np
import requests
from datetime import datetime
from tqdm import tqdm
import time

class HistoricalFundamentalsCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.symbol_data_cache = {}  # 심볼별 전체 데이터 캐시
        
    def fetch_all_data_for_symbol(self, symbol):
        """심볼의 모든 과거 데이터를 한 번에 가져와서 캐시"""
        if symbol in self.symbol_data_cache:
            return
            
        print(f"\n📊 Fetching all historical data for {symbol}...")
        
        # 1. Financial Ratios
        ratios_url = f"{self.base_url}/ratios/{symbol}?limit=20&apikey={self.api_key}"
        try:
            response = requests.get(ratios_url)
            ratios_data = response.json()
            ratios_df = pd.DataFrame(ratios_data) if ratios_data else pd.DataFrame()
        except:
            ratios_df = pd.DataFrame()
        
        # 2. Financial Growth
        growth_url = f"{self.base_url}/financial-growth/{symbol}?limit=20&apikey={self.api_key}"
        try:
            response = requests.get(growth_url)
            growth_data = response.json()
            growth_df = pd.DataFrame(growth_data) if growth_data else pd.DataFrame()
        except:
            growth_df = pd.DataFrame()
        
        # 3. Income Statement (EPS용)
        income_url = f"{self.base_url}/income-statement/{symbol}?limit=20&apikey={self.api_key}"
        try:
            response = requests.get(income_url)
            income_data = response.json()
            income_df = pd.DataFrame(income_data) if income_data else pd.DataFrame()
        except:
            income_df = pd.DataFrame()
        
        # 캐시에 저장
        self.symbol_data_cache[symbol] = {
            'ratios': ratios_df,
            'growth': growth_df,
            'income': income_df
        }
        
        time.sleep(0.5)  # API 제한 방지
    
    def get_fundamentals_for_year(self, symbol, year):
        """특정 연도의 모든 재무 데이터 반환"""
        if symbol not in self.symbol_data_cache:
            return None
            
        data = self.symbol_data_cache[symbol]
        result = {}
        
        # Ratios
        if not data['ratios'].empty and 'date' in data['ratios'].columns:
            mask = data['ratios']['date'].str[:4] == str(year)
            if mask.any():
                year_data = data['ratios'][mask].iloc[0]
                result['pb_ratio'] = year_data.get('priceToBookRatio', np.nan)
                result['roe'] = year_data.get('returnOnEquity', np.nan) * 100 if year_data.get('returnOnEquity') else np.nan
                result['operating_margin'] = year_data.get('operatingProfitMargin', np.nan) * 100 if year_data.get('operatingProfitMargin') else np.nan
                result['debt_equity_ratio'] = year_data.get('debtEquityRatio', np.nan)
        
        # Growth
        if not data['growth'].empty and 'date' in data['growth'].columns:
            mask = data['growth']['date'].str[:4] == str(year)
            if mask.any():
                year_data = data['growth'][mask].iloc[0]
                growth = year_data.get('epsgrowth', year_data.get('netIncomeGrowth', np.nan))
                if growth and growth != np.nan:
                    result['earnings_growth'] = growth * 100
        
        # Income (EPS)
        if not data['income'].empty and 'date' in data['income'].columns:
            mask = data['income']['date'].str[:4] == str(year)
            if mask.any():
                year_data = data['income'][mask].iloc[0]
                result['eps'] = year_data.get('eps', np.nan)
        
        return result

def add_fundamentals_to_episodes_fast(input_file, output_file):
    """더 빠른 방법으로 재무 데이터 추가"""
    
    print("📊 Loading enriched episodes...")
    df = pd.read_csv(input_file)
    df['entry_date'] = pd.to_datetime(df['entry_datetime']).dt.date
    df['exit_date'] = pd.to_datetime(df['exit_datetime']).dt.date
    
    print(f"  - Loaded {len(df):,} episodes")
    
    # 고유 심볼 추출
    symbols = df['symbol'].unique()
    print(f"  - Unique symbols: {len(symbols)}")
    
    # FMP API 초기화
    api_key = "4pdQ35BjBbXNe2saE8cX8zxMiQKgxvdy"
    collector = HistoricalFundamentalsCollector(api_key)
    
    # 1단계: 모든 심볼의 데이터를 먼저 가져오기
    print("\n📥 Step 1: Fetching all historical data for each symbol...")
    for symbol in tqdm(symbols, desc="Fetching symbol data"):
        collector.fetch_all_data_for_symbol(symbol)
    
    # 2단계: 각 에피소드에 데이터 적용
    print("\n🔧 Step 2: Applying fundamentals to episodes...")
    
    # 새로운 컬럼 초기화
    fundamental_metrics = ['pe_ratio', 'pb_ratio', 'roe', 'operating_margin', 
                          'debt_equity_ratio', 'earnings_growth']
    
    for prefix in ['entry_', 'exit_']:
        for metric in fundamental_metrics:
            df[f'{prefix}{metric}'] = np.nan
    
    # 배치 처리
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing episodes"):
        symbol = row['symbol']
        entry_year = row['entry_date'].year
        exit_year = row['exit_date'].year
        
        # Entry 시점 데이터
        entry_data = collector.get_fundamentals_for_year(symbol, entry_year)
        if entry_data:
            # P/E Ratio 계산
            if 'eps' in entry_data and pd.notna(entry_data['eps']) and entry_data['eps'] > 0:
                df.at[idx, 'entry_pe_ratio'] = row['entry_price'] / entry_data['eps']
            
            for metric in ['pb_ratio', 'roe', 'operating_margin', 'debt_equity_ratio', 'earnings_growth']:
                if metric in entry_data:
                    df.at[idx, f'entry_{metric}'] = entry_data[metric]
        
        # Exit 시점 데이터
        exit_data = collector.get_fundamentals_for_year(symbol, exit_year)
        if exit_data:
            # P/E Ratio 계산
            if 'eps' in exit_data and pd.notna(exit_data['eps']) and exit_data['eps'] > 0:
                df.at[idx, 'exit_pe_ratio'] = row['exit_price'] / exit_data['eps']
            
            for metric in ['pb_ratio', 'roe', 'operating_margin', 'debt_equity_ratio', 'earnings_growth']:
                if metric in exit_data:
                    df.at[idx, f'exit_{metric}'] = exit_data[metric]
    
    # 통계 출력
    print("\n📊 Fundamental data coverage:")
    for metric in fundamental_metrics:
        entry_coverage = df[f'entry_{metric}'].notna().sum() / len(df) * 100
        exit_coverage = df[f'exit_{metric}'].notna().sum() / len(df) * 100
        print(f"  {metric}: entry={entry_coverage:.1f}%, exit={exit_coverage:.1f}%")
    
    # 저장
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")
    
    # 샘플 출력
    print("\n📋 Sample data:")
    sample_cols = ['symbol', 'entry_date', 'entry_pe_ratio', 'entry_roe', 
                   'exit_date', 'exit_pe_ratio', 'exit_roe', 'return_pct']
    print(df[sample_cols].head(5).to_string(index=False))
    
    return df

if __name__ == "__main__":
    enriched_df = add_fundamentals_to_episodes_fast(
        '../results/final/enriched_trading_episodes.csv',
        '../results/final/enriched_trading_episodes_with_fundamentals.csv'
    )