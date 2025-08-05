import pandas as pd
import numpy as np
import json
import yfinance as yf
from datetime import datetime

def analyze_all_greek_isins(file_path, top_n=50):
    """더 많은 그리스 ISIN 분석"""
    print("📊 Analyzing Greek ISINs...")
    
    # 상위 거래 ISIN 리스트
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 전체 데이터의 날짜 범위 파악
    date_min = df['timestamp'].min()
    date_max = df['timestamp'].max()
    print(f"  - Date range: {date_min.date()} to {date_max.date()}")
    
    isin_counts = df['ISIN'].value_counts().head(top_n)
    
    isin_profiles = {}
    
    for i, (isin, count) in enumerate(isin_counts.items()):
        if i % 10 == 0:
            print(f"  Progress: {i}/{top_n}")
            
        # 해당 ISIN 데이터만 추출
        isin_data = df[df['ISIN'] == isin].copy()
        isin_data['timestamp'] = pd.to_datetime(isin_data['timestamp'])
        isin_data['price_per_unit'] = isin_data['totalValue'] / isin_data['units']
        
        # 일별 평균 가격
        daily_prices = isin_data.groupby(isin_data['timestamp'].dt.date)['price_per_unit'].mean()
        
        if len(daily_prices) > 20:
            # 로그 수익률
            log_returns = np.log(daily_prices / daily_prices.shift(1)).dropna()
            
            # 변동성 (연간화)
            annual_vol = log_returns.std() * np.sqrt(252) * 100
            
            # 평균 수익률
            annual_return = log_returns.mean() * 252 * 100
            
            # 거래 빈도
            date_range = (isin_data['timestamp'].max() - isin_data['timestamp'].min()).days
            trading_freq = len(isin_data) / max(date_range, 1)
            
            # 평균 거래 규모
            avg_trade_value = isin_data['totalValue'].mean()
            
            isin_profiles[isin] = {
                'volatility': round(annual_vol, 2),
                'return': round(annual_return, 2),
                'trading_freq': round(trading_freq, 2),
                'avg_trade_value': round(avg_trade_value, 2),
                'n_trades': count,
                'n_customers': isin_data['customerID'].nunique(),
                'date_min': isin_data['timestamp'].min(),
                'date_max': isin_data['timestamp'].max()
            }
    
    return isin_profiles, date_min, date_max

def get_diverse_us_stocks():
    """다양한 미국 주식 리스트 - 200개로 확대"""
    return [
        # Mega Cap Tech (10)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'ORCL', 'CSCO', 'INTC', 'IBM',
        
        # Large Cap Growth (15)
        'TSLA', 'NFLX', 'ADBE', 'CRM', 'AMD', 'AVGO', 'QCOM', 'TXN', 'MU', 'AMAT',
        'LRCX', 'KLAC', 'ASML', 'NOW', 'INTU',
        
        # Financials (20)
        'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'SCHW', 'BLK', 'SPGI', 'AXP',
        'COF', 'USB', 'PNC', 'TFC', 'BK', 'STT', 'TROW', 'NTRS', 'FITB', 'RF',
        
        # Healthcare (20)
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'LLY', 'MRK', 'CVS', 'BMY', 'AMGN',
        'GILD', 'DHR', 'MDT', 'ABT', 'SYK', 'ISRG', 'BSX', 'ELV', 'CI', 'HUM',
        
        # Consumer (20)
        'WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'TGT', 'LOW', 'COST', 'PEP', 'KO',
        'PG', 'CL', 'MDLZ', 'MO', 'PM', 'KMB', 'GIS', 'K', 'HSY', 'STZ',
        
        # Industrials (15)
        'BA', 'CAT', 'UPS', 'HON', 'GE', 'MMM', 'LMT', 'RTX', 'DE', 'UNP',
        'FDX', 'EMR', 'ETN', 'ITW', 'CSX',
        
        # High Volatility / Meme Stocks (25)
        'COIN', 'RIOT', 'MARA', 'PLTR', 'HOOD', 'UPST', 'GME', 'AMC', 'SOFI', 'WISH',
        'CLOV', 'BB', 'NOK', 'SNDL', 'TLRY', 'ACB', 'CGC', 'SPCE', 'LCID', 'RIVN',
        'NKLA', 'RIDE', 'WKHS', 'GOEV', 'FSR',
        
        # Mid Cap Tech (20)
        'ROKU', 'SNAP', 'PINS', 'ZM', 'DOCU', 'OKTA', 'TWLO', 'NET', 'DDOG', 'SNOW',
        'CRWD', 'ZS', 'PANW', 'FTNT', 'CYBR', 'S', 'ESTC', 'MDB', 'CFLT', 'U',
        
        # Crypto/Blockchain (10)
        'MSTR', 'HIVE', 'HUT', 'BITF', 'BTBT', 'ARBK', 'CIFR', 'CLSK', 'DGHI', 'BTDR',
        
        # Energy (15)
        'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'EOG', 'PSX', 'MPC', 'VLO', 'PXD',
        'DVN', 'HES', 'FANG', 'CTRA', 'MRO',
        
        # REITs (10)
        'SPG', 'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'DLR', 'O', 'WELL', 'AVB',
        
        # Utilities (10)
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'ED', 'XEL',
        
        # Materials (10)
        'LIN', 'APD', 'FCX', 'NEM', 'CTVA', 'DD', 'ECL', 'SHW', 'GOLD', 'NUE',
        
        # Biotech (15)
        'MRNA', 'BNTX', 'NVAX', 'BIIB', 'REGN', 'VRTX', 'ALXN', 'SGEN', 'INCY', 'BMRN',
        'ALNY', 'RARE', 'IONS', 'EXAS', 'TECH',
        
        # EV/Clean Energy (10)
        'NIO', 'XPEV', 'LI', 'QS', 'CHPT', 'BLNK', 'EVGO', 'LEV', 'PTRA', 'HYLN',
        
        # ETFs (10)
        'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'ARKK', 'ARKW', 'ARKQ', 'ARKG'
    ]

def analyze_us_characteristics(tickers, start_date, end_date):
    """미국 주식 특성 분석 - 그리스 데이터와 동일한 기간"""
    print(f"\n📈 Analyzing {len(tickers)} US stocks...")
    print(f"  - Date range: {start_date} to {end_date}")
    
    profiles = {}
    failed = []
    
    # 배치 다운로드 - 정확한 날짜 범위 지정
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=True)
    
    for ticker in tickers:
        try:
            if len(data.columns.levels) > 0 and ticker in data.columns.levels[0]:
                ticker_data = data[ticker]['Close'].dropna()
            else:
                ticker_data = data['Close'].dropna() if len(tickers) == 1 else pd.Series()
                
            if len(ticker_data) > 50:
                log_returns = np.log(ticker_data / ticker_data.shift(1)).dropna()
                
                profiles[ticker] = {
                    'volatility': round(log_returns.std() * np.sqrt(252) * 100, 2),
                    'return': round(log_returns.mean() * 252 * 100, 2),
                    'price_mean': round(ticker_data.mean(), 2),
                    'liquidity': len(ticker_data)  # 거래일수로 유동성 대체
                }
        except:
            failed.append(ticker)
            
    if failed:
        print(f"  ⚠️ Failed to analyze: {failed}")
        
    return profiles

def create_unique_mapping(greek_profiles, us_profiles):
    """각 ISIN에 고유한 미국 주식 매핑"""
    print("\n🔗 Creating unique mappings...")
    
    mappings = {}
    used_tickers = set()
    
    # 그리스 ISIN을 변동성 순으로 정렬
    sorted_isins = sorted(greek_profiles.items(), 
                         key=lambda x: x[1]['volatility'], 
                         reverse=True)
    
    # 미국 주식도 변동성 순으로 정렬
    sorted_us = sorted(us_profiles.items(), 
                      key=lambda x: x[1]['volatility'], 
                      reverse=True)
    
    for isin, isin_props in sorted_isins:
        best_match = None
        min_score = float('inf')
        
        for ticker, us_props in sorted_us:
            # 이미 사용된 ticker는 건너뛰기
            if ticker in used_tickers:
                continue
                
            # 특성 차이 계산
            vol_diff = abs(isin_props['volatility'] - us_props['volatility'])
            ret_diff = abs(isin_props['return'] - us_props['return'])
            
            # 종합 점수
            score = vol_diff + ret_diff * 0.5
            
            if score < min_score:
                min_score = score
                best_match = ticker
        
        if best_match:
            mappings[isin] = {
                'us_ticker': best_match,
                'match_score': round(min_score, 2),
                'isin_vol': isin_props['volatility'],
                'us_vol': us_profiles[best_match]['volatility'],
                'isin_ret': isin_props['return'],
                'us_ret': us_profiles[best_match]['return']
            }
            used_tickers.add(best_match)
            
            print(f"{isin} → {best_match} (vol: {isin_props['volatility']}% → {us_profiles[best_match]['volatility']}%)")
    
    return mappings

def main():
    # 1. 그리스 ISIN 분석 (상위 50개)
    greek_profiles, date_min, date_max = analyze_all_greek_isins('/Users/inter4259/Desktop/transactions.csv', top_n=50)
    
    # 2. 미국 주식 분석 - 그리스 데이터와 동일한 기간
    us_tickers = get_diverse_us_stocks()
    us_profiles = analyze_us_characteristics(us_tickers, start_date=date_min.strftime('%Y-%m-%d'), 
                                            end_date=date_max.strftime('%Y-%m-%d'))
    
    # 3. 1:1 고유 매핑 생성
    unique_mappings = create_unique_mapping(greek_profiles, us_profiles)
    
    # 4. 결과 저장
    with open('../results/mappings/unique_isin_mapping.json', 'w') as f:
        json.dump(unique_mappings, f, indent=2)
    
    print(f"\n✅ Created unique mappings for {len(unique_mappings)} ISINs")
    print(f"📁 Saved to ../results/mappings/unique_isin_mapping.json")
    
    # 매핑 요약
    print("\n📊 Mapping Summary:")
    df = pd.DataFrame(unique_mappings).T
    print(f"Average volatility difference: {(df['isin_vol'] - df['us_vol']).abs().mean():.2f}%")
    print(f"Unique US stocks used: {df['us_ticker'].nunique()}")
    
    return unique_mappings

if __name__ == "__main__":
    mappings = main()