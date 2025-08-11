import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from tqdm import tqdm

def calculate_technical_indicators(hist_data, target_date, symbol):
    """특정 날짜 시점의 기술적 지표 계산"""
    
    # 시간대 처리를 위해 날짜만 사용
    hist_to_date = hist_data[hist_data.index.date <= target_date.date()].copy()
    
    if len(hist_to_date) < 5:  # 최소 5일 데이터 필요
        return None
    
    try:
        current_price = hist_to_date['Close'].iloc[-1]
        current_volume = hist_to_date['Volume'].iloc[-1]
        
        # 1. 모멘텀 (5, 20, 60일)
        momentum_5d = (current_price / hist_to_date['Close'].iloc[-6] - 1) * 100 if len(hist_to_date) >= 6 else np.nan
        momentum_20d = (current_price / hist_to_date['Close'].iloc[-21] - 1) * 100 if len(hist_to_date) >= 21 else np.nan
        momentum_60d = (current_price / hist_to_date['Close'].iloc[-61] - 1) * 100 if len(hist_to_date) >= 61 else np.nan
        
        # 2. 이동평균 대비 괴리율
        ma_5 = hist_to_date['Close'].rolling(window=5).mean().iloc[-1] if len(hist_to_date) >= 5 else np.nan
        ma_20 = hist_to_date['Close'].rolling(window=20).mean().iloc[-1] if len(hist_to_date) >= 20 else np.nan
        ma_60 = hist_to_date['Close'].rolling(window=60).mean().iloc[-1] if len(hist_to_date) >= 60 else np.nan
        
        ma_dev_5 = (current_price / ma_5 - 1) * 100 if not np.isnan(ma_5) else np.nan
        ma_dev_20 = (current_price / ma_20 - 1) * 100 if not np.isnan(ma_20) else np.nan
        ma_dev_60 = (current_price / ma_60 - 1) * 100 if not np.isnan(ma_60) else np.nan
        
        # 3. 거래량 변화율
        avg_vol_5 = hist_to_date['Volume'].iloc[-5:].mean() if len(hist_to_date) >= 5 else np.nan
        avg_vol_20 = hist_to_date['Volume'].iloc[-20:].mean() if len(hist_to_date) >= 20 else np.nan
        avg_vol_60 = hist_to_date['Volume'].iloc[-60:].mean() if len(hist_to_date) >= 60 else np.nan
        
        vol_change_5 = (current_volume / avg_vol_5 - 1) * 100 if avg_vol_5 > 0 else np.nan
        vol_change_20 = (current_volume / avg_vol_20 - 1) * 100 if avg_vol_20 > 0 else np.nan
        vol_change_60 = (current_volume / avg_vol_60 - 1) * 100 if avg_vol_60 > 0 else np.nan
        
        # 4. 변동성 (연간화된 표준편차)
        returns = hist_to_date['Close'].pct_change().dropna()
        
        vol_5 = returns.iloc[-5:].std() * np.sqrt(252) * 100 if len(returns) >= 5 else np.nan
        vol_20 = returns.iloc[-20:].std() * np.sqrt(252) * 100 if len(returns) >= 20 else np.nan
        vol_60 = returns.iloc[-60:].std() * np.sqrt(252) * 100 if len(returns) >= 60 else np.nan
        
        # 5. 52주 고점 대비 현재 가격 비율 (가능한 데이터만큼만)
        if len(hist_to_date) >= 252:
            high_52w = hist_to_date['High'].iloc[-252:].max()
            ratio_52w_high = (current_price / high_52w) * 100
        else:
            # 52주 미만이면 가능한 기간의 최고점 사용
            high_period = hist_to_date['High'].max()
            ratio_52w_high = (current_price / high_period) * 100
        
        return {
            'momentum_5d': round(momentum_5d, 2),
            'momentum_20d': round(momentum_20d, 2),
            'momentum_60d': round(momentum_60d, 2),
            'ma_dev_5d': round(ma_dev_5, 2),
            'ma_dev_20d': round(ma_dev_20, 2),
            'ma_dev_60d': round(ma_dev_60, 2),
            'vol_change_5d': round(vol_change_5, 2),
            'vol_change_20d': round(vol_change_20, 2),
            'vol_change_60d': round(vol_change_60, 2),
            'volatility_5d': round(vol_5, 2),
            'volatility_20d': round(vol_20, 2),
            'volatility_60d': round(vol_60, 2),
            'ratio_52w_high': round(ratio_52w_high, 2)
        }
        
    except Exception as e:
        print(f"Error calculating indicators for {symbol} on {target_date}: {e}")
        return None

def calculate_indicators_with_holding_period(hist_data, target_date, symbol, holding_days):
    """기술적 지표 계산 (보유 기간과 관계없이 모든 지표 계산)"""
    
    # 시간대 처리를 위해 날짜만 사용
    hist_to_date = hist_data[hist_data.index.date <= target_date.date()].copy()
    
    if len(hist_to_date) < 5:  # 최소 5일 데이터 필요
        return None
    
    try:
        current_price = hist_to_date['Close'].iloc[-1]
        current_volume = hist_to_date['Volume'].iloc[-1]
        
        # 1. 모멘텀 (5, 20, 60일) - 보유 기간과 관계없이 모두 계산
        momentum_5d = (current_price / hist_to_date['Close'].iloc[-6] - 1) * 100 if len(hist_to_date) >= 6 else np.nan
        momentum_20d = (current_price / hist_to_date['Close'].iloc[-21] - 1) * 100 if len(hist_to_date) >= 21 else np.nan
        momentum_60d = (current_price / hist_to_date['Close'].iloc[-61] - 1) * 100 if len(hist_to_date) >= 61 else np.nan
        
        # 2. 이동평균 대비 괴리율
        ma_5 = hist_to_date['Close'].rolling(window=5).mean().iloc[-1] if len(hist_to_date) >= 5 else np.nan
        ma_20 = hist_to_date['Close'].rolling(window=20).mean().iloc[-1] if len(hist_to_date) >= 20 else np.nan
        ma_60 = hist_to_date['Close'].rolling(window=60).mean().iloc[-1] if len(hist_to_date) >= 60 else np.nan
        
        ma_dev_5 = (current_price / ma_5 - 1) * 100 if not np.isnan(ma_5) else np.nan
        ma_dev_20 = (current_price / ma_20 - 1) * 100 if not np.isnan(ma_20) else np.nan
        ma_dev_60 = (current_price / ma_60 - 1) * 100 if not np.isnan(ma_60) else np.nan
        
        # 3. 거래량 변화율
        avg_vol_5 = hist_to_date['Volume'].iloc[-5:].mean() if len(hist_to_date) >= 5 else np.nan
        avg_vol_20 = hist_to_date['Volume'].iloc[-20:].mean() if len(hist_to_date) >= 20 else np.nan
        avg_vol_60 = hist_to_date['Volume'].iloc[-60:].mean() if len(hist_to_date) >= 60 else np.nan
        
        vol_change_5 = (current_volume / avg_vol_5 - 1) * 100 if avg_vol_5 > 0 and not np.isnan(avg_vol_5) else np.nan
        vol_change_20 = (current_volume / avg_vol_20 - 1) * 100 if avg_vol_20 > 0 and not np.isnan(avg_vol_20) else np.nan
        vol_change_60 = (current_volume / avg_vol_60 - 1) * 100 if avg_vol_60 > 0 and not np.isnan(avg_vol_60) else np.nan
        
        # 4. 변동성 (연간화된 표준편차)
        returns = hist_to_date['Close'].pct_change().dropna()
        
        vol_5 = returns.iloc[-5:].std() * np.sqrt(252) * 100 if len(returns) >= 5 else np.nan
        vol_20 = returns.iloc[-20:].std() * np.sqrt(252) * 100 if len(returns) >= 20 else np.nan
        vol_60 = returns.iloc[-60:].std() * np.sqrt(252) * 100 if len(returns) >= 60 else np.nan
        
        # 5. 52주 고점 대비 현재 가격 비율 (가능한 데이터만큼만)
        if len(hist_to_date) >= 252:
            high_52w = hist_to_date['High'].iloc[-252:].max()
            ratio_52w_high = (current_price / high_52w) * 100
        else:
            # 52주 미만이면 가능한 기간의 최고점 사용
            high_period = hist_to_date['High'].max()
            ratio_52w_high = (current_price / high_period) * 100
        
        return {
            'momentum_5d': round(momentum_5d, 2) if not np.isnan(momentum_5d) else np.nan,
            'momentum_20d': round(momentum_20d, 2) if not np.isnan(momentum_20d) else np.nan,
            'momentum_60d': round(momentum_60d, 2) if not np.isnan(momentum_60d) else np.nan,
            'ma_dev_5d': round(ma_dev_5, 2) if not np.isnan(ma_dev_5) else np.nan,
            'ma_dev_20d': round(ma_dev_20, 2) if not np.isnan(ma_dev_20) else np.nan,
            'ma_dev_60d': round(ma_dev_60, 2) if not np.isnan(ma_dev_60) else np.nan,
            'vol_change_5d': round(vol_change_5, 2) if not np.isnan(vol_change_5) else np.nan,
            'vol_change_20d': round(vol_change_20, 2) if not np.isnan(vol_change_20) else np.nan,
            'vol_change_60d': round(vol_change_60, 2) if not np.isnan(vol_change_60) else np.nan,
            'volatility_5d': round(vol_5, 2) if not np.isnan(vol_5) else np.nan,
            'volatility_20d': round(vol_20, 2) if not np.isnan(vol_20) else np.nan,
            'volatility_60d': round(vol_60, 2) if not np.isnan(vol_60) else np.nan,
            'ratio_52w_high': round(ratio_52w_high, 2)
        }
        
    except Exception as e:
        print(f"Error calculating indicators for {symbol} on {target_date}: {e}")
        return None

def fetch_market_benchmark_data(start_date, end_date):
    """S&P 500 벤치마크 데이터 수집"""
    
    print("📈 Fetching market benchmark data (S&P 500)...")
    
    # S&P 500 ETF (SPY) 사용
    spy = yf.Ticker("SPY")
    
    # 여유있게 90일 더 이전부터 데이터 가져오기
    extended_start = start_date - timedelta(days=90)
    
    try:
        # 일일 데이터 가져오기
        spy_data = spy.history(start=extended_start, end=end_date + timedelta(days=5))
        
        if spy_data.empty:
            print("  ⚠️ Could not fetch SPY data")
            return None
            
        # 수익률 계산
        spy_data['daily_return'] = spy_data['Close'].pct_change()
        
        # 이동평균 수익률 계산
        spy_data['ma_return_5d'] = spy_data['daily_return'].rolling(5).mean()
        spy_data['ma_return_20d'] = spy_data['daily_return'].rolling(20).mean()
        
        # 누적 수익률 (기간별)
        spy_data['cum_return_5d'] = (spy_data['Close'] / spy_data['Close'].shift(5) - 1)
        spy_data['cum_return_20d'] = (spy_data['Close'] / spy_data['Close'].shift(20) - 1)
        
        # 변동성
        spy_data['volatility_20d'] = spy_data['daily_return'].rolling(20).std() * np.sqrt(252)
        
        print(f"  ✅ Market data fetched: {len(spy_data)} trading days")
        
        return spy_data
        
    except Exception as e:
        print(f"  ⚠️ Error fetching market data: {e}")
        return None

def enrich_episodes_with_indicators(episodes_file, output_file):
    """거래 에피소드에 기술적 지표 추가"""
    
    print("📊 Loading trading episodes...")
    episodes_df = pd.read_csv(episodes_file)
    episodes_df['entry_datetime'] = pd.to_datetime(episodes_df['entry_datetime'])
    episodes_df['exit_datetime'] = pd.to_datetime(episodes_df['exit_datetime'])
    
    print(f"  - Loaded {len(episodes_df)} episodes")
    
    # 1. 시장 벤치마크 데이터 가져오기
    min_date = episodes_df['entry_datetime'].min()
    max_date = episodes_df['exit_datetime'].max()
    market_data = fetch_market_benchmark_data(min_date, max_date)
    
    # 2. 필요한 심볼과 날짜 범위 추출
    symbols = episodes_df['symbol'].unique()
    print(f"\n📈 Unique symbols: {len(symbols)}")
    
    # 3. 각 심볼별로 필요한 전체 기간의 데이터 다운로드
    print("\n💾 Downloading historical data for all symbols...")
    symbol_data = {}
    
    for symbol in tqdm(symbols, desc="Downloading"):
        # 가장 이른 entry_date와 가장 늦은 exit_date 찾기
        symbol_episodes = episodes_df[episodes_df['symbol'] == symbol]
        min_date = symbol_episodes['entry_datetime'].min() - timedelta(days=400)  # 충분한 과거 데이터
        max_date = symbol_episodes['exit_datetime'].max() + timedelta(days=5)
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=min_date, end=max_date)
            
            if not hist.empty:
                symbol_data[symbol] = hist
            else:
                print(f"  ⚠️ No data for {symbol}")
        except Exception as e:
            print(f"  ⚠️ Error downloading {symbol}: {e}")
    
    # 4. 각 에피소드에 대해 entry/exit 시점의 지표 계산
    print("\n🔄 Calculating technical indicators for each episode...")
    
    # 새로운 컬럼들을 위한 리스트
    entry_indicators = []
    exit_indicators = []
    market_indicators = []
    
    for idx, row in tqdm(episodes_df.iterrows(), total=len(episodes_df), desc="Processing"):
        symbol = row['symbol']
        
        if symbol not in symbol_data:
            # 데이터가 없는 경우 NaN으로 채우기
            entry_indicators.append({})
            exit_indicators.append({})
            market_indicators.append({})
            continue
        
        hist = symbol_data[symbol]
        
        # 보유 기간 가져오기
        holding_days = row['holding_period_days']
        
        # Entry 시점 지표 (보유 기간 고려)
        entry_ind = calculate_indicators_with_holding_period(hist, row['entry_datetime'], symbol, holding_days)
        if entry_ind:
            # 'entry_' 접두사 추가
            entry_ind = {f'entry_{k}': v for k, v in entry_ind.items()}
        else:
            entry_ind = {}
        entry_indicators.append(entry_ind)
        
        # Exit 시점 지표 (보유 기간 고려)
        exit_ind = calculate_indicators_with_holding_period(hist, row['exit_datetime'], symbol, holding_days)
        if exit_ind:
            # 'exit_' 접두사 추가
            exit_ind = {f'exit_{k}': v for k, v in exit_ind.items()}
        else:
            exit_ind = {}
        exit_indicators.append(exit_ind)
        
        # 시장 벤치마크 지표 추가
        market_ind = {}
        if market_data is not None:
            entry_date = row['entry_datetime'].date()
            exit_date = row['exit_datetime'].date()
            
            # 진입 시점 시장 상황
            if entry_date in market_data.index.date:
                market_entry = market_data[market_data.index.date == entry_date].iloc[0]
                market_ind['market_entry_ma_return_5d'] = round(market_entry['ma_return_5d'] * 100, 2) if pd.notna(market_entry['ma_return_5d']) else 0
                market_ind['market_entry_ma_return_20d'] = round(market_entry['ma_return_20d'] * 100, 2) if pd.notna(market_entry['ma_return_20d']) else 0
                market_ind['market_entry_cum_return_5d'] = round(market_entry['cum_return_5d'] * 100, 2) if pd.notna(market_entry['cum_return_5d']) else 0
                market_ind['market_entry_cum_return_20d'] = round(market_entry['cum_return_20d'] * 100, 2) if pd.notna(market_entry['cum_return_20d']) else 0
                market_ind['market_entry_volatility_20d'] = round(market_entry['volatility_20d'] * 100, 2) if pd.notna(market_entry['volatility_20d']) else 0
            
            # 보유 기간 중 시장 수익률
            if entry_date in market_data.index.date and exit_date in market_data.index.date:
                entry_close = market_data[market_data.index.date == entry_date]['Close'].iloc[0]
                exit_close = market_data[market_data.index.date == exit_date]['Close'].iloc[0]
                market_return = (exit_close / entry_close - 1) * 100
                market_ind['market_return_during_holding'] = round(market_return, 2)
        
        market_indicators.append(market_ind)
    
    # 5. 지표를 DataFrame에 추가
    entry_df = pd.DataFrame(entry_indicators)
    exit_df = pd.DataFrame(exit_indicators)
    market_df = pd.DataFrame(market_indicators)
    
    # 원본 데이터와 결합
    enriched_df = pd.concat([episodes_df, entry_df, exit_df, market_df], axis=1)
    
    # 6. 초과 수익률 계산
    if 'market_return_during_holding' in enriched_df.columns:
        enriched_df['excess_return'] = enriched_df['return_pct'] - enriched_df['market_return_during_holding']
        print(f"\n📊 Market-adjusted performance calculated")
        print(f"  - Average market return: {enriched_df['market_return_during_holding'].mean():.2f}%")
        print(f"  - Average excess return: {enriched_df['excess_return'].mean():.2f}%")
    
    # 7. 추가 계산: 지표 변화량 (exit - entry)
    print("\n📊 Calculating indicator changes...")
    
    indicator_names = ['momentum_5d', 'momentum_20d', 'momentum_60d', 
                      'ma_dev_5d', 'ma_dev_20d', 'ma_dev_60d',
                      'vol_change_5d', 'vol_change_20d', 'vol_change_60d',
                      'volatility_5d', 'volatility_20d', 'volatility_60d',
                      'ratio_52w_high']
    
    for ind in indicator_names:
        entry_col = f'entry_{ind}'
        exit_col = f'exit_{ind}'
        change_col = f'change_{ind}'
        
        if entry_col in enriched_df.columns and exit_col in enriched_df.columns:
            enriched_df[change_col] = enriched_df[exit_col] - enriched_df[entry_col]
    
    # 8. 저장
    enriched_df.to_csv(output_file, index=False)
    
    # 9. 통계 출력
    print(f"\n✅ Enriched data saved to {output_file}")
    print(f"\n📊 Data Summary:")
    print(f"  - Total episodes: {len(enriched_df)}")
    print(f"  - Original columns: {len(episodes_df.columns)}")
    print(f"  - Enriched columns: {len(enriched_df.columns)}")
    print(f"  - New indicator columns: {len(enriched_df.columns) - len(episodes_df.columns)}")
    
    # 데이터 완전성 체크
    print(f"\n📊 Data Completeness:")
    for col in ['entry_momentum_20d', 'entry_volatility_20d', 'entry_ratio_52w_high', 'market_return_during_holding']:
        if col in enriched_df.columns:
            completeness = enriched_df[col].notna().sum() / len(enriched_df) * 100
            print(f"  - {col}: {completeness:.1f}% complete")
    
    return enriched_df

if __name__ == "__main__":
    enriched_df = enrich_episodes_with_indicators(
        '../results/final/industry_based_trading_episodes.csv',
        '../results/final/enriched_trading_episodes.csv'
    )