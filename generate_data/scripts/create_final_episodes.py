import pandas as pd
import numpy as np
import json
from datetime import datetime
import yfinance as yf

def create_comprehensive_trading_data(transaction_file, mapping_file, output_file='../results/final/final_trading_episodes.csv'):
    """최종 거래 데이터 생성"""
    
    # 1. 매핑 정보 로드
    print("📂 Loading mapping information...")
    with open(mapping_file, 'r') as f:
        isin_mapping = json.load(f)
    
    print(f"  - Loaded mappings for {len(isin_mapping)} ISINs")
    
    # 2. 거래 데이터 로드
    print("\n📊 Loading transaction data...")
    df = pd.read_csv(transaction_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['price_per_unit'] = df['totalValue'] / df['units']
    
    # 매핑된 ISIN만 필터링
    mapped_isins = set(isin_mapping.keys())
    df_mapped = df[df['ISIN'].isin(mapped_isins)]
    
    print(f"  - Total transactions: {len(df)}")
    print(f"  - Mapped transactions: {len(df_mapped)} ({len(df_mapped)/len(df)*100:.1f}%)")
    
    # 3. Buy-Sell 매칭
    print("\n🔄 Matching Buy-Sell transactions...")
    episodes = []
    
    # 고객별, ISIN별로 그룹화
    grouped = df_mapped.groupby(['customerID', 'ISIN'])
    
    for (customer_id, isin), group in grouped:
        us_ticker = isin_mapping[isin]['us_ticker']
        
        # Buy와 Sell 분리
        buys = group[group['transactionType'] == 'Buy'].sort_values('timestamp')
        sells = group[group['transactionType'] == 'Sell'].sort_values('timestamp')
        
        # FIFO 매칭
        buy_queue = []
        
        for _, buy in buys.iterrows():
            buy_queue.append({
                'timestamp': buy['timestamp'],
                'units': buy['units'],
                'price': buy['price_per_unit'],
                'total_value': buy['totalValue']
            })
        
        for _, sell in sells.iterrows():
            remaining_units = sell['units']
            sell_timestamp = sell['timestamp']
            sell_price = sell['price_per_unit']
            
            while remaining_units > 0 and buy_queue:
                # 매도 시점 이전의 매수만 선택
                eligible_buys = [b for b in buy_queue if b['timestamp'] < sell_timestamp]
                
                if not eligible_buys:
                    break
                
                buy = eligible_buys[0]
                buy_idx = buy_queue.index(buy)
                
                # 매칭할 수량 결정
                matched_units = min(buy['units'], remaining_units)
                
                # 보유 기간 계산
                holding_days = (sell_timestamp - buy['timestamp']).days
                
                # 유효한 거래만 (1일 ~ 1000일)
                if 1 <= holding_days <= 1000:
                    return_pct = (sell_price - buy['price']) / buy['price']
                    
                    # 포지션 크기 (거래 금액 기반으로 현실적으로 설정)
                    position_value = matched_units * buy['price']
                    if position_value < 1000:
                        position_size = 0.05
                    elif position_value < 5000:
                        position_size = 0.1
                    elif position_value < 10000:
                        position_size = 0.2
                    elif position_value < 20000:
                        position_size = 0.3
                    else:
                        position_size = 0.4
                    
                    episode = {
                        'entry_datetime': buy['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'exit_datetime': sell_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'entry_price': round(buy['price'], 2),
                        'exit_price': round(sell_price, 2),
                        'position_size_pct': position_size,
                        'return_pct': round(return_pct, 4),
                        'holding_period_days': holding_days,
                        'symbol': us_ticker,
                        'original_isin': isin
                    }
                    
                    episodes.append(episode)
                
                # 매칭된 수량 차감
                remaining_units -= matched_units
                buy['units'] -= matched_units
                
                if buy['units'] <= 0:
                    buy_queue.pop(buy_idx)
    
    print(f"  - Created {len(episodes)} trading episodes")
    
    # 4. 미국 주식의 실제 가격 사용
    print("\n💰 Getting actual US stock prices for each trade...")
    episodes_df = pd.DataFrame(episodes)
    
    # 심볼별로 그룹화하여 효율적으로 처리
    symbols = episodes_df['symbol'].unique()
    price_cache = {}
    
    # 각 심볼별로 필요한 날짜 범위의 실제 가격 데이터 가져오기
    for symbol in symbols:
        symbol_episodes = episodes_df[episodes_df['symbol'] == symbol]
        date_min = pd.to_datetime(symbol_episodes['entry_datetime'].min())
        date_max = pd.to_datetime(symbol_episodes['exit_datetime'].max())
        
        print(f"  Fetching {symbol} prices from {date_min.date()} to {date_max.date()}")
        
        try:
            # 실제 날짜 범위의 주가 데이터 가져오기
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=date_min - pd.Timedelta(days=5), 
                                end=date_max + pd.Timedelta(days=5))
            
            if not hist.empty:
                price_cache[symbol] = hist['Close']
            else:
                print(f"    ⚠️ No data for {symbol}")
        except Exception as e:
            print(f"    ⚠️ Error fetching {symbol}: {e}")
    
    # 각 거래에 대해 실제 날짜의 가격 사용
    valid_episodes = []
    skipped_no_price = 0
    skipped_no_data = 0
    
    for idx, episode in episodes_df.iterrows():
        symbol = episode['symbol']
        if symbol not in price_cache:
            skipped_no_data += 1
            continue
            
        if symbol in price_cache:
            prices = price_cache[symbol]
            entry_date = pd.to_datetime(episode['entry_datetime']).date()
            exit_date = pd.to_datetime(episode['exit_datetime']).date()
            
            # 해당 날짜 또는 가장 가까운 거래일의 가격 찾기
            entry_price = None
            exit_price = None
            
            # entry_date의 가격 찾기
            for i in range(5):  # 최대 5일까지 앞뒤로 검색
                try_date = entry_date + pd.Timedelta(days=i)
                if try_date in prices.index.date:
                    entry_price = prices[prices.index.date == try_date].iloc[0]
                    break
                try_date = entry_date - pd.Timedelta(days=i)
                if try_date in prices.index.date:
                    entry_price = prices[prices.index.date == try_date].iloc[0]
                    break
            
            # exit_date의 가격 찾기
            for i in range(5):
                try_date = exit_date + pd.Timedelta(days=i)
                if try_date in prices.index.date:
                    exit_price = prices[prices.index.date == try_date].iloc[0]
                    break
                try_date = exit_date - pd.Timedelta(days=i)
                if try_date in prices.index.date:
                    exit_price = prices[prices.index.date == try_date].iloc[0]
                    break
            
            # 실제 가격이 모두 있는 경우만 사용
            if entry_price is not None and exit_price is not None:
                # 실제 수익률 계산
                actual_return = (exit_price - entry_price) / entry_price
                
                valid_episode = episode.copy()
                valid_episode['entry_price'] = round(entry_price, 2)
                valid_episode['exit_price'] = round(exit_price, 2)
                valid_episode['return_pct'] = round(actual_return, 4)
                
                valid_episodes.append(valid_episode)
            else:
                skipped_no_price += 1
    
    print(f"\n📊 Episode filtering summary:")
    print(f"  - Total input episodes: {len(episodes_df)}")
    print(f"  - Skipped {skipped_no_data} episodes (no data for symbol)")
    print(f"  - Skipped {skipped_no_price} episodes (no price on specific dates)")
    print(f"  - Valid episodes: {len(valid_episodes)}")
    
    episodes_df = pd.DataFrame(valid_episodes)
    
    # 5. 최종 데이터 정리 및 저장
    final_df = episodes_df[['entry_datetime', 'exit_datetime', 'entry_price', 
                           'exit_price', 'position_size_pct', 'return_pct', 
                           'holding_period_days', 'symbol']]
    
    final_df = final_df.sort_values('entry_datetime').reset_index(drop=True)
    
    # CSV 저장
    final_df.to_csv(output_file, index=False)
    
    # 6. 통계 출력
    print(f"\n📊 Final Trading Data Summary:")
    print(f"Total episodes: {len(final_df)}")
    print(f"Unique symbols: {final_df['symbol'].nunique()}")
    print(f"Date range: {final_df['entry_datetime'].min()} to {final_df['exit_datetime'].max()}")
    print(f"\nPerformance metrics:")
    print(f"  - Average return: {final_df['return_pct'].mean():.2%}")
    print(f"  - Win rate: {(final_df['return_pct'] > 0).mean():.2%}")
    print(f"  - Average holding: {final_df['holding_period_days'].mean():.1f} days")
    print(f"  - Median holding: {final_df['holding_period_days'].median():.0f} days")
    
    # 심볼별 분포
    print(f"\n📈 Top 10 symbols by trade count:")
    symbol_counts = final_df['symbol'].value_counts().head(10)
    for symbol, count in symbol_counts.items():
        avg_ret = final_df[final_df['symbol'] == symbol]['return_pct'].mean()
        print(f"  {symbol}: {count} trades (avg return: {avg_ret:.2%})")
    
    print(f"\n✅ Saved {len(final_df)} episodes to {output_file}")
    
    return final_df

if __name__ == "__main__":
    df = create_comprehensive_trading_data(
        '/Users/inter4259/Desktop/transactions.csv',
        '../results/mappings/unique_isin_mapping.json',
        '../results/final/final_trading_episodes.csv'
    )