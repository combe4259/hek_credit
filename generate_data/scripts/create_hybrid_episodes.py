import pandas as pd
import numpy as np
import json
from datetime import datetime
import yfinance as yf

def create_hybrid_trading_data(transaction_file, mapping_file,
                               customer_info_file='/Users/inter4259/Downloads/FAR-Trans/customer_information.csv',
                               output_file='../results/final/hybrid_trading_episodes.csv'):
    """하이브리드 매핑과 실제 가격 데이터를 사용한 거래 데이터 생성"""
    
    # 1. 매핑 정보 로드
    print("📂 Loading mapping information...")
    with open(mapping_file, 'r') as f:
        isin_mapping = json.load(f)
    
    print(f"  - Loaded mappings for {len(isin_mapping)} ISINs")
    
    # 2. 고객 투자 능력 데이터 로드
    print("\n💰 Loading customer investment capacity data...")
    customer_df = pd.read_csv(customer_info_file)
    print(f"  - Customer data shape: {customer_df.shape}")
    print(f"  - Columns: {list(customer_df.columns)}")
    
    # EUR to USD 환율 (대략적인 평균값 사용)
    EUR_TO_USD = 1.1
    
    # 투자 능력 매핑 (EUR → USD 변환)
    capacity_mapping_eur = {
        'CAP_LT30K': 20000,  # Less than 30K EUR
        'Predicted_CAP_LT30K': 20000,
        'CAP_30K_80K': 55000,  # 30K-80K EUR 중간값
        'Predicted_CAP_30K_80K': 55000,
        'CAP_80K_300K': 190000,  # 80K-300K EUR 중간값
        'Predicted_CAP_80K_300K': 190000,
        'CAP_GT300K': 500000,  # Greater than 300K EUR
        'Predicted_CAP_GT300K': 500000,
        'Not_Available': 100000  # 기본값
    }
    
    # USD로 변환된 투자 능력 매핑
    capacity_mapping_usd = {k: v * EUR_TO_USD for k, v in capacity_mapping_eur.items()}
    
    # 고객별 투자 능력 딕셔너리 생성 (USD 기준, 최신 정보 사용)
    customer_capacity_usd = {}
    # timestamp로 정렬하여 최신 정보 사용
    customer_df_sorted = customer_df.sort_values('timestamp')
    for _, row in customer_df_sorted.iterrows():
        customer_id = row['customerID']
        capacity_type = row['investmentCapacity']
        customer_capacity_usd[customer_id] = capacity_mapping_usd.get(capacity_type, 110000)  # 기본값도 USD로
    
    print(f"  - Loaded investment capacity for {len(customer_capacity_usd)} customers")
    
    # 투자 능력 분포 출력
    capacity_dist = customer_df['investmentCapacity'].value_counts()
    print("\n  Investment capacity distribution (USD):")
    for cap_type, count in capacity_dist.items():
        if cap_type in capacity_mapping_usd:
            print(f"    {cap_type}: {count} customers (${capacity_mapping_usd[cap_type]:,.0f} USD)")
    
    # 3. 거래 데이터 로드
    print("\n📊 Loading transaction data...")
    df = pd.read_csv(transaction_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['price_per_unit'] = df['totalValue'] / df['units']
    
    # 매핑된 ISIN만 필터링
    mapped_isins = set(isin_mapping.keys())
    df_mapped = df[df['ISIN'].isin(mapped_isins)]
    
    print(f"  - Total transactions: {len(df)}")
    print(f"  - Mapped transactions: {len(df_mapped)} ({len(df_mapped)/len(df)*100:.1f}%)")
    
    # 4. Buy-Sell 매칭
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
                    # 임시 포지션 크기 (나중에 USD 가격으로 재계산)
                    position_size = 0.0  # 일단 0으로 설정
                    
                    episode = {
                        'entry_datetime': buy['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'exit_datetime': sell_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'position_size_pct': position_size,
                        'holding_period_days': holding_days,
                        'symbol': us_ticker,
                        'original_isin': isin,
                        'greek_ticker': isin_mapping[isin].get('greek_ticker', ''),
                        'greek_name': isin_mapping[isin].get('english_name', isin_mapping[isin].get('greek_name', '')),
                        'industry': isin_mapping[isin].get('industry', isin_mapping[isin].get('sector', '')),
                        'units': matched_units,  # 거래 수량 저장
                        'customer_id': customer_id  # 고객 ID 저장
                    }
                    
                    episodes.append(episode)
                
                # 매칭된 수량 차감
                remaining_units -= matched_units
                buy['units'] -= matched_units
                
                if buy['units'] <= 0:
                    buy_queue.pop(buy_idx)
    
    print(f"  - Created {len(episodes)} trading episodes")
    
    # 5. 미국 주식의 실제 가격 사용
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
            
        prices = price_cache[symbol]
        entry_date = pd.to_datetime(episode['entry_datetime']).date()
        exit_date = pd.to_datetime(episode['exit_datetime']).date()
        
        # 해당 날짜 또는 가장 가까운 거래일의 가격 찾기
        entry_price = None
        exit_price = None
        
        # entry_date의 가격 찾기
        for i in range(5):
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
            
            # USD 기준 포지션 크기 계산
            position_value_usd = episode['units'] * entry_price  # USD 기준
            customer_id = episode['customer_id']
            
            # 고객의 USD 투자 능력 가져오기
            if customer_id in customer_capacity_usd:
                customer_total_usd = customer_capacity_usd[customer_id]
            else:
                customer_total_usd = 110000  # 기본값 (100,000 EUR * 1.1)
            
            # 실제 포트폴리오 비중 계산 (USD 기준)
            position_size_pct = min(position_value_usd / customer_total_usd, 1.0)
            
            # 너무 작은 포지션은 최소값 설정
            if position_size_pct < 0.001:
                position_size_pct = 0.001
            
            valid_episode = episode.copy()
            valid_episode['entry_price'] = round(entry_price, 2)
            valid_episode['exit_price'] = round(exit_price, 2)
            valid_episode['return_pct'] = round(actual_return, 4)
            valid_episode['position_size_pct'] = round(position_size_pct, 4)  # USD 기준으로 재계산된 값
            
            # customer_id와 units는 최종 출력에서 제외하므로 삭제
            del valid_episode['customer_id']
            del valid_episode['units']
            
            valid_episodes.append(valid_episode)
        else:
            skipped_no_price += 1
    
    print(f"\n📊 Episode filtering summary:")
    print(f"  - Total input episodes: {len(episodes_df)}")
    print(f"  - Skipped {skipped_no_data} episodes (no data for symbol)")
    print(f"  - Skipped {skipped_no_price} episodes (no price on specific dates)")
    print(f"  - Valid episodes: {len(valid_episodes)}")
    
    episodes_df = pd.DataFrame(valid_episodes)
    
    # 6. 최종 데이터 정리 및 저장
    final_df = episodes_df[['entry_datetime', 'exit_datetime', 'entry_price', 
                           'exit_price', 'position_size_pct', 'return_pct', 
                           'holding_period_days', 'symbol', 'greek_ticker', 'greek_name', 'industry']]
    
    final_df = final_df.sort_values('entry_datetime').reset_index(drop=True)
    
    # CSV 저장
    final_df.to_csv(output_file, index=False)
    
    # 7. 통계 출력
    print(f"\n📊 Final Trading Data Summary:")
    print(f"Total episodes: {len(final_df)}")
    print(f"Unique symbols: {final_df['symbol'].nunique()}")
    print(f"Date range: {final_df['entry_datetime'].min()} to {final_df['exit_datetime'].max()}")
    
    print(f"\nPerformance metrics:")
    print(f"  - Average return: {final_df['return_pct'].mean():.2%}")
    print(f"  - Win rate: {(final_df['return_pct'] > 0).mean():.2%}")
    print(f"  - Average holding: {final_df['holding_period_days'].mean():.1f} days")
    print(f"  - Median holding: {final_df['holding_period_days'].median():.0f} days")
    
    print(f"\nPosition size metrics:")
    print(f"  - Average position size: {final_df['position_size_pct'].mean():.1%} of portfolio")
    print(f"  - Median position size: {final_df['position_size_pct'].median():.1%} of portfolio")
    print(f"  - Max position size: {final_df['position_size_pct'].max():.1%} of portfolio")
    print(f"  - Min position size: {final_df['position_size_pct'].min():.1%} of portfolio")
    
    # Industry별 통계
    print(f"\n📊 By Industry:")
    industry_stats = final_df.groupby('industry').agg({
        'return_pct': ['count', 'mean'],
        'symbol': 'nunique'
    }).round(4)
    print(industry_stats)
    
    # 심볼별 통계
    print(f"\n📈 Top 10 symbols by trade count:")
    symbol_stats = final_df.groupby('symbol').agg({
        'return_pct': ['count', 'mean'],
        'greek_name': 'first'
    })
    symbol_stats.columns = ['count', 'avg_return', 'greek_name']
    top_symbols = symbol_stats.sort_values('count', ascending=False).head(10)
    
    for symbol, row in top_symbols.iterrows():
        print(f"  {symbol} ({row['greek_name']}): {row['count']} trades (avg return: {row['avg_return']:.2%})")
    
    print(f"\n✅ Saved {len(final_df)} episodes to {output_file}")
    
    return final_df

if __name__ == "__main__":
    df = create_hybrid_trading_data(
        '/Users/inter4259/Desktop/transactions.csv',
        '../results/mappings/industry_based_mapping.json',
        '/Users/inter4259/Downloads/FAR-Trans/customer_information.csv',
        '../results/final/industry_based_trading_episodes.csv'
    )