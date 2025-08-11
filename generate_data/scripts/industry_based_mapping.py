import pandas as pd
import numpy as np
import json
import yfinance as yf
from datetime import datetime

def load_all_data():
    """모든 필요한 데이터 로드"""
    print("📊 Loading all data...")
    
    # 1. 상세 회사 정보 (Industry 포함)
    detailed_info = pd.read_csv('/Users/inter4259/Desktop/greek_companies_detailed.csv')
    print(f"  - Loaded {len(detailed_info)} detailed company records")
    
    # 2. 가격 데이터
    prices = pd.read_csv('/Users/inter4259/Downloads/FAR-Trans/close_prices.csv')
    prices['timestamp'] = pd.to_datetime(prices['timestamp'])
    print(f"  - Loaded {len(prices)} price records")
    
    # 3. 거래 데이터
    transactions = pd.read_csv('/Users/inter4259/Desktop/transactions.csv')
    print(f"  - Loaded {len(transactions)} transactions")
    
    return detailed_info, prices, transactions

def get_industry_based_pools():
    """Industry 기준 미국 주식 풀"""
    return {
        # Energy 섹터 추가 (Engery 오타 대응)
        'Energy': [
            'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'EOG', 'PSX', 'MPC', 'VLO', 'PXD',
            'DVN', 'HES', 'FANG', 'CTRA', 'MRO', 'APA', 'HAL', 'BKR', 'SU', 'CNQ',
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'ED', 'XEL'
        ],
        
        # 은행업
        'Banking': [
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'BK',
            'COF', 'STT', 'NTRS', 'FITB', 'RF', 'KEY', 'HBAN', 'MTB', 'CFG', 'ALLY',
            'ZION', 'CMA', 'PBCT', 'FHN', 'WAL', 'SNV', 'EWBC', 'PACW', 'WBS'
        ],
        
        # 중앙은행 (특수) - ETF로 대체
        'Central Bank': [
            'GOVT', 'SHY', 'IEF', 'TLT', 'AGG', 'BND'
        ],
        
        # 금융 서비스업
        'Financial Services': [
            'BLK', 'SCHW', 'SPGI', 'CME', 'ICE', 'MSCI', 'NDAQ', 'CBOE', 'MKTX', 'MCO',
            'AMP', 'BEN', 'IVZ', 'TROW', 'SEIC', 'AMG', 'JEF', 'EVR', 'LAZ', 'PJT'
        ],
        
        # 금융 지주회사
        'Financial Holding Company': [
            'BRK-B', 'BRK-A', 'BAM', 'BN', 'L', 'Y', 'BX', 'KKR', 'APO', 'ARES',
            'CG', 'OWL', 'HLI', 'MAIN', 'ARCC'
        ],
        
        # 석유정제업
        'Oil Refining': [
            'XOM', 'CVX', 'PSX', 'MPC', 'VLO', 'HFC', 'PBF', 'CVI', 'DK', 'DINO',
            'PARR', 'ALJ', 'CLMT'
        ],
        
        # 전력업
        'Power Generation': [
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'ED', 'XEL',
            'ES', 'ETR', 'EIX', 'DTE', 'CNP', 'CMS', 'WEC', 'AES', 'PPL', 'FE'
        ],
        
        # 전력 송전업
        'Power Transmission': [
            'AEP', 'ETR', 'ITC', 'OTTR', 'EIX', 'SO', 'DUK', 'XEL'
        ],
        
        # 신재생에너지
        'Renewable Energy': [
            'NEE', 'AES', 'BEP', 'NOVA', 'RUN', 'ENPH', 'SEDG', 'FSLR', 'SPWR', 'CSIQ',
            'DQ', 'JKS', 'PLUG', 'BE', 'CWEN', 'AY', 'ORA', 'SHLS', 'HASI', 'EVA'
        ],
        
        # 상하수도업
        'Water & Sewage': [
            'AWK', 'AWR', 'CWT', 'WTR', 'SJW', 'MSEX', 'YORW', 'ARTNA', 'GWRS', 'CDZI'
        ],
        
        # 통신업
        'Telecommunications': [
            'T', 'VZ', 'TMUS', 'CMCSA', 'CHTR', 'LUMN', 'FYBR', 'CABO', 'ATNI', 'USM',
            'TDS', 'SHEN', 'CNSL', 'LBRDA', 'LBRDK', 'WOW', 'ATUS'
        ],
        
        # 항공업
        'Airlines': [
            'LUV', 'DAL', 'UAL', 'AAL', 'ALK', 'JBLU', 'SAVE', 'HA', 'ULCC', 'MESA',
            'SKYW', 'ALGT', 'CPA', 'RJET', 'ZNH'
        ],
        
        # 건설업
        'Construction': [
            'CAT', 'DE', 'VMC', 'MLM', 'EME', 'FLR', 'PWR', 'MTZ', 'GVA', 'DY',
            'ACM', 'AAON', 'AZZ', 'CENX', 'CX', 'GBX', 'HAYN', 'HEES', 'IEX', 'IIIN'
        ],
        
        # 건설·부동산업
        'Construction & Real Estate': [
            'CBRE', 'JLL', 'CWK', 'MMI', 'KBH', 'LEN', 'DHI', 'PHM', 'TOL', 'MDC',
            'CCS', 'GRBK', 'MHO', 'TMHC', 'TPH', 'CLC', 'MTH', 'LGIH', 'CTO', 'FOR'
        ],
        
        # 산업·에너지 복합기업
        'Industrial & Energy Conglomerate': [
            'GE', 'HON', 'MMM', 'DHR', 'ITW', 'ETN', 'EMR', 'ROK', 'ROP', 'AME',
            'DOV', 'PH', 'XYL', 'IR', 'TT', 'CMI', 'PNR', 'GWW', 'FAST', 'AOS'
        ],
        
        # 부동산 개발업
        'Real Estate Development': [
            'SPG', 'PLD', 'PSA', 'EQIX', 'DLR', 'O', 'WELL', 'AVB', 'EQR', 'ESS',
            'MAA', 'UDR', 'CPT', 'AIV', 'EXR', 'CUBE', 'FR', 'BXP', 'VNO', 'SLG',
            'KIM', 'REG', 'FRT', 'MAC', 'SKT', 'WPC', 'NNN', 'STOR', 'STAG', 'REXR'
        ],
        
        # 증권거래소
        'Stock Exchange': [
            'CME', 'ICE', 'NDAQ', 'CBOE', 'MKTX', 'IBKR', 'VIRT', 'BGCP', 'OPY'
        ],
        
        # 정보통신기술
        'Information & Communication Technology': [
            'CSCO', 'ANET', 'JNPR', 'FFIV', 'CIEN', 'INFN', 'LITE', 'COMM', 'VIAV', 'EXTR',
            'CALX', 'DZSI', 'RBBN', 'CLFD', 'PDFS', 'CAMP', 'RESN', 'ALLT', 'WSTG', 'DT'
        ],
        
        # 소매유통업
        'Retail': [
            'WMT', 'TGT', 'COST', 'HD', 'LOW', 'CVS', 'WBA', 'KR', 'ACI', 'SFM',
            'TJX', 'ROST', 'DG', 'DLTR', 'FIVE', 'BURL', 'PSMT', 'OLLI', 'BIG', 'BJ'
        ],
        
        # 완구·잡화 소매업
        'Toy & Household Retail': [
            'TGT', 'WMT', 'DG', 'DLTR', 'FIVE', 'BBY', 'BIG', 'OLLI', 'BURL', 'TJX',
            'BBWI', 'FL', 'DKS', 'ASO', 'BGFV', 'HIBB', 'SPWH', 'GCO', 'SCVL', 'ZUMZ'
        ],
        
        # 복권·게임업
        'Lottery & Gaming': [
            'DKNG', 'PENN', 'MGM', 'WYNN', 'LVS', 'CZR', 'BYD', 'GDEN', 'RSI', 'IGT',
            'SGMS', 'EVRI', 'AGS', 'ACHR', 'GMBL', 'BALY', 'MCRI', 'CNTY', 'FLL', 'GNOG'
        ],
        
        # 복권·베팅업 (동일)
        'Lottery & Betting': [
            'DKNG', 'PENN', 'MGM', 'WYNN', 'LVS', 'CZR', 'BYD', 'GDEN', 'RSI', 'IGT'
        ],
        
        # 플라스틱 제조업
        'Plastic Manufacturing': [
            'DOW', 'LYB', 'WLK', 'CC', 'AVNT', 'TREX', 'SOLV', 'HWKN', 'KOP', 'DNMR',
            'EVA', 'POL', 'CBT', 'CYH', 'PCT', 'ASPN', 'KRNT', 'TSE', 'OLN', 'HCC'
        ],
        
        # 금속 제조업
        'Metal Manufacturing': [
            'FCX', 'NEM', 'AA', 'CLF', 'X', 'NUE', 'STLD', 'RS', 'ATI', 'CMC',
            'SCHN', 'WOR', 'CRS', 'ZEUS', 'HAYN', 'CENX', 'KALU', 'ARNC', 'TG', 'MP',
            'VALE', 'RIO', 'BHP', 'TMST', 'TX', 'SID', 'GGB', 'CMP', 'USAP', 'SYNL'
        ],
        
        # 음료 제조업
        'Beverage Manufacturing': [
            'KO', 'PEP', 'MNST', 'KDP', 'STZ', 'TAP', 'SAM', 'BUD', 'FIZZ', 'CELH',
            'COKE', 'PRMW', 'NAPA', 'ZVIA', 'STKL', 'SHOT', 'LNDC', 'BREW', 'WEST', 'CCEP'
        ],
        
        # 투자 지주회사
        'Investment Holding Company': [
            'BRK-B', 'BRK-A', 'BKNG', 'MKL', 'WRB', 'RE', 'KNSL', 'BDT', 'LFCF', 'Y',
            'L', 'PSHG', 'STNG', 'SACH', 'GAIN', 'GLAD', 'NMFC', 'TCPC', 'OXSQ', 'GBDC'
        ],
        
        # 투자 펀드
        'Investment Fund': [
            'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'IVV', 'VEA', 'VWO', 'EFA',
            'AGG', 'BND', 'LQD', 'HYG', 'EMB', 'TLT', 'IEF', 'SHY', 'TIP', 'MUB',
            'GLD', 'SLV', 'USO', 'UNG', 'DBO', 'PDBC', 'DJP', 'RJI', 'GSG', 'DBC'
        ]
    }

def analyze_greek_stocks(detailed_info, prices):
    """그리스 주식 변동성 분석"""
    print("\n📈 Analyzing Greek stocks...")
    
    greek_profiles = {}
    
    for _, company in detailed_info.iterrows():
        isin = company['ISIN_Code']
        
        # 가격 데이터 찾기
        isin_prices = prices[prices['ISIN'] == isin].copy()
        
        if len(isin_prices) < 50:
            continue
            
        isin_prices = isin_prices.sort_values('timestamp')
        
        # 로그 수익률 계산
        isin_prices['returns'] = np.log(isin_prices['closePrice'] / isin_prices['closePrice'].shift(1))
        returns = isin_prices['returns'].dropna()
        
        # 변동성 계산 (연간화)
        annual_vol = returns.std() * np.sqrt(252) * 100
        annual_return = returns.mean() * 252 * 100
        
        greek_profiles[isin] = {
            'ticker': company['Ticker'],
            'english_name': company['English_Name'],
            'industry': company['Industry'],
            'volatility': round(annual_vol, 2),
            'return': round(annual_return, 2),
            'n_prices': len(isin_prices),
            'date_min': isin_prices['timestamp'].min(),
            'date_max': isin_prices['timestamp'].max()
        }
        
        print(f"  {isin} ({company['Ticker']}) - {company['Industry']} - Vol: {annual_vol:.1f}%")
    
    return greek_profiles

def analyze_us_stocks_by_industry(industry_pools, start_date, end_date):
    """Industry별 미국 주식 분석"""
    print(f"\n📈 Analyzing US stocks by industry...")
    print(f"  - Date range: {start_date} to {end_date}")
    
    us_profiles = {}
    
    # 모든 티커 수집
    all_tickers = set()
    for industry_tickers in industry_pools.values():
        all_tickers.update(industry_tickers)
    
    all_tickers = list(all_tickers)
    print(f"  - Total unique tickers: {len(all_tickers)}")
    
    # 배치 다운로드
    data = yf.download(all_tickers, start=start_date, end=end_date, group_by='ticker', progress=True)
    
    for ticker in all_tickers:
        try:
            if len(data.columns.levels) > 0 and ticker in data.columns.levels[0]:
                ticker_data = data[ticker]['Close'].dropna()
            else:
                ticker_data = data['Close'].dropna() if len(all_tickers) == 1 else pd.Series()
                
            if len(ticker_data) > 50:
                log_returns = np.log(ticker_data / ticker_data.shift(1)).dropna()
                
                # 해당 티커가 속한 industry들 찾기
                ticker_industries = []
                for industry, tickers in industry_pools.items():
                    if ticker in tickers:
                        ticker_industries.append(industry)
                
                us_profiles[ticker] = {
                    'industries': ticker_industries,
                    'volatility': round(log_returns.std() * np.sqrt(252) * 100, 2),
                    'return': round(log_returns.mean() * 252 * 100, 2),
                    'price_mean': round(ticker_data.mean(), 2),
                    'n_days': len(ticker_data)
                }
        except:
            pass
    
    return us_profiles

def create_industry_based_mapping(greek_profiles, us_profiles, industry_pools):
    """Industry 기준 매핑"""
    print("\n🔗 Creating industry-based mappings...")
    
    # Industry 이름 정정 매핑
    industry_corrections = {
        'Engery': 'Energy',
        'Energy': 'Energy',  # 에너지 관련
        'Oil': 'Oil Refining',
        'Aviation': 'Airlines',
        'Lottery': 'Lottery & Gaming',
        'ICT': 'Information & Communication Technology',
        'Plastics Manufacturing': 'Plastic Manufacturing'
    }
    
    mappings = {}
    used_tickers = set()
    
    for isin, greek_props in greek_profiles.items():
        industry = greek_props['industry']
        # Industry 이름 정정
        corrected_industry = industry_corrections.get(industry, industry)
        
        print(f"\n{isin} ({greek_props['ticker']}) - {greek_props['english_name']}")
        print(f"  Industry: {industry} → {corrected_industry}")
        print(f"  Greek volatility: {greek_props['volatility']}%")
        
        # 해당 industry의 미국 주식 풀에서 매칭
        industry_candidates = []
        
        if corrected_industry in industry_pools:
            for ticker in industry_pools[corrected_industry]:
                if ticker in us_profiles and ticker not in used_tickers:
                    us_data = us_profiles[ticker]
                    if corrected_industry in us_data['industries']:
                        industry_candidates.append((ticker, us_data))
        
        if not industry_candidates:
            print(f"  ⚠️ No candidates found in {corrected_industry}")
            continue
        
        # 변동성 기준으로 최적 매칭 찾기
        best_match = None
        min_vol_diff = float('inf')
        
        for ticker, us_props in industry_candidates:
            vol_diff = abs(greek_props['volatility'] - us_props['volatility'])
            
            if vol_diff < min_vol_diff:
                min_vol_diff = vol_diff
                best_match = ticker
        
        if best_match:
            mappings[isin] = {
                'us_ticker': best_match,
                'greek_ticker': greek_props['ticker'],
                'english_name': greek_props['english_name'],
                'industry': corrected_industry,
                'greek_vol': greek_props['volatility'],
                'us_vol': us_profiles[best_match]['volatility'],
                'vol_diff': round(min_vol_diff, 2),
                'greek_return': greek_props['return'],
                'us_return': us_profiles[best_match]['return']
            }
            used_tickers.add(best_match)
            
            print(f"  → Matched to {best_match} (US vol: {us_profiles[best_match]['volatility']}%, diff: {min_vol_diff:.2f}%)")
    
    return mappings

def main():
    # 1. 모든 데이터 로드
    detailed_info, prices, transactions = load_all_data()
    
    # 2. 날짜 범위 확인
    date_min = prices['timestamp'].min()
    date_max = prices['timestamp'].max()
    print(f"\n📅 Data date range: {date_min.date()} to {date_max.date()}")
    
    # 3. 그리스 주식 분석 (detailed_info에 있는 회사만)
    greek_profiles = analyze_greek_stocks(detailed_info, prices)
    print(f"\n✅ Analyzed {len(greek_profiles)} Greek companies")
    
    # 4. Industry 기준 미국 주식 풀
    industry_pools = get_industry_based_pools()
    
    # 5. 미국 주식 분석
    us_profiles = analyze_us_stocks_by_industry(
        industry_pools,
        start_date=date_min.strftime('%Y-%m-%d'),
        end_date=date_max.strftime('%Y-%m-%d')
    )
    
    # 6. Industry 기준 매핑
    mappings = create_industry_based_mapping(greek_profiles, us_profiles, industry_pools)
    
    # 7. 결과 저장
    with open('../results/mappings/industry_based_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)
    
    # 8. 결과 요약
    print(f"\n✅ Created mappings for {len(mappings)} companies")
    
    if mappings:
        vol_diffs = [m['vol_diff'] for m in mappings.values()]
        print(f"\n📊 Mapping Quality:")
        print(f"  - Average volatility difference: {np.mean(vol_diffs):.2f}%")
        print(f"  - Max volatility difference: {max(vol_diffs):.2f}%")
        print(f"  - Min volatility difference: {min(vol_diffs):.2f}%")
        
        # Industry별 통계
        industry_counts = {}
        for mapping in mappings.values():
            industry = mapping['industry']
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        
        print(f"\n📊 Mappings by industry:")
        for industry, count in sorted(industry_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {industry}: {count}")
        
        # 주요 매핑 예시
        print(f"\n📌 Key mappings:")
        key_tickers = ['ΜΥΤΙΛ', 'ΑΡΑΙΓ', 'ΟΤΕ', 'ΔΕΗ', 'ΕΤΕ']
        for mapping in mappings.values():
            if mapping['greek_ticker'] in key_tickers:
                print(f"  {mapping['greek_ticker']} ({mapping['english_name']}) → {mapping['us_ticker']} ({mapping['industry']}, vol diff: {mapping['vol_diff']}%)")
    
    return mappings

if __name__ == "__main__":
    mappings = main()