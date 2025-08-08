import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import time

def add_future_returns(
    input_csv: str = "news_combined_features.csv",
    output_csv: str = "news_with_future_returns.csv",
    return_days: int = 3  # 3거래일 후 수익률
):
    """
    뉴스 데이터에 미래 수익률 컬럼 추가하고, 학습에 필요한 컬럼만 최종 저장
    """

    # 티커 매핑 (BRK → BRK-B, BF → BF-B)
    ticker_mapping = {
        'BRK': 'BRK-B',  # 버크셔 해서웨이 클래스B
        'BF': 'BF-B'     # 브라운 포만 클래스B
    }

    def map_ticker(ticker: str) -> str:
        """티커 매핑 함수"""
        return ticker_mapping.get(ticker, ticker)

    print(f"📊 뉴스 데이터에 {return_days}일 후 수익률 추가 중...")

    # 1. 데이터 로드
    df = pd.read_csv(input_csv)
    original_columns = df.columns.tolist() # 원본 컬럼 저장
    print(f"✅ {len(df)}개 뉴스 로드")

    # 2. 날짜 컬럼 확인 및 변환
    date_col = 'news_date' if 'news_date' in df.columns else 'date'
    df[date_col] = pd.to_datetime(df[date_col])

    # 3. 결과 저장용 리스트
    results = []

    # 4. 티커별로 처리 (API 호출 최적화)
    unique_tickers = df['ticker'].unique()
    ticker_data_cache = {}

    print(f"📈 {len(unique_tickers)}개 티커의 주가 데이터 수집 중...")

    for ticker in tqdm(unique_tickers, desc="티커별 데이터 수집"):
        try:
            mapped_ticker = map_ticker(ticker)
            ticker_df = df[df['ticker'] == ticker]
            min_date = ticker_df[date_col].min()
            max_date = ticker_df[date_col].max()
            
            start_date = min_date - timedelta(days=5)
            end_date = max_date + timedelta(days=return_days + 15) # 여유 기간 증가
            
            stock = yf.Ticker(mapped_ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if len(hist) < 10:
                print(f"⚠️ {ticker}: 주가 데이터 부족")
                continue
                
            ticker_data_cache[ticker] = hist
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ {ticker} 데이터 수집 실패: {e}")
            continue
    
    print(f"✅ {len(ticker_data_cache)}개 티커 주가 데이터 수집 완료")

    # 5. 각 뉴스별로 수익률 계산
    print("📊 뉴스별 미래 수익률 계산 중...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="수익률 계산"):
        ticker = row['ticker']
        news_date = row[date_col]
        result = row.to_dict()
        
        # 기본값 설정
        result['news_day_close'] = np.nan
        result[f'future_{return_days}d_close'] = np.nan
        result[f'future_return_{return_days}d'] = np.nan
        result['return_calculation_status'] = 'no_stock_data'

        if ticker in ticker_data_cache:
            hist = ticker_data_cache[ticker]
            try:
                # 필터링을 통해 검색 범위 축소
                relevant_hist = hist[hist.index >= news_date.tz_localize(hist.index.tz)]
                if not relevant_hist.empty:
                    # 뉴스 당일 또는 그 이후의 첫 거래일 및 종가
                    news_day_data = relevant_hist.iloc[0]
                    news_day_close = news_day_data['Close']
                    
                    # return_days 거래일 후의 데이터 확인
                    if len(relevant_hist) > return_days:
                        future_day_data = relevant_hist.iloc[return_days]
                        future_close = future_day_data['Close']

                        if news_day_close > 0:
                            future_return = (future_close - news_day_close) / news_day_close
                            result.update({
                                'news_day_close': news_day_close,
                                f'future_{return_days}d_close': future_close,
                                f'future_return_{return_days}d': future_return,
                                'return_calculation_status': 'success'
                            })
                        else:
                             result['return_calculation_status'] = 'zero_price_error'
                    else:
                        result['return_calculation_status'] = 'future_day_not_found'
                else:
                    result['return_calculation_status'] = 'news_day_not_found'
            except Exception as e:
                result['return_calculation_status'] = f'error: {str(e)}'

        results.append(result)

    # 6. 결과 데이터프레임 생성
    result_df = pd.DataFrame(results)

    # 7. 최종 저장할 컬럼 선택 <--- ✨수정된 핵심 부분✨
    final_columns = original_columns + [f'future_return_{return_days}d']
    # 혹시 모를 오류를 방지하기 위해 실제 존재하는 컬럼만 한 번 더 확인
    final_columns_to_keep = [col for col in final_columns if col in result_df.columns]
    final_save_df = result_df[final_columns_to_keep]

    # 8. 최종 결과 저장
    final_save_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # 9. 결과 요약 (요약 통계는 전체 데이터로 계산)
    print(f"\n✅ 최종 데이터 저장 완료!")
    print(f"📁 저장 파일: {output_csv}")
    print(f"📋 최종 파일에는 학습에 필요한 {len(final_save_df.columns)}개 컬럼만 포함됩니다.")
    
    status_counts = result_df['return_calculation_status'].value_counts()
    print(f"\n📊 처리 결과 요약:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}개")
    
    if 'success' in status_counts:
        success_df = result_df[result_df['return_calculation_status'] == 'success']
        returns = success_df[f'future_return_{return_days}d']
        print(f"\n📈 {return_days}일 후 수익률 통계 (성공 건 대상):")
        print(f"  평균: {returns.mean():.4f} ({returns.mean()*100:.2f}%)")
        print(f"  표준편차: {returns.std():.4f} ({returns.std()*100:.2f}%)")
        print(f"  최소값: {returns.min():.4f} ({returns.min()*100:.2f}%)")
        print(f"  최대값: {returns.max():.4f} ({returns.max()*100:.2f}%)")
        
        positive_count = len(returns[returns > 0])
        negative_count = len(returns[returns < 0])
        print(f"  수익률 > 0: {positive_count}개 ({positive_count/len(returns)*100:.1f}%)")
        print(f"  수익률 < 0: {negative_count}개 ({negative_count/len(returns)*100:.1f}%)")
    
    return final_save_df

if __name__ == "__main__":
    # 3일 후 수익률로 계산하여 학습에 필요한 컬럼만 저장
    add_future_returns(
        input_csv="news_combined_features.csv",
        output_csv="news_with_future_returns.csv", 
        return_days=3
    )