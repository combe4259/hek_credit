"""
데이터 캐시 생성 스크립트 (v3.3: 최종 Excel 파일 기반 - 빈 행 문제 해결)
- 사용자가 제공한 Excel(.xlsx) 파일의 비어있는 첫 행을 건너뛰고 데이터를 수집합니다.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime
import time
from tqdm import tqdm
import joblib
import os
from google.colab import drive

# --- 설정 ---
drive.mount('/content/drive', force_remount=True)

# ❗️사용자의 실제 Excel 파일 경로
MAPPING_FILE_PATH = "/content/drive/MyDrive/sp500_korean_stocks_with_symbols.xlsx"

# 새로 생성할 캐시 파일 경로
CACHE_FILE_PATH = "/content/drive/MyDrive/yfinance_data_cache.joblib"
# 데이터 수집 시작 날짜
START_DATE = datetime(2018, 1, 1)

def create_yfinance_cache_from_excel():
    """사용자의 Excel 파일에서 티커 목록을 읽어와 yfinance 캐시를 생성합니다."""

    print("🚀 사용자 Excel 기반 데이터 캐싱 시작...")

    # --- 1. Excel 파일에서 티커 목록 읽기 ---
    try:
        # ▼▼▼ [핵심 수정] header=1 옵션을 추가하여 세 번째 줄을 열 이름으로 지정 ▼▼▼
        # 비어있는 첫 행과 제목 행을 건너뜁니다.
        df_map = pd.read_excel(MAPPING_FILE_PATH, sheet_name=0, header=1)
        # ▲▲▲ 수정 완료 ▲▲▲

        # 'Symbol' 열에 비어있는 값이 있을 경우 제거하고, 문자열로 변환
        df_map = df_map.dropna(subset=['Symbol'])
        df_map['Symbol'] = df_map['Symbol'].astype(str)
        # 심볼의 '.'을 '-'로 표준화
        df_map['Symbol'] = df_map['Symbol'].str.replace('.', '-', regex=False)
        tickers = df_map['Symbol'].unique().tolist()
        print(f"✅ Excel 파일에서 {len(tickers)}개의 고유 티커를 성공적으로 읽었습니다.")
        print(f"   - 샘플 티커: {tickers[:5]}")
    except FileNotFoundError:
        print(f"❌ 매핑 파일을 찾을 수 없습니다: {MAPPING_FILE_PATH}")
        print("   - 파일 이름과 경로가 정확한지 다시 확인해주세요.")
        return
    except KeyError:
        print("❌ 파일에서 'Symbol' 열을 찾을 수 없습니다. 엑셀 파일의 3행에 'Symbol' 열이 있는지 확인해주세요.")
        return
    except Exception as e:
        print(f"❌ 매핑 파일을 읽는 중 오류 발생: {e}")
        return

    # --- 2. yfinance에서 데이터 수집 및 캐싱 ---
    ticker_data_cache = {}
    end_date = datetime.now()

    for ticker in tqdm(tickers, desc="티커별 데이터 수집 및 캐싱"):
        if not ticker or pd.isna(ticker):
            continue

        retries = 3
        while retries > 0:
            try:
                hist = yf.Ticker(ticker).history(start=START_DATE, end=end_date)
                if not hist.empty:
                    ticker_data_cache[ticker] = hist
                else:
                    print(f"⚠️ {ticker}: yfinance에서 빈 데이터를 반환했습니다.")
                break
            except Exception as e:
                retries -= 1
                if retries > 0:
                    print(f"❌ {ticker} 데이터 로드 실패: {e}. 5초 후 재시도... ({retries}회 남음)")
                    time.sleep(5)
                else:
                    print(f"❌ {ticker} 데이터 로드 최종 실패: {e}.")
        time.sleep(1)

    # --- 3. 캐시 파일 저장 ---
    if not ticker_data_cache:
        print("❌ 캐시할 데이터가 없습니다. 프로그램을 종료합니다.")
        return

    joblib.dump(ticker_data_cache, CACHE_FILE_PATH)
    print("\n" + "="*50)
    print(f"🎉 새로운 yfinance 데이터 캐싱 완료!")
    print(f"   - 저장 경로: {CACHE_FILE_PATH}")
    print(f"   - 최종 캐시된 티커 수: {len(ticker_data_cache)}개")
    print("="*50)

# --- 스크립트 실행 ---
if __name__ == "__main__":
    create_yfinance_cache_from_excel()