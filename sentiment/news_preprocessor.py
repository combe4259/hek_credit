"""
뉴스 전처리 + 한국어 종목명 → 영어 티커 매핑 (단순화 버전)
"""

import json
import pandas as pd
import re
import os
from typing import Dict, List, Optional
from rapidfuzz import fuzz, process
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NewsPreprocessor:
    """뉴스 전처리 & 종목 매핑 클래스"""

    def __init__(self, excel_file: str = "data/src/sp500_korean_stocks_with_symbols.xlsx"):
        self.stock_mapping = {}
        self.load_stock_mapping(excel_file)

    def load_stock_mapping(self, excel_file: str):
        """엑셀 파일에서 매핑 딕셔너리 생성"""
        try:
            df = pd.read_excel(excel_file)
            df.columns = ['korean_name', 'symbol']  # 첫 열: 한국어, 두 번째 열: 영어

            # NaN 제거 후 strip
            df = df.dropna()
            for _, row in df.iterrows():
                korean = str(row['korean_name']).strip()
                symbol = str(row['symbol']).strip()
                if korean and symbol:
                    self.stock_mapping[korean] = symbol

            print(f"✅ 매핑 로드 완료: {len(self.stock_mapping)}개")

        except Exception as e:
            print(f"❌ 엑셀 로드 실패: {e}")

    def clean_content(self, text: str) -> str:
        """뉴스 내용 정리 (불필요한 공백/특수문자 + 광고/잡다한 문구 제거)"""
        if not text:
            return ""

        # 기본 정리
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r'\s+', ' ', text)

        # 게티이미지/광고/버튼 텍스트 제거
        unwanted_patterns = [
            r"사진\s*=\s*게티이미지\w*",       # "사진=게티이미지뱅크"
            r"게티이미지뱅크",                # "게티이미지뱅크"
            r"구독\s*\d+[명]*",               # "구독 6,565명"
            r"응원\s*\d+",                   # "응원 76,297"
            r"(좋아요|슬퍼요|화나요|팬이에요|후속기사 원해요)\s*\d*",  # 감정 버튼
            r"무료만화\s*보러가기",            # "무료만화 보러가기"
            r"(프로야구|바로가기)",           # "프로야구 바로가기"
            r"언론사\s*홈\s*바로가기",         # 언론사 홈 바로가기
            r"기자\s*구독",                   # "배우근 기자 구독"
        ]

        for pattern in unwanted_patterns:
            text = re.sub(pattern, "", text)

        return text.strip()

    def map_stock(self, stock_name: str) -> Optional[str]:
        """한국어 종목명 → 영어 티커 (정확 매칭 → 퍼지 매칭 순서)"""
        if stock_name in self.stock_mapping:
            return self.stock_mapping[stock_name]

        # 퍼지 매칭 (cutoff 85 이상만)
        match = process.extractOne(stock_name, self.stock_mapping.keys(), scorer=fuzz.partial_ratio)
        if match and match[1] >= 85:
            print(f"📍 퍼지 매칭: '{stock_name}' → '{match[0]}' ({self.stock_mapping[match[0]]})")
            return self.stock_mapping[match[0]]

        return None

    def process_news(self, input_file: str = "data/src/newsDB.news.json"):
        """뉴스 JSON 처리 + 매핑"""
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as e:
            print(f"❌ 뉴스 JSON 로드 실패: {e}")
            return [], {}

        print(f"📊 뉴스 {len(raw_data)}개 처리 시작")

        processed_data = []
        stats = {"mapped": 0, "failed": 0}

        for i, item in enumerate(raw_data):
            stock_kor = item.get("stock", "").strip()
            content = self.clean_content(item.get("content", ""))

            # 날짜 처리
            published_raw = item.get("published_at", {}).get("$date", None)
            if published_raw:
                news_date = datetime.fromisoformat(published_raw.replace("Z", "+00:00")).date()
                
                # 1년 이전 뉴스 필터링 (2024년 8월 5일 이후만)
                cutoff_date = datetime(2024, 8, 5).date()
                if news_date < cutoff_date:
                    continue
                    
                news_date = news_date.isoformat()
            else:
                news_date = None

            if not stock_kor or not news_date:
                continue

            ticker = self.map_stock(stock_kor)
            if ticker:
                stats["mapped"] += 1
            else:
                stats["failed"] += 1
                ticker = stock_kor  # 매핑 실패 시 원본 그대로 저장

            processed_data.append({
                "original_stock": stock_kor,
                "ticker": ticker,
                "news_date": news_date,
                "content": content
            })

            if (i + 1) % 1000 == 0:
                print(f"  진행상황: {i+1}/{len(raw_data)} 처리 완료")

        print(f"\n✅ 매핑 완료: 성공 {stats['mapped']} / 실패 {stats['failed']}")
        return processed_data, stats

    def save_results(self, processed_data: List[Dict]):
        """결과 저장 (JSON + CSV + 티커별 카운트)"""
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

        # 저장할 폴더 경로
        folder_path = "data"
        os.makedirs(folder_path, exist_ok=True)
        
        # 전체 JSON 저장
        json_file = os.path.join(folder_path, f"news_processed_{ts}.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"💾 전체 JSON 저장 → {json_file}")

        # 매핑 성공 데이터만 CSV 저장
        mapped_data = [d for d in processed_data if d['ticker'] in self.stock_mapping.values()]
        df = pd.DataFrame(mapped_data)
        csv_file = os.path.join(folder_path, f"news_mapped_{ts}.csv")
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        print(f"📊 매핑 성공 CSV 저장 → {csv_file} (총 {len(df)}개)")

        # 티커별 뉴스 개수 저장
        ticker_counts = df['ticker'].value_counts()
        count_file = os.path.join(folder_path, f"ticker_counts_{ts}.csv")
        ticker_counts.to_csv(count_file, header=['news_count'])
        print(f"📈 티커별 카운트 저장 → {count_file}")


def main():
    processor = NewsPreprocessor("data/src/sp500_korean_stocks_with_symbols.xlsx")
    processed_data, stats = processor.process_news("data/src/newsDB.news.json")

    if processed_data:
        processor.save_results(processed_data)

if __name__ == "__main__":
    main()
