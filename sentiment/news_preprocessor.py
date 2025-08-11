"""
뉴스 전처리 + 한국어 종목명 → 영어 티커 매핑
"""

import json
import pandas as pd
import re
import os
from typing import Dict, List, Optional
from rapidfuzz import fuzz, process
from datetime import datetime, timedelta
import warnings
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasketch import MinHash, MinHashLSH
warnings.filterwarnings('ignore')

# .env 파일에서 환경변수 로드 (루트 디렉토리)
load_dotenv("../.env")

class NewsPreprocessor:
    """뉴스 전처리 & 종목 매핑 클래스"""

    def __init__(self, excel_file: str = "data/src/sp500_korean_stocks_with_symbols.xlsx", debug: bool = False):
        self.debug = debug
        self.stock_mapping = {}
        self.load_stock_mapping(excel_file)

        # <<< 필터링 규칙용 키워드 및 패턴 정의 >>>
        self.PROMOTION_KEYWORDS = ["할인", "특가", "이벤트", "프로모션", "무료", "증정"]
        self.INVESTMENT_KEYWORDS = [
            "실적", "전망", "목표주가", "투자의견", "매수", "매도", "중립", "상승", "하락",
            "수익률", "애널리스트", "리포트", "컨센서스", "어닝", "가이던스", "공급", "계약", "M&A",
            "실적발표", "주가", "투자", "인수", "성장", "수주",
            "배당", "자본금", "시장점유율", "분기실적", "영업이익", "순이익", "매출",
            "주식", "증권", "거래량", "유상증자", "재무제표", "재무구조", "IPO", "상장",
            "투자자", "펀드", "헤지펀드", "신규사업", "사업확장", "신제품", "리스크", "위험",
            "경쟁사", "환율", "금리", "경제지표", "주총", "배당금", "유통주식수", "대주주",
            "지분변동", "자회사", "합작법인", "신용등급", "채권", "부채비율"
]
        self.POLITICAL_KEYWORDS = [
            "정부", "정책", "규제", "관세", "선거", "대통령", "국회", "법안", "공화당", "민주당",
            "트럼프", "바이든", "의회"
        ]
        self.ENUMERATION_PATTERNS = [r"등을", r"등이", r"비롯한"]

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
        """1차 전처리: 불필요한 공백/특수문자 + 광고/잡다한 문구 제거"""
        if not isinstance(text, str) or not text:
            return ""

        # 줄바꿈 및 공백 정리
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r'\s+', ' ', text)

        # 게티이미지/광고/버튼 등 불필요한 문구 패턴 제거
        unwanted_pattern = re.compile(
            r"사진\s*=\s*게티이미지\w*|"            # 사진=게티이미지뱅크 등
            r"게티이미지뱅크|"
            r"구독\s*\d+[명]*|"                    # 구독 6,565명
            r"응원\s*\d+|"                        # 응원 76,297
            r"(좋아요|슬퍼요|화나요|팬이에요|후속기사 원해요)\s*\d*|"  # 감정 버튼
            r"무료만화\s*보러가기|"
            r"(프로야구|바로가기)|"
            r"언론사\s*홈\s*바로가기|"
            r"기자\s*구독|"
            r"\*?\s*재판매\s*및\s*DB\s*금지\s*\*?|"  # *재판매 및 DB 금지*
            r"\*?\s*재판매\s*금지\s*\*?|"            # *재판매 금지*
            r"사진\s*제공\s*=\s*[^\s]+|"            # 사진 제공=회사명
            r"이\s*기사는\s*\d{4}년\d{2}월\d{2}일.*?선공개.*?|"   # 프리미엄 콘텐츠 관련
            r"재생\s*\d+[,\d]*\s*\d{1,2}:\d{2}|"   # 재생 7,188 04:26
            r"\d{1,2}:\d{2}\s*-\s*|"                # 04:26 - 
            r"\|?\s*[가-힣]{2,4}\s*기자|"           # 최병태 기자, 김수현 기자 등
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"  # 이메일 패턴
            r"(공식계정\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
        )
        
        text = unwanted_pattern.sub("", text)

        # 대괄호 [] 안의 내용 제거
        text = re.sub(r"\[[^\]]*\]", "", text)

        # 저작권 표시 제거 (ⓒ출처명)
        text = re.sub(r"ⓒ[^\s]+", "", text)

        # 특수 기호 제거
        text = re.sub(r"[①⑦⑩◆◇■●▶▷△◀▲▼★※]", "", text)

        # 사진 관련 태그 제거
        text = re.sub(r"<사진=.*?>", "", text)
        text = re.sub(r'/(사진|그래픽|영상)=.*?기자', '', text)
        text = re.sub(r"(그래픽|사진|기사|인터뷰|Cover Story)\s*=\s*[^,.\s]+", "", text)
        text = re.sub(r"사진:.*?(\.|$)", "", text)
        text = re.sub(r"〈사진=.*?〉", "", text)

        # 저작권 출처 + 기자명 + 날짜 시간 패턴 제거 (추가)
        text = re.sub(
            r"ⓒ\s*[^\d\s]+(?:\s*[가-힣]{2,4})?\s*\d{1,2}일\s*오?전?\s*\d{1,2}시\s*\d{1,2}분",
            "", text
        )

        # URL 제거
        text = re.sub(r"https?://[^\s]+", "", text)

        # 양 옆 ===== 5개 이상으로 둘러쌓인 텍스트
        text = re.sub(r"={5,}.*?={5,}", "", text, flags=re.DOTALL)

        # '인사이트' 포함 단어 제거
        text = re.sub(r"\b[가-힣]+인사이트\b", "", text)

        # 따옴표 통일
        text = re.sub(r"[“”‘’\"']", '"', text)

        # 감성분석에 불필요한 특수문자 제거 (단, 문장부호 유지)
        text = re.sub(r"[^\w\s.,!?\"']", " ", text)

        # 기사 게재일, 출처 문구 제거
        text = re.sub(r"이 기사는 .*?에 게재된 기사입니다\.", "", text)

        # 1) 깨진 문자(�) UTF-8 인코딩/디코딩해서 제거
        text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

        text = text.replace("\\", "")

        # 최종 공백 정리
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def map_stock(self, stock_name: str) -> Optional[str]:
        """한국어 종목명 → 영어 티커 (정확 매칭 → 퍼지 매칭 순서)"""
        if stock_name in self.stock_mapping:
            return self.stock_mapping[stock_name]

        # 퍼지 매칭 (cutoff 85 이상만)
        match = process.extractOne(stock_name, self.stock_mapping.keys(), scorer=fuzz.partial_ratio)
        if match and match[1] >= 85:
            if self.debug:
                print(f"📍 퍼지 매칭: '{stock_name}' → '{match[0]}' ({self.stock_mapping[match[0]]})")
            return self.stock_mapping[match[0]]

        return None
    
    def get_minhash(self, text, num_perm=128):
        m = MinHash(num_perm=num_perm)
        tokens = set(text.split())  # 간단 토큰화
        for token in tokens:
            m.update(token.encode('utf8'))
        return m
    
    def filter_similar_news(self, news_list, threshold=0.85):
        """뉴스 본문 유사도 기반 중복 뉴스 제거"""
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        unique_news = []
        minhashes = []
        
        for i, news in enumerate(news_list):
            m = self.get_minhash(news['content'])
            minhashes.append(m)
            
        for i, m in enumerate(minhashes):
            # 현재 LSH에 삽입된 것 중 유사한 뉴스 검색
            similar_keys = lsh.query(m)

            # 자기 자신 key i가 있다면 제거 (안 들어가있으면 보통 없지만 안전장치)
            similar_keys = [k for k in similar_keys if k != i]

            if similar_keys:
                # 유사한 뉴스 이미 있음 → 중복이므로 skip
                continue

            lsh.insert(i, m)
            unique_news.append(news_list[i])
                
        return unique_news

    # <<< 뉴스 관련성 필터링 로직 >>>
    def is_relevant(self, title: str, content: str, stock_kor: str, ticker: Optional[str]) -> (bool, str):
        """뉴스 기사의 투자 관련성을 판단하여 (True/False, 사유)를 반환"""
        if not title and not content:
            return False, "내용_없음"
            
        # 1. 즉시 제거 규칙 (Fast Fail)
        if any(keyword in title for keyword in self.PROMOTION_KEYWORDS):
            return False, "광고성_제목"

        for pattern in self.ENUMERATION_PATTERNS:
            if re.search(f"{re.escape(stock_kor)}\s*{pattern}", content):
                # 주변 50자 이내에 투자 키워드가 없다면 단순 나열로 간주
                match = re.search(f"{re.escape(stock_kor)}\s*{pattern}", content)
                if match:
                    start, end = max(0, match.start() - 50), match.end() + 50
                    context_window = content[start:end]
                    if not any(kw in context_window for kw in self.INVESTMENT_KEYWORDS):
                        return False, "단순_나열"
        
        # 2. 관련성 점수 계산
        score = 0
        stock_name_in_title = stock_kor in title or (ticker and ticker in title)
        if stock_name_in_title:
            score += 3

        stock_mentions = content.count(stock_kor) + (content.count(ticker) if ticker else 0)
        score += stock_mentions

        found_invest_keywords = {keyword for keyword in self.INVESTMENT_KEYWORDS if keyword in content}
        score += len(found_invest_keywords) * 2

        stock_name_in_title = stock_kor in title or (ticker and ticker in title)

        if stock_mentions < 2 and not stock_name_in_title:
            # 주변 맥락 검사 대신 투자 키워드만 검사
            if len(found_invest_keywords) >= 2:
                return True, "투자키워드_있어서_통과"
            else:
                return False, "언급_부족_및_맥락_없음"

        if score < 2:
            return False, f"관련성_점수_낮음({score}점)"

        return True, f"관련성_높음({score}점)"
    
    def validate_cleaned_text(self, original: str, cleaned: str) -> Dict:
        """정제된 텍스트 품질 검증"""
        if not original or not cleaned:
            return {"status": "empty", "length_reduction": 0}
        
        original_len = len(original)
        cleaned_len = len(cleaned)
        length_reduction = (original_len - cleaned_len) / original_len
        
        # 너무 많이 줄어들면 경고
        if length_reduction > 0.7:  # 70% 이상 줄어들면
            status = "over_cleaned"
        elif length_reduction < 0.1:  # 10% 미만으로 줄어들면
            status = "under_cleaned"
        else:
            status = "good"
        
        return {
            "status": status,
            "length_reduction": round(length_reduction, 2),
            "original_length": original_len,
            "cleaned_length": cleaned_len
        }
    
    def _filter_by_ticker_count(self, data: List[Dict], min_news_count: int = 5) -> List[Dict]:
        """티커별 최소 뉴스 개수 필터링"""
        df = pd.DataFrame(data)
        counts = df['ticker'].value_counts()
        valid_tickers = counts[counts >= min_news_count].index.tolist()
        if self.debug:
            dropped = counts[counts < min_news_count]
            print(f"🚫 제거된 티커: {list(dropped.index)} (뉴스 수: {dropped.to_dict()})")
        filtered = df[df['ticker'].isin(valid_tickers)]
        return filtered.to_dict(orient='records') 

    def load_news_data(self, file_path: str) -> List[Dict]:
        """뉴스 데이터 파일 로드"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"✅ 뉴스 데이터 로드 완료: {len(data)}건")
            return data
        except Exception as e:
            print(f"❌ 뉴스 데이터 로드 실패: {e}")
            return []

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

    def process_news_parallel(self, raw_data: List[Dict], max_workers: int = 5) -> tuple[List[Dict], Dict]:
        """멀티스레드를 사용한 뉴스 병렬 처리"""
        results = []
        status_counter = {"good": 0, "over_cleaned": 0, "under_cleaned": 0, "empty": 0}
        
        print(f"🚀 멀티스레드 처리 시작 (워커: {max_workers}개, 총 {len(raw_data)}건)")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_news, item, self) for item in raw_data]
            
            for i, future in enumerate(as_completed(futures), 1):
                result, status = future.result()
                if status in status_counter:
                    status_counter[status] += 1
                else:
                    status_counter[status] = 1
                
                if result:
                    results.append(result)
                
                if i % 10 == 0 or i == len(raw_data):
                    progress = (i / len(raw_data)) * 100
                    print(f"  진행상황: {i}/{len(raw_data)} ({progress:.1f}%) 처리 완료", flush=True)
        
        print(f"✅ 병렬 처리 완료: {len(results)}건")
        print(f"   텍스트 품질: {status_counter}")
        
        return results, status_counter
    

def process_single_news(item, processor: NewsPreprocessor):
    if processor.debug:
        print(f"▶ 뉴스 처리 시작: {item.get('stock', '')[:10]}...", flush=True)
    stock_kor = item.get("stock", "").strip()
    content_original = item.get("content", "")
    title_original = item.get("title", "")

    # 기본 유효성 검사
    if not stock_kor or not content_original or not title_original:
        return None, "missing_data"
    
    published_raw = item.get("published_at", {}).get("$date", None)
    try:
        news_date = datetime.fromisoformat(published_raw.replace("Z", "+00:00")).date()
        if news_date < datetime(2024, 1, 1).date():
            return None, "date_filtered"
        news_date_str = news_date.isoformat()
    except:
        return None, "date_error"
    
    # 1차 룰셋 정제
    content_cleaned = processor.clean_content(content_original)
    title_cleaned = processor.clean_content(title_original)

    if not stock_kor or not content_cleaned or len(content_cleaned.strip()) < 50 or len(content_cleaned.split('.')) < 2:
        return None, "content_too_short"
    
    ticker = processor.map_stock(stock_kor) or stock_kor
    
    # 관련성 필터링
    is_rel, reason = processor.is_relevant(title_cleaned, content_cleaned, stock_kor, ticker)
    if not is_rel:
        if processor.debug:
            print(f"  ❌ 필터링됨: '{title_cleaned[:30]}...' (사유: {reason})", flush=True)
        return None, f"filtered_{reason}"
    
    quality = processor.validate_cleaned_text(content_original, content_cleaned)

    return {
        "original_stock": stock_kor,
        "ticker": ticker,
        "news_date": news_date_str,
        "content": content_cleaned,
        "url": item.get("url", "").strip()
    }, quality["status"]


def main():
    """메인 실행 함수: 뉴스 전처리 파이프라인"""
    # 1. 전처리기 초기화
    processor = NewsPreprocessor(
        excel_file="data/src/sp500_korean_stocks_with_symbols.xlsx", 
        debug=True,
    )
    
    # 2. 뉴스 데이터 로드
    raw_data = processor.load_news_data("data/src/newsDB.news.json")
    if not raw_data:
        return
    
    # 3. 병렬 처리로 뉴스 전처리
    results, status_counter = processor.process_news_parallel(raw_data, max_workers=5)

    # 중복 뉴스 필터링
    print(f"🔍 중복 뉴스 필터링 전: {len(results)}건")
    deduped_results = processor.filter_similar_news(results, threshold=0.85)
    print(f"🔍 중복 뉴스 필터링 후: {len(deduped_results)}건")

    
    # 4. 티커별 최소 뉴스 개수 필터링 & 결과 저장
    filtered_results = processor._filter_by_ticker_count(deduped_results, min_news_count=5)
    processor.save_results(filtered_results)
    
if __name__ == "__main__":
    main()
