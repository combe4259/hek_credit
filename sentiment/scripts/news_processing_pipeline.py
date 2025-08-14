# ==================================================================================
# 뉴스 전처리 파이프라인
# 새로 크롤링된 뉴스 → 감성분석 → FinBERT 임베딩 → 학습 데이터 형태로 변환
# ==================================================================================

import pandas as pd
import numpy as np
import os
import sys
import re
from datetime import datetime
from pymongo import MongoClient
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# 상위 디렉토리 추가 (sentiment_analysis.py 사용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class NewsProcessor:
    def __init__(self):
        self.mongo_uri = os.getenv('MONGODB_URI', 'mongodb+srv://julk0206:%23Sooyeon2004@hek.yqi7d9x.mongodb.net')
        self.client = None
        
    def connect_mongodb(self):
        """MongoDB 연결"""
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client.newsDB
            self.collection = self.db.news
            print("MongoDB 연결 완료")
            return True
        except Exception as e:
            print(f"MongoDB 연결 실패: {e}")
            return False
    
    # 전처리
    def clean_content(self, content):
        if not content:
            return ""
        
        content_str = str(content)
        
        # 프로모션/광고성 키워드 제거
        promo_keywords = [
            r'.*무료체험.*', r'.*체험신청.*', r'.*가입하기.*', r'.*클릭.*',
            r'.*바로가기.*', r'.*신청.*', r'.*이벤트.*', r'.*할인.*',
            r'.*혜택.*', r'.*특가.*', r'.*기회.*'
        ]
        
        for keyword in promo_keywords:
            content_str = re.sub(keyword, '', content_str, flags=re.IGNORECASE)
        
        # 불필요한 텍스트 패턴 제거
        unwanted_patterns = [
            r'\[.*?기자\]',  # [기자명]
            r'\[.*?특파원\]',  # [특파원명] 
            r'\[.*?=.*?\]',  # [지역=기자]
            r'Copyright.*',  # 저작권 표시
            r'copyrights?.*',  # 저작권 표시 (소문자)
            r'무단.*전재.*',  # 무단 전재 금지
            r'재배포.*금지.*',  # 재배포 금지
            r'배포.*금지.*',  # 배포 금지  
            r'※.*',  # 주석
            r'▶.*',  # 화살표로 시작하는 링크
            r'▲.*사진.*',  # 사진 설명
            r'▼.*사진.*',  # 사진 설명
            r'사진.*=.*',  # 사진 캡션
            r'그래픽.*=.*',  # 그래픽 캡션
            r'자료.*=.*',  # 자료 캡션
            r'\(사진.*\)',  # (사진설명)
            r'\(자료.*\)',  # (자료설명)
            r'\(그래픽.*\)',  # (그래픽설명)
            r'▷.*바로가기',  # 링크 안내
            r'▷.*자세히',  # 링크 안내
            r'관련기사.*',  # 관련기사 안내
            r'이전기사.*',  # 이전기사 안내
            r'다음기사.*',  # 다음기사 안내
            r'기자.*@.*\..*',  # 이메일 주소
            r'연락처.*\d{3}-\d{3,4}-\d{4}',  # 전화번호
            r'.*구독.*',  # 구독 안내
            r'.*팔로우.*',  # 팔로우 안내
            r'.*좋아요.*',  # 좋아요 안내
            r'.*공유하기.*',  # 공유 안내
            r'.*페이스북.*',  # SNS 관련
            r'.*트위터.*',  # SNS 관련
            r'.*인스타그램.*',  # SNS 관련
            r'※.*이 기사는.*',  # 기사 출처 표시
            r'▶.*이 기사는.*',  # 기사 출처 표시
        ]
        
        for pattern in unwanted_patterns:
            content_str = re.sub(pattern, '', content_str, flags=re.IGNORECASE)
        
        # HTML 태그 및 특수문자 정리
        content_str = re.sub(r'<[^>]+>', '', content_str)  # HTML 태그
        content_str = re.sub(r'&[a-z]+;', ' ', content_str)  # HTML
        content_str = re.sub(r'[^\w\s가-힣]', ' ', content_str)  # 특수문자 (한글, 영어, 숫자, 공백만 유지)
        content_str = re.sub(r'\s+', ' ', content_str)  # 연속된 공백을 하나로
        content_str = content_str.strip()
        
        # 너무 짧은 컨텐츠 필터링
        if len(content_str) < 20:
            return ""
        
        return content_str

    def get_new_news_from_mongo(self, limit=None):
        """MongoDB에서 새로운 뉴스 가져오기"""
        try:
            if not self.client:
                if not self.connect_mongodb():
                    return pd.DataFrame()
            
            # 최신 뉴스부터 가져오기
            query = {}
            if limit:
                cursor = self.collection.find(query).sort("created_at", -1).limit(limit)
            else:
                cursor = self.collection.find(query).sort("created_at", -1)
            
            news_data = list(cursor)
            
            if not news_data:
                print("새로운 뉴스가 없습니다.")
                return pd.DataFrame()
            
            df = pd.DataFrame(news_data)
            print(f"MongoDB에서 {len(df)}개 뉴스 로드 완료")
            
            # 필요한 컬럼만 선택하고 이름 변경
            required_columns = {
                'stock': 'original_stock',
                'title': 'title',
                'content': 'content',
                'url': 'url',
                'press': 'press',
                'published_at': 'news_date',
                'created_at': 'crawled_at'
            }
            
            # 컬럼 이름 변경
            for old_name, new_name in required_columns.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # 필수 컬럼 체크
            missing_cols = [col for col in required_columns.values() if col not in df.columns]
            if missing_cols:
                print(f"누락된 컬럼: {missing_cols}")
            
            # 컨텐츠 정제 적용
            print("뉴스 컨텐츠 정제 중")
            df['content'] = df['content'].apply(self.clean_content)
            
            # 정제 후 빈 컨텐츠 제거
            original_count = len(df)
            df = df[df['content'].str.len() > 0]
            filtered_count = len(df)
            
            if original_count != filtered_count:
                print(f"컨텐츠 정제 후 {original_count - filtered_count}개 뉴스 필터링됨")
            
            return df[list(required_columns.values())].copy()
            
        except Exception as e:
            print(f"MongoDB 데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    # 감성분석
    def run_sentiment_analysis(self, df):
        print("감성분석 시작...")
        
        try:
            # 감성분석을 위한 워커 함수들
            def init_worker():
                global nlp_pipeline
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
                model_name = "snunlp/KR-FinBert-SC"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                nlp_pipeline = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # CPU 사용
                    top_k=None,
                    max_length=512,
                    truncation=True
                )
            
            def worker_process(content):
                global nlp_pipeline
                try:
                    if not content or len(str(content).strip()) < 10:
                        return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
                    
                    # 짧은 텍스트는 전체를 한번에 처리
                    if len(str(content)) < 500:
                        result = nlp_pipeline(str(content))
                        scores = {x['label']: x['score'] for x in result}
                    else:
                        # 긴 텍스트는 문장 단위로 처리 후 평균
                        from nltk.tokenize import sent_tokenize
                        sentences = sent_tokenize(str(content))[:5]  # 최대 5문장만
                        all_scores = []
                        
                        for sent in sentences:
                            if len(sent.strip()) > 5:
                                result = nlp_pipeline(sent)
                                scores = {x['label']: x['score'] for x in result}
                                all_scores.append(scores)
                        
                        if all_scores:
                            # 평균 계산
                            scores = {}
                            for label in ['positive', 'negative', 'neutral']:
                                scores[label] = np.mean([s.get(label, 0) for s in all_scores])
                        else:
                            scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
                    
                    positive = scores.get('positive', 0.0)
                    negative = scores.get('negative', 0.0)
                    neutral = scores.get('neutral', 0.0)
                    
                    # 정규화
                    total = positive + negative + neutral
                    if total > 0:
                        positive /= total
                        negative /= total
                        neutral /= total
                    
                    return {
                        'positive': positive,
                        'negative': negative, 
                        'neutral': neutral
                    }
                    
                except Exception as e:
                    print(f"감성분석 오류: {e}")
                    return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            
            # NLTK 데이터 다운로드
            try:
                import nltk
                nltk.data.find('tokenizers/punkt_tab')
            except:
                import nltk
                nltk.download('punkt_tab', quiet=True)
                nltk.download('punkt', quiet=True)
            
            # 병렬 처리로 감성분석 실행
            contents = df['content'].fillna('').tolist()
            
            with ProcessPoolExecutor(max_workers=2, initializer=init_worker) as executor:
                futures = [executor.submit(worker_process, content) for content in contents]
                
                results = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="감성분석"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"감성분석 실행 오류: {e}")
                        results.append({'positive': 0.0, 'negative': 0.0, 'neutral': 0.0})
            
            # 결과를 DataFrame에 추가
            sentiment_df = pd.DataFrame(results)
            for col in ['positive', 'negative', 'neutral']:
                df[col] = sentiment_df[col]
            
            # 감성 점수 계산
            df['sentiment_score'] = df['positive'] - df['negative']
            
            print("감성분석 완료")
            return df
            
        except Exception as e:
            print(f"감성분석 실패: {e}")
            # 실패시 기본값으로 채우기
            df['positive'] = 0.5
            df['negative'] = 0.3
            df['neutral'] = 0.2
            df['sentiment_score'] = 0.2
            return df
    
    def add_finbert_embeddings(self, df):
        print("FinBERT 임베딩 생성 중")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # FinBERT 모델 로드
            model_name = "jhgan/ko-sroberta-multitask"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            
            embeddings = []
            
            for content in tqdm(df['content'].fillna(''), desc="임베딩 생성"):
                try:
                    if not content or len(str(content).strip()) < 10:
                        # 빈 컨텐츠는 제로 벡터
                        embedding = np.zeros(768)
                    else:
                        # 토큰화 및 임베딩 생성
                        inputs = tokenizer(str(content)[:512], return_tensors="pt", 
                                         truncation=True, padding=True, max_length=512)
                        
                        with torch.no_grad():
                            outputs = model(**inputs)
                            # [CLS] 토큰의 임베딩 사용
                            embedding = outputs.last_hidden_state[0][0].numpy()
                    
                    embeddings.append(embedding)
                    
                except Exception as e:
                    print(f"임베딩 생성 오류: {e}")
                    embeddings.append(np.zeros(768))
            
            # FinBERT 컬럼 추가
            for i in range(768):
                df[f'finbert_{i}'] = [emb[i] if len(emb) > i else 0 for emb in embeddings]
            
            print("FinBERT 임베딩 완료")
            return df
            
        except Exception as e:
            print(f"FinBERT 임베딩 실패: {e}")
            print("기본값으로 임베딩 생성")
            
            # 실패시 랜덤 임베딩 생성
            for i in range(768):
                df[f'finbert_{i}'] = np.random.normal(0, 0.01, len(df))
            
            return df
    
    def add_technical_scores(self, df):
        """기술적 분석 점수 계산 (yfinance 사용)"""
        print("📈 기술적 분석 점수 계산 중...")
        
        try:
            import yfinance as yf
            
            # 종목 매핑 파일에서 ticker 정보 가져오기
            mapping_file = os.path.join(os.path.dirname(__file__), 'data/sp500_korean_stocks_with_symbols.xlsx')
            if os.path.exists(mapping_file):
                df_mapping = pd.read_excel(mapping_file, header=1)
                df_mapping.dropna(subset=['Symbol'], inplace=True)
                name_to_ticker = pd.Series(df_mapping.Symbol.values, 
                                         index=df_mapping['Korean Name'].str.strip()).to_dict()
            else:
                # 기본 매핑
                name_to_ticker = {
                    '엔비디아': 'NVDA', '삼성전자': '005930.KS', '애플': 'AAPL',
                    '마이크로소프트': 'MSFT', '구글': 'GOOGL', '테슬라': 'TSLA'
                }
            
            # ticker 컬럼 추가
            df['ticker'] = df['original_stock'].str.strip('$').map(name_to_ticker).fillna('UNKNOWN')
            
            # 각 종목별로 기술적 점수 계산
            technical_scores = []
            unique_tickers = df['ticker'].unique()
            
            for ticker in tqdm(unique_tickers, desc="기술적 분석"):
                if ticker == 'UNKNOWN':
                    ticker_scores = {'rule_score': 50.0, 'momentum_score': 50.0, 'volume_score': 50.0}
                else:
                    try:
                        # 30일간 주가 데이터 가져오기
                        stock = yf.Ticker(ticker)
                        hist = stock.history(period="1mo")
                        
                        if len(hist) > 10:
                            # RSI 계산 (간단 버전)
                            closes = hist['Close'].values
                            gains = []
                            losses = []
                            for i in range(1, len(closes)):
                                change = closes[i] - closes[i-1]
                                if change > 0:
                                    gains.append(change)
                                    losses.append(0)
                                else:
                                    gains.append(0)
                                    losses.append(-change)
                            
                            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
                            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
                            
                            if avg_loss != 0:
                                rs = avg_gain / avg_loss
                                rsi = 100 - (100 / (1 + rs))
                            else:
                                rsi = 50
                            
                            # 모멘텀 점수 (최근 5일 vs 이전 5일)
                            if len(closes) >= 10:
                                recent_avg = np.mean(closes[-5:])
                                previous_avg = np.mean(closes[-10:-5])
                                momentum = ((recent_avg - previous_avg) / previous_avg) * 100 + 50
                                momentum = np.clip(momentum, 0, 100)
                            else:
                                momentum = 50
                            
                            # 거래량 점수 (최근 거래량 vs 평균 거래량)
                            volumes = hist['Volume'].values
                            if len(volumes) >= 5:
                                recent_volume = np.mean(volumes[-5:])
                                avg_volume = np.mean(volumes)
                                if avg_volume > 0:
                                    volume_score = min((recent_volume / avg_volume) * 50, 100)
                                else:
                                    volume_score = 50
                            else:
                                volume_score = 50
                            
                            ticker_scores = {
                                'rule_score': rsi,
                                'momentum_score': momentum,
                                'volume_score': volume_score
                            }
                        else:
                            ticker_scores = {'rule_score': 50.0, 'momentum_score': 50.0, 'volume_score': 50.0}
                            
                    except Exception as e:
                        print(f"⚠️ {ticker} 데이터 가져오기 실패: {e}")
                        ticker_scores = {'rule_score': 50.0, 'momentum_score': 50.0, 'volume_score': 50.0}
                
                technical_scores.append({'ticker': ticker, **ticker_scores})
            
            # 점수를 DataFrame에 매핑
            scores_df = pd.DataFrame(technical_scores)
            df = df.merge(scores_df, on='ticker', how='left')
            
            print("기술적 분석 점수 계산 완료")
            return df
            
        except Exception as e:
            print(f"기술적 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            
            df['ticker'] = df['original_stock'].str.strip('$').str[:4]
            df['rule_score'] = 50.0
            df['momentum_score'] = 50.0
            df['volume_score'] = 50.0
            return df
    
    # 섹터 및 시가총액 정보 추가
    def add_sector_and_market_cap(self, df):
        print("섹터 및 시가총액 정보 추가 중")
        
        try:
            import yfinance as yf
            
            unique_tickers = df['ticker'].unique()
            sector_data = []
            
            for ticker in tqdm(unique_tickers, desc="섹터/시총 정보"):
                if ticker == 'UNKNOWN':
                    sector_info = {'ticker': ticker, 'sector': 'Unknown', 'market_cap': 100000000000}
                else:
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        
                        sector_info = {
                            'ticker': ticker,
                            'sector': info.get('sector', 'Unknown'),
                            'market_cap': info.get('marketCap', 100000000000)
                        }
                    except Exception as e:
                        print(f"{ticker} 정보 가져오기 실패: {e}")
                        sector_info = {'ticker': ticker, 'sector': 'Unknown', 'market_cap': 100000000000}
                
                sector_data.append(sector_info)
            
            # 정보를 DataFrame에 매핑
            sector_df = pd.DataFrame(sector_data)
            df = df.merge(sector_df, on='ticker', how='left')
            
            print("✅ 섹터 및 시가총액 정보 추가 완료")
            return df
            
        except Exception as e:
            print(f"섹터 정보 추가 실패: {e}")
            df['sector'] = 'Unknown'
            df['market_cap'] = 100000000000
            return df
    
    def save_processed_data(self, df):
        """처리된 데이터 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # 기존 data 폴더에 저장
        data_dir = './data'
        os.makedirs(data_dir, exist_ok=True)
        
        filename = f"news_full_features_robust_{timestamp}.csv"
        filepath = os.path.join(data_dir, filename)
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"처리된 데이터 저장 완료: {filepath}")
        
        # 최신 파일로 심볼릭 링크 생성
        latest_filepath = os.path.join(data_dir, "news_full_features_robust.csv")
        try:
            if os.path.exists(latest_filepath):
                os.remove(latest_filepath)
            df.to_csv(latest_filepath, index=False, encoding='utf-8-sig')
            print(f"최신 데이터 링크 업데이트: {latest_filepath}")
        except Exception as e:
            print(f"심볼릭 링크 생성 실패: {e}")
        
        return filepath

def main():
    print("=" * 60)
    print("뉴스 데이터 전처리 파이프라인 시작")
    print("=" * 60)
    
    processor = NewsProcessor()
    
    try:
        # 1. MongoDB에서 새로운 뉴스 가져오기
        print("📡 MongoDB에서 뉴스 데이터 로드 중...")
        df = processor.get_new_news_from_mongo()
        
        if df.empty:
            print("처리할 뉴스가 없습니다.")
            return
        
        print(f"{len(df)}개 뉴스 로드 완료")
        
        # 2. 감성분석 실행
        df = processor.run_sentiment_analysis(df)
        
        # 3. FinBERT 임베딩 추가
        df = processor.add_finbert_embeddings(df)
        
        # 4. 기술적 분석 점수 추가 (ticker 포함)
        df = processor.add_technical_scores(df)
        
        # 5. 섹터 및 시가총액 정보 추가
        df = processor.add_sector_and_market_cap(df)
        
        # 6. sentence_count 컬럼 추가 (누락된 경우)
        if 'sentence_count' not in df.columns:
            df['sentence_count'] = df['content'].fillna('').str.split('.').str.len()
        
        # 7. 처리된 데이터 저장
        saved_path = processor.save_processed_data(df)
        
        print("=" * 60)
        print("뉴스 데이터 전처리 파이프라인 완료!")
        print(f"저장된 파일: {saved_path}")
        print(f"총 처리된 뉴스: {len(df)}개")
        print("=" * 60)
        
    except Exception as e:
        print(f"파이프라인 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if processor.client:
            processor.client.close()

if __name__ == "__main__":
    main()