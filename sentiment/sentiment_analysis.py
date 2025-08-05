import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import multiprocessing

# NLTK 데이터 확인 및 다운로드 (필요시만)
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 데이터 다운로드 중...")
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)

def process_article(content, nlp_pipeline, batch_size=64):
    """하나의 기사 본문(content)을 문장별로 나눠 감성분석하고 평균값 반환"""
    
    # 빠른 전처리
    if not content or len(str(content).strip()) < 10:
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'sentence_count': 0}
    
    sentences = sent_tokenize(str(content))
    if len(sentences) == 0:
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'sentence_count': 0}

    sentence_scores = []

    # 문장 단위 batch 처리
    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i:i + batch_size]
        try:
            batch_results = nlp_pipeline(batch_sents)
            for res in batch_results:
                scores = {x['label']: x['score'] for x in res}
                positive = scores.get('positive', 0.0)
                negative = scores.get('negative', 0.0)
                neutral = scores.get('neutral', 0.0)

                total = positive + negative + neutral
                if total > 0:
                    positive /= total
                    negative /= total
                    neutral /= total

                sentence_scores.append({
                    'positive': positive,
                    'negative': negative,
                    'neutral': neutral
                })
        except Exception as e:
            print(f"Error in batch: {e}")
            sentence_scores.extend([{'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}] * len(batch_sents))

    # 문장별 점수 평균내기
    pos_avg = sum(s['positive'] for s in sentence_scores) / len(sentence_scores)
    neg_avg = sum(s['negative'] for s in sentence_scores) / len(sentence_scores)
    neu_avg = sum(s['neutral'] for s in sentence_scores) / len(sentence_scores)
    sent_count = len(sentences)

    return {'positive': pos_avg, 
            'negative': neg_avg, 
            'neutral': neu_avg,
            'sentence_count': sent_count
            }

def main():
    """메인 실행 함수"""
    # 1) 모델 로드 (한 번만)
    print("🤖 모델 로딩 중...")
    from load_model import nlp_pipeline
    print("✅ 모델 로드 완료")
    
    # 2) 데이터 불러오기
    import glob
    import os
    
    csv_files = glob.glob("data/news_mapped_*.csv")
    if csv_files:
        latest_file = max(csv_files, key=os.path.getctime)
        print(f"📂 최신 전처리 파일 사용: {latest_file}")
        df = pd.read_csv(latest_file)
    else:
        raise FileNotFoundError("전처리된 뉴스 파일을 찾을 수 없습니다.")
    
    print(f"📊 총 {len(df)}개 뉴스 처리 시작")
    
    # 3) 순차 처리 (안정성 우선)
    all_results = []
    contents = df["content"].tolist()
    
    # 진행률 표시 개선
    for i, content in enumerate(tqdm(contents, desc="감성분석 진행")):
        result = process_article(content, nlp_pipeline, batch_size=64)
        all_results.append(result)
        
        # 1000개마다 진행상황 출력
        if (i + 1) % 1000 == 0:
            print(f"진행: {i+1}/{len(contents)} 완료")
    
    print("✅ 감성분석 완료")
    
    # 4) 결과 합치기
    scores_df = pd.DataFrame(all_results)
    df = pd.concat([df, scores_df], axis=1)
    
    # 단일 점수화
    df['sentiment_score'] = df['positive'] - df['negative']
    
    # 5) 저장
    df.to_csv("news_sentiment_probabilities.csv", index=False, encoding="utf-8-sig")
    print("📁 news_sentiment_probabilities.csv 저장 완료!")

if __name__ == "__main__":
    main()