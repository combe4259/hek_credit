"""
BERT 임베딩만 새로 생성하여 기존 CSV에 추가하는 스크립트
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sys
import os

# Google Drive 마운트 및 경로 설정
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
module_path = '/content/drive/MyDrive/'
if module_path not in sys.path:
    sys.path.append(module_path)

print("🧠 KR-FinBERT 임베딩 모델 로딩 중...")
embedding_model = SentenceTransformer('snunlp/KR-FinBERT')
print("✅ 임베딩 모델 로딩 완료!")

def generate_embeddings_from_csv(input_csv_path, output_csv_path):
    """기존 CSV 파일을 불러와 BERT 임베딩을 추가한 뒤 저장합니다."""
    print(f"📁 기존 파일 로딩 중: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
        print(f"✅ {len(df)}개 데이터 로드 완료!")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {input_csv_path}")
        return

    # 'content' 컬럼을 BERT 임베딩으로 변환
    print("🧠 BERT 임베딩 생성 시작...")
    contents = df['content'].fillna('').tolist()

    embeddings = embedding_model.encode(
        contents,
        convert_to_numpy=True,
        show_progress_bar=True,
        truncation=True,
        max_length=512
    )

    # 임베딩을 새로운 DataFrame으로 변환
    embedding_df = pd.DataFrame(embeddings, index=df.index)
    embedding_df.columns = [f'finbert_{i}' for i in range(embedding_df.shape[1])]

    # 기존 데이터프레임과 임베딩 데이터프레임을 합치기
    final_df = pd.concat([df, embedding_df], axis=1)

    # 최종 파일 저장
    print(f"🎉 최종 파일 저장 완료: {output_csv_path}")
    final_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"📈 최종 컬럼: {list(final_df.columns)}")

if __name__ == "__main__":
    INPUT_FILE = "/content/drive/MyDrive/news_sentiment_probabilities.csv" # 기존 파일 경로를 여기에 입력하세요
    OUTPUT_FILE = "/content/drive/MyDrive/news_with_embeddings.csv"
    generate_embeddings_from_csv(INPUT_FILE, OUTPUT_FILE)