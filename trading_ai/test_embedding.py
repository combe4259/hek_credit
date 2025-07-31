import torch
import torch.nn as nn
import numpy as np
import faiss

# 간단한 임베딩 테스트
class SimpleEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(15, 64)
    
    def forward(self, x):
        return self.fc(x)

# 테스트
print("PyTorch 버전:", torch.__version__)
print("FAISS 설치 확인...")

try:
    # 작은 데이터로 테스트
    embedder = SimpleEmbedder()
    test_data = torch.randn(100, 15)
    
    print("임베딩 생성 테스트...")
    with torch.no_grad():
        embeddings = embedder(test_data).numpy()
    print(f"✅ 임베딩 생성 성공: {embeddings.shape}")
    
    # FAISS 테스트
    print("FAISS 인덱스 테스트...")
    index = faiss.IndexFlatL2(64)
    index.add(embeddings.astype(np.float32))
    print(f"✅ FAISS 인덱스 생성 성공: {index.ntotal}개")
    
except Exception as e:
    print(f"❌ 오류 발생: {e}")