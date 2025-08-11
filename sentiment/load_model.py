from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 모델 로드
model_name = "snunlp/KR-FinBert-SC"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 감성 분석 pipeline 준비
nlp_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
    max_length=512,   # 모델 허용 최대 길이
    truncation=True   # 512 넘으면 자동 자르기
)
