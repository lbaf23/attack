import transformers

# model_path = 'echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid'
# model_path = 'mrm8488/bert-mini-finetuned-age_news-classification'
# model_path = 'xaqren/sentiment_analysis'
# model_path = 'lannelin/bert-imdb-1hidden'
# model_path = 'gchhablani/bert-base-cased-finetuned-mnli'
model_path = "rohanrajpal/bert-base-multilingual-codemixed-cased-sentiment"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = transformers.pipeline('text-classification', tokenizer=tokenizer, model=model)

classifier("I'm watching a movie")
