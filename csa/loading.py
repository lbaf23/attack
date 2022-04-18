import transformers

model_path = 'hfl/chinese-bert-wwm'

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)


classifier = transformers.pipeline('text-classification', tokenizer=tokenizer, model=model)

s0 = "我讨厌那部电影"
s1 = "我喜欢那部电影"

classifier([s0, s1])
