import transformers


model_path = "roberta-large-mnli"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = transformers.pipeline('text-classification', tokenizer=tokenizer, model=model)

s1 = "Which city is the capital of France?, Where is the capital of France?"

classifier(s1)
