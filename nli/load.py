import transformers

model_path = "cross-encoder/nli-distilroberta-base"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer)

# entailment label 0
s0 = "An old man with a package poses in front of an advertisement. A man poses in front of an ad."

# neutral label 1
s1 = "An old man with a package poses in front of an advertisement. A man poses in front of an ad for beer."

# contradiction label 2
s2 = "One tan girl with a wool hat is running and leaning over an object, while another person in a wool hat is sitting on the ground. A boy runs into a wall."

classifier([s0, s1, s2])
