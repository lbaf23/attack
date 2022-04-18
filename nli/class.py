import transformers


# model_path = "roberta-large-mnli"
model_path = "textattack/bert-base-uncased-snli"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)


classifier = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer)

# s1 = "A soccer game with multiple males playing. Some men are playing a sport."
# s2 = "A soccer game with multiple males playing. Computer Science is very popular nowadays."

# entailment label 0
s0 = "A man, woman, and child enjoying themselves on a beach. A family of three is at the beach."

# neutral label 1
s1 = "A Little League team tries to catch a runner sliding into a base in an afternoon game. A team is trying to score the games winning out."

# contradiction label 2
s2 = "Two women, holding food carryout containers, hug. Two groups of rival gang members flipped each other off."


classifier([s0, s1, s2])
