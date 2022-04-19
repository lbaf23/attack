import OpenAttack as oa
import datasets
import transformers


def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }


def sa_attack(model_path):
    dataset = datasets.load_dataset("sst", split="train[:10]").map(function=dataset_mapping)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, output_hidden_states=False)
    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)
    attacker = oa.attackers.PWWSAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False, progress_bar=True)
    return res


# model_path = 'echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid'
# model_path = 'lannelin/bert-imdb-1hidden'
model_path = "rohanrajpal/bert-base-multilingual-codemixed-cased-sentiment"

sa_attack(model_path)
