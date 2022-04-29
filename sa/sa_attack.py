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

    #model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, output_hidden_states=False)
    model = transformers.TFAutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2,
                                                                            output_hidden_states=False)

    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)
    attacker = oa.attackers.PWWSAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False, progress_bar=True)
    return res

#model_path = 'sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english'
#model_path = 'bhadresh-savani/distilbert-base-uncased-sentiment-sst2'
model_path = 'echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid'
# model_path = 'lannelin/bert-imdb-1hidden'

sa_attack(model_path)
