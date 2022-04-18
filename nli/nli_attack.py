import OpenAttack as oa
import datasets
import transformers


def dataset_mapping(x):
    return {
        "x": x["premise"] + x["hypothesis"],
        "y": x["label"],
    }


def sa_attack(model_path):
    dataset = datasets.load_dataset("snli", split="test[:10]").map(function=dataset_mapping)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, output_hidden_states=False)
    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)
    attacker = oa.attackers.PWWSAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=True)
    return res


model_path = "sentence-transformers/bert-base-nli-mean-tokens"


sa_attack(model_path)
