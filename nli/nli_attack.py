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
    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings)

    attacker = oa.attackers.PWWSAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=True, progress_bar=True)

    attacker = oa.attackers.DeepWordBugAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=True, progress_bar=True)

    attacker = oa.attackers.GANAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=True, progress_bar=True)

    attacker = oa.attackers.PSOAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=True, progress_bar=True)

    attacker = oa.attackers.HotFlipAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=True, progress_bar=True)

    return res


model_path = "cross-encoder/nli-distilroberta-base"
sa_attack(model_path)
