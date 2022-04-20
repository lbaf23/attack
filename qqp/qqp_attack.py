import OpenAttack as oa
import datasets
import transformers


def qqp_mapping(x):
    return {
        "x": x["question1"] + ", " + x["question2"],
        "y": x["label"],
    }



def qqp_attack(model_path):
    dataset = datasets.load_dataset("glue", "qqp", split="train[:10]").map(function=qqp_mapping)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, output_hidden_states=False)

    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings)
    attacker = oa.attackers.PWWSAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False, progress_bar=True)

    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings)
    attacker = oa.attackers.DeepWordBugAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False, progress_bar=True)

    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings)
    attacker = oa.attackers.GANAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False, progress_bar=True)

    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings)
    attacker = oa.attackers.PSOAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False, progress_bar=True)

    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.roberta.embeddings.word_embeddings)
    attacker = oa.attackers.HotFlipAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False, progress_bar=True)

    return res


model_path = 'howey/roberta-large-qqp'

qqp_attack(model_path)
