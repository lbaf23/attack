import OpenAttack as oa
import datasets
import transformers


def sa_attack(model_path):
    dataset = datasets.load_from_disk("datasets/ChnSentiCorp")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, output_hidden_states=False)
    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)

    attacker = oa.attackers.PWWSAttacker(lang="chinese")
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False)
    print(res)

    attacker = oa.attackers.GeneticAttacker(lang="chinese")
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False)
    print(res)

    attacker = oa.attackers.TextFoolerAttacker(lang="chinese")
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False)
    print(res)

    return res


model_path = 'hfl/chinese-bert-wwm'


sa_attack(model_path)
