import OpenAttack as oa
import datasets
import transformers


def csa_attack(model_path):
    dataset = datasets.load_from_disk("datasets/ChnSentiCorp")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, output_hidden_states=False)
    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)

    success = 0
    total = 0

    attacker = oa.attackers.PWWSAttacker(lang="chinese")
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False, progress_bar=True)
    print(res)
    success = success + res.get("Successful Instances")
    total = total + res.get("Total Attacked Instances")


    attacker = oa.attackers.GeneticAttacker(lang="chinese")
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False, progress_bar=True)
    print(res)
    success = success + res.get("Successful Instances")
    total = total + res.get("Total Attacked Instances")


    attacker = oa.attackers.TextFoolerAttacker(lang="chinese")
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False, progress_bar=True)
    print(res)
    success = success + res.get("Successful Instances")
    total = total + res.get("Total Attacked Instances")

    score = (total - success) * 100.0 / total
    return score


model_path = 'hfl/chinese-bert-wwm'

res = csa_attack(model_path)
print(res)
