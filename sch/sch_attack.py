import OpenAttack as oa
import datasets
import transformers




def sch_attack(model_path):
    dataset = datasets.load_dataset("glue", "qqp", split="train[:10]")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)
    attacker = oa.attackers.PWWSAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=True)
    return res


model_path = "textattack/bert-base-uncased-QQP"

sch_attack(model_path)
