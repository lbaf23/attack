import OpenAttack as oa
import datasets
import transformers


def ner_attack(model_path):
    # load some examples of SST-2 for evaluation
    dataset = datasets.load_from_disk("datasets/conll2003")
    # choose the costomized classifier as the victim model

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)

    # choose PWWS as the attacker and initialize it with default parameters
    attacker = oa.attackers.PWWSAttacker()
    # prepare for attacking
    attack_eval = oa.AttackEval(attacker, victim)
    # launch attacks and print attack results
    res = attack_eval.eval(dataset, visualize=False)
    return res


ner_attack("dslim/bert-base-NER")
