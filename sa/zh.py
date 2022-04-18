import OpenAttack as oa
import datasets
import transformers


def dataset_mapping(x):
    return {
        "x": x["review_body"],
        "y": x["stars"],
    }


def sa_attack(model_path):
    dataset = datasets.load_dataset("amazon_reviews_multi",'zh',split="train[:20]").map(function=dataset_mapping)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

    victim = oa.loadVictim("BERT.AMAZON_ZH")
    #victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)

    attacker = oa.attackers.PWWSAttacker(lang='chinese')
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=True)
    return res


sa_attack("hfl/chinese-bert-wwm-ext")
