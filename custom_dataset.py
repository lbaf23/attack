import OpenAttack as oa
import datasets
import transformers


def sst_dataset(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }

    
def imdb_dataset(x):
    return {
        "x": x["text"],
        "y": x["label"],
    }


def sa_attack(model_path):
    dataset = datasets.Dataset.from_dict({
        "x": [
            "I hate this movie.",
            "I like this apple."
        ],
        "y": [
            0, # 0 for negative
            1, # 1 for positive
        ]
    })

    dataset = datasets.load_from_disk('datasets/sst')
    #datasets = datasets.load_dataset("sst", split="train[:5]").map(function=sst_dataset)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, output_hidden_states=False)
    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)

    rate = 0
    result = []

    print("-->PWWS Start")
    attacker = oa.attackers.PWWSAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False)
    print("-->PWWS Finished")

    result.append({
        "attacker": "PWWSAttacker",
        "result": res
    })
    rate = rate + res.get("Attack Success Rate")

    print("-->TextBugger Start")
    attacker = oa.attackers.TextBuggerAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False)
    print("-->TextBugger Finished")

    result.append({
        "attacker": "TextBuggerAttacker",
        "result": res
    })

    rate = rate + res.get("Attack Success Rate")

    print("-->GAN Start")
    attacker = oa.attackers.GANAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False)
    print("-->GAN Finished")

    result.append({
        "attacker": "GANAttacker",
        "result": res
    })

    rate = rate + res.get("Attack Success Rate")

    score = (1 - rate / 3) * 100
    return score, result

model_path = "echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid"

print(sa_attack(model_path))
