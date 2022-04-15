import OpenAttack as oa
import datasets
import transformers


def sa_attack(model_path, token):
    dataset = datasets.load_from_disk('datasets/sst', keep_in_memory=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_auth_token=token)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                            use_auth_token=token,
                                                                            num_labels=2,
                                                                            output_hidden_states=False)

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

    print("-->DeepWordBugAttacker Start")
    attacker = oa.attackers.DeepWordBugAttacker()
    attack_eval = oa.AttackEval(attacker, victim)
    res = attack_eval.eval(dataset, visualize=False)
    print("-->DeepWordBugAttacker Finished")

    result.append({
        "attacker": "DeepWordBugAttacker",
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


model_path = "lbaf23/bert-test"
token = ""

print(sa_attack(model_path, token))
