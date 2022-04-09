import OpenAttack as oa
import datasets
import transformers

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }
    
# attack a model
def sa_attack(model_path):

    # load some examples of SST-2 for evaluation
    dataset = datasets.load_dataset("sst", split="train[:5]").map(function=dataset_mapping)
    # choose the costomized classifier as the victim model

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, output_hidden_states=False)

    victim = oa.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)

    # choose PWWS as the attacker and initialize it with default parameters
    attacker = oa.attackers.PWWSAttacker()
    # prepare for attacking
    attack_eval = oa.AttackEval(attacker, victim)
    # launch attacks and print attack results 
    res = attack_eval.eval(dataset, visualize=False)
    return res
