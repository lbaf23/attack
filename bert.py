import OpenAttack as oa
import datasets
import transformers

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }
    
# load some examples of SST-2 for evaluation
dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)
# choose the costomized classifier as the victim model

model_name = "bert-base-uncased"

tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
model = transformers.BertModel.from_pretrained(model_name)



victim = oa.classifiers.TransformersClassifier(model, tokenizer, transformers.BertModel.bert.embeddings.word_embeddings)


# choose PWWS as the attacker and initialize it with default parameters
attacker = oa.attackers.PWWSAttacker()
# prepare for attacking
attack_eval = oa.AttackEval(attacker, victim)
# launch attacks and print attack results 
res = attack_eval.eval(dataset, visualize=False)
print(res)
