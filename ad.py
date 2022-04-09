import OpenAttack
import torch
import datasets
import tqdm

from OpenAttack.text_process.tokenizer import PunctTokenizer

tokenizer = PunctTokenizer()

class MyClassifier(OpenAttack.Classifier):
    def __init__(self, model, vocab) -> None:
        self.model = model
        self.vocab = vocab

    def get_prob(self, sentences):
        with torch.no_grad():
            token_ids = make_batch_tokens([
                tokenizer.tokenize(sent, pos_tagging=False) for sent in sentences
            ], self.vocab)
            token_ids = torch.LongTensor(token_ids)
            return self.model(token_ids).cpu().numpy()

    def get_pred(self, sentences):
        return self.get_prob(sentences).argmax(axis=1)


# Design a feedforward neural network as the the victim sentiment analysis model
def make_model(vocab_size):
    """
    see `tutorial - pytorch <https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#define-the-model>`__
    """
    import torch.nn as nn
    class TextSentiment(nn.Module):
        def __init__(self, vocab_size, embed_dim=32, num_class=2):
            super().__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
            self.fc = nn.Linear(embed_dim, num_class)
            self.softmax = nn.Softmax(dim=1)
            self.init_weights()

        def init_weights(self):
            initrange = 0.5
            self.embedding.weight.data.uniform_(-initrange, initrange)
            self.fc.weight.data.uniform_(-initrange, initrange)
            self.fc.bias.data.zero_()

        def forward(self, text):
            embedded = self.embedding(text, None)
            return self.softmax(self.fc(embedded))
    return TextSentiment(vocab_size)

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
        "tokens":  tokenizer.tokenize(x["sentence"], pos_tagging=False)
    }

# Choose SST-2 as the dataset
def prepare_data():
    vocab = {
        "<UNK>": 0,
        "<PAD>": 1
    }
    dataset = datasets.load_dataset("sst").map(function=dataset_mapping).remove_columns(["label", "sentence", "tree"])
    for dataset_name in ["train", "validation", "test"]:
        for inst in dataset[dataset_name]:
            for token in inst["tokens"]:
                if token not in vocab:
                    vocab[token] = len(vocab)
    return dataset["train"], dataset["validation"], dataset["test"], vocab

def make_batch_tokens(tokens_list, vocab):
    batch_x = [
        [
            vocab[token] if token in vocab else vocab["<UNK>"]
                for token in tokens
        ] for tokens in tokens_list
    ]
    max_len = max( [len(tokens) for tokens in tokens_list] )
    batch_x = [
        sentence + [vocab["<PAD>"]] * (max_len - len(sentence))
            for sentence in batch_x
    ]
    return batch_x

# Batch data
def make_batch(data, vocab):
    batch_x = make_batch_tokens(data["tokens"], vocab)
    batch_y = data["y"]
    return torch.LongTensor(batch_x), torch.LongTensor(batch_y)

# Train the victim model for one epoch
def train_epoch(model, dataset, vocab, batch_size=128, learning_rate=5e-3):
    dataset = dataset.shuffle()
    model.train()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    avg_loss = 0
    for start in range(0, len(dataset), batch_size):
        train_x, train_y = make_batch(dataset[start: start + batch_size], vocab)
        pred = model(train_x)
        loss = criterion(pred.log(), train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    return avg_loss / len(dataset)

def eval_classifier_acc(dataset, victim):
    correct = 0
    for inst in dataset:
        correct += (victim.get_pred( [inst["x"]] )[0] == inst["y"])
    return correct / len(dataset)

# Train the victim model and conduct evaluation
def train_model(model, data_train, data_valid, vocab, num_epoch=10):
    mx_acc = None
    mx_model = None
    for i in range(num_epoch):
        loss = train_epoch(model, data_train, vocab)
        victim = MyClassifier(model, vocab)
        accuracy = eval_classifier_acc(data_valid, victim)
        print("Epoch %d: loss: %lf, accuracy %lf" % (i, loss, accuracy))
        if mx_acc is None or mx_acc < accuracy:
            mx_model = model.state_dict()
    model.load_state_dict(mx_model)
    return model

# Launch adversarial attacks and generate adversarial examples
def attack(classifier, dataset, attacker = OpenAttack.attackers.PWWSAttacker()):
    attack_eval = OpenAttack.AttackEval(
        attacker,
        classifier,
    )
    correct_samples = [
        inst for inst in dataset if classifier.get_pred( [inst["x"]] )[0] == inst["y"]
    ]

    accuracy = len(correct_samples) / len(dataset)

    adversarial_samples = {
        "x": [],
        "y": [],
        "tokens": []
    }

    for result in tqdm.tqdm(attack_eval.ieval(correct_samples), total=len(correct_samples)):
        if result["success"]:
            adversarial_samples["x"].append(result["result"])
            adversarial_samples["y"].append(result["data"]["y"])
            adversarial_samples["tokens"].append(tokenizer.tokenize(result["result"], pos_tagging=False))

    attack_success_rate = len(adversarial_samples["x"]) / len(correct_samples)

    print("Accuracy: %lf%%\nAttack success rate: %lf%%" % (accuracy * 100, attack_success_rate * 100))

    return datasets.Dataset.from_dict(adversarial_samples)

def main():
    print("Loading data")
    train, valid, test, vocab = prepare_data() # Load dataset
    model = make_model(len(vocab)) # Design a victim model

    print("Training")
    trained_model = train_model(model, train, valid, vocab) # Train the victim model

    print("Generating adversarial samples (this step will take dozens of minutes)")
    victim = MyClassifier(trained_model, vocab) # Wrap the victim model
    adversarial_samples = attack(victim, train) # Conduct adversarial attacks and generate adversarial examples

    print("Adversarially training classifier")
    print(train.features)
    print(adversarial_samples.features)

    new_dataset = {
        "x": [],
        "y": [],
        "tokens": []
    }
    for it in train:
        new_dataset["x"].append( it["x"] )
        new_dataset["y"].append( it["y"] )
        new_dataset["tokens"].append( it["tokens"] )

    for it in adversarial_samples:
        new_dataset["x"].append( it["x"] )
        new_dataset["y"].append( it["y"] )
        new_dataset["tokens"].append( it["tokens"] )

    finetune_model = train_model(trained_model, datasets.Dataset.from_dict(new_dataset), valid, vocab) # Retrain the classifier with additional adversarial examples

    print("Testing enhanced model (this step will take dozens of minutes)")
    attack(victim, train) # Re-attack the victim model to measure the effect of adversarial training