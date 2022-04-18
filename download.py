import datasets


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


def download_sst(len):
    dataset = datasets.load_dataset("sst", split="train[:%s]" % len).map(function=sst_dataset)
    dataset.save_to_disk('datasets/sst')


def download_conll(len):
    dataset = datasets.load_dataset("conll2003", split="train[:%s]" % len)
    dataset.save_to_disk('datasets/conll2003')



# download_sst(20)
download_conll(20)

