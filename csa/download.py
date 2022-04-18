import datasets


def ChnSentiCorp_dataset(x):
    return {
        "x": x["text"],
        "y": x["label"],
    }


def download_ChnSentiCorp(len):
    dataset = datasets.load_dataset("seamew/ChnSentiCorp", split="train[:%s]" % len).map(function=ChnSentiCorp_dataset)
    dataset.save_to_disk('datasets/ChnSentiCorp')


def download_nltk():
    import nltk
    nltk.download('omw-1.4')


download_ChnSentiCorp(10)
