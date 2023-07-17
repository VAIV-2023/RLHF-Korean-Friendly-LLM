"""Korean Balanced Evaluation of Significant Tasks"""


import csv

import pandas as pd

import datasets


_CITATAION = """\
@misc{https://doi.org/10.48550/arxiv.2204.04541,
  doi = {10.48550/ARXIV.2204.04541},
  url = {https://arxiv.org/abs/2204.04541},
  author = {Kim, Dohyeong and Jang, Myeongjun and Kwon, Deuk Sin and Davis, Eric},
  title = {KOBEST: Korean Balanced Evaluation of Significant Tasks},
  publisher = {arXiv},
  year = {2022},
}
"""

_DESCRIPTION = """\
    The dataset contains data for KoBEST dataset
"""

_URL = "https://github.com/SKT-LSL/KoBEST_datarepo/raw/main"


_DATA_URLS = {
    "boolq": {
        "train": _URL + "/v1.0/BoolQ/train.tsv",
        "dev": _URL + "/v1.0/BoolQ/dev.tsv",
        "test": _URL + "/v1.0/BoolQ/test.tsv",
    },
    "copa": {
        "train": _URL + "/v1.0/COPA/train.tsv",
        "dev": _URL + "/v1.0/COPA/dev.tsv",
        "test": _URL + "/v1.0/COPA/test.tsv",
    },
    "sentineg": {
        "train": _URL + "/v1.0/SentiNeg/train.tsv",
        "dev": _URL + "/v1.0/SentiNeg/dev.tsv",
        "test": _URL + "/v1.0/SentiNeg/test.tsv",
        "test_originated": _URL + "/v1.0/SentiNeg/test.tsv",
    },
    "hellaswag": {
        "train": _URL + "/v1.0/HellaSwag/train.tsv",
        "dev": _URL + "/v1.0/HellaSwag/dev.tsv",
        "test": _URL + "/v1.0/HellaSwag/test.tsv",
    },
    "wic": {
        "train": _URL + "/v1.0/WiC/train.tsv",
        "dev": _URL + "/v1.0/WiC/dev.tsv",
        "test": _URL + "/v1.0/WiC/test.tsv",
    },
}

_LICENSE = "CC-BY-SA-4.0"


class KoBESTConfig(datasets.BuilderConfig):
    """Config for building KoBEST"""

    def __init__(self, description, data_url, citation, url, **kwargs):
        """
        Args:
            description: `string`, brief description of the dataset
            data_url: `dictionary`, dict with url for each split of data.
            citation: `string`, citation for the dataset.
            url: `string`, url for information about the dataset.
            **kwrags: keyword arguments frowarded to super
        """
        super(KoBESTConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.description = description
        self.data_url = data_url
        self.citation = citation
        self.url = url


class KoBEST(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        KoBESTConfig(name=name, description=_DESCRIPTION, data_url=_DATA_URLS[name], citation=_CITATAION, url=_URL)
        for name in ["boolq", "copa", 'sentineg', 'hellaswag', 'wic']
    ]
    BUILDER_CONFIG_CLASS = KoBESTConfig

    def _info(self):
        features = {}
        if self.config.name == "boolq":
            labels = ["False", "True"]
            features["paragraph"] = datasets.Value("string")
            features["question"] = datasets.Value("string")
            features["label"] = datasets.features.ClassLabel(names=labels)

        if self.config.name == "copa":
            labels = ["alternative_1", "alternative_2"]
            features["premise"] = datasets.Value("string")
            features["question"] = datasets.Value("string")
            features["alternative_1"] = datasets.Value("string")
            features["alternative_2"] = datasets.Value("string")
            features["label"] = datasets.features.ClassLabel(names=labels)

        if self.config.name == "wic":
            labels = ["False", "True"]
            features["word"] = datasets.Value("string")
            features["context_1"] = datasets.Value("string")
            features["context_2"] = datasets.Value("string")
            features["label"] = datasets.features.ClassLabel(names=labels)

        if self.config.name == "hellaswag":
            labels = ["ending_1", "ending_2", "ending_3", "ending_4"]

            features["context"] = datasets.Value("string")
            features["ending_1"] = datasets.Value("string")
            features["ending_2"] = datasets.Value("string")
            features["ending_3"] = datasets.Value("string")
            features["ending_4"] = datasets.Value("string")
            features["label"] = datasets.features.ClassLabel(names=labels)

        if self.config.name == "sentineg":
            labels = ["negative", "positive"]
            features["sentence"] = datasets.Value("string")
            features["label"] = datasets.features.ClassLabel(names=labels)

        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=datasets.Features(features), homepage=_URL, citation=_CITATAION
        )

    def _split_generators(self, dl_manager):

        train = dl_manager.download_and_extract(self.config.data_url["train"])
        dev = dl_manager.download_and_extract(self.config.data_url["dev"])
        test = dl_manager.download_and_extract(self.config.data_url["test"])

        if self.config.data_url.get("test_originated"):
            test_originated = dl_manager.download_and_extract(self.config.data_url["test_originated"])

            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train, "split": "train"}),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev, "split": "dev"}),
                datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test, "split": "test"}),
                datasets.SplitGenerator(name="test_originated", gen_kwargs={"filepath": test_originated, "split": "test_originated"}),
            ]

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train, "split": "train"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev, "split": "dev"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test, "split": "test"}),
        ]

    def _generate_examples(self, filepath, split):
        if self.config.name == "boolq":
            df = pd.read_csv(filepath, sep="\t")
            df = df.dropna()
            df = df[['Text', 'Question', 'Answer']]

            df = df.rename(columns={
                'Text': 'paragraph',
                'Question': 'question',
                'Answer': 'label',
            })
            df['label'] = [0 if str(s) == 'False' else 1 for s in df['label'].tolist()]

        elif self.config.name == "copa":
            df = pd.read_csv(filepath, sep="\t")
            df = df.dropna()
            df = df[['sentence', 'question', '1', '2', 'Answer']]

            df = df.rename(columns={
                'sentence': 'premise',
                'question': 'question',
                '1': 'alternative_1',
                '2': 'alternative_2',
                'Answer': 'label',
            })
            df['label'] = [i-1 for i in df['label'].tolist()]

        elif self.config.name == "wic":
            df = pd.read_csv(filepath, sep="\t")
            df = df.dropna()
            df = df[['Target', 'SENTENCE1', 'SENTENCE2', 'ANSWER']]

            df = df.rename(columns={
                'Target': 'word',
                'SENTENCE1': 'context_1',
                'SENTENCE2': 'context_2',
                'ANSWER': 'label',
            })
            df['label'] = [0 if str(s) == 'False' else 1 for s in df['label'].tolist()]

        elif self.config.name == "hellaswag":
            df = pd.read_csv(filepath, sep="\t")
            df = df.dropna()
            df = df[['context', 'choice1', 'choice2', 'choice3', 'choice4', 'label']]

            df = df.rename(columns={
                'context': 'context',
                'choice1': 'ending_1',
                'choice2': 'ending_2',
                'choice3': 'ending_3',
                'choice4': 'ending_4',
                'label': 'label',
            })

        elif self.config.name == "sentineg":
            df = pd.read_csv(filepath, sep="\t")
            df = df.dropna()

            if split == "test_originated":
                df = df[['Text_origin', 'Label_origin']]

                df = df.rename(columns={
                    'Text_origin': 'sentence',
                    'Label_origin': 'label',
                })
            else:
                df = df[['Text', 'Label']]

                df = df.rename(columns={
                    'Text': 'sentence',
                    'Label': 'label',
                })

        else:
            raise NotImplementedError

        for id_, row in df.iterrows():
            features = {key: row[key] for key in row.keys()}
            yield id_, features


if __name__ == "__main__":
    dataset = datasets.load_dataset("kobest_v1.py", 'sentineg', ignore_verifications=True)
    ds = dataset['test_originated']
    print(ds)

    # for task in ['boolq', 'copa', 'wic', 'hellaswag', 'sentineg']:
    #     dataset = datasets.load_dataset("kobest_v1.py", task, ignore_verifications=True)
    #     print(dataset)
    #     print(dataset['train']['label'])


