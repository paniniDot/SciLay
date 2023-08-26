"""SciLay Dataset."""

import gzip
import json
import os
import random
import string

import datasets

_HOMEPAGE = ""

_CITATION = """
"""

_DESCRIPTION = """
SCILAY comprises 46,486 instances, each representing a scientific article in the biomedical domain. 
Each instance in the dataset includes the following components:
    - plain_text: Containing a plain language summary of the scientific article. This section is written in a simple and accessible language, and is intended to be understandable by a wide audience.
    - technical_text: This section contains the abstract of the scientific article. It provides a detailed and technical description of the research conducted in the article.
In addition to the textual content, each instance is associated with the following metadata:
    - Keywords: Keywords that capture the main topics and themes addressed in the article.
    - Journal: The journal in which the article is published, providing context about the source of the research.
    - DOI (Digital Object Identifier): A unique identifier for the article, facilitating easy referencing.
The main objective of the SCILAY dataset is to support the development and evaluation of text summarization models that can effectively simplify complex scientific language while retaining the essential information. 
"""

_LICENSE = "Creative Commons Attribution 4.0 International"

_SPLIT_NAMES = {datasets.Split.TRAIN: "train", datasets.Split.VALIDATION: "validation", datasets.Split.TEST: "test"}
_URL = "data/{version}/{split_name}.zip"

_DOI = "doi"
_PMCID = "pmcid"
_SUMMARY = "plain_text"
_ABSTRACT = "technical_text"
_JOURNAL = "journal"
_TOPICS = "topics"
_KEYWORDS = "keywords"

_JOURNALS = {
    "NC": "Nature Communications",
    "A": "Animals : an Open Access Journal from MDPI",
    "NIHR": "NIHR Journals Library",
    "PLGEN": "PLoS Genetics",
    "PLPAT": "PLoS Pathogens",
    "PLCB": "PLoS Computational Biology",
    "PLNTD": "PLoS Neglected Tropical Diseases",
    "B": "Biology",
    "I": "Insects",
    "EL": "eLife",
    "PLB": "PLoS Biology",
    "CB": "Communications Biology",
    "SD": "Scientific Data",
    "MBIO": "mBio",
    "C": "Cancers",
    "OTHER": "Others"
}

# Available versions:
# 1.0.0 cased raw strings.

_VERSION = "1.0.0"

class SciLayConfig(datasets.BuilderConfig):
    """BuilderConfig for SciLay."""

    def __init__(self, journals="all", version=_VERSION, **kwargs):
        """BuilderConfig for SciLay.
        Args:
            journals (str or list, default 'all'): List of journal names. Either 'all' or a combination
                of {'NC', 'A', 'NIHR', 'PLGEN', 'PLPAT', 'PLCB', 'PLNTD', 'B', 'I', 'EL', 'PLB', 'CB', 'SD', 'MBIO', 'C', 'OTHER'}.
            **kwargs: keyword arguments forwarded to super.
        """
        if isinstance(journals, str):
            journals = [journals]
        name = "+".join(journals)
        if name == "all":
            journals = list(_JOURNALS) 
        if version != _VERSION:
            name = f"{name}-{version}"
        super().__init__(name=name, version=version, **kwargs)
        self.journals = journals

class SciLay(datasets.GeneratorBasedBuilder):
    """SciLay datasets."""

    BUILDER_CONFIG_CLASS = SciLayConfig
    BUILDER_CONFIGS = [
        SciLayConfig(
            journals="all",
            description="Articles from all journals.",
        ),
    ] + [
        SciLayConfig(
            journals=k,
            description=f"Articles from journals {k}: {v}",
        )
        for k, v in sorted(_JOURNALS.items())
    ]
    DEFAULT_CONFIG_NAME = "all"
    VERSION = _VERSION

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                _DOI: datasets.Value("string"),
                _PMCID: datasets.Value("string"),
                _SUMMARY: datasets.Value("string"),
                _ABSTRACT: datasets.Value("string"),
                _JOURNAL: datasets.Value("string"),
                _TOPICS: datasets.Sequence(datasets.Value("string")),
                _KEYWORDS: datasets.Sequence(datasets.Value("string"))
            }),
            supervised_keys=(_ABSTRACT, _SUMMARY),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls = {
            split: _URL.format(version=self.config.version, split_name=split_name)
            for split, split_name in _SPLIT_NAMES.items()
        }
        dl_paths = dl_manager.download_and_extract(urls)
        paths = {
            split: [
                dl_manager.iter_files(os.path.join(dl_paths[split], split_name, code)) for code in self.config.journals
            ]
            for split, split_name in _SPLIT_NAMES.items()
        }
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"paths": paths[split]},
            )
            for split in _SPLIT_NAMES
        ]

    def _generate_examples(self, paths=None):
        """Yields examples."""
        for paths_per_journal in paths:
            for path in paths_per_journal:
                with open(path, "rb") as fin:
                    for row in fin:
                        json_obj = json.loads(row)
                        unique_key = f"{json_obj[_DOI]}_{generate_random_string()}"
                        yield unique_key, {
                            _DOI: json_obj[_DOI],
                            _PMCID: json_obj[_PMCID],
                            _SUMMARY: json_obj[_SUMMARY],
                            _ABSTRACT: json_obj[_ABSTRACT],
                            _JOURNAL: json_obj[_JOURNAL],
                            _TOPICS: json_obj[_TOPICS],
                            _KEYWORDS: json_obj[_KEYWORDS]
                        }


def generate_random_string(length=10):
    letters = string.ascii_lowercase + string.ascii_uppercase
    return ''.join(random.choice(letters) for _ in range(length))
