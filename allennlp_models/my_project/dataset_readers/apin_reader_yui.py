from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, SpacyTokenizer

import logging
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)

class Apin_ReaderV2(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or SpacyTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, source_text : str,
                         target_text : str) -> Instance:

        source_tokens = self.tokenizer.tokenize(source_text)
        target_tokens = self.tokenizer.tokenize(target_text)
        source_field = TextField(source_tokens, self.token_indexers)
        target_field = TextField(target_tokens, self.token_indexers)

        return Instance({"source_tokens": source_field, "target_tokens": target_field})

    def _read(self, data_path : List[str]) -> Iterable[Instance]:
        src_train_path = data_path[0]
        tgt_train_path = data_path[1]

        with open(src_train_path) as src, open(tgt_train_path) as tgt:
            for line_src, line_tgt in zip(src, tgt):
                yield self.text_to_instance(line_src, line_tgt)

#
# src_train_path = "/home/alvinwatner/NQG_Interrogative_Phrases/data/squad-src-train-interro-repanswer.txt"
# tgt_train_path = "/home/alvinwatner/NQG_Interrogative_Phrases/data/squad-tgt-train-interro-repanswer.txt"
#
# src_train_path = "/home/alvinwatner/NQG_Interrogative_Phrases/data/toy_data/results_preprocess/squad-src-train-interro-repanswer.txt"
# tgt_train_path = "/home/alvinwatner/NQG_Interrogative_Phrases/data/toy_data/results_preprocess/squad-tgt-train-interro-repanswer.txt"
# dataset_reader = Apin_ReaderV2(max_tokens=64)
# instances = list(dataset_reader.read([src_train_path, tgt_train_path]))
#
# for instance in instances[:10]:
#     print(instance)
