from typing import List, Dict
from allennlp_models.rc import ApinReader
from allennlp.common.util import ensure_list
from allennlp_models.rc.models import ApinSeq2Seq
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data import Instance, Vocabulary, Field
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder

from tests import FIXTURES_ROOT


def build_vocab(instances : List[Instance]) -> Vocabulary:
    print("bulding vocabulary...")
    return Vocabulary.from_instances(instances)

reader = ApinReader()
instances = reader.read(FIXTURES_ROOT / "rc" / "squad.json")
instances = ensure_list(instances)
vocab = build_vocab(instances)
vocab_size = vocab.get_vocab_size("tokens")

embedder = BasicTextFieldEmbedder(
    {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)}
)
LSTMencoder = LstmSeq2SeqEncoder(input_size=10,
                                 hidden_size=100,
                                 num_layers=1
                                 )

model = ApinSeq2Seq(vocab,
                      source_embedder = embedder,
                      encoder = LSTMencoder,
                      max_decoding_steps = 12,)
model.training = False
# fields: Dict[str, Field] = {}
vocab_size = vocab.get_vocab_size("tokens")
print(f"vocab_size = {vocab_size}")
counter = 0
for instance in instances:
    if counter == 1:
        # fields["source_tokens"] = instance['source_tokens']
        # outputs = model.forward_on_instance(Instance(fields))
        outputs = model.forward_on_instance(instance)

        # print(f"yoman3 : {instance['target_tokens']}")
        break
    else:
        counter += 1




