from typing import Iterable, List, Tuple
from tests import FIXTURES_ROOT

from allennlp.common import Params
from allennlp.common.util import ensure_list

from allennlp_models.rc.models import BidirectionalAttentionFlow as bidaf

from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.matrix_attention import LinearMatrixAttention
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder

from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
)

from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer

serialization_dir = "/home/alvinwatner/allennlp-models/allennlp_models"

def build_vocab(instances : Iterable[Instance]) -> Vocabulary:
    print("Building vocabulary...")
    return Vocabulary.from_instances(instances)

def build_model(vocab : Vocabulary) -> Model:
    print("Building the model...")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=200, num_embeddings=vocab_size)}
    )

    phrase_layer = LstmSeq2SeqEncoder(input_size = 200,
                               hidden_size = 100,
                               num_layers = 1,
                               bidirectional=True)


    matrix_attn    = LinearMatrixAttention(200, 200, "x,y,x*y")
    modeling_layer = LstmSeq2SeqEncoder(input_size = 800,
                                   hidden_size = 100,
                                   num_layers = 2,
                                   bidirectional = True,
                                   dropout = 0.2
                                   )
    span_end_encoder = LstmSeq2SeqEncoder(input_size = 1400,
                                          hidden_size = 100,
                                          num_layers = 1
                                          )

    return bidaf(vocab,
                 embedder,
                 2,
                 phrase_layer,
                 matrix_attn,
                 modeling_layer,
                 span_end_encoder
                 )

def build_dataset_reader() -> DatasetReader:
    return DatasetReader.from_params(Params({"type": "squad1"}))

def read_data(reader : DatasetReader) ->  List[Instance]:
    print("Reading data...")
    train_data = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad.json"))
    dev_data   = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad.json"))
    return train_data, dev_data

'''Let's create a data loader, indeed a simple one'''
def build_data_loaders(train_data : List[Instance],
                       dev_data   : List[Instance]) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, batch_size = 8, shuffle = True)
    dev_loader   = SimpleDataLoader(dev_data, batch_size = 8, shuffle = False)

    return train_loader, dev_loader

def build_trainer(model             : Model,
                  serialization_dir : str,
                  train_loader      : DataLoader,
                  dev_loader        : DataLoader) -> Trainer:

    parameters = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    trainer    = GradientDescentTrainer(
        model                  = model,
        serialization_dir      = serialization_dir,
        data_loader            = train_loader,
        validation_data_loader = dev_loader,
        num_epochs             = 5,
        optimizer              = AdamOptimizer(parameters)
    )

    return trainer

def run_training_loop():
    dataset_reader       = build_dataset_reader()
    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)

    train_loader, dev_loader = build_data_loaders(train_data, dev_data)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)
    
    trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
    print("Starting training")
    trainer.train()
    print("Finished training")

    return model, dataset_reader

# run_training_loop()


# Today task
#1) modify the input
#2) build your own model
#3) train on toy data

# Today goals
#1) modify the input (check)
#2) build your own model (check)
#3) train on toy data (check)

dataset_reader = build_dataset_reader()
train_data, _ = read_data(dataset_reader)

vocab = build_vocab(train_data)
# model = build_model(vocab)