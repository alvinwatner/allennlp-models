from typing import List, Tuple

from allennlp_models.my_project import ApinReader
from allennlp_models.my_project.models import ApinSeq2Seq
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from allennlp.models import Model

from allennlp.common.util import ensure_list

from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
)

from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer
from tests import FIXTURES_ROOT

serialization_dir = "/home/alvinwatner/allennlp-models/allennlp_models/my_project/results"

def read_data(reader : DatasetReader) -> List[Instance]:
    train_data = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad.json"))
    dev_data = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad.json"))
    return train_data, dev_data

def build_dataset_reader() -> DatasetReader:
    return ApinReader()

def build_vocab(instances : List[Instance]) -> Vocabulary:
    print("bulding vocabulary...")
    return Vocabulary.from_instances(instances)

def build_data_loaders(train_data: List[Instance],
                       dev_data: List[Instance]) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, batch_size=8, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, batch_size=8, shuffle=False)

    return train_loader, dev_loader

def build_trainer(model: Model,
                  serialization_dir: str,
                  train_loader: DataLoader,
                  dev_loader: DataLoader) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=AdamOptimizer(parameters)
    )

    return trainer

def build_model(vocab : Vocabulary) -> Model:
    vocab_size = Vocabulary.get_vocab_size(vocab)
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)}
    )
    LSTMencoder = LstmSeq2SeqEncoder(input_size=10,
                                     hidden_size=100,
                                     num_layers=1
                                     )
    model = ApinSeq2Seq(vocab,
                        source_embedder =embedder,
                        encoder =LSTMencoder,
                        max_decoding_steps = 10)
    return model

def run_training_loop():
    dataset_reader = build_dataset_reader()
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
reader = build_dataset_reader()

train_data = ensure_list(reader.read(FIXTURES_ROOT / "rc" / "squad.json"))







