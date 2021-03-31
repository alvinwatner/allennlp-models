from typing import List, Tuple

from allennlp_models.my_project import ApinReader
from allennlp_models.my_project.models import ApinSeq2Seq
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from allennlp.models import Model

from allennlp.common.util import ensure_list
from allennlp_models.my_project.dataset_readers.apin_reader_yui import Apin_ReaderV2

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

src_train_path = "/home/alvinwatner/NQG_Interrogative_Phrases/data/toy_data/results_preprocess/squad-src-train-interro-repanswer.txt"
tgt_train_path = "/home/alvinwatner/NQG_Interrogative_Phrases/data/toy_data/results_preprocess/squad-tgt-train-interro-repanswer.txt"

src_val_path = "/home/alvinwatner/NQG_Interrogative_Phrases/data/toy_data/results_preprocess/squad-src-val-interro-repanswer.txt"
tgt_val_path = "/home/alvinwatner/NQG_Interrogative_Phrases/data/toy_data/results_preprocess/squad-tgt-val-interro-repanswer.txt"

def read_data(reader : DatasetReader) -> List[Instance]:
    train_data = ensure_list(reader.read([src_train_path, tgt_train_path]))
    dev_data = ensure_list(reader.read([src_val_path, tgt_val_path]))
    return train_data, dev_data

def build_dataset_reader() -> DatasetReader:
    return Apin_ReaderV2()

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

run_training_loop()








