{

  "dataset_reader": {
    "type": "apin_reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "character_tokenizer": {
          "byte_encoding": "utf-8",
          "start_tokens": [259],
          "end_tokens": [260]
        },
        "min_padding_length": 5
      }
    }
  },
  "train_data_path": "/home/alvinwatner/allennlp-models/test_fixtures/rc/squad.json",
  "validation_data_path": "/home/alvinwatner/allennlp-models/test_fixtures/rc/squad.json",
  "model": {
    "type": "apin_seq2seq",
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz",
          "embedding_dim": 100,
          "trainable": false
        },
        "token_characters": {
          "type": "character_encoding",
          "embedding": {
            "num_embeddings": 262,
            "embedding_dim": 16
          },
          "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": 100,
            "ngram_filter_sizes": [5]
          },
          "dropout": 0.2
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 200,
      "hidden_size": 100,
      "num_layers": 1
    },
    "max_decoding_steps": 10,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 2
    }
  },

  "trainer": {
    "num_epochs": 4,
    "optimizer": {
      "type": "adam",
      "betas": [0.9, 0.9]
    }
  }
}
