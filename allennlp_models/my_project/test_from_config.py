# from allennlp_models.my_project import ApinReader
# from allennlp_models.my_project.models import ApinSeq2Seq
# from allennlp.commands.train import train_model_from_file
#
# config_filename = "/home/alvinwatner/allennlp-models/training_config/my_project/my_model_trained_on_my_dataset.jsonnet"
# serialization_dir = "/home/alvinwatner/allennlp-models/allennlp_models/my_project/results"
#
# train_model_from_file(
#     config_filename, serialization_dir, file_friendly_logging=True, force=True
# )

from allennlp_models.my_project.dataset_readers.corenlp import CoreNLP

corenlp = CoreNLP()
text = "What is going on?"
if isinstance(text, str):
    print(text)
# intero, non_intero = corenlp.forward(text)
# print(f"intero = {intero}")
# print(f"non-intero = {non_intero}")