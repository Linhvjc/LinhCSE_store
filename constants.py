# import py_vncorenlp
import yaml
from pyvi import ViTokenizer

train_config = './configs/train.yml'
eval_config = './configs/eval.yml'

test_path = r'./mydata/test/benchmark_id.csv'
corpus_path = r'./mydata/test/corpus.json'
output_path = r'./output'

def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def segment_pyvi(sentence):
    sentence["text"]= ViTokenizer.tokenize(sentence['text'])
    return sentence