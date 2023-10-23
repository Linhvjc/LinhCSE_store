# import py_vncorenlp
import yaml
from pyvi import ViTokenizer

train_config = './configs/train.yml'
eval_config = './configs/eval.yml'
benchmark_config = './configs/benchmark.yml'

test_path = r'./data/benchmark/benchmark_id.csv'
corpus_path = r'./data/benchmark/corpus.json'
output_path = r'./output'

CLS_MAPPING = {
    "VoVanPhuc/bge-base-vi": "cls",
    "keepitreal/vietnamese-sbert": "cls", 
    "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base": "avg", 
    "VoVanPhuc/unsup-SimCSE-VietNamese-phobert-base": "avg", 
    "./runs/best_t10": "avg"
}

THRESHOLD_FALSE_NEGATIVE = 0.7 

def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def segment_pyvi(sentence):
    sentence["text"]= ViTokenizer.tokenize(sentence['text'])
    return sentence

def segment_pyvi_csv(sentence):
    sentence["anchor"]= ViTokenizer.tokenize(sentence['anchor'])
    sentence["positive"]= ViTokenizer.tokenize(sentence['positive'])
    return sentence