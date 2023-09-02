# import py_vncorenlp
import yaml
from pyvi import ViTokenizer
# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/home/link/spaces/LinhCSE/assets/vncorenlp')
train_config = '/home/link/spaces/LinhCSE/configs/train.yml'
eval_config = '/home/link/spaces/LinhCSE/configs/eval.yml'

def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def segment_pyvi(sentence):
    sentence["text"]= ViTokenizer.tokenize(sentence['text'])
    return sentence