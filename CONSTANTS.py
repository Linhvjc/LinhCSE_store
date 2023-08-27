import py_vncorenlp
import yaml
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/home/link/spaces/LinhCSE/assets/vncorenlp')
train_config = '/home/link/spaces/LinhCSE/configs/train.yml'
eval_config = ''

def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config