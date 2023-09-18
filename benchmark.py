from typing import Optional
from transformers import AutoModel, PhobertTokenizer, AutoTokenizer
import torch
import tqdm 
from datasets import load_dataset
from pyvi import ViTokenizer
from sentence_transformers.util import semantic_search
import pandas as pd

from constants import load_config, benchmark_config
from eval import Evaluation


class ModifyEvaluation(Evaluation):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
    
    def embedding(self, batch_size: int, sentences: list):
        return self._embedding(batch_size, sentences)
    
    

class Benchmark:
    def __init__(self) -> None:
        self.args = load_config(benchmark_config)
        self.raw = self.args['raw']
        self.k = self.args['k']
        self.thredsold = self.args['thredsold']
        self.output_path = self.args['output_path']
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(self.args['model_path']).to(self.device)
        self.model.eval()
        if 'phobert' in self.args['model_path']:
            self.tokenizer = PhobertTokenizer.from_pretrained(
                self.args['model_path'])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args['model_path'])
        self.modify_eval = ModifyEvaluation(self.tokenizer)
    

    def create_benchmark(self):
        datasets = load_dataset('text', data_files={'sentences': self.raw})
        datasets = datasets.map(lambda sample: {'segment': ViTokenizer.tokenize(sample['text'])})
        encoded_queries = self.modify_eval.embedding(batch_size=128,
                                                     sentences=datasets['sentences']['segment'])
        encoded_corpus = encoded_queries.clone().detach()

        predict = semantic_search(query_embeddings=encoded_queries,
                                  corpus_embeddings=encoded_corpus,
                                  query_chunk_size=100,
                                  corpus_chunk_size=encoded_corpus.shape[0],
                                  top_k=self.k+1)
        
        final_result = {'Query':[], 'Result':[], 'Number of relevant query': []}
        for i, sample in enumerate(predict):
            result = []
            query_sentence = datasets['sentences'][i]['text']
            
            for sentence in sample:
                retrieval_sentence = datasets['sentences'][sentence['corpus_id']]['text']
                if sentence['score'] >= self.thredsold and \
                retrieval_sentence.lower() != query_sentence.lower() and \
                retrieval_sentence.lower() not in result:
                    result.append(datasets['sentences'][sentence['corpus_id']]['text'])
            
            if len(result) > 0:
                final_result['Query'].append(query_sentence)
                final_result['Result'].append(result)
                final_result['Number of relevant query'].append(len(result))

        df = pd.DataFrame(final_result)
        df.to_csv(self.output_path, index=False)
        
if __name__ == '__main__':
    benchmark = Benchmark()
    benchmark.create_benchmark()
    
