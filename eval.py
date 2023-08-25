# System libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

# Installed libraries
from transformers import pipeline, AutoModel, PhobertTokenizer, AutoTokenizer
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean
from torch.nn.functional import cosine_similarity
import torch.nn as nn
from utils.fronts import Font
from tqdm import tqdm
import time
import py_vncorenlp
from sentence_transformers.util import semantic_search


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:[ %(levelname)s ]:\t%(message)s ')
font = Font()
class Evaluation:
    def __init__(self, 
                 test_path:str, 
                 model_path:str, 
                 corpus_path: dict, 
                 batch_size: int = 64, 
                 df_columns_name:list = ["Query", "Result"]) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(model_path).to(self.device) 
        self.model.eval()
        self.test_path = test_path
        self.batch_size = batch_size
        self.df = pd.read_csv(test_path)
        self.df_columns_name = df_columns_name
        self.y_pred = None
        self.y_true = None
        self.num_true_label = None
        if 'phobert' in model_path:
            self.tokenizer = PhobertTokenizer.from_pretrained(model_path) 
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logging.warning(font.warning_text('You choose another model that is not "phobert". Its may cause the problem of tokenizer'))
        with open(corpus_path) as user_file:
            self.corpus = eval(user_file.read())
        
        print(font.inline_text('-'))
        logging.warning(font.warning_text('Please check your parameter'))
        logging.info(font.info_text(f"Test path: {test_path}"))
        logging.info(font.info_text(f"Model path (or name): {model_path}"))
        logging.info(font.info_text(f"Corpus path: {corpus_path}"))
        logging.info(font.info_text(f"Batch size: {batch_size}"))
        logging.info(font.info_text(f"Df columns name: {df_columns_name}"))
        print(font.inline_text('-'))
    
    def _embedding(self, batch_size:int, sentences:list):
        """
        Sentences embedding
        
        Args:
            batch_size: A number of samples in a batch
            sentences: A list of sentences to embedding
        
        Returns:
            A list of sentences embedded
        """
        batch_size = batch_size if batch_size < len(sentences) else len(sentences)
        batchs = [sentences[i: i+ batch_size] for i in range(0, len(sentences), batch_size)] 
        encoded_sentences = torch.tensor([]).to(self.device)
        self.model.eval()
        for batch in tqdm(batchs): 
            tokenizer_batch = self.tokenizer.batch_encode_plus(batch, padding=True, truncation=True, return_tensors='pt')
            tokenizer_batch = tokenizer_batch.to(self.device)
            with torch.no_grad():
                encoded_batch = self.model(**tokenizer_batch)
                encoded_batch = encoded_batch.last_hidden_state.mean(dim=1).squeeze()
                encoded_batch = torch.reshape(encoded_batch, (encoded_batch.shape[0],-1)) \
                if len(batch) !=1 else (torch.reshape(encoded_batch, (1,-1)))
                
                encoded_sentences = torch.cat((encoded_sentences, encoded_batch), dim = 0)   #replace with vstack or hstack
        return encoded_sentences
                
    def _calculate_top_k(self, y_true:list , y_pred:list, k:int) -> float:
        """
        Calculate the score of top k retrieval
        
        Args:
            y_true: List of true label
            y_pred: List of predict label
            k: Top k relevant sentences
        
        Returns:
            The average score of top k 
        """
        
        logging.info(font.underline_text(f"Calculate Recall@{k}"))
        result = []
        num_true_label = []
        for i, sample in enumerate(y_pred):
            total_true_label = len(y_true[i])
            true_positive = len(set(sample) & set(y_true[i]))
            result.append(true_positive/total_true_label)
            num_true_label.append(f"{true_positive}/{total_true_label}")
        self.num_true_label = num_true_label
        return mean(result)
    
    def _inference(self, k:int, test_path:str = None, batch_size:int = None) -> list:
        """
        Makes the prediction
        
        Args:
            k: Top k relevant sentences
            test_path[Optional]: Path of the test file
            batch_size[Optional]: A number of samples in a batch
        
        Returns:
            The prediction of samples by top k
        """
        import CONSTANTS
        if not test_path: 
            test_path = self.test_path
        if not batch_size: 
            batch_size = self.batch_size
        
        queries_id = self.df[self.df_columns_name[0]].tolist()
        queries_text = [self.corpus[id] for id in queries_id]
        queries_text = [" ".join(CONSTANTS.rdrsegmenter.word_segment(query)) for query in queries_text]
        
        logging.info(f"{font.underline_text('Embedding query')}")
        encoded_queries = self._embedding(batch_size=batch_size, sentences=queries_text)
        logging.info(f"Encoded_queries size: {encoded_queries.shape}")
        
        corpus_text = np.array(list(self.corpus.values())).flatten().tolist()
        corpus_text = [" ".join(CONSTANTS.rdrsegmenter.word_segment(c)) for c in corpus_text]
        corpus_id = np.array(list(self.corpus.keys())).flatten()
        
        logging.info(f"{font.underline_text('Embedding corpus')}")
        encoded_corpus = self._embedding(batch_size=batch_size, sentences=corpus_text)
        logging.info(f"Encoded_corpus size: {encoded_corpus.shape}")
        
        logging.info(f"{font.underline_text('Calculate consine similarity')}")
        
        queries_batchs = np.array_split(encoded_queries.cpu(), batch_size, axis=0)
        corpus_batchs = np.array_split(encoded_corpus.cpu(), batch_size, axis=0)
        corpus_batchs = [corpus_batch.to(self.device) for corpus_batch in corpus_batchs]
        
        queries_batchs = [query_batch.unsqueeze(1) for query_batch in queries_batchs]
        corpus_batchs = [corpus_batch.unsqueeze(0) for corpus_batch in corpus_batchs]
        
        # print(corpus_batchs[0].device)
        # result = torch.tensor([]).to(self.device)
        # for query_batch in tqdm(queries_batchs):
        #     batch_result = torch.tensor([]).to(self.device)
        #     for corpus_batch in corpus_batchs:
        #         similarity_score = cosine_similarity(query_batch.to(self.device), corpus_batch, dim = -1)
        #         similarity_score.to(self.device)
        #         batch_result = torch.cat((batch_result, similarity_score), dim=1)
        #     result = torch.cat((result, batch_result), axis=0)
        # indices = np.argsort(-np.array(result.cpu()), axis=1)[:, 1:k+1]
        # predict = corpus_id[indices].tolist()
        
        predict = semantic_search(query_embeddings = encoded_queries,
                                  corpus_embeddings  = encoded_corpus,
                                  query_chunk_size = 100,
                                  corpus_chunk_size = encoded_corpus.shape[0],
                                  top_k = k+1)
        predict = [[corpus_id[item['corpus_id']] for item in sample] for sample in predict]
            
        logging.info(f"{font.underline_text('Calculate consine similarity [Done]')}")
        
        return predict
        
    def evaluation(self, k:int, test_path:str = None, batch_size:int = None) -> float:
        """
        Makes the Evaluation
        
        Args:
            k: Top k relevant sentences
            test_path[Optional]: Path of the test file
            batch_size[Optional]: A number of samples in a batch
        
        Returns:
            The evaluation of samples by top k
        """
        logging.info(f"{font.bool_text('Evaluation')}")
        if not test_path: test_path = self.test_path
        if not batch_size: batch_size = self.batch_size
        
        y_pred = self._inference(k = k, test_path=test_path, batch_size = batch_size)
        
        y_true = self.df[self.df_columns_name[1]].tolist()
        y_true = [eval(label) for label in y_true]
        
        self.y_true = y_true
        self.y_pred = y_pred
        
        recall_top_k = self._calculate_top_k(y_true=y_true, y_pred=y_pred, k=k)
        print(font.bool_text(f"""
              =========================================================================
              |                             Recall@{k}: {recall_top_k}                |
              =========================================================================
              """))
        return recall_top_k
    
    def export_output(self):
        def _mapping_sample(sample):
            result = [self.corpus[id] for id in sample]
            return result
        
        queries = [self.corpus[id] for id in self.df[self.df_columns_name[0]]]
        y_true_string = [_mapping_sample(sample) for sample in self.y_true]
        y_pred_string = [_mapping_sample(sample) for sample in self.y_pred]
        
        df_output = pd.DataFrame({
            'query': queries,
            'true_label': y_true_string,
            'predict_label':y_pred_string,
            'num_true_label': self.num_true_label
        })
        df_output.to_csv('/home/link/spaces/LinhCSE/output.csv', index=False)
        logging.info(font.info_text(f"Export output.csv file Successfully"))

if __name__ == '__main__':
    # model_path = r'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base'
    # model_path = r'VoVanPhuc/unsup-SimCSE-VietNamese-phobert-base'
    # model_path = r'keepitreal/vietnamese-sbert'
    model_path = r'/home/link/spaces/LinhCSE/runs/best_24_08'
    test_path = r'/home/link/spaces/LinhCSE/mydata/test/benchmark_id.csv'
    corpus_path = r'/home/link/spaces/LinhCSE/mydata/test/corpus.json'
    
    evaluation = Evaluation(test_path=test_path,
                            model_path=model_path,
                            corpus_path=corpus_path,
                            batch_size= 512
                            )
    result = evaluation.evaluation(k = 10)
    evaluation.export_output()