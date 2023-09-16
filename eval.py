import os

import torch
from transformers import AutoModel, PhobertTokenizer, AutoTokenizer
import pandas as pd
import numpy as np
from statistics import mean
from tqdm import tqdm
from sentence_transformers.util import semantic_search
from typing import Optional
from pyvi import ViTokenizer

from utils.fronts import Font
from constants import load_config, eval_config

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s:[ %(levelname)s ]:\t%(message)s ')
font = Font()


class Evaluation:
    def __init__(self,
                 args: Optional[dict] = None,
                 df_columns_name: Optional[list] = None,
                 tokenizer = None) -> None:
        self.args = args or load_config(eval_config)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(
            self.args['model_path']).to(self.device)
        self.model.eval()
        
        self.df = pd.read_csv(self.args['test_path'])
        self.df_columns_name = df_columns_name or ["Query", "Result"]
        self.y_pred = None
        self.y_true = None
        self.num_true_label = None
        
        if tokenizer:
            self.tokenizer = tokenizer
        elif 'phobert' in self.args['model_path']:
            self.tokenizer = PhobertTokenizer.from_pretrained(
                self.args['model_path'])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args['model_path'])
            logging.warning(font.warning_text(
                'You choose another model that is not "phobert". Its may cause the problem of tokenizer'))
        
        with open(self.args['corpus_path']) as user_file:
            self.corpus = eval(user_file.read())
        self.encoded_queries = None
        self.encoded_corpus = None

        print(font.inline_text('-'))
        logging.warning(font.warning_text('Please check your parameter'))
        logging.info(font.info_text(f"Test path: {self.args['test_path']}"))
        logging.info(font.info_text(
            f"Model path (or name): {self.args['model_path']}"))
        logging.info(font.info_text(
            f"Corpus path: {self.args['corpus_path']}"))
        logging.info(font.info_text(f"Batch size: {self.args['batch_size']}"))
        logging.info(font.info_text(f"Df columns name: {df_columns_name}"))
        print(font.inline_text('-'))

    def _embedding(self, batch_size: int, sentences: list) -> torch.Tensor:
        """
        Sentences embedding

        Args:
            batch_size: A number of samples in a batch
            sentences: A list of sentences to embedding

        Return:
            A list of sentences embedded
        """
        batch_size = batch_size if batch_size < len(sentences) else len(sentences)
        batchs = [sentences[i: i + batch_size]
                  for i in range(0, len(sentences), batch_size)]
        encoded_sentences = torch.tensor([]).to(self.device)
        self.model.eval()
        for batch in tqdm(batchs):
            tokenizer_batch = self.tokenizer.batch_encode_plus(batch, 
                                                               padding=True, 
                                                               truncation=True, 
                                                               return_tensors='pt')
            tokenizer_batch = tokenizer_batch.to(self.device)
            with torch.no_grad():
                encoded_batch = self.model(**tokenizer_batch)
                encoded_batch = encoded_batch.last_hidden_state.mean(dim=1).squeeze()
                encoded_batch = torch.reshape(encoded_batch, (encoded_batch.shape[0], -1)) \
                    if len(batch) != 1 else (torch.reshape(encoded_batch, (1, -1)))
                # replace with vstack or hstack
                encoded_sentences = torch.cat((encoded_sentences, encoded_batch), dim=0)
        return encoded_sentences

    def _calculate_top_k(self, y_true: list, y_pred: list, k: int) -> float:
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

    def _inference(self, k: int, test_path: Optional[str] = None, batch_size: Optional[int] = None) -> list:
        """
        Makes the prediction

        Args:
            k: Top k relevant sentences
            test_path[Optional]: Path of the test file
            batch_size[Optional]: A number of samples in a batch

        Returns:
            The prediction of samples by top k
        """
        test_path = test_path or self.test_path
        batch_size = batch_size or self.batch_size
        if self.encoded_queries is None:
            queries_id = self.df[self.df_columns_name[0]].tolist()
            queries_text = [self.corpus[id] for id in queries_id]
            queries_text = [ViTokenizer.tokenize(query) for query in queries_text]

            logging.info(f"{font.underline_text('Embedding query')}")
            self.encoded_queries = self._embedding(batch_size=batch_size, 
                                                   sentences=queries_text)
        logging.info(f"Encoded_queries size: {self.encoded_queries.shape}")

        corpus_id = np.array(list(self.corpus.keys())).flatten()
        if self.encoded_corpus is None:
            corpus_text = np.array(list(self.corpus.values())).flatten().tolist()
            corpus_text = [ViTokenizer.tokenize(c) for c in corpus_text]
            logging.info(f"{font.underline_text('Embedding corpus')}")
            self.encoded_corpus = self._embedding(batch_size=batch_size, 
                                                  sentences=corpus_text)
        logging.info(f"Encoded_corpus size: {self.encoded_corpus.shape}")

        logging.info(f"{font.underline_text('Calculate consine similarity')}")

        # queries_batchs = np.array_split(self.encoded_queries.cpu(), batch_size, axis=0)
        # corpus_batchs = np.array_split(self.encoded_corpus.cpu(), batch_size, axis=0)
        # corpus_batchs = [corpus_batch.to(self.device)
        #                  for corpus_batch in corpus_batchs]

        # queries_batchs = [query_batch.unsqueeze(1) for query_batch in queries_batchs]
        # corpus_batchs = [corpus_batch.unsqueeze(0) for corpus_batch in corpus_batchs]

        predict = semantic_search(query_embeddings=self.encoded_queries,
                                  corpus_embeddings=self.encoded_corpus,
                                  query_chunk_size=100,
                                  corpus_chunk_size=self.encoded_corpus.shape[0],
                                  top_k=k+1)
        predict = [[corpus_id[item['corpus_id']]
                    for item in sample] for sample in predict]

        logging.info(f"{font.underline_text('Calculate consine similarity [Done]')}")

        return predict

    def evaluation(self, k: Optional[int] = None, test_path: Optional[str] = None, batch_size: Optional[int] = None) -> float:
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
        test_path = test_path or self.args['test_path']
        batch_size = batch_size or self.args['batch_size']
        k = k or self.args['k']

        y_pred = self._inference(k=k, 
                                 test_path=test_path, 
                                 batch_size=batch_size)

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

        self._export_output(k=k)
        return recall_top_k

    def _export_output(self, path: Optional[str] = None, k: Optional[int] = None):
        """
        Export output of benchmark
        """
        path = path or self.args['output_path']
        k = k or self.args['k']
        output_csv_path = os.path.join(path, f"output_recall_{k}.txt")

        queries = [self.corpus[id] for id in self.df[self.df_columns_name[0]]]
        y_true_string = [[self.corpus[id] for id in sample]
                         for sample in self.y_true]
        y_pred_string = [[self.corpus[id] for id in sample]
                         for sample in self.y_pred]

        df_output = pd.DataFrame({
            'query': queries,
            'true_label': y_true_string,
            'predict_label': y_pred_string,
            'num_true_label': self.num_true_label
        })
        df_output.to_csv(output_csv_path, index=False)
        logging.info(font.info_text(f"Export output_recall_{k}.txt file Successfully"))


def main():
    evaluation = Evaluation()
    evaluation.evaluation()


if __name__ == '__main__':
    main()
