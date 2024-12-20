# miracle_fr


import datasets
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from operator import itemgetter
from beir.reranking import Rerank
import pathlib, os
import datetime
import logging
import random
from beir import util, LoggingHandler
from beir.retrieval import models
from sentence_transformers import models as sentence_models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
import logging
import pathlib, os
from transformers import BertTokenizer, BertModel
import torch
from ranx import compare
from ranx import Qrels, Run, compare, fuse, optimize_fusion
from monot5 import MonoT5
from transformers import AutoModelForSeq2SeqLM
random.seed(42)
import torch
import logging
from typing import Dict, List


torch.cuda.set_device('cuda:1')

logger = logging.getLogger(__name__)

logging.basicConfig(filename='answer_fusion_sum_t5_base_miracle_french.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )

logger = logging.getLogger('main')


class TaskVector():
        def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
            """Initializes the task vector from a pretrained and a finetuned checkpoints.
            
            This can either be done by passing two state dicts (one corresponding to the
            pretrained model, and another to the finetuned model), or by directly passying in
            the task vector state dict.
            """
            if vector is not None:
                self.vector = vector
            else:
                assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
                with torch.no_grad():
                    pretrained_state_dict = AutoModelForSeq2SeqLM.from_pretrained(pretrained_checkpoint).state_dict()
                    finetuned_state_dict = AutoModelForSeq2SeqLM.from_pretrained(finetuned_checkpoint).state_dict()
                    
                    self.vector = {}
                    for key in list(pretrained_state_dict.keys()):
                        #if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        #    continue
                        self.vector[key] = finetuned_state_dict[key].cuda() - pretrained_state_dict[key].cuda()
        
        def __add__(self, other):
            """Add two task vectors together."""
            with torch.no_grad():
                new_vector = {}
                for key in self.vector:
                    if key not in other.vector:
                        print(f'Warning, key {key} is not present in both task vectors.')
                        continue
                    new_vector[key] = self.vector[key] + other.vector[key]
            return TaskVector(vector=new_vector)

        def __radd__(self, other):
            if other is None or isinstance(other, int):
                return self
            return self.__add__(other)

        def __neg__(self):
            """Negate a task vector."""
            with torch.no_grad():
                new_vector = {}
                for key in self.vector:
                    new_vector[key] = - self.vector[key]
            return TaskVector(vector=new_vector)

        def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
            """Apply a task vector to a pretrained model."""
            with torch.no_grad():
                pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_checkpoint)
                new_state_dict = {}
                pretrained_state_dict = pretrained_model.state_dict()
                for key in list(pretrained_state_dict.keys()):
                    if key not in self.vector:
                        print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                        continue
                    new_state_dict[key] = pretrained_state_dict[key].cuda() + scaling_coef * self.vector[key].cuda()
            pretrained_model.load_state_dict(new_state_dict, strict=False)
            return pretrained_model, new_state_dict





lang='fr'  # or any of the 16 languages
miracl = datasets.load_dataset('miracl/miracl', lang)

# training set:
# for data in miracl['train']:  # or 'dev', 'testA'
#   query_id = data['query_id']
#   query = data['query']
#   positive_passages = data['positive_passages']
#   negative_passages = data['negative_passages']
  
#   for entry in positive_passages: # OR 'negative_passages'
#     docid = entry['docid']
#     title = entry['title']
#     text = entry['text']


lang='fr'  # or any of the 16 languages
miracl_corpus = datasets.load_dataset('miracl/miracl-corpus', lang)

test_queries = {}

for qid in miracl['dev']:
    test_queries[qid['query_id']] =  qid['query']


test_qrels = {}

for qid in miracl['dev']:
    relevant = {}
    for doc in qid['negative_passages']:
        relevant[doc['docid']] = 0

    for doc in qid['positive_passages']:
        relevant[doc['docid']] = 1
    
    test_qrels[qid['query_id']] = relevant


hostname = "localhost" #localhost
index_name = 'miracl'
initialize = True # False

test_corpus = {}

for el in miracl_corpus['train']:
    test_corpus[el['docid']] = {'title': el['title'],'text': el['text']}

import json
with open('./corpus_miracl_fr.json', 'w') as f:
    json.dump(test_corpus, f)

language = 'french'
model = BM25(index_name=index_name, hostname=hostname, language=language, initialize=initialize)
retriever = EvaluateRetrieval(model)

results_bm25_test = retriever.retrieve(test_corpus, test_queries)

with open('./results_bm25_miracl_fr.json', 'w') as f:
    json.dump(results_bm25_test, f)

lista_results_combined = []

for coef in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:    # 0.1,0.2,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9, 1.5

    cross_encoder_model = MonoT5('unicamp-dl/mt5-base-en-msmarco', token_false='▁no', token_true='▁yes')

    task_vector = TaskVector('google/mt5-base', 'airKlizz/mt5-base-wikinewssum-french')  #     airKlizz/mt5-base-wikinewssum-english

    # Generating Extended and Multilingual Summaries with Pre-trained Transformers, LREC 2022

    sum_model, new_state_dict = task_vector.apply_to('unicamp-dl/mt5-base-en-msmarco', scaling_coef=coef)

    sum_model.to('cuda')

    #cross_encoder_model.model.parameters = sum_model.to('cuda')

    cross_encoder_model.model.load_state_dict(new_state_dict, strict=False)

    # base_model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')

    # Rerank top-100 results using the reranker provided

    #import json
    #with open('./bm25_large_test_trec_covid.json', 'w') as f:
    #    json.dump(results_bm25_test, f)

    reranker = Rerank(cross_encoder_model, batch_size=128)

    test_rerank_results = reranker.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)

    bm25_run_test = Run(results_bm25_test, name='BM25')

    #test_qrels_correct = {q: test_qrels[q] for q in results_bm25_test.keys()}
    
    #test_qrels = Qrels(test_qrels_correct)  

    #with open('./t5_large_test_trec_covid.json', 'w') as f:
    #    json.dump(test_rerank_results, f)

    test_t5_rels = Run(test_rerank_results, name = 'MT5-sum-lambda-'+str(coef))

    all_combined_test_run = fuse(
        runs=[bm25_run_test, test_t5_rels],
        norm="min-max",
        method="wsum",
        params={'weights': (0.5, 0.5)},
    )
    all_combined_test_run.name = 'BM25 + MT5-sum-lambda-' + str(coef)

    lista_results_combined.append(all_combined_test_run)



lista_results_combined.append(bm25_run_test)

test_qrels_correct = {q: test_qrels[q] for q in results_bm25_test.keys()}
    
test_qrels = Qrels(test_qrels_correct)  

report = compare(
        qrels=test_qrels,
        runs=lista_results_combined,
        metrics=['precision@1', 'ndcg@3', 'ndcg@10', 'recall@100', 'map@100'],
        max_p=0.01/3  # P-value threshold, 3 tests
    )

logger.info(f'\n{report}')

print(report)

lista_results_combined = []

###########################################################################################################################àà

for coef in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:    # 0.1,0.2,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9, 1.5

    cross_encoder_model = MonoT5('unicamp-dl/mt5-base-en-msmarco', token_false='▁no', token_true='▁yes')

    task_vector = TaskVector('airKlizz/mt5-base-wikinewssum-english', 'airKlizz/mt5-base-wikinewssum-french')  #     airKlizz/mt5-base-wikinewssum-english

    # Generating Extended and Multilingual Summaries with Pre-trained Transformers, LREC 2022

    sum_model, new_state_dict = task_vector.apply_to('unicamp-dl/mt5-base-en-msmarco', scaling_coef=coef)

    sum_model.to('cuda')

    #cross_encoder_model.model.parameters = sum_model.to('cuda')

    cross_encoder_model.model.load_state_dict(new_state_dict, strict=False)

    # base_model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')

    # Rerank top-100 results using the reranker provided

    #import json
    #with open('./bm25_large_test_trec_covid.json', 'w') as f:
    #    json.dump(results_bm25_test, f)

    reranker = Rerank(cross_encoder_model, batch_size=128)

    test_rerank_results = reranker.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)

    bm25_run_test = Run(results_bm25_test, name='BM25')

    #test_qrels_correct = {q: test_qrels[q] for q in results_bm25_test.keys()}
    
    #test_qrels = Qrels(test_qrels_correct)  

    #with open('./t5_large_test_trec_covid.json', 'w') as f:
    #    json.dump(test_rerank_results, f)

    test_t5_rels = Run(test_rerank_results, name = 'MT5-sum-lambda-'+str(coef))

    all_combined_test_run = fuse(
        runs=[bm25_run_test, test_t5_rels],
        norm="min-max",
        method="wsum",
        params={'weights': (0.5, 0.5)},
    )
    all_combined_test_run.name = 'BM25 + MT5-sum-lambda-' + str(coef)

    lista_results_combined.append(all_combined_test_run)



lista_results_combined.append(bm25_run_test)

test_qrels_correct = {q: test_qrels[q] for q in results_bm25_test.keys()}
    
test_qrels = Qrels(test_qrels_correct)  

report = compare(
        qrels=test_qrels,
        runs=lista_results_combined,
        metrics=['precision@1', 'ndcg@3', 'ndcg@10', 'recall@100', 'map@100'],
        max_p=0.01/3  # P-value threshold, 3 tests
    )

logger.info(f'\n{report}')

print(report)
