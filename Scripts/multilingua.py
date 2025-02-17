# german/multi-lingua

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

import logging
from typing import Dict, List

torch.cuda.set_device('cuda:1')

logger = logging.getLogger(__name__)

#Parent class for any reranking model
# class Rerank:
    
#     def __init__(self, model, batch_size: int = 128, **kwargs):
#         self.cross_encoder = model
#         self.batch_size = batch_size
#         self.rerank_results = {}
        
#     def rerank(self, 
#                corpus: Dict[str, Dict[str, str]], 
#                queries: Dict[str, str],
#                results: Dict[str, Dict[str, float]],
#                top_k: int) -> Dict[str, Dict[str, float]]:
        
#         sentence_pairs, pair_ids = [], []
        
#         for query_id in results:
#             if len(results[query_id]) > top_k:
#                 for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
#                     pair_ids.append([query_id, doc_id])
#                     corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
#                     sentence_pairs.append([queries[query_id], corpus_text])
            
#             else:
#                 for doc_id in results[query_id]:
#                     pair_ids.append([query_id, doc_id])
#                     corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
#                     sentence_pairs.append([queries[query_id], corpus_text])

#         #### Starting to Rerank using cross-attention
#         logging.info("Starting To Rerank Top-{}....".format(top_k))
#         rerank_scores = [float(score) for score in self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)]

#         #### Reranking results
#         self.rerank_results = {query_id: {} for query_id in results}
#         for pair, score in zip(pair_ids, rerank_scores):
#             query_id, doc_id = pair[0], pair[1]
#             self.rerank_results[query_id][doc_id] = score

#         return self.rerank_results 
#### Just some code to print debug information to stdout

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
        

dataset = 'germanquad'

logging.basicConfig(filename='answer_fusion_sum_t5_base_multilingua_'+dataset+'.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO
                    )

logger = logging.getLogger('main')
#### /print debug information to stdout



#### Download dbpedia-entity.zip dataset and unzip the dataset

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

test_corpus, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")

#test_queries = {q: test_queries[q] for q in list(test_queries.keys())[:int(20*len(test_queries)/100)]}

#test_qrels = {q: test_qrels[q] for q in list(test_queries.keys())}

#########################################
#### (1) RETRIEVE Top-100 docs using BM25
#########################################

#### Provide parameters for Elasticsearch
hostname = "localhost" #localhost
index_name = dataset
initialize = True # False

language = "german"

#extract(base_model_path = 'google-t5/t5-base', chat_model_path='razent/SciFive-base-Pubmed_PMC', output_path = './', l1=1)

#sum_model = AutoModel.from_pretrained('castorini/monot5-base-msmarco')

#sum_model = add_chat_vector(sum_model, './', 1)

lista_results_combined = []

number_of_shards = 1
model = BM25(index_name=index_name, hostname=hostname, language=language, initialize=initialize, number_of_shards=number_of_shards)
retriever = EvaluateRetrieval(model)

results_bm25_test = retriever.retrieve(test_corpus, test_queries)

for coef in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5]:    # 0.1,0.2,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9, 1.5

    cross_encoder_model = MonoT5('unicamp-dl/mt5-base-en-msmarco', token_false='▁no', token_true='▁yes')

    task_vector = TaskVector('google/mt5-base', 'airKlizz/mt5-base-wikinewssum-german')  #     airKlizz/mt5-base-wikinewssum-english

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

#########################################################################################################
# cross_encoder_model = MonoT5('castorini/monot5-base-msmarco', token_false='▁false', token_true='▁true')


# #### Or use MiniLM, TinyBERT etc. CE models (https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)
# # cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# # cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

# reranker = Rerank(cross_encoder_model, batch_size=128)

# # Rerank top-100 results using the reranker provided
# rerank_results = reranker.rerank(corpus, queries, results, top_k=100)

# #### Evaluate your retrieval using NDCG@k, MAP@K ...
# ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, [1,3,10])


# results_dic['t5-msmarco'] = {'ndcg10':ndcg['NDCG@10']}

# lista_results.append(rerank_results)


##############################################################################################################################

# cross_encoder_model = MonoT5('razent/SciFive-large-Pubmed_PMC', token_false='▁false', token_true='▁true')

# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=1)
# retriever = EvaluateRetrieval(model)

# #### Retrieve dense results (format of results is identical to qrels)


#     ######### Test #########################################################################

#     # 'castorini/monot5-base-med-msmarco'

# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
# retriever = EvaluateRetrieval(model)

# results_bm25_test = retriever.retrieve(test_corpus, test_queries)

# bm25_run_test = Run(results_bm25_test, name='BM25')

# reranker = Rerank(cross_encoder_model, batch_size=128)

# test_rerank_results = reranker.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)


#     #test_qrels_correct = {q: test_qrels[q] for q in results_bm25_test.keys()}
    
#     #test_qrels = Qrels(test_qrels_correct)  

# test_t5_rels = Run(test_rerank_results, name = 'Sci-T5')

# all_combined_test_run = fuse(
#         runs=[bm25_run_test, test_t5_rels],
#         norm="min-max",
#         method="wsum",
#         params={'weights': (0.5, 0.5)},
#     )
# all_combined_test_run.name = 'BM25 + Sci-T5' 

# lista_results_combined.append(all_combined_test_run)
# ###########################################################################################################################

# print(results_dic)

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
