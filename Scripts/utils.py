# utils

import json
import logging
import os
import random
import subprocess
from beir.retrieval import models
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
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, AutoModelForSequenceClassification, AutoModelForCausalLM
import logging
import pathlib, os
from transformers import BertTokenizer, BertModel
import torch
from ranx import compare
from ranx import Qrels, Run, compare, fuse, optimize_fusion
from monot5 import MonoT5
from transformers import AutoModelForSeq2SeqLM
import numpy as np
import torch
import tqdm
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import Dict, List

logger = logging.getLogger(__name__)

def seed_everything(seed: int):
    logger.info(f'Setting global random seed to {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True




class TaskVectorT5():
        def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, ablation = None, layer = None):
            """Initializes the task vector from a pretrained and a finetuned checkpoints.
            
            This can either be done by passing two state dicts (one corresponding to the
            pretrained model, and another to the finetuned model), or by directly passying in
            the task vector state dict.
            """
            self.ablation = ablation
            self.layer = layer

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
                    if self.ablation is not None:
                        if self.ablation=='add':
                            if self.layer in key:
                                new_state_dict[key] = pretrained_state_dict[key].cuda() + scaling_coef * self.vector[key].cuda()
                            else:
                                new_state_dict[key] = pretrained_state_dict[key].cuda()
                        elif self.ablation=='remove':
                            if self.layer in key:
                                new_state_dict[key] = pretrained_state_dict[key].cuda()
                            else:
                                new_state_dict[key] = pretrained_state_dict[key].cuda() + scaling_coef * self.vector[key].cuda()
                    else:
                        new_state_dict[key] = pretrained_state_dict[key].cuda() + scaling_coef * self.vector[key].cuda()
            pretrained_model.load_state_dict(new_state_dict, strict=False)
            return pretrained_model, new_state_dict


class TaskVectorLLama():
        def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, ablation = None, layer = None):
            """Initializes the task vector from a pretrained and a finetuned checkpoints.
            
            This can either be done by passing two state dicts (one corresponding to the
            pretrained model, and another to the finetuned model), or by directly passying in
            the task vector state dict.
            """
            self.ablation = ablation
            self.layer = layer
            if vector is not None:
                self.vector = vector
            else:
                assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
                with torch.no_grad():
                    pretrained_state_dict = AutoModelForCausalLM.from_pretrained(pretrained_checkpoint).state_dict()
                    finetuned_state_dict = AutoModelForCausalLM.from_pretrained(finetuned_checkpoint).state_dict()
                    
                    self.vector = {}
                    for key in list(pretrained_state_dict.keys()):
                        #if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        #    continue
                        self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
        
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
                pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_checkpoint)
                new_state_dict = {}
                pretrained_state_dict = pretrained_model.state_dict()
                for key in list(pretrained_state_dict.keys()):
                    if key not in self.vector:
                        print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                        continue
                    if self.ablation is not None:
                        if self.ablation=='add':
                            if self.layer in key:
                                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
                            else:
                                new_state_dict[key] = pretrained_state_dict[key] #.cuda()
                        elif self.ablation=='remove':
                            if self.layer in key:
                                new_state_dict[key] = pretrained_state_dict[key] #.cuda()
                            else:
                                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
                    else:
                        new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
                    #new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
            pretrained_model.load_state_dict(new_state_dict, strict=False)
            return pretrained_model, new_state_dict




class TaskVectorBERT():
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
                    pretrained_state_dict = AutoModel.from_pretrained(pretrained_checkpoint).state_dict()
                    finetuned_state_dict = AutoModel.from_pretrained(finetuned_checkpoint).state_dict()
                    
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
                pretrained_model = CrossEncoder(pretrained_checkpoint, max_length=512)
                new_state_dict = {}
                #for mod in ['query','doc']:
                    # if mod=='query': 
                    #     pretrained_state_dict = pretrained_model.model.q_model.state_dict()
                    #     print(pretrained_model.model.q_model)
                    # elif mod=='doc':
                pretrained_state_dict = pretrained_model.model.model.state_dict()
                for key in list(pretrained_state_dict.keys()):
                    if key.replace('bert.','') not in self.vector:
                        print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                        continue
                    new_state_dict[key] = pretrained_state_dict[key].cuda() + scaling_coef * self.vector[key.replace('bert.','')].cuda()
                    # if mod=='query':
                    #     pretrained_state_dict = pretrained_model.model.q_model.load_state_dict(new_state_dict, strict=False)
                    # elif mod=='doc':
                    #     pretrained_state_dict = pretrained_model.model.doc_model.load_state_dict(new_state_dict, strict=False)
                pretrained_model.model.model.load_state_dict(new_state_dict, strict=False)
            return pretrained_model, new_state_dict


class TaskVectorMinilm():
        def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, ablation = None, layer = None):
            """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
            This can either be done by passing two state dicts (one corresponding to the
            pretrained model, and another to the finetuned model), or by directly passying in
            the task vector state dict.
            """
            self.ablation = ablation
            self.layer = layer
            if vector is not None:
                self.vector = vector
            else:
                assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
                with torch.no_grad():
                    pretrained_state_dict = AutoModel.from_pretrained(pretrained_checkpoint).state_dict()
                    finetuned_state_dict = AutoModel.from_pretrained(finetuned_checkpoint).state_dict()
                    
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
                pretrained_model = models.SentenceBERT(pretrained_checkpoint)
                new_state_dict = {}
                for mod in ['query','doc']:
                    if mod=='query': 
                        pretrained_state_dict = pretrained_model.q_model.state_dict()
                        #print(pretrained_model.model.q_model)
                    elif mod=='doc':
                        pretrained_state_dict = pretrained_model.doc_model.state_dict()
                    for key in list(pretrained_state_dict.keys()):
                        if key.replace('0.auto_model.','') not in self.vector:
                            print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                            continue
                        if self.ablation is not None:
                            if self.ablation=='add':
                                if self.layer in key.replace('0.auto_model.',''):
                                    new_state_dict[key] = pretrained_state_dict[key].cuda() + scaling_coef * self.vector[key.replace('0.auto_model.','')].cuda()
                                else:
                                    new_state_dict[key] = pretrained_state_dict[key].cuda()
                            elif self.ablation=='remove':
                                if self.layer in key:
                                    new_state_dict[key] = pretrained_state_dict[key].cuda()
                                else:
                                    new_state_dict[key] = pretrained_state_dict[key].cuda() + scaling_coef * self.vector[key.replace('0.auto_model.','')].cuda()
                        else:
                            new_state_dict[key] = pretrained_state_dict[key].cuda() + scaling_coef * self.vector[key].cuda()
                        #new_state_dict[key] = pretrained_state_dict[key].cuda() + scaling_coef * self.vector[key.replace('0.auto_model.','')].cuda()
                    if mod=='query':
                        pretrained_state_dict = pretrained_model.q_model.load_state_dict(new_state_dict, strict=False)
                    elif mod=='doc':
                        pretrained_state_dict = pretrained_model.doc_model.load_state_dict(new_state_dict, strict=False)
                    #pretrained_model.model.model.load_state_dict(new_state_dict, strict=False)
            return pretrained_model, new_state_dict

class RerankBert:
    
    def __init__(self, model, batch_size: int = 128, **kwargs):
        self.cross_encoder = model
        self.batch_size = batch_size
        self.rerank_results = {}
        
    def rerank(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:
        
        sentence_pairs, pair_ids = [], []
        
        for query_id in results:
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])
            
            else:
                for doc_id in results[query_id]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])

        #### Starting to Rerank using cross-attention
        logging.info("Starting To Rerank Top-{}....".format(top_k))
        res = self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)
        rerank_scores = []
        for r in res:
            try:
                s = r[1]
            except:
                s = r
            rerank_scores.append(float(s))

        #### Reranking results
        self.rerank_results = {query_id: {} for query_id in results}
        for pair, score in zip(pair_ids, rerank_scores):
            query_id, doc_id = pair[0], pair[1]
            self.rerank_results[query_id][doc_id] = score

        return self.rerank_results 


class RerankT5:
    
    def __init__(self, model, batch_size: int = 128, **kwargs):
        self.cross_encoder = model
        self.batch_size = batch_size
        self.rerank_results = {}
        
    def rerank(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:
        
        sentence_pairs, pair_ids = [], []
        
        for query_id in results:
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])
            
            else:
                for doc_id in results[query_id]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])

        #### Starting to Rerank using cross-attention
        logging.info("Starting To Rerank Top-{}....".format(top_k))
        rerank_scores = [float(score) for score in self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)]

        #### Reranking results
        self.rerank_results = {query_id: {} for query_id in results}
        for pair, score in zip(pair_ids, rerank_scores):
            query_id, doc_id = pair[0], pair[1]
            self.rerank_results[query_id][doc_id] = score

        return self.rerank_results 
