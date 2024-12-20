# biobert-reranking

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
from transformers import AutoModelForSeq2SeqLM
random.seed(42)

torch.cuda.set_device('cuda:0')

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
        def apply_to(self, pretrained_checkpoint, scaling_coef=1.0, coef_ir =1.0):
            """Apply a task vector to a pretrained model."""
            with torch.no_grad():
                pretrained_model = CrossEncoder(pretrained_checkpoint)
                new_state_dict = {}
                pretrained_state_dict = pretrained_model.model.model.state_dict()
                for key in list(pretrained_state_dict.keys()):
                    if key.replace('bert.','') not in self.vector:
                        print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                        continue
                    new_state_dict[key] = coef_ir*pretrained_state_dict[key].cuda() + scaling_coef * self.vector[key.replace('bert.','')].cuda()
            pretrained_model.model.model.load_state_dict(new_state_dict, strict=False)
            return pretrained_model, new_state_dict
        
for df_name in ['scifact','nfcorpus']: #
    dataset = df_name

    logging.basicConfig(filename='answer_fusion_sum_bert_'+dataset+'.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )

    logger = logging.getLogger('main')
    #### /print debug information to stdout


    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)


    if dataset=='scifact':

        val_corpus, val_queries, val_qrels = GenericDataLoader(data_path).load(split="train")

        val_queries = {q: val_queries[q] for q in list(val_queries.keys())[:int(20*len(val_queries)/100)]}

        val_qrels = {q: val_qrels[q] for q in list(val_queries.keys())}

    if dataset=='nfcorpus':

        val_corpus, val_queries, val_qrels = GenericDataLoader(data_path).load(split="dev")
        
    test_corpus, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")


    hostname = "localhost" #localhost
    index_name = dataset
    initialize = True # False
    lista_results_combined = []

    for coef in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:  #,
        

    ######### Validation ##############################################################################

        model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
        retriever = EvaluateRetrieval(model)

    #### Retrieve dense results (format of results is identical to qrels)
        results_bm25_val = retriever.retrieve(val_corpus, val_queries)

        bm25_run_val = Run(results_bm25_val, name='BM25')

        #base_model = sentence_models.Transformer('sentence-transformers/msmarco-distilbert-base-dot-prod-v3')

        task_vector = TaskVector('sentence-transformers/all-MiniLM-L6-v2', 'menadsa/BioS-MiniLM')

        base_model, new_state_dict = task_vector.apply_to('cross-encoder/ms-marco-MiniLM-L-6-v2', scaling_coef=coef, coef_ir = 1)

        base_model.model.model.to('cuda')

        #base_model.load_state_dict(new_state_dict)


        #model = DRES(models.SentenceBERT("sentence-transformers/msmarco-distilbert-base-dot-prod-v3"), batch_size=128)
        #base_model.model.load_state_dict(new_state_dict, strict=False)

        #pooling_model = sentence_models.Pooling(768, 'mean')
        #s_model = SentenceTransformer(modules=[base_model, pooling_model])

        #modelb = models.SentenceBERT('sentence-transformers/msmarco-distilbert-base-dot-prod-v3')
        #modelb.doc_model = s_model
        #modelb.q_model = s_model
        #cross_encoder_model = CrossEncoder('nboost/pt-bert-base-uncased-msmarco')
        #cross_encoder_model.model.model = s_model


        # sebastian-hofstaetter/distilbert-cat-margin_mse-T2-msmarco
        #modelb = DRES(modelb, batch_size=128)
        #retrieverb = Rerank(cross_encoder_model, batch_size=128) 

        cross_encoder_model = base_model
        #cross_encoder_model = CrossEncoder('nboost/pt-bert-base-uncased-msmarco')
        #cross_encoder_model.model.model = s_model

#### Or use MiniLM, TinyBERT etc. CE models (https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)
# cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

        rerankerb = Rerank(cross_encoder_model, batch_size=128)

# Rerank top-100 results using the reranker provided
        
        try:

            val_rerank_results = rerankerb.rerank(val_corpus, val_queries, results_bm25_val, top_k=100)
        except:
            import ipdb
            ipdb.set_trace()


        val_t5_rels = Run(val_rerank_results, name = 'MiniLM-sum-lambda_kn-'+str(coef)+'-lambda-ir-'+str(coef_ir))

        val_qrels_correct = {q: val_qrels[q] for q in results_bm25_val.keys()}
        
        val_qrels = Qrels(val_qrels_correct) 

        try: 

            all_best_params = optimize_fusion(
            qrels=val_qrels,
            runs=[bm25_run_val, val_t5_rels],
            norm="min-max",
            method="wsum",
            metric="ndcg@10",  # The metric to maximize during optimization
            return_optimization_report=True
        )
            
        except:
            import ipdb
            ipdb.set_trace()


        modelbm = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
        retriever = EvaluateRetrieval(modelbm)

        results_bm25_test = retriever.retrieve(test_corpus, test_queries)

        bm25_run_test = Run(results_bm25_test, name='BM25')

        retrieverb = Rerank(cross_encoder_model, batch_size=128)
        test_rerank_results = retrieverb.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)


        test_t5_rels = Run(test_rerank_results, name = 'MiniLM-sum-lambda-'+str(coef)+'-lambda-ir-'+str(coef_ir))

        all_combined_test_run = fuse(
            runs=[bm25_run_test, test_t5_rels],
            norm="min-max",
            method="wsum",
            params=all_best_params[0],
        )
        all_combined_test_run.name = 'BM25 + MiniLM-sum-lambda-' + str(coef) +'-lambda-ir-'+str(coef_ir)


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