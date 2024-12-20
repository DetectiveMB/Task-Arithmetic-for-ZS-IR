from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
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
from monot5 import MonoT5
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

        def apply_to(self, pretrained_checkpoint, layer, encoder, scaling_coef=1.0):
            """Apply a task vector to a pretrained model."""
            with torch.no_grad():
                pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_checkpoint)
                new_state_dict = {}
                pretrained_state_dict = pretrained_model.state_dict()
                for key in list(pretrained_state_dict.keys()):
                    if key not in self.vector:
                        print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                        continue
                    for l in layer:
                        for submodel in encoder:
                            if (str(l)+'.layer' in key) and (submodel in key):
                                new_state_dict[key] = pretrained_state_dict[key].cuda() + scaling_coef * self.vector[key].cuda()
                            else:
                                new_state_dict[key] = pretrained_state_dict[key].cuda()

            pretrained_model.load_state_dict(new_state_dict, strict=False)
            return pretrained_model, new_state_dict
        
df_name = 'scifact' #
dataset = df_name

logging.basicConfig(filename='answer_fusion_sum_t5_ablation'+dataset+'.log',
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

    
test_corpus, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")


hostname = "localhost" #localhost
index_name = dataset
initialize = True # False

coef = 0.7


        ######### Validation ##############################################################################

model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model)

    #### Retrieve dense results (format of results is identical to qrels)

        #base_model = sentence_models.Transformer('sentence-transformers/msmarco-distilbert-base-dot-prod-v3')

cross_encoder_model = MonoT5('castorini/monot5-base-msmarco', token_false='▁false', token_true='▁true')

task_vector = TaskVector('google-t5/t5-base', 'razent/SciFive-base-Pubmed_PMC')



for submodel in [['encoder'],['decoder'],['encoder','decoder']]:

    lista_results_combined = []

    logger.info(f'\nWe focus on the submodel: {submodel[0]}')

    logger.info('\nParto a rimuove da ultimo: prima ci sono tutti e 12 i layer, poi ci sono primi 11 e rimuovo ultimo ecc')

    for l in range(12):

        sum_model, new_state_dict = task_vector.apply_to('castorini/monot5-base-msmarco', scaling_coef=coef, layer = list(range(12-l)), encoder = submodel)

        sum_model.to('cuda')

        cross_encoder_model.model.load_state_dict(new_state_dict, strict=False)



        rerankerb = Rerank(cross_encoder_model, batch_size=128)

# Rerank top-100 results using the reranker provided


        modelbm = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
        retriever = EvaluateRetrieval(modelbm)

        results_bm25_test = retriever.retrieve(test_corpus, test_queries)

        bm25_run_test = Run(results_bm25_test, name='BM25')

        retrieverb = Rerank(cross_encoder_model, batch_size=128)
        test_rerank_results = retrieverb.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)


        test_t5_rels = Run(test_rerank_results, name = 'T5-sommo_fino_layer-'+str(12-l))


        all_combined_test_run = fuse(
runs=[bm25_run_test, test_t5_rels],
norm="min-max",
method="wsum",
params={'weights': (0.5, 0.5)},
)
        all_combined_test_run.name = 'BM25 + T5-sommo_fino_layer-'+str(12-l)

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



####################################################à

    lista_results_combined = []


    logger.info(f'\n{submodel[0]}')

    logger.info('\nParto ad aggiungere dal primo: prima sommo il layer zero, poi fino a uno, poi fino al secondo layer ecc')

    for l in range(12):

        sum_model, new_state_dict = task_vector.apply_to('castorini/monot5-base-msmarco', scaling_coef=coef, layer = list(range(l)), encoder = submodel)

        sum_model.to('cuda')

        cross_encoder_model.model.load_state_dict(new_state_dict, strict=False)

        rerankerb = Rerank(cross_encoder_model, batch_size=128)

    # Rerank top-100 results using the reranker provided


        modelbm = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
        retriever = EvaluateRetrieval(modelbm)

        results_bm25_test = retriever.retrieve(test_corpus, test_queries)

        bm25_run_test = Run(results_bm25_test, name='BM25')

        retrieverb = Rerank(cross_encoder_model, batch_size=128)
        test_rerank_results = retrieverb.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)


        test_t5_rels = Run(test_rerank_results, name = 'T5-sum-sommo_fino_layer-'+str(l))


        all_combined_test_run = fuse(
    runs=[bm25_run_test, test_t5_rels],
    norm="min-max",
    method="wsum",
    params={'weights': (0.5, 0.5)},
    )
        all_combined_test_run.name = 'BM25 + T5-sum-sommo_fino_layer-'+str(l)

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


    ###############################################################################################

    lista_results_combined = []

    logger.info(f'\n{submodel[0]}')

    logger.info('\nParto ad aggiungere dal primo, sommo sempre un layer: prima sommo il layer zero, poi solo layer uno, poi solo layer due ecc')

    for l in range(12):

        sum_model, new_state_dict = task_vector.apply_to('castorini/monot5-base-msmarco', scaling_coef=coef, layer = [l], encoder = submodel)

        sum_model.to('cuda')

        cross_encoder_model.model.load_state_dict(new_state_dict, strict=False)

        rerankerb = Rerank(cross_encoder_model, batch_size=128)

    # Rerank top-100 results using the reranker provided


        modelbm = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
        retriever = EvaluateRetrieval(modelbm)

        results_bm25_test = retriever.retrieve(test_corpus, test_queries)

        bm25_run_test = Run(results_bm25_test, name='BM25')

        retrieverb = Rerank(cross_encoder_model, batch_size=128)
        test_rerank_results = retrieverb.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)


        test_t5_rels = Run(test_rerank_results, name = 'T5-sum-sommo_solo_layer-'+str(l))


        all_combined_test_run = fuse(
    runs=[bm25_run_test, test_t5_rels],
    norm="min-max",
    method="wsum",
    params={'weights': (0.5, 0.5)},
    )
        all_combined_test_run.name = 'BM25 + T5-sum-sommo_solo_layer-'+str(l)

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