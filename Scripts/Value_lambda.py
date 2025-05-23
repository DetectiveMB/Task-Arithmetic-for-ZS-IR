# Value_lambda

from utils import seed_everything, TaskVectorT5, TaskVectorBERT, RerankT5, TaskVectorLLama, RerankBert
import json
from llama_ir import LLamaRank
import logging
import os
import click
import torch
import tqdm
import pathlib
import beir
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.reranking import Rerank
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from beir.retrieval import models
from ranx import compare
from ranx import Qrels, Run, compare, fuse, optimize_fusion
from monot5 import MonoT5


def applyBM25(dataset, test_corpus, test_queries, name = 'test'):

    hostname = "localhost" #localhost
    index_name = dataset
    initialize = True # False

    number_of_shards = 1

    if dataset=='germanquad':
        language = "german"
        model = BM25(index_name=index_name, hostname=hostname, language=language, initialize=initialize, number_of_shards=number_of_shards)
    elif dataset in ['trec-covid', 'scidocs', 'dbpedia-entity']:
        model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    else:
        model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

    retriever = EvaluateRetrieval(model)

    results_bm25_test = retriever.retrieve(test_corpus, test_queries)
    #with open('./bm25_'+name+'_'+dataset+'.json', 'w') as f:
    #    json.dump(results_bm25_test, f)

    return results_bm25_test


def apply_reranking(dataset, model_name, device, model_base_path, model_vector_minus_path, model_vector_plus_path, test_corpus, test_queries, results_bm25_test, coef, name = 'test'):
    
    if 't5' in model_name:
        if (dataset=='germanquad') or ('mt5' in model_name):
            token_false='▁no'
            token_true='▁yes'
        else:
            token_false='▁false'
            token_true='▁true'
                
        cross_encoder_model = MonoT5(model_base_path, token_false=token_false, token_true=token_true)

        task_vector = TaskVectorT5(model_vector_minus_path, model_vector_plus_path)

        sum_model, new_state_dict = task_vector.apply_to(model_base_path, scaling_coef=coef)

        sum_model.to(device)

        cross_encoder_model.model.load_state_dict(new_state_dict, strict=False)

        reranker = RerankT5(cross_encoder_model, batch_size=128)
        
    if 'BERT' in model_name:

        task_vector = TaskVectorBERT(model_vector_minus_path, model_vector_plus_path)

        base_model, new_state_dict = task_vector.apply_to(model_base_path, scaling_coef=coef)

        #base_model.model.doc_model.to(device)
        #base_model.model.q_model.to(device)

        cross_encoder_model = base_model

        #reranker = EvaluateRetrieval(base_model, score_function="cos_sim", k_values=[1,3,5,10,100])
        #rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=100)

        #print(rerank_results)

        reranker = RerankBert(cross_encoder_model, batch_size=128)

    if 'Llama' in model_name:

        cross_encoder_model = LLamaRank(model_base_path, device = device)

        task_vector = TaskVectorLLama(model_vector_minus_path, model_vector_plus_path)

        sum_model, new_state_dict = task_vector.apply_to(model_base_path, scaling_coef=coef)

        cross_encoder_model.model.load_state_dict(new_state_dict, strict=False)

        reranker = Rerank(cross_encoder_model, batch_size=128)

    
    test_rerank_results = reranker.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)

    #with open('./'+model_name+'_'+name+'_'+dataset+'rerank.json', 'w') as f:
    #    json.dump(test_rerank_results, f)

    return test_rerank_results

    

@click.command()
@click.option(
    "--dataset",
    type=str,
    required=True
)
@click.option(
    "--output_folder",
    type=str,
    default = './Risultati'
)
@click.option(
    "--model_name",
    type=str,
    required=True
)
@click.option(
    "--model_base_path",
    type=str,
    required=True
)
@click.option(
    "--model_vector_plus_path",
    type=str,
    required=True
)
@click.option(
    "--model_vector_minus_path",
    type=str,
    required=True
)
@click.option(
    "--device",
    type=str,
    required=True,
    default='cuda:1'
)
@click.option(
    "--seed",
    type=int,
    default=42
)
# @click.option(
#     "--until",
#     type=bool,
#     default=True
# )




def main(dataset, output_folder, model_name, device, model_base_path, model_vector_minus_path, model_vector_plus_path, seed):
    if seed:
        seed_everything(seed)

    #os.makedirs('../logs', exist_ok=True)
    logging.basicConfig(filename='./Ablation_lambda_long_fusion_'+model_name+'_'+dataset+'.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )

    logger = logging.getLogger('main')
    torch.cuda.set_device(device)

    logger.info(f'Loading dataset: {dataset}')

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)


    #test_corpus, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")

    assert dataset in ['scifact', 'nfcorpus', 'fiqa', 'dbpedia-entity','fever'], "model in BEIR without VALIDATION SET"

    if dataset=='scifact':

        val_corpus, val_queries, val_qrels = GenericDataLoader(data_path).load(split="train")

        val_queries = {q: val_queries[q] for q in list(val_queries.keys())[:int(20*len(val_queries)/100)]} # 

        val_qrels = {q: val_qrels[q] for q in list(val_queries.keys())}

    else:

        val_corpus, val_queries, val_qrels = GenericDataLoader(data_path).load(split="dev")

    results_bm25_val = applyBM25(dataset, val_corpus, val_queries, name = 'val')
    bm25_run_val = Run(results_bm25_val, name='BM25')
    
    lambda_weight_values = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]  

    lista_results_combined = []

    logging.info('Starting cycle over all lambda values')

    val_qrels_correct = {q: val_qrels[q] for q in results_bm25_val.keys()}
        
    val_qrels = Qrels(val_qrels_correct)  

    all_best_params=[{'weights': (0.5, 0.5)}]

    for coef in lambda_weight_values:
        logging.info(f'Value of lambda: {coef}')
        val_rerank_results  = apply_reranking(dataset, model_name, device, model_base_path, model_vector_minus_path, model_vector_plus_path, val_corpus, val_queries, results_bm25_val, coef, 'val')
        val_t5_rels = Run(val_rerank_results, name = model_name+'-sum-lambda-'+str(coef))
        #logger.info(coef)
        #ndcg, _map, recall, precision, hole = dense_retriever.evaluate(val_qrels, val_rerank_results,  k_values=[1,3,5,10,100])
        #test_rerank_results = apply_reranking(dataset, model_name, device, model_base_path, model_vector_minus_path, model_vector_plus_path, test_corpus, test_queries, results_bm25_test, coef, 'test')

        #test_t5_rels = Run(test_rerank_results, name = model_name+'-sum-lambda-'+str(coef))
        
        all_combined_test_run = fuse(
        runs=[bm25_run_val, val_t5_rels],
        norm="min-max",
        method="wsum",
        params=all_best_params[0],
    )
        all_combined_test_run.name = 'BM25 +' + model_name+'-sum-lambda-'+str(coef)

        lista_results_combined.append(all_combined_test_run)

        #logger.info(f'\n{all_combined_test_run}')

################################################# Baselines ############################################################################

# bm25_run_val
    lista_results_combined.append(bm25_run_val)

    report = compare(
            qrels=val_qrels,
            runs=lista_results_combined,
            metrics=['precision@10', 'ndcg@3', 'ndcg@10', 'recall@100', 'map@100'],
            max_p=0.01/3  # P-value threshold, 18 test, rispetto agli altri
        )

    logger.info('Final Report with dataset:')
    logger.info(f'\n{dataset}')
    logger.info('Final Report with models:')
    logger.info(f'\n{model_name}')
    logger.info(f'\n{model_base_path}')
    logger.info(f'\n{model_vector_minus_path}')
    logger.info(f'\n{model_vector_plus_path}')


    logger.info(f'\n{report}')

    print(report)


if __name__ == '__main__':
    main()