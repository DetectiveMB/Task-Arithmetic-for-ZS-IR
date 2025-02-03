# run_model_sum_weights

from utils import seed_everything, TaskVectorT5, TaskVectorBERT, RerankT5, TaskVectorLLama, RerankBert, TaskVectorMinilm
import json
import logging
import os
import click
import torch
import tqdm
import pathlib
import beir
from beir import util, LoggingHandler
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.datasets.data_loader import GenericDataLoader
from beir.reranking import Rerank
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from beir.retrieval import models
from ranx import compare
from ranx import Qrels, Run, compare, fuse, optimize_fusion
from monot5 import MonoT5
from llama_ir import LLamaRank
from Sentence_bert import encode_corpus, encode_queries, cos_sim, biencoder_reranking


def applyBM25(dataset, test_corpus, test_queries, name = 'test'):

    hostname = "localhost" #localhost
    index_name = dataset
    initialize = True # False

    number_of_shards = 1

    if dataset=='germanquad':
        language = "german"
        model = BM25(index_name=index_name, hostname=hostname, language=language, initialize=initialize, number_of_shards=number_of_shards)
    elif dataset in ['trec-covid', 'scidocs']:
        model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    else:
        model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

    retriever = EvaluateRetrieval(model)

    results_bm25_test = retriever.retrieve(test_corpus, test_queries)
    #with open('./bm25_'+name+'_'+dataset+'.json', 'w') as f:
    #    json.dump(results_bm25_test, f)

    return results_bm25_test


def apply_reranking(dataset, model_name, device, model_base_path, model_vector_minus_path, model_vector_plus_path, test_corpus, test_queries, results_bm25_test, coef, mod, ablation, layer, name = 'test'):
    test_rerank_results = {}
    if 't5' in model_name:
        if dataset=='germanquad':
            token_false='▁no'
            token_true='▁yes'
        else:
            token_false='▁false'
            token_true='▁true'
                
        cross_encoder_model = MonoT5(model_base_path, token_false=token_false, token_true=token_true)

        task_vector = TaskVectorT5(model_vector_minus_path, model_vector_plus_path, ablation = ablation, layer = layer)

        sum_model, new_state_dict = task_vector.apply_to(model_base_path, scaling_coef=coef)

        sum_model.to(device)

        cross_encoder_model.model.load_state_dict(new_state_dict, strict=False)

        reranker = RerankT5(cross_encoder_model, batch_size=128)
        test_rerank_results = reranker.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)
        
    if 'BERT' in model_name:

        task_vector = TaskVectorBERT(model_vector_minus_path, model_vector_plus_path)

        base_model, new_state_dict = task_vector.apply_to(model_base_path, scaling_coef=coef)

        #base_model.model.doc_model.to(device)
        #base_model.model.q_model.to(device)

        cross_encoder_model = base_model
        reranker = RerankBert(cross_encoder_model, batch_size=128)
        test_rerank_results = reranker.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)

        #reranker = EvaluateRetrieval(cross_encoder_model, score_function="cos_sim")

    if 'MiniLM' in model_name:
        task_vector = TaskVectorMinilm(model_vector_minus_path, model_vector_plus_path, ablation = ablation, layer = layer)

        base_model, new_state_dict = task_vector.apply_to(model_base_path, scaling_coef=coef)

        base_model.doc_model.to(device)
        base_model.q_model.to(device)

        test_rerank_results = biencoder_reranking(base_model, test_corpus, test_queries, results_bm25_test, mod, top_k=100)


    if 'Llama' in model_name:

        cross_encoder_model = LLamaRank(model_base_path, device = device)

        task_vector = TaskVectorLLama(model_vector_minus_path, model_vector_plus_path, ablation = ablation, layer = layer)

        sum_model, new_state_dict = task_vector.apply_to('zyznull/RankingGPT-llama2-7b', scaling_coef=coef)

        cross_encoder_model.model.load_state_dict(new_state_dict, strict=False)

        reranker = Rerank(cross_encoder_model, batch_size=128)

        test_rerank_results = reranker.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)

    #with open('./'+model_name+'_'+name+'_'+dataset+'rerank.json', 'w') as f:
    #    json.dump(test_rerank_results, f)

    return test_rerank_results


def evaluate_baselines(dataset, path, model_name, device, model_base_path, model_vector_minus_path, model_vector_plus_path, test_corpus, test_queries, results_bm25_test, coef, mod, name = 'test'):
    if 't5' in model_name:
        if dataset=='germanquad':
            token_false='▁no'
            token_true='▁yes'
        else:
            token_false='▁false'
            token_true='▁true'
                
        cross_encoder_model = MonoT5(path, token_false=token_false, token_true=token_true)

        reranker = RerankT5(cross_encoder_model, batch_size=128)
        rerank_results = reranker.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)
    
    if 'Llama' in model_name:
        if path==model_vector_minus_path:
            cross_encoder_model = cross_encoder_model.model.to('cpu')
            logger.info('model to cpu')
            
        cross_encoder_model = LLamaRank(path, device=device)
        reranker = Rerank(cross_encoder_model, batch_size=128)
        rerank_results = reranker.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)

    if 'MiniLM' in model_name:
        base_model = models.SentenceBERT(path)
        base_model.doc_model.to(device)
        base_model.q_model.to(device)

        rerank_results = biencoder_reranking(base_model, test_corpus, test_queries, results_bm25_test, mod, top_k=100)
    
    return rerank_results
    

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
    "--alfa",
    type=float,
    required=True
)
@click.option(
    "--seed",
    type=int,
    default=42
)
@click.option(
    "--mod",
    type=str,
    default='dot'
)
@click.option(
    "--ablation",
    type=str,
    default='add'
)



def main(dataset, output_folder, model_name, device, alfa, model_base_path, model_vector_minus_path, model_vector_plus_path, seed, mod, ablation):
    if seed:
        seed_everything(seed)

    #os.makedirs('../logs', exist_ok=True)
    logging.basicConfig(filename='./Ablation_layer'+model_name+'_'+dataset+'.log',
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


    test_corpus, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")

    assert dataset in ['scifact', 'nfcorpus', 'trec-covid', 'scidocs', 'germanquad', 'climate-fever', 'fiqa', 'dbpedia-entity', 'climate-fever'], "model not in BEIR"

    if dataset=='scifact':

        val_corpus, val_queries, val_qrels = GenericDataLoader(data_path).load(split="train")

        val_queries = {q: val_queries[q] for q in list(val_queries.keys())[:int(20*len(val_queries)/100)]}

        val_qrels = {q: val_qrels[q] for q in list(val_queries.keys())}

    elif (dataset=='nfcorpus') or (dataset=='fiqa') or (dataset=='dbpedia-entity'):

        val_corpus, val_queries, val_qrels = GenericDataLoader(data_path).load(split="dev")


    else:
        val_corpus, val_queries, val_qrels = {},{},{}


    if dataset in ['scifact','nfcorpus','fiqa','dbpedia-entity']:
        results_bm25_val = applyBM25(dataset, val_corpus, val_queries, name = 'val')
        bm25_run_val = Run(results_bm25_val, name='BM25')
    else:
        results_bm25_val = {}


    results_bm25_test = applyBM25(dataset, test_corpus, test_queries, name = 'test')

    bm25_run_test = Run(results_bm25_test, name='BM25')

    lambda_weight_values = [alfa]  # 0 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1, lambda_weight_values

    coef = alfa
    
    lista_results_combined = []

    logging.info('Starting cycle over all lambda values')

    if ('t5' in model_name) or ('roberta' in model_name):
        num_layers = 12
    if 'Llama' in model_name:
        num_layers= 32

    test_qrels_correct = {q: test_qrels[q] for q in results_bm25_test.keys()}
        
    test_qrels = Qrels(test_qrels_correct) 

    dictionary_values = {'MiniLM_roberta':{'scifact':[{'weights': (0.6, 0.4)}],'nfcorpus':[{'weights': (0.8, 0.2)}]}, 
            't5_base':{'scifact':[{'weights': (0.4, 0.6)}],'nfcorpus':[{'weights': (0.7, 0.3)}]},
            'Llama-2-7b':{'scifact':[{'weights': (0.2, 0.8)}],'nfcorpus':[{'weights': (0.7, 0.3)}]}}
    

    for layer in list(range(num_layers)):
        logging.info(f'Value of layer: {str(layer)}')
        if (dataset in ['scifact','nfcorpus','fiqa','dbpedia-entity']):
            best_parameters = dictionary_values[model_name][dataset][0]
        else:
            best_parameters = [{'weights': (0.5, 0.5)}][0]

        # 'distil':{'scifact':[{'weights': (0.5, 0.5)}],'nfcorpus':[{'weights': (0.5, 0.5)}]},
        #     val_rerank_results = apply_reranking(dataset, model_name, device, model_base_path, model_vector_minus_path, model_vector_plus_path, val_corpus, val_queries, results_bm25_val, coef, mod, 'val')
        #     val_t5_rels = Run(val_rerank_results, name = model_name+'-sum-lambda-'+str(coef))
        #     val_qrels_correct = {q: val_qrels[q] for q in results_bm25_val.keys()}
        
        #     val_qrels = Qrels(val_qrels_correct)  

        #     all_best_params = optimize_fusion(
        #     qrels=val_qrels,
        #     runs=[bm25_run_val, val_t5_rels],
        #     norm="min-max",
        #     method="wsum",
        #     metric="ndcg@10",  # The metric to maximize during optimization
        #     return_optimization_report=True
        #     )

        #     logger.info(f'best parameters with all two: {all_best_params[0]}')
        #     logger.info(f'\n{all_best_params[1]}')
        
        # elif (dataset=='dbpedia_entity') and (model_name=='t5_base'):
        #     all_best_params=[{'weights': (0.2, 0.8)}]
        # else:
        #     all_best_params=[{'weights': (0.5, 0.5)}]
        test_rerank_results = apply_reranking(dataset, model_name, device, model_base_path, model_vector_minus_path, model_vector_plus_path, test_corpus, test_queries, results_bm25_test, coef, mod, ablation, str(layer), 'test')

        test_t5_rels = Run(test_rerank_results, name = model_name+'-layer-'+str(layer))

        all_combined_test_run = fuse(
        runs=[bm25_run_test, test_t5_rels],
        norm="min-max",
        method="wsum",
        params=best_parameters,
    )
        all_combined_test_run.name = 'BM25 +' + model_name+'-lambda_'+ablation+'_layer_'+str(layer)

        lista_results_combined.append(all_combined_test_run)

        if (layer%10==0) and ('Llama' in model_name):
            report = compare(
            qrels=test_qrels,
            runs=lista_results_combined,
            metrics=['ndcg@3', 'ndcg@10'],  #'precision@10',  'recall@100', 'map@100'
            max_p=0.01  # P-value threshold, 4 tests: con BM25, base model, base retrieval (i.e. lambda coef = 0), base domain specific
            )

            logger.info(f'\n{report}')

            print(report)

            lista_results_combined = []


################################################# Baselines ############################################################################
    # logging.info('Running baselines, i.e. models minus and plus, and BM25')

    # for path in [model_vector_plus_path, model_vector_minus_path]:  #model_vector_plus_path
    #     logger.info(f'\n Working with baseline: {path}')
    #     # if 't5' in model_name:
    #     #     if dataset=='germanquad':
    #     #         token_false='▁no'
    #     #         token_true='▁yes'
    #     #     else:
    #     #         token_false='▁false'
    #     #         token_true='▁true'
                
    #     #     cross_encoder_model = MonoT5(path, token_false=token_false, token_true=token_true)

    #     #     reranker = RerankT5(cross_encoder_model, batch_size=128)
    #     #     val_rerank_results = reranker.rerank(val_corpus, val_queries, results_bm25_val, top_k=100)

    #     # if 'BERT' in model_name:
    #     #     #task_vector = TaskVectorBERT(model_vector_minus_path, model_vector_plus_path)

    #     #     #base_model, new_state_dict = task_vector.apply_to(model_base_path, scaling_coef=coef)

    #     # #base_model.model.doc_model.to(device)
    #     # #base_model.model.q_model.to(device)
    #     #     cross_encoder_model = CrossEncoder(path, max_length=512)

    #     #     #cross_encoder_model = base_model

    #     # #reranker = EvaluateRetrieval(base_model, score_function="cos_sim", k_values=[1,3,5,10,100])
    #     # #rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=100)

    #     # #print(rerank_results)

    #     #     reranker = RerankBert(cross_encoder_model, batch_size=128)
    #     #     val_rerank_results = reranker.rerank(val_corpus, val_queries, results_bm25_val, top_k=100)

    #         #reranker = EvaluateRetrieval(cross_encoder_model, score_function="cos_sim")

    #     # if 'Llama' in model_name:
    #     #     if path==model_vector_minus_path:
    #     #         cross_encoder_model = cross_encoder_model.model.to('cpu')
    #     #         logger.info('model to cpu')
            
    #     #     cross_encoder_model = LLamaRank(path, device=device)
    #     #     reranker = Rerank(cross_encoder_model, batch_size=128)
    #     #     val_rerank_results = reranker.rerank(val_corpus, val_queries, results_bm25_val, top_k=100)

    #     if dataset in ['scifact','nfcorpus','fiqa','dbpedia-entity']:
    #         val_rerank_results = evaluate_baselines(dataset, path, model_name, device, model_base_path, model_vector_minus_path, model_vector_plus_path, val_corpus, val_queries, results_bm25_val, coef, mod, name = 'val')
    #         val_t5_rels = Run(val_rerank_results, name = model_name+'-sum-lambda-'+str(coef))
    #         val_qrels_correct = {q: val_qrels[q] for q in results_bm25_val.keys()}
        
    #         val_qrels = Qrels(val_qrels_correct)  

    #         all_best_params = optimize_fusion(
    #         qrels=val_qrels,
    #         runs=[bm25_run_val, val_t5_rels],
    #         norm="min-max",
    #         method="wsum",
    #         metric="ndcg@10",  # The metric to maximize during optimization
    #         return_optimization_report=True
    #         )

    #         logger.info(f'best parameters with all two: {all_best_params[0]}')
    #         logger.info(f'\n{all_best_params[1]}')

    #     else:
    #         all_best_params=[{'weights': (0.5, 0.5)}]

    #     test_rerank_results = evaluate_baselines(dataset, path, model_name, device, model_base_path, model_vector_minus_path, model_vector_plus_path, test_corpus, test_queries, results_bm25_test, coef, mod, name = 'test')
    #     #test_rerank_results = reranker.rerank(test_corpus, test_queries, results_bm25_test, mod, top_k=100)

    #     if path==model_vector_minus_path:
    #         base_version = '_original_version'
    #     if path==model_vector_plus_path:
    #         base_version = '_domain_specific'

    #     test_t5_rels = Run(test_rerank_results, name = model_name + base_version)

    #     all_combined_test_run = fuse(
    #     runs=[bm25_run_test, test_t5_rels],
    #     norm="min-max",
    #     method="wsum",
    #     params=all_best_params[0])

        
    #     all_combined_test_run.name = 'BM25 +' + model_name + base_version

    #     lista_results_combined.append(all_combined_test_run)


    # ########################### BM25    

    # lista_results_combined.append(bm25_run_test)

     

  
    logger.info('Final Report with dataset:')
    logger.info(f'\n{dataset}')
    logger.info('Final Report with models:')
    logger.info(f'\n{model_name}')
    logger.info(f'\n{model_base_path}')


    report = compare(
            qrels=test_qrels,
            runs=lista_results_combined,
            metrics=['ndcg@3', 'ndcg@10'],  #'precision@10',  'recall@100', 'map@100'
            max_p=0.01  # P-value threshold, 4 tests: con BM25, base model, base retrieval (i.e. lambda coef = 0), base domain specific
            )

    logger.info(f'\n{report}')

    print(report)
    # logger.info(f'\n{model_vector_minus_path}')
    # logger.info(f'\n{model_vector_plus_path}')





if __name__ == '__main__':
    main()