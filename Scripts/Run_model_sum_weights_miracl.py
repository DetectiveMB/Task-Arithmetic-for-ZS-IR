# Run_model_sum_weights_miracl
import datasets
from utils import seed_everything, TaskVectorT5, TaskVectorBERT, RerankT5, RerankBert
import json
import logging
import os
import click
import torch
import tqdm
import pathlib
import beir
#from beir import util, LoggingHandler
#from beir.datasets.data_loader import GenericDataLoader
from beir.reranking import Rerank
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from beir.retrieval import models
from ranx import compare
from ranx import Qrels, Run, compare, fuse, optimize_fusion
from monot5 import MonoT5


def applyBM25(language, test_corpus, test_queries, name = 'test'):

    hostname = "localhost" #localhost
    index_name = 'miracl_'+language
    initialize = True # False

    number_of_shards = 1

    model = BM25(index_name=index_name, hostname=hostname, language=language, initialize=initialize, number_of_shards=number_of_shards)

    retriever = EvaluateRetrieval(model)

    results_bm25_test = retriever.retrieve(test_corpus, test_queries)
    #with open('./bm25_'+name+'_'+dataset+'.json', 'w') as f:
    #    json.dump(results_bm25_test, f)

    return results_bm25_test


def apply_reranking(model_name, device, model_base_path, model_vector_minus_path, model_vector_plus_path, test_corpus, test_queries, results_bm25_test, coef, name = 'test'):
    
    if 't5' in model_name:
        token_false='▁no'
        token_true='▁yes'
            
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
        reranker = RerankBert(cross_encoder_model, batch_size=128)

    # if 'MiniLM' in model_name:

    #     task_vector = TaskVectorBERT(model_vector_minus_path, model_vector_plus_path)

    #     base_model, new_state_dict = task_vector.apply_to(model_base_path, scaling_coef=coef)

    #     base_model.model.model.to(device)

    #     cross_encoder_model = base_model

    #     reranker = Rerank(cross_encoder_model, batch_size=128)

    test_rerank_results = reranker.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)

    #with open('./'+model_name+'_'+name+'_'+dataset+'rerank.json', 'w') as f:
    #    json.dump(test_rerank_results, f)

    return test_rerank_results

    

@click.command()
@click.option(
    "--language",
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
    default='cuda:0'
)
@click.option(
    "--seed",
    type=int,
    default=42
)
@click.option(
    "--alfa",
    type=float,
    default=0.5
)




def main(language, output_folder, alfa, model_name, device, model_base_path, model_vector_minus_path, model_vector_plus_path, seed):
    if seed:
        seed_everything(seed)

    #os.makedirs('../logs', exist_ok=True)
    logging.basicConfig(filename='./Answer_fusion_'+model_name+'_miracle_'+language+'.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )

    logger = logging.getLogger('main')
    torch.cuda.set_device(device)

    logger.info(f'Loading dataset miracl in: {language}')

    if language=='spanish':
        lang='es'  # or any of the 16 languages
    if language=='french':
        lang='fr'
    if language=='english':
        lang='en'

    miracl = datasets.load_dataset('miracl/miracl', lang)
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

    test_corpus = {}

    for el in miracl_corpus['train']:
        test_corpus[el['docid']] = {'title': el['title'],'text': el['text']}


    results_bm25_test = applyBM25(language, test_corpus, test_queries, name = 'test')

    bm25_run_test = Run(results_bm25_test, name='BM25')

    lambda_weight_values = [0,alfa]   
    
    lista_results_combined = []

    logging.info('Starting cycle over all lambda values')

    for coef in lambda_weight_values:
        logging.info(f'Value of lambda: {coef}')

        all_best_params=[{'weights': (0.5, 0.5)}]

        test_rerank_results = apply_reranking(model_name, device, model_base_path, model_vector_minus_path, model_vector_plus_path, test_corpus, test_queries, results_bm25_test, coef, 'test')

        test_t5_rels = Run(test_rerank_results, name = model_name+'-sum-lambda-'+str(coef))

        all_combined_test_run = fuse(
        runs=[bm25_run_test, test_t5_rels],
        norm="min-max",
        method="wsum",
        params=all_best_params[0],
    )
        all_combined_test_run.name = 'BM25 +' + model_name+'-sum-lambda-'+str(coef)

        lista_results_combined.append(all_combined_test_run)


    logging.info('Running baselines, i.e. models minus and plus, and BM25')

    for path in [model_vector_minus_path,model_vector_plus_path]:
        if 't5' in model_name:
            token_false='▁no'
            token_true='▁yes'

                
            cross_encoder_model = MonoT5(path, token_false=token_false, token_true=token_true)

            reranker = RerankT5(cross_encoder_model, batch_size=128)
        
        if 'BERT' in model_name:
            cross_encoder_model = CrossEncoder(path)
            reranker = RerankBert(cross_encoder_model, batch_size=128)

        all_best_params=[{'weights': (0.5, 0.5)}]


        test_rerank_results = reranker.rerank(test_corpus, test_queries, results_bm25_test, top_k=100)

        if path==model_vector_minus_path:
            base_version = '_original_version'
        if path==model_vector_plus_path:
            base_version = '_domain_specific'

        test_t5_rels = Run(test_rerank_results, name = model_name + base_version)

        all_combined_test_run = fuse(
        runs=[bm25_run_test, test_t5_rels],
        norm="min-max",
        method="wsum",
        params=all_best_params[0])

        
        all_combined_test_run.name = 'BM25 +' + model_name + base_version

        lista_results_combined.append(all_combined_test_run)



    lista_results_combined.append(bm25_run_test)

    test_qrels_correct = {q: test_qrels[q] for q in results_bm25_test.keys()}
        
    test_qrels = Qrels(test_qrels_correct)  

    report = compare(
            qrels=test_qrels,
            runs=lista_results_combined,
            metrics=['precision@10', 'ndcg@3', 'ndcg@10', 'recall@100', 'map@100'],
            max_p=0.01/4  # P-value threshold, 4 tests: con BM25, base model, base retrieval (i.e. lambda coef = 0), base domain specific
        )
    logger.info('Final Report with dataset miracl in language:')
    logger.info(f'\n{language}')
    logger.info('Final Report with models:')
    logger.info(f'\n{model_name}')
    logger.info(f'\n{model_base_path}')
    logger.info(f'\n{model_vector_minus_path}')
    logger.info(f'\n{model_vector_plus_path}')


    logger.info(f'\n{report}')

    print(report)


if __name__ == '__main__':
    main()

