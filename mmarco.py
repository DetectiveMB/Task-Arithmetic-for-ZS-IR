# mmarco

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from operator import itemgetter
#from beir.reranking import Rerank
import pathlib, os
import datetime
import logging
from datasets import load_dataset
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
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import json
import logging
random.seed(42)

logger = logging.getLogger(__name__)

logging.basicConfig(filename='answer_fusion_sum_t5_base_italiano_trec_covid_transl.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO
                    )

logger = logging.getLogger('main')

# dataset = load_dataset('unicamp-dl/mmarco', 'italian', trust_remote_code = True)

# query = load_dataset('unicamp-dl/mmarco', 'queries-italian', trust_remote_code = True)

# corpus = []
# q = []

# for i in range(5):
#     corpus.append(dataset['train'][i]['positive'])
#     corpus.append(dataset['train'][i]['negative'])
#     q.append(dataset['train'][i]['query'])

    #### /print debug information to stdout

    

    #### Download dbpedia-entity.zip dataset and unzip the dataset

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format('nfcorpus')
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

test_corpus, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")

test_translated_corpus = {}

test_translated_queries = {}

device    = torch.device('cuda:1')
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-it')
model     = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-it').to(device).eval()

######## Corpus

# for d in tqdm(test_corpus.keys()):

#     text = test_corpus[d]['text']
#     title = test_corpus[d]['title']

#     tokenized_text = tokenizer.prepare_seq2seq_batch(
#                 text, 
#                 max_length=512,
#                 return_tensors="pt")

#     tokenized_title = tokenizer.prepare_seq2seq_batch(
#                 title, 
#                 max_length=512,
#                 return_tensors="pt")

#     with torch.no_grad():
#         translated = model.generate(
#                     input_ids=tokenized_text['input_ids'].to(device), 
#                     max_length=512, 
#                     num_beams=5,
#                     do_sample=False)

#         translated_documents = tokenizer.batch_decode(
#                     translated, 
#                     skip_special_tokens=True)

#         translated = model.generate(
#                     input_ids=tokenized_title['input_ids'].to(device), 
#                     max_length=512, 
#                     num_beams=5,
#                     do_sample=False)

#         translated_title = tokenizer.batch_decode(
#                     translated, 
#                     skip_special_tokens=True)




#     test_translated_corpus[d] = {'text':translated_documents, 'title':translated_title}
   

#with open('./nfcorpus_corpus_italian.json', 'w') as f:
#            json.dump(test_translated_corpus, f)

with open('./nfcorpus_corpus_italian.json') as f_in:
    test_translated_corpus = json.load(f_in)

################### Query

# for d in tqdm(test_queries.keys()):

#     text = test_queries[d]
#     #title = test_corpus[d]['title']

#     tokenized_text = tokenizer.prepare_seq2seq_batch(
#                 text, 
#                 max_length=512,
#                 return_tensors="pt")

#     with torch.no_grad():
#         translated = model.generate(
#                     input_ids=tokenized_text['input_ids'].to(device), 
#                     max_length=512, 
#                     num_beams=5,
#                     do_sample=False)

#         translated_documents = tokenizer.batch_decode(
#                     translated, 
#                     skip_special_tokens=True)




#     test_translated_queries[d] = translated_documents


# with open('./nfcorpus_queries_italian.json', 'w') as f:
#             json.dump(test_translated_queries, f)

with open('./nfcorpus_queries_italian.json') as f_in:
    test_translated_queries = json.load(f_in)

################################## retrieval

hostname = "localhost" #localhost
index_name = 'trec-covid'
initialize = True # False

language = "italian"

#extract(base_model_path = 'google-t5/t5-base', chat_model_path='razent/SciFive-base-Pubmed_PMC', output_path = './', l1=1)

#sum_model = AutoModel.from_pretrained('castorini/monot5-base-msmarco')

#sum_model = add_chat_vector(sum_model, './', 1)

lista_results_combined = []

number_of_shards = 1
model = BM25(index_name=index_name, hostname=hostname, language=language, initialize=initialize, number_of_shards=number_of_shards)
retriever = EvaluateRetrieval(model)

results_bm25_test = retriever.retrieve(test_translated_corpus, test_translated_queries)

for coef in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5]:    # 0.1,0.2,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9, 1.5

    cross_encoder_model = MonoT5('unicamp-dl/mt5-base-en-msmarco', token_false='▁no', token_true='▁yes')

    task_vector = TaskVector('airKlizz/mt5-base-wikinewssum-english', 'airKlizz/mt5-base-wikinewssum-italian')  #     airKlizz/mt5-base-wikinewssum-english

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

    test_rerank_results = reranker.rerank(test_translated_corpus, test_translated_queries, results_bm25_test, top_k=100)

    bm25_run_test = Run(results_bm25_test, name='BM25')

    #test_qrels_correct = {q: test_qrels[q] for q in results_bm25_test.keys()}
    
    #test_qrels = Qrels(test_qrels_correct)  

    #with open('./t5_large_test_trec_covid.json', 'w') as f:
    #    json.dump(test_rerank_results, f)

    test_t5_rels = Run(test_rerank_results, name = 'IT5-sum-lambda-'+str(coef))

    all_combined_test_run = fuse(
        runs=[bm25_run_test, test_t5_rels],
        norm="min-max",
        method="wsum",
        params={'weights': (0.5, 0.5)},
    )
    all_combined_test_run.name = 'BM25 + IT5-sum-lambda-' + str(coef)

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

for coef in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5]:    # 0.1,0.2,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9, 1.5

    cross_encoder_model = MonoT5('unicamp-dl/mt5-base-en-msmarco', token_false='▁no', token_true='▁yes')

    task_vector = TaskVector('google/mt5-base', 'airKlizz/mt5-base-wikinewssum-italian')  #     airKlizz/mt5-base-wikinewssum-english

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

    test_rerank_results = reranker.rerank(test_translated_corpus, test_translated_queries, results_bm25_test, top_k=100)

    bm25_run_test = Run(results_bm25_test, name='BM25')

    #test_qrels_correct = {q: test_qrels[q] for q in results_bm25_test.keys()}
    
    #test_qrels = Qrels(test_qrels_correct)  

    #with open('./t5_large_test_trec_covid.json', 'w') as f:
    #    json.dump(test_rerank_results, f)

    test_t5_rels = Run(test_rerank_results, name = 'IT5-sum-lambda-'+str(coef))

    all_combined_test_run = fuse(
        runs=[bm25_run_test, test_t5_rels],
        norm="min-max",
        method="wsum",
        params={'weights': (0.5, 0.5)},
    )
    all_combined_test_run.name = 'BM25 + IT5-sum-lambda-' + str(coef)

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